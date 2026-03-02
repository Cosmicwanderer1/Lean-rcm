"""
Expert Iteration 训练器
@author ygw
更新日期: 2026-02-28

第四阶段 (Phase 4) 的核心训练模块：
通过 MAGC-MCTS 自我博弈 (Self-Play) 采样成功的证明轨迹，
提取为高质量训练数据，迭代更新策略模型参数。

Expert Iteration 循环:
    for iteration in range(max_iterations):
        1. self_play_sampling: 用当前策略模型 + MAGC-MCTS 搜索大量定理
        2. trajectory_extraction: 从成功证明中提取 (state, thought, tactic) 三元组
        3. verification_filter: 通过 Pantograph 验证提取的轨迹
        4. sft_update: 在新数据上微调（重放 + 新数据混合）
        5. evaluation: 在验证集上评估 Pass@k

与 Phase 3 (SFT) 的关系:
    - Phase 3 = 从人工/教师标注数据学习基础能力
    - Phase 4 = 从自身搜索结果迭代提升
    - Expert Iteration 本质上是 "自持续改进" (Self-Improving)

技术栈:
    - trl.SFTTrainer + LoRA 进行微调
    - MAGC-MCTS 进行证明搜索
    - Pantograph 进行形式化验证
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer as TrlSFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.common.utils import (
    load_yaml, save_jsonl, load_jsonl, iter_jsonl,
    ensure_dir, set_seed, setup_logging
)

logger = logging.getLogger("expert_iteration")


# ================================================================
# 轨迹提取
# ================================================================

class TrajectoryExtractor:
    """
    从 MAGC-MCTS 搜索结果中提取训练轨迹

    每个成功的证明搜索路径被分解为:
        (state_before, thought, tactic, state_after) 四元组

    过滤策略:
        - 去除 sorry / admit 策略
        - 去除空策略 / 空状态
        - 去重（基于 state + tactic 的哈希）
        - 可选: 只保留最短路径轨迹

    参数:
        min_tactic_length: 策略文本的最小长度
        deduplicate: 是否对轨迹去重
        keep_shortest_only: 是否只保留最短证明路径
    """

    def __init__(self,
                 min_tactic_length: int = 2,
                 deduplicate: bool = True,
                 keep_shortest_only: bool = False):
        """
        初始化轨迹提取器

        参数:
            min_tactic_length: 策略文本最小长度
            deduplicate: 是否去重
            keep_shortest_only: 是否只保留最短路径
        """
        self.min_tactic_length = min_tactic_length
        self.deduplicate = deduplicate
        self.keep_shortest_only = keep_shortest_only
        self._seen_hashes = set()

    def extract_from_search_result(self,
                                   theorem_name: str,
                                   proof_tactics: List[str],
                                   states: List[str]) -> List[Dict[str, str]]:
        """
        从一次搜索结果中提取训练样本

        参数:
            theorem_name: 定理名称
            proof_tactics: 成功的策略序列
            states: 对应的状态序列（比 tactics 多一个初始状态）

        返回:
            List[Dict]: 提取的训练样本列表
        """
        samples = []

        for i, tactic in enumerate(proof_tactics):
            tactic = tactic.strip()
            if not tactic or len(tactic) < self.min_tactic_length:
                continue
            if tactic.lower() in ("sorry", "admit"):
                continue

            state_before = states[i].strip() if i < len(states) else ""
            state_after = states[i + 1].strip() if i + 1 < len(states) else "no goals"

            if not state_before:
                continue

            # 去重
            if self.deduplicate:
                sample_hash = hash(f"{state_before}|||{tactic}")
                if sample_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(sample_hash)

            samples.append({
                "theorem": theorem_name,
                "state_before": state_before,
                "tactic": tactic,
                "state_after": state_after,
                "thought": "",  # 由 thought 回标模块填充
            })

        return samples

    def extract_batch(self,
                      search_results: List[Dict[str, Any]]
                      ) -> List[Dict[str, str]]:
        """
        批量提取训练样本

        参数:
            search_results: 搜索结果列表，每项包含:
                - theorem: 定理名称
                - tactics: 策略序列
                - states: 状态序列
                - success: 是否成功

        返回:
            List[Dict]: 所有提取的训练样本
        """
        all_samples = []

        for result in search_results:
            if not result.get("success", False):
                continue

            tactics = result.get("tactics", [])
            states = result.get("states", [])
            theorem = result.get("theorem", "unknown")

            samples = self.extract_from_search_result(theorem, tactics, states)
            all_samples.extend(samples)

        logger.info(
            f"批量提取完成: {len(search_results)} 个搜索结果 → "
            f"{len(all_samples)} 个训练样本"
        )
        return all_samples

    def reset_dedup(self):
        """重置去重缓存"""
        self._seen_hashes.clear()


# ================================================================
# Expert Iteration 数据集
# ================================================================

class ExpertIterationDataset:
    """
    Expert Iteration 训练数据集

    混合两类数据:
    1. 基础 SFT 数据（Phase 3 的原始训练集）— 防止灾难性遗忘
    2. 自我博弈新数据（当前迭代的搜索结果）— 提供增量学习信号

    参数:
        base_data_path: Phase 3 基础 SFT 数据路径
        new_samples: 新的自我博弈样本
        system_prompt: 系统提示词
        base_ratio: 基础数据的混合比例 (0~1)
        max_base_samples: 基础数据最大条数
    """

    def __init__(self,
                 base_data_path: str = "",
                 new_samples: List[Dict[str, str]] = None,
                 system_prompt: str = "",
                 base_ratio: float = 0.3,
                 max_base_samples: int = 5000):
        """
        初始化混合数据集

        参数:
            base_data_path: 基础 SFT 数据路径
            new_samples: 新的训练样本
            system_prompt: 系统提示词
            base_ratio: 基础数据比例
            max_base_samples: 基础数据最大条数
        """
        self.system_prompt = system_prompt or (
            "You are a Lean 4 theorem proving assistant. Given a proof state, "
            "first provide a brief reasoning (Thought) explaining your strategy, "
            "then output the exact Lean 4 tactic to apply."
        )
        self.base_ratio = base_ratio

        # 加载基础数据
        base_data = []
        if base_data_path and os.path.exists(base_data_path):
            raw = list(iter_jsonl(base_data_path))
            if max_base_samples > 0 and len(raw) > max_base_samples:
                import random
                raw = random.sample(raw, max_base_samples)
            base_data = raw
            logger.info(f"基础数据加载: {len(base_data)} 条")

        # 新数据
        if new_samples is None:
            new_samples = []
        logger.info(f"新增数据: {len(new_samples)} 条")

        # 混合
        self.samples = self._mix_data(base_data, new_samples)
        logger.info(f"混合后总计: {len(self.samples)} 条")

    def _mix_data(self,
                  base: List[Dict],
                  new: List[Dict]) -> List[Dict]:
        """
        混合基础数据和新数据

        参数:
            base: 基础 SFT 数据
            new: 新的自我博弈数据

        返回:
            List[Dict]: 混合后的数据
        """
        if not base:
            return new
        if not new:
            return base

        # 按比例采样基础数据
        target_base_count = int(len(new) * self.base_ratio / (1 - self.base_ratio))
        target_base_count = min(target_base_count, len(base))

        import random
        sampled_base = random.sample(base, target_base_count) if target_base_count > 0 else []

        mixed = sampled_base + new
        random.shuffle(mixed)
        return mixed

    def to_hf_dataset(self) -> Dataset:
        """
        转换为 HuggingFace Dataset (messages 格式)

        返回:
            Dataset: 包含 messages 字段的数据集
        """
        records = []
        for item in self.samples:
            state = item.get("state_before", "").strip()
            thought = item.get("thought", "").strip()
            tactic = item.get("tactic", "").strip()

            if not state or not tactic:
                continue

            # 构造 assistant 回复
            if thought:
                assistant_content = f"[Thought] {thought}\n[Tactic] {tactic}"
            else:
                assistant_content = f"[Tactic] {tactic}"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": state},
                {"role": "assistant", "content": assistant_content},
            ]
            records.append({"messages": messages})

        return Dataset.from_list(records)


# ================================================================
# Expert Iteration 训练器
# ================================================================

class ExpertIterationTrainer:
    """
    Expert Iteration 训练器

    实现完整的 Expert Iteration 循环:
    1. 使用当前策略模型 + MAGC-MCTS 搜索定理
    2. 从成功证明中提取训练轨迹
    3. 混合基础数据进行 SFT 更新
    4. 评估并保存检查点

    参数:
        config: 训练配置字典
        verifier: PantographVerifier 实例
        generator: ThoughtCoSTacticGenerator 实例
        mcts: MAGCMCTS 实例
    """

    def __init__(self,
                 config: Dict[str, Any],
                 verifier=None,
                 generator=None,
                 mcts=None):
        """
        初始化 Expert Iteration 训练器

        参数:
            config: 配置字典，包含:
                model.base_model_path: 基座模型路径
                model.lora_path: LoRA 路径
                training.num_iterations: 迭代轮数
                training.learning_rate: 学习率
                training.batch_size: 批次大小
                training.base_data_path: Phase 3 基础训练数据路径
                training.base_ratio: 基础数据混合比例
                search.theorems_per_iteration: 每轮搜索的定理数
                search.timeout_per_theorem: 每个定理的搜索超时
                output_dir: 输出目录
            verifier: PantographVerifier
            generator: ThoughtCoSTacticGenerator
            mcts: MAGCMCTS
        """
        self.config = config
        self.verifier = verifier
        self.generator = generator
        self.mcts = mcts

        # 模型配置
        model_cfg = config.get("model", {})
        self.base_model_path = model_cfg.get(
            "base_model_path",
            "/root/autodl-tmp/models/DeepSeek-Prover-V2-7B"
        )
        self.lora_path = model_cfg.get("lora_path", "")

        # 训练配置
        train_cfg = config.get("training", {})
        self.num_iterations = train_cfg.get("num_iterations", 5)
        self.learning_rate = train_cfg.get("learning_rate", 5e-6)
        self.batch_size = train_cfg.get("batch_size", 4)
        self.gradient_accumulation = train_cfg.get("gradient_accumulation", 4)
        self.base_data_path = train_cfg.get("base_data_path", "")
        self.base_ratio = train_cfg.get("base_ratio", 0.3)
        self.max_steps_per_iter = train_cfg.get("max_steps_per_iter", 500)

        # 搜索配置
        search_cfg = config.get("search", {})
        self.theorems_per_iter = search_cfg.get("theorems_per_iteration", 100)
        self.timeout_per_theorem = search_cfg.get("timeout_per_theorem", 300)

        # 输出
        self.output_dir = config.get("output_dir", "checkpoints/expert_iteration")
        ensure_dir(self.output_dir)

        # 工具
        self.extractor = TrajectoryExtractor()

        # 历史记录
        self.iteration_history: List[Dict[str, Any]] = []

    # ================================================================
    # 核心训练循环
    # ================================================================

    def train(self, theorem_pool: List[Dict[str, str]]):
        """
        执行完整的 Expert Iteration 训练

        参数:
            theorem_pool: 待搜索的定理池，每项包含:
                - name: 定理名称
                - type: 类型表达式
                - description: 自然语言描述（可选）
        """
        logger.info(f"Expert Iteration 启动: {self.num_iterations} 轮, "
                    f"定理池 {len(theorem_pool)} 个")

        for iteration in range(self.num_iterations):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Expert Iteration 第 {iteration + 1}/{self.num_iterations} 轮")
            logger.info(f"{'=' * 60}")

            iter_start = time.time()
            iter_output = os.path.join(self.output_dir, f"iter_{iteration + 1}")
            ensure_dir(iter_output)

            # Step 1: 自我博弈采样
            logger.info("[Step 1] 自我博弈采样...")
            search_results = self._self_play_sampling(theorem_pool, iteration)

            # Step 2: 轨迹提取
            logger.info("[Step 2] 轨迹提取...")
            new_samples = self.extractor.extract_batch(search_results)

            # 保存本轮搜索结果和轨迹
            results_path = os.path.join(iter_output, "search_results.jsonl")
            samples_path = os.path.join(iter_output, "training_samples.jsonl")
            save_jsonl(search_results, results_path)
            save_jsonl(new_samples, samples_path)

            success_count = sum(1 for r in search_results if r.get("success"))
            logger.info(
                f"  搜索完成: {success_count}/{len(search_results)} 成功, "
                f"提取 {len(new_samples)} 个训练样本"
            )

            if not new_samples:
                logger.warning("  无新样本，跳过训练步骤")
                continue

            # Step 3: SFT 更新
            logger.info("[Step 3] SFT 更新...")
            train_metrics = self._sft_update(new_samples, iter_output)

            # Step 4: 更新模型引用
            new_lora_path = os.path.join(iter_output, "final")
            if os.path.exists(new_lora_path):
                self.lora_path = new_lora_path
                # 如果有 generator, 更新其 LoRA 路径
                if self.generator:
                    self.generator.lora_path = new_lora_path

            iter_elapsed = time.time() - iter_start

            # 记录历史
            iter_record = {
                "iteration": iteration + 1,
                "search_total": len(search_results),
                "search_success": success_count,
                "new_samples": len(new_samples),
                "train_metrics": train_metrics,
                "elapsed_seconds": iter_elapsed,
            }
            self.iteration_history.append(iter_record)

            # 保存历史
            history_path = os.path.join(self.output_dir, "iteration_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.iteration_history, f, indent=2, ensure_ascii=False)

            logger.info(
                f"  第 {iteration + 1} 轮完成: "
                f"成功率={success_count / max(len(search_results), 1):.2%}, "
                f"样本数={len(new_samples)}, "
                f"耗时={iter_elapsed:.0f}s"
            )

        logger.info(f"\nExpert Iteration 完成! 共 {self.num_iterations} 轮")

    # ================================================================
    # 自我博弈采样
    # ================================================================

    def _self_play_sampling(self,
                            theorem_pool: List[Dict[str, str]],
                            iteration: int) -> List[Dict[str, Any]]:
        """
        使用当前模型 + MAGC-MCTS 搜索定理

        参数:
            theorem_pool: 定理池
            iteration: 当前迭代编号

        返回:
            List[Dict]: 搜索结果
        """
        import random
        # 从池中采样
        sample_size = min(self.theorems_per_iter, len(theorem_pool))
        sampled = random.sample(theorem_pool, sample_size)

        results = []
        for i, theorem in enumerate(sampled):
            name = theorem.get("name", f"thm_{i}")
            thm_type = theorem.get("type", "")
            desc = theorem.get("description", "")

            if not thm_type:
                continue

            logger.debug(f"  搜索 [{i + 1}/{sample_size}]: {name}")

            try:
                # 使用 MAGC-MCTS 搜索
                if self.mcts:
                    proof_tactics = self.mcts.search(thm_type, desc)
                else:
                    proof_tactics = None

                result = {
                    "theorem": name,
                    "type": thm_type,
                    "success": proof_tactics is not None,
                    "tactics": proof_tactics or [],
                    "states": [],  # 由验证器填充
                }

                # 如果成功，回放一次获取 states
                if proof_tactics and self.verifier:
                    states = self._replay_proof(thm_type, proof_tactics)
                    result["states"] = states

                results.append(result)

            except Exception as e:
                logger.warning(f"  搜索异常 {name}: {e}")
                results.append({
                    "theorem": name,
                    "type": thm_type,
                    "success": False,
                    "tactics": [],
                    "states": [],
                    "error": str(e),
                })

        return results

    def _replay_proof(self, theorem_type: str,
                      tactics: List[str]) -> List[str]:
        """
        回放证明以获取中间状态序列

        参数:
            theorem_type: 定理类型表达式
            tactics: 策略序列

        返回:
            List[str]: 状态序列
        """
        states = []
        try:
            init = self.verifier.goal_start(theorem_type)
            if not init:
                return states

            state_id = init.get("stateId", -1)
            sp = self.verifier.goal_print(state_id)
            initial_state = self._extract_goals(sp)
            states.append(initial_state)

            for tactic in tactics:
                result = self.verifier.goal_tactic(state_id, tactic)
                if result and result.get("is_valid", False):
                    state_id = result.get("new_state_id", -1)
                    sp = self.verifier.goal_print(state_id)
                    state_str = self._extract_goals(sp)
                    states.append(state_str)
                else:
                    break

        except Exception as e:
            logger.debug(f"回放异常: {e}")

        return states

    # ================================================================
    # SFT 更新
    # ================================================================

    def _sft_update(self,
                    new_samples: List[Dict[str, str]],
                    output_dir: str) -> Dict[str, Any]:
        """
        在新数据上执行 SFT 微调

        参数:
            new_samples: 新的训练样本
            output_dir: 本轮输出目录

        返回:
            Dict: 训练指标
        """
        # 构建混合数据集
        dataset = ExpertIterationDataset(
            base_data_path=self.base_data_path,
            new_samples=new_samples,
            base_ratio=self.base_ratio,
        )
        hf_dataset = dataset.to_hf_dataset()

        if len(hf_dataset) == 0:
            logger.warning("数据集为空，跳过训练")
            return {}

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA 配置
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # 如果已有 LoRA 权重，先加载再继续训练
        if self.lora_path and os.path.exists(self.lora_path):
            from peft import PeftModel
            logger.info(f"加载已有 LoRA 权重: {self.lora_path}")
            model = PeftModel.from_pretrained(
                model.base_model.model, self.lora_path
            )
            model.train()

        # 训练配置
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps_per_iter,
            logging_steps=10,
            save_steps=self.max_steps_per_iter,
            bf16=True,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            max_seq_length=2048,
        )

        # 训练
        trainer = TrlSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            args=sft_config,
        )

        train_result = trainer.train()

        # 保存
        final_path = os.path.join(output_dir, "final")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # 清理 GPU 内存
        del model, trainer
        torch.cuda.empty_cache()

        metrics = {
            "train_loss": train_result.training_loss
            if hasattr(train_result, "training_loss") else None,
            "samples_count": len(hf_dataset),
        }

        logger.info(f"SFT 更新完成: loss={metrics.get('train_loss')}, "
                    f"样本数={len(hf_dataset)}")

        return metrics

    # ================================================================
    # 辅助工具
    # ================================================================

    @staticmethod
    def _extract_goals(state_print_result: Any) -> str:
        """从 goal.print 结果提取目标文本"""
        if not state_print_result:
            return ""
        goals = state_print_result.get("goals", [])
        if not goals:
            return "no goals"
        lines = []
        for g in goals:
            if isinstance(g, str):
                lines.append(g)
            elif isinstance(g, dict):
                lines.append(g.get("target", g.get("goal", str(g))))
        return "\n".join(lines)


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行启动 Expert Iteration"""
    import argparse
    parser = argparse.ArgumentParser(description="Expert Iteration 训练")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML)")
    parser.add_argument("--theorems", type=str, required=True,
                        help="定理池文件路径 (JSONL)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    setup_logging("expert_iteration")
    set_seed(args.seed)

    config = load_yaml(args.config)
    theorem_pool = list(iter_jsonl(args.theorems))

    logger.info(f"配置: {args.config}")
    logger.info(f"定理池: {len(theorem_pool)} 个定理")

    trainer = ExpertIterationTrainer(config=config)
    trainer.train(theorem_pool)


if __name__ == "__main__":
    main()
