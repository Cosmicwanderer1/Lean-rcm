"""
监督微调训练器 — SFT Phase 1
@author ygw
更新日期: 2026-02-12

基于 trl.SFTTrainer + LoRA 实现 Thought-CoS-Tactic 范式的监督微调。
Phase 1 目标: 输入 State，输出 Thought + Tactic。
基座模型: DeepSeek-Prover-V2-7B
"""

import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer as TrlSFTTrainer

# 项目内部工具
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.common.utils import load_yaml, load_jsonl, iter_jsonl, ensure_dir, set_seed

logger = logging.getLogger("sft_trainer")


# ================================================================
# 数据集类: 将 JSONL 转换为 trl 所需的 messages 格式
# ================================================================

class Phase1Dataset:
    """
    Phase 1 SFT 数据集
    将 thought_dataset_cleaned.jsonl 转换为 chat messages 格式

    输入格式 (每条 JSONL):
        - state_before: 证明状态
        - thought: 推理过程
        - tactic: Lean4 策略

    输出格式 (trl messages):
        - system: 系统提示词
        - user: state_before
        - assistant: [Thought] thought \n[Tactic] tactic

    参数:
        data_path (str): JSONL 数据文件路径
        system_prompt (str): 系统提示词
        max_samples (int): 最大样本数，0 表示不限制
    """

    def __init__(self, data_path: str, system_prompt: str, max_samples: int = 0):
        self.data_path = data_path
        self.system_prompt = system_prompt
        self.max_samples = max_samples
        self.data = self._load_and_convert()
        logger.info(f"Phase1Dataset 加载完成: {len(self.data)} 条样本, 来源: {data_path}")

    def _load_and_convert(self) -> List[Dict[str, Any]]:
        """
        加载 JSONL 并转换为 messages 格式

        返回:
            List[Dict]: 包含 messages 字段的字典列表
        """
        processed = []
        for i, item in enumerate(iter_jsonl(self.data_path)):
            if self.max_samples > 0 and i >= self.max_samples:
                break

            state_before = item.get("state_before", "").strip()
            thought = item.get("thought", "").strip()
            tactic = item.get("tactic", "").strip()

            # 跳过空字段或 sorry
            if not state_before or not tactic or tactic.lower() == "sorry":
                continue

            # 构建 assistant 回复: Thought + Tactic 结构化输出
            if thought:
                assistant_content = f"[Thought] {thought}\n[Tactic] {tactic}"
            else:
                assistant_content = f"[Tactic] {tactic}"

            processed.append({
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": state_before},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

        return processed

    def to_hf_dataset(self) -> Dataset:
        """
        转换为 HuggingFace Dataset

        返回:
            Dataset: HF Dataset 对象
        """
        return Dataset.from_list(self.data)

    def train_val_split(self, val_ratio: float = 0.05, seed: int = 42) -> tuple:
        """
        划分训练集和验证集

        参数:
            val_ratio (float): 验证集比例
            seed (int): 随机种子

        返回:
            tuple: (train_dataset, val_dataset) HF Dataset 对象
        """
        full_ds = self.to_hf_dataset()
        split = full_ds.train_test_split(test_size=val_ratio, seed=seed)
        logger.info(
            f"数据集划分: 训练集 {len(split['train'])} 条, "
            f"验证集 {len(split['test'])} 条"
        )
        return split["train"], split["test"]


class ErrorCorrectionDataset:
    """
    错误修正数据集，将 error_correction_train.jsonl 转换为 chat messages 格式

    输入格式 (每条 JSONL):
        - state_before: 证明状态
        - error_tactic: 错误策略
        - error_message: 错误信息
        - original_tactic: 正确策略
        - repair_hint: 修复提示
        - thought: 推理过程

    输出格式 (trl messages):
        - system: 系统提示词
        - user: state_before + 错误策略 + 错误信息
        - assistant: [Thought] 修复推理 \n[Tactic] 正确策略

    参数:
        data_path (str): JSONL 数据文件路径
        system_prompt (str): 系统提示词
    """

    EC_SYSTEM_SUFFIX = (
        "\nAdditionally, you can fix incorrect tactics. "
        "When given a proof state with an erroneous tactic and its error message, "
        "provide the corrected tactic."
    )

    def __init__(self, data_path: str, system_prompt: str):
        self.data_path = data_path
        self.system_prompt = system_prompt + self.EC_SYSTEM_SUFFIX
        self.data = self._load_and_convert()
        logger.info(f"ErrorCorrectionDataset 加载完成: {len(self.data)} 条样本")

    def _load_and_convert(self) -> List[Dict[str, Any]]:
        """
        加载 JSONL 并转换为 messages 格式

        返回:
            List[Dict]: 包含 messages 字段的字典列表
        """
        processed = []
        for item in iter_jsonl(self.data_path):
            state_before = item.get("state_before", "").strip()
            error_tactic = item.get("error_tactic", "").strip()
            error_msg = item.get("error_message", "").strip()
            original_tactic = item.get("original_tactic", "").strip()
            thought = item.get("thought", "").strip()
            repair_hint = item.get("repair_hint", "").strip()

            if not state_before or not original_tactic:
                continue

            # 用户输入: 状态 + 错误策略 + 错误信息
            user_content = (
                f"{state_before}\n\n"
                f"[Error Tactic] {error_tactic}\n"
                f"[Error Message] {error_msg}"
            )

            # 助手输出: 修复推理 + 正确策略
            if thought and repair_hint:
                assistant_content = (
                    f"[Thought] {thought} {repair_hint}\n"
                    f"[Tactic] {original_tactic}"
                )
            elif thought:
                assistant_content = f"[Thought] {thought}\n[Tactic] {original_tactic}"
            else:
                assistant_content = f"[Tactic] {original_tactic}"

            processed.append({
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

        return processed

    def to_hf_dataset(self) -> Dataset:
        """
        转换为 HuggingFace Dataset

        返回:
            Dataset: HF Dataset 对象
        """
        return Dataset.from_list(self.data)


# ================================================================
# 训练器类: 基于 trl.SFTTrainer + LoRA
# ================================================================

class SFTTrainer:
    """
    监督微调训练器，封装 trl.SFTTrainer + LoRA

    支持功能:
        - 从 YAML 配置文件加载全部参数
        - LoRA 低秩适配微调
        - 4-bit 量化加载 (QLoRA，可选)
        - 自动 train/val 划分
        - wandb 日志集成
        - 断点续训

    参数:
        config_path (str): YAML 配置文件路径
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_yaml(config_path)
        self.model = None
        self.tokenizer = None
        self._trainer = None

    def _build_lora_config(self) -> Optional[LoraConfig]:
        """
        根据配置构建 LoRA 参数

        返回:
            LoraConfig: LoRA 配置对象，未启用时返回 None
        """
        lora_cfg = self.config.get("lora", {})
        if not lora_cfg.get("enabled", False):
            return None

        return LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    def _load_model_and_tokenizer(self, use_4bit: bool = False):
        """
        加载基座模型和分词器

        参数:
            use_4bit (bool): 是否使用 4-bit 量化 (QLoRA 模式)
        """
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("base_model", "deepseek-ai/DeepSeek-Prover-V2-7B")
        local_path = model_cfg.get("local_model_path", "")
        dtype_str = model_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        # 优先使用本地模型路径
        if local_path and os.path.isdir(local_path):
            model_name = local_path
            logger.info(f"使用本地模型: {local_path}")
        else:
            logger.info(f"本地路径不存在，从 HuggingFace 加载: {model_name}")

        logger.info(f"加载分词器: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        # DeepSeek 模型需要设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 模型加载参数
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        # 4-bit 量化配置 (QLoRA)
        if use_4bit:
            logger.info("启用 4-bit 量化 (QLoRA 模式)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"

        logger.info(f"加载模型: {model_name}, dtype={dtype_str}, 4bit={use_4bit}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 应用 LoRA
        lora_config = self._build_lora_config()
        if lora_config is not None:
            logger.info(
                f"应用 LoRA: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"target={lora_config.target_modules}"
            )
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def _build_sft_config(self, phase: str = "phase1") -> SFTConfig:
        """
        根据配置构建 SFTConfig

        参数:
            phase (str): 训练阶段 ("phase1" 或 "phase2")

        返回:
            SFTConfig: trl 训练配置
        """
        train_cfg = self.config.get("training", {}).get(phase, {})
        ckpt_cfg = self.config.get("checkpoint", {})
        log_cfg = self.config.get("logging", {})
        lr_cfg = self.config.get("lr_scheduler", {})
        optim_cfg = self.config.get("optimizer", {})

        output_dir = ckpt_cfg.get("output_dir", f"checkpoints/sft_{phase}")
        ensure_dir(output_dir)

        return SFTConfig(
            output_dir=output_dir,
            # 训练超参
            num_train_epochs=train_cfg.get("epochs", 3),
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            # 精度
            bf16=train_cfg.get("bf16", True),
            # 优化器与调度器
            optim=optim_cfg.get("type", "adamw_torch"),
            lr_scheduler_type=lr_cfg.get("type", "cosine"),
            # 日志
            logging_steps=train_cfg.get("logging_steps", 50),
            # 评估与保存
            eval_strategy="steps",
            eval_steps=train_cfg.get("eval_steps", 500),
            save_strategy="steps",
            save_steps=train_cfg.get("save_steps", 500),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # wandb
            report_to="wandb" if log_cfg.get("use_wandb", False) else "tensorboard",
            run_name=log_cfg.get("run_name", f"sft-{phase}"),
            # 断点续训
            resume_from_checkpoint=ckpt_cfg.get("resume_from", None),
            # 其他
            seed=self.config.get("data", {}).get("seed", 42),
            dataloader_num_workers=self.config.get("data", {}).get("num_workers", 4),
            remove_unused_columns=False,
        )

    def train_phase1(self, use_4bit: bool = False, max_samples: int = 0):
        """
        执行 Phase 1 训练: State → Thought + Tactic

        参数:
            use_4bit (bool): 是否使用 4-bit 量化 (QLoRA)
            max_samples (int): 最大样本数，0 表示全量
        """
        data_cfg = self.config.get("data", {})
        prompt_cfg = self.config.get("prompts", {})

        # 设置随机种子
        seed = data_cfg.get("seed", 42)
        set_seed(seed)

        # 1. 加载模型
        self._load_model_and_tokenizer(use_4bit=use_4bit)

        # 2. 加载预划分的数据集
        system_prompt = prompt_cfg.get("phase1_system",
            "You are a Lean 4 theorem proving assistant.")

        train_path = data_cfg.get("train_dataset",
            "data/processed/cos_dataset/train.jsonl")
        val_path = data_cfg.get("val_dataset",
            "data/processed/cos_dataset/val.jsonl")

        logger.info(f"加载训练集: {train_path}")
        train_ds = Phase1Dataset(
            data_path=train_path,
            system_prompt=system_prompt,
            max_samples=max_samples,
        )

        logger.info(f"加载验证集: {val_path}")
        val_ds = Phase1Dataset(
            data_path=val_path,
            system_prompt=system_prompt,
            max_samples=0,
        )

        train_dataset = train_ds.to_hf_dataset()
        eval_dataset = val_ds.to_hf_dataset()

        # 2.1 可选: 混合错误修正数据
        if data_cfg.get("use_error_correction", False):
            ec_path = data_cfg.get("error_correction_dataset",
                "data/processed/error_correction/error_correction_train.jsonl")
            logger.info(f"混合错误修正数据: {ec_path}")
            ec_ds = ErrorCorrectionDataset(
                data_path=ec_path,
                system_prompt=system_prompt,
            )
            ec_hf = ec_ds.to_hf_dataset()
            from datasets import concatenate_datasets
            train_dataset = concatenate_datasets([train_dataset, ec_hf])
            logger.info(f"混合后训练集: {len(train_dataset)} 条")

        # 3. 构建训练配置
        sft_config = self._build_sft_config(phase="phase1")

        # 4. 创建 trl.SFTTrainer
        logger.info("创建 SFTTrainer 实例...")
        model_cfg = self.config.get("model", {})
        self._trainer = TrlSFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            max_seq_length=model_cfg.get("max_length", 2048),
        )

        # 5. 开始训练
        logger.info("=" * 60)
        logger.info("Phase 1 训练开始")
        logger.info(f"  训练样本: {len(train_dataset)}")
        logger.info(f"  验证样本: {len(eval_dataset)}")
        logger.info(f"  Epochs: {sft_config.num_train_epochs}")
        logger.info(f"  Batch size: {sft_config.per_device_train_batch_size}")
        logger.info(f"  Grad accum: {sft_config.gradient_accumulation_steps}")
        logger.info(f"  LR: {sft_config.learning_rate}")
        logger.info("=" * 60)

        ckpt_cfg = self.config.get("checkpoint", {})
        resume_from = ckpt_cfg.get("resume_from", None)
        self._trainer.train(resume_from_checkpoint=resume_from)

        # 6. 保存最终模型
        self._save_final_model()

        logger.info("Phase 1 训练完成!")

    def _save_final_model(self):
        """
        保存最终训练好的模型和分词器
        LoRA 模式下保存 adapter，全量模式下保存完整模型
        """
        ckpt_cfg = self.config.get("checkpoint", {})
        output_dir = ckpt_cfg.get("output_dir", "checkpoints/sft_phase1")
        final_dir = os.path.join(output_dir, "final")
        ensure_dir(final_dir)

        lora_cfg = self.config.get("lora", {})
        if lora_cfg.get("enabled", False):
            # LoRA: 保存 adapter 权重
            logger.info(f"保存 LoRA adapter 到: {final_dir}")
            self.model.save_pretrained(final_dir)
        else:
            # 全量: 保存完整模型
            logger.info(f"保存完整模型到: {final_dir}")
            self._trainer.save_model(final_dir)

        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"分词器已保存到: {final_dir}")

    def evaluate(self, eval_dataset_path: Optional[str] = None) -> Dict[str, float]:
        """
        评估模型性能

        参数:
            eval_dataset_path (str): 评估数据集路径，None 则使用训练时的验证集

        返回:
            Dict[str, float]: 评估指标 (eval_loss 等)
        """
        if self._trainer is None:
            raise RuntimeError("请先调用 train_phase1() 进行训练")

        logger.info("开始评估...")
        metrics = self._trainer.evaluate()
        logger.info(f"评估结果: {metrics}")
        return metrics


# ================================================================
# 命令行入口
# ================================================================

def main():
    """
    命令行入口函数

    用法:
        python -m src.trainer.sft_trainer --config configs/training_sft.yaml
        python -m src.trainer.sft_trainer --config configs/training_sft.yaml --use-4bit
        python -m src.trainer.sft_trainer --config configs/training_sft.yaml --max-samples 1000
    """
    import argparse

    parser = argparse.ArgumentParser(description="RTAP SFT Phase 1 训练器")
    parser.add_argument(
        "--config", type=str, default="configs/training_sft.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--use-4bit", action="store_true", default=False,
        help="启用 4-bit 量化 (QLoRA 模式，降低显存需求)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="最大训练样本数，0 表示全量训练"
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置 wandb 项目 (如果启用)
    config = load_yaml(args.config)
    log_cfg = config.get("logging", {})
    if log_cfg.get("use_wandb", False):
        os.environ.setdefault("WANDB_PROJECT", log_cfg.get("project_name", "rtap-v3-sft"))

    # 创建训练器并执行
    trainer = SFTTrainer(config_path=args.config)
    trainer.train_phase1(
        use_4bit=args.use_4bit,
        max_samples=args.max_samples,
    )

    # 输出评估结果
    metrics = trainer.evaluate()
    logger.info(f"最终评估指标: {metrics}")


if __name__ == "__main__":
    main()
