"""
数据增强模块
@author ygw
更新日期: 2026-02-06

W8 核心模块：通过错误注入和合成定理生成扩充训练数据。

功能一：错误注入 (Error Injection)
- 对正确的 (State, Tactic) 对注入 4 类错误：
  1. tactic_typo: 策略拼写错误（如 simp → simpp）
  2. wrong_tactic: 替换为语义不匹配的策略
  3. argument_error: 参数交换/删除/添加
  4. missing_step: 跳过中间证明步骤
- 产出 ETR (Error-Tactic Repair) 训练数据

功能二：合成定理生成 (Synthetic Theorem Generation)
- 基于模板生成简单定理（代数、逻辑、不等式、集合论）
- 通过 Pantograph 验证可证明性
- 扩充训练集覆盖面

技术产出:
- error_injection.jsonl: 错误注入数据集
- synthetic_theorems.jsonl: 合成定理数据集
- augmented_dataset.jsonl: 合并后的增强数据集
"""

import os
import sys
import json
import copy
import time
import random
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_json, save_jsonl, load_jsonl, ensure_dir,
    setup_logging, ProgressTracker, get_timestamp,
    batch_iter, set_seed, compute_hash, deduplicate_by_key
)

logger = logging.getLogger(__name__)


# ================================================================
# 数据结构定义
# ================================================================

@dataclass
class ErrorRecord:
    """
    错误注入记录

    属性:
        original_tactic: 原始正确策略
        error_tactic: 注入错误后的策略
        error_type: 错误类型 (tactic_typo / wrong_tactic / argument_error / missing_step)
        state_before: 错误发生时的证明状态
        state_after: 正确策略执行后的状态（用于修复目标）
        theorem_name: 所属定理名称
        thought: 对应的 Thought 文本
        repair_hint: 修复提示（可选）
    """
    original_tactic: str = ""
    error_tactic: str = ""
    error_type: str = ""
    state_before: str = ""
    state_after: str = ""
    theorem_name: str = ""
    thought: str = ""
    repair_hint: str = ""


@dataclass
class SyntheticTheorem:
    """
    合成定理记录

    属性:
        theorem_statement: 定理声明（Lean4 格式）
        category: 定理类别 (algebra / logic / inequality / set_theory)
        proof_tactics: 证明策略序列
        difficulty: 难度等级 (easy / medium)
        template_id: 来源模板编号
        is_verified: 是否经过 Pantograph 验证
    """
    theorem_statement: str = ""
    category: str = ""
    proof_tactics: List[str] = field(default_factory=list)
    difficulty: str = "easy"
    template_id: str = ""
    is_verified: bool = False


# ================================================================
# 合成定理模板库
# ================================================================

# 每个模板包含: theorem (Lean4 声明), proof (策略序列), category, difficulty
SYNTHETIC_TEMPLATES = {
    "algebra": [
        {
            "theorem": "theorem add_comm_nat (a b : ℕ) : a + b = b + a",
            "proof": ["omega"],
            "difficulty": "easy",
            "variants": [
                ("a b c : ℕ", "a + b + c = c + b + a"),
                ("a b : ℕ", "a + b + 0 = b + a"),
            ]
        },
        {
            "theorem": "theorem add_assoc_nat (a b c : ℕ) : (a + b) + c = a + (b + c)",
            "proof": ["omega"],
            "difficulty": "easy",
            "variants": [
                ("a b c d : ℕ", "(a + b) + (c + d) = a + (b + (c + d))"),
            ]
        },
        {
            "theorem": "theorem mul_comm_nat (a b : ℕ) : a * b = b * a",
            "proof": ["ring"],
            "difficulty": "easy",
            "variants": [
                ("a b c : ℕ", "a * b * c = c * b * a"),
            ]
        },
        {
            "theorem": "theorem mul_zero_nat (a : ℕ) : a * 0 = 0",
            "proof": ["ring"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem mul_one_nat (a : ℕ) : a * 1 = a",
            "proof": ["ring"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem distrib_left_nat (a b c : ℕ) : a * (b + c) = a * b + a * c",
            "proof": ["ring"],
            "difficulty": "medium",
            "variants": [
                ("a b c : ℕ", "(a + b) * c = a * c + b * c"),
            ]
        },
        {
            "theorem": "theorem sq_expansion (a b : ℕ) : (a + b) * (a + b) = a * a + 2 * a * b + b * b",
            "proof": ["ring"],
            "difficulty": "medium",
            "variants": []
        },
    ],
    "logic": [
        {
            "theorem": "theorem and_comm_prop (P Q : Prop) (h : P ∧ Q) : Q ∧ P",
            "proof": ["exact ⟨h.2, h.1⟩"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem or_intro_left (P Q : Prop) (h : P) : P ∨ Q",
            "proof": ["exact Or.inl h"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem or_intro_right (P Q : Prop) (h : Q) : P ∨ Q",
            "proof": ["exact Or.inr h"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem modus_ponens (P Q : Prop) (hp : P) (hpq : P → Q) : Q",
            "proof": ["exact hpq hp"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem double_neg_intro (P : Prop) (h : P) : ¬¬P",
            "proof": ["intro hn", "exact hn h"],
            "difficulty": "medium",
            "variants": []
        },
        {
            "theorem": "theorem contrapositive (P Q : Prop) (h : P → Q) : ¬Q → ¬P",
            "proof": ["intro hnq hp", "exact hnq (h hp)"],
            "difficulty": "medium",
            "variants": []
        },
    ],
    "inequality": [
        {
            "theorem": "theorem nat_zero_le (a : ℕ) : 0 ≤ a",
            "proof": ["omega"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem nat_le_add (a b : ℕ) : a ≤ a + b",
            "proof": ["omega"],
            "difficulty": "easy",
            "variants": [
                ("a b : ℕ", "b ≤ a + b"),
            ]
        },
        {
            "theorem": "theorem nat_succ_pos (n : ℕ) : 0 < n + 1",
            "proof": ["omega"],
            "difficulty": "easy",
            "variants": [
                ("n : ℕ", "0 < n.succ"),
            ]
        },
        {
            "theorem": "theorem nat_le_trans (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c",
            "proof": ["omega"],
            "difficulty": "medium",
            "variants": []
        },
    ],
    "set_theory": [
        {
            "theorem": "theorem set_subset_refl (α : Type*) (s : Set α) : s ⊆ s",
            "proof": ["intro x hx", "exact hx"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem set_inter_subset_left (α : Type*) (s t : Set α) : s ∩ t ⊆ s",
            "proof": ["intro x hx", "exact hx.1"],
            "difficulty": "easy",
            "variants": []
        },
        {
            "theorem": "theorem set_subset_union_left (α : Type*) (s t : Set α) : s ⊆ s ∪ t",
            "proof": ["intro x hx", "exact Or.inl hx"],
            "difficulty": "easy",
            "variants": []
        },
    ],
}

# 随机参数名池（用于合成定理变体生成）
PARAM_NAMES = ["x", "y", "z", "m", "n", "p", "q", "r", "k", "w"]


# ================================================================
# 错误注入器
# ================================================================

class ErrorInjector:
    """
    错误注入器：对正确的 Tactic 注入可控错误，生成 ETR 训练数据。

    支持 4 种错误类型:
    1. tactic_typo - 策略名拼写错误
    2. wrong_tactic - 替换为不匹配的策略
    3. argument_error - 参数变异（交换/删除/添加）
    4. missing_step - 跳过中间步骤

    使用示例:
        injector = ErrorInjector(config["augmentation"]["error_injection"])
        error_records = injector.inject(dataset)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化错误注入器

        参数:
            config: error_injection 配置字典
        """
        self.injection_ratio = config.get("injection_ratio", 0.3)

        # 解析错误类型及权重
        error_types_config = config.get("error_types", {})
        self.error_weights = {}
        self.error_configs = {}
        for etype, econf in error_types_config.items():
            self.error_weights[etype] = econf.get("weight", 0.25)
            self.error_configs[etype] = econf

        # 归一化权重
        total_weight = sum(self.error_weights.values())
        if total_weight > 0:
            self.error_weights = {
                k: v / total_weight for k, v in self.error_weights.items()
            }

        # 构建拼写错误映射表
        typo_conf = self.error_configs.get("tactic_typo", {})
        self.typo_map = typo_conf.get("typo_map", {})

        # 构建策略替换池
        wrong_conf = self.error_configs.get("wrong_tactic", {})
        self.replacement_pool = wrong_conf.get("replacement_pool", [
            "simp", "ring", "norm_num", "omega", "trivial", "assumption"
        ])

        # 参数变异方式
        arg_conf = self.error_configs.get("argument_error", {})
        self.mutations = arg_conf.get("mutations", [
            "swap_args", "drop_arg", "add_random_arg"
        ])

        # 统计
        self.stats = {
            "total_input": 0, "total_injected": 0,
            "tactic_typo": 0, "wrong_tactic": 0,
            "argument_error": 0, "missing_step": 0,
        }

    def inject(self, dataset: List[Dict]) -> List[Dict]:
        """
        对数据集执行错误注入

        v2: step_index=0 的样本注入所有 4 种错误类型（4x 放大），
        因为只有 step_index=0 不需要 Pantograph 重放前置步骤，验证成功率最高。
        step_index>0 的样本仍然随机分配 1 种错误类型。
        @author ygw 2026-03-01

        参数:
            dataset: W7 产出的 thought_dataset 样本列表

        返回:
            List[Dict]: 错误注入记录列表（每条包含 error_tactic + original_tactic）
        """
        self.stats["total_input"] = len(dataset)
        all_error_types = list(self.error_weights.keys())

        # 分离 step_index=0 和 step_index>0 的样本
        step0_indices = [i for i, s in enumerate(dataset) if s.get("step_index", -1) == 0]
        other_indices = [i for i, s in enumerate(dataset) if s.get("step_index", -1) != 0]

        logger.info(f"错误注入: 总样本 {len(dataset)}, "
                     f"step_index=0: {len(step0_indices)}, "
                     f"step_index>0: {len(other_indices)}")

        results = []

        # step_index=0: 每个样本注入所有 4 种错误类型（最大化 Pantograph 验证率）
        for idx in step0_indices:
            sample = dataset[idx]
            for error_type in all_error_types:
                record = self._inject_single(sample, error_type)
                if record:
                    results.append(record)
                    self.stats["total_injected"] += 1
                    self.stats[error_type] = self.stats.get(error_type, 0) + 1

        # step_index>0: 随机分配 1 种错误类型（验证率较低但仍有价值）
        error_type_list = self._distribute_error_types(len(other_indices))
        for idx, error_type in zip(other_indices, error_type_list):
            sample = dataset[idx]
            record = self._inject_single(sample, error_type)
            if record:
                results.append(record)
                self.stats["total_injected"] += 1
                self.stats[error_type] = self.stats.get(error_type, 0) + 1

        step0_count = sum(1 for r in results if r.get("step_index", -1) == 0)
        logger.info(f"错误注入完成: {len(results)} 条记录 "
                     f"(step_index=0: {step0_count})")
        for etype in ["tactic_typo", "wrong_tactic", "argument_error", "missing_step"]:
            logger.info(f"  {etype}: {self.stats.get(etype, 0)}")

        return results

    def _distribute_error_types(self, count: int) -> List[str]:
        """
        按权重分配错误类型

        参数:
            count: 需要分配的总数

        返回:
            List[str]: 错误类型列表
        """
        types = list(self.error_weights.keys())
        weights = [self.error_weights[t] for t in types]
        result = random.choices(types, weights=weights, k=count)
        return result

    def _inject_single(self, sample: Dict, error_type: str) -> Optional[Dict]:
        """
        对单个样本注入指定类型的错误

        参数:
            sample: 原始样本
            error_type: 错误类型

        返回:
            Dict: 错误注入记录，失败返回 None
        """
        original_tactic = sample.get("tactic", "")
        if not original_tactic:
            return None

        error_tactic = ""
        repair_hint = ""

        if error_type == "tactic_typo":
            error_tactic, repair_hint = self._inject_typo(original_tactic)
        elif error_type == "wrong_tactic":
            error_tactic, repair_hint = self._inject_wrong_tactic(original_tactic)
        elif error_type == "argument_error":
            error_tactic, repair_hint = self._inject_argument_error(original_tactic)
        elif error_type == "missing_step":
            error_tactic, repair_hint = self._inject_missing_step(original_tactic)

        # 确保错误策略与原始策略不同
        if not error_tactic or error_tactic == original_tactic:
            return None

        record = {
            "original_tactic": original_tactic,
            "error_tactic": error_tactic,
            "error_type": error_type,
            "state_before": sample.get("state_before", ""),
            "state_after": sample.get("state_after", ""),
            "theorem_name": sample.get("theorem_name", ""),
            "theorem_full_name": sample.get("theorem_full_name", ""),
            "step_index": sample.get("step_index", -1),
            "thought": sample.get("thought", ""),
            "repair_hint": repair_hint,
            "source": "error_injection",
        }
        return record

    def _inject_typo(self, tactic: str) -> Tuple[str, str]:
        """
        注入拼写错误

        参数:
            tactic: 原始策略文本

        返回:
            Tuple[str, str]: (错误策略, 修复提示)
        """
        tactic_name = tactic.split()[0] if tactic else ""

        # 优先使用预定义的拼写错误映射
        if tactic_name in self.typo_map:
            typo_list = self.typo_map[tactic_name]
            if typo_list:
                typo = random.choice(typo_list)
                error_tactic = tactic.replace(tactic_name, typo, 1)
                return error_tactic, f"typo: '{typo}' should be '{tactic_name}'"

        # 通用拼写错误：随机字符变异
        if len(tactic_name) >= 3:
            chars = list(tactic_name)
            mutation_type = random.choice(["swap", "duplicate", "delete"])
            if mutation_type == "swap" and len(chars) >= 2:
                # 交换相邻字符
                pos = random.randint(0, len(chars) - 2)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            elif mutation_type == "duplicate":
                # 重复一个字符
                pos = random.randint(0, len(chars) - 1)
                chars.insert(pos, chars[pos])
            elif mutation_type == "delete" and len(chars) >= 4:
                # 删除一个字符
                pos = random.randint(1, len(chars) - 2)
                chars.pop(pos)

            typo_name = "".join(chars)
            if typo_name != tactic_name:
                error_tactic = tactic.replace(tactic_name, typo_name, 1)
                return error_tactic, f"typo: '{typo_name}' should be '{tactic_name}'"

        return "", ""

    def _inject_wrong_tactic(self, tactic: str) -> Tuple[str, str]:
        """
        替换为错误的策略

        参数:
            tactic: 原始策略文本

        返回:
            Tuple[str, str]: (错误策略, 修复提示)
        """
        tactic_name = tactic.split()[0] if tactic else ""

        # 从替换池中选择一个不同的策略
        candidates = [t for t in self.replacement_pool if t != tactic_name]
        if not candidates:
            return "", ""

        wrong = random.choice(candidates)
        # 保留原始参数部分
        parts = tactic.split(maxsplit=1)
        if len(parts) > 1:
            error_tactic = f"{wrong} {parts[1]}"
        else:
            error_tactic = wrong

        return error_tactic, f"wrong tactic: should use '{tactic_name}' instead of '{wrong}'"

    def _inject_argument_error(self, tactic: str) -> Tuple[str, str]:
        """
        注入参数错误

        参数:
            tactic: 原始策略文本

        返回:
            Tuple[str, str]: (错误策略, 修复提示)
        """
        parts = tactic.split()
        if len(parts) < 2:
            # 无参数，无法注入参数错误，回退到拼写错误
            return self._inject_typo(tactic)

        tactic_name = parts[0]
        args = parts[1:]
        mutation = random.choice(self.mutations)

        if mutation == "swap_args" and len(args) >= 2:
            # 交换两个参数的位置
            i, j = random.sample(range(len(args)), 2)
            args[i], args[j] = args[j], args[i]
            hint = f"swapped args: positions {i+1} and {j+1}"
        elif mutation == "drop_arg" and len(args) >= 2:
            # 删除一个参数
            drop_idx = random.randint(0, len(args) - 1)
            dropped = args.pop(drop_idx)
            hint = f"missing arg: arg {drop_idx+1} '{dropped}' was removed"
        elif mutation == "add_random_arg":
            # 添加一个随机参数
            random_arg = random.choice(PARAM_NAMES)
            insert_pos = random.randint(0, len(args))
            args.insert(insert_pos, random_arg)
            hint = f"extra arg: inserted '{random_arg}' at position {insert_pos+1}"
        else:
            return "", ""

        error_tactic = tactic_name + " " + " ".join(args)
        return error_tactic, hint

    def _inject_missing_step(self, tactic: str) -> Tuple[str, str]:
        """
        模拟缺少步骤的错误：将当前策略替换为终止策略

        v2: sorry 在 Lean4 中永远成功（axiom），导致 Pantograph 验证
        标记为 error_tactic_succeeded。改用 done/rfl/exact _ 等
        在大多数上下文中会真正失败的策略。
        @author ygw 2026-03-01

        参数:
            tactic: 原始策略文本

        返回:
            Tuple[str, str]: (错误策略, 修复提示)
        """
        # v2: 不再使用 sorry（Lean4 axiom，永远成功）
        # 改用 done/rfl/exact _，这些策略在目标未解决时会报错
        replacements = ["done", "rfl", "exact _"]
        error_tactic = random.choice(replacements)
        hint = (f"missing step: used '{error_tactic}' placeholder, "
                f"should be '{tactic}'")
        return error_tactic, hint


# ================================================================
# 合成定理生成器
# ================================================================

class SyntheticTheoremGenerator:
    """
    合成定理生成器：基于模板生成简单定理，扩充训练集覆盖面。

    生成策略:
    1. 从模板库中按类别权重采样
    2. 对模板进行变体扩展（参数重命名、表达式变换）
    3. 可选：通过 Pantograph 验证可证明性
    4. 为每条合成定理生成对应的训练样本

    使用示例:
        generator = SyntheticTheoremGenerator(config["augmentation"]["synthetic_theorems"])
        synthetic_records = generator.generate()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化合成定理生成器

        参数:
            config: synthetic_theorems 配置字典
        """
        self.num_theorems = config.get("num_theorems", 5000)

        # 类别权重
        categories_config = config.get("categories", {})
        self.category_weights = {}
        for cat_name, cat_conf in categories_config.items():
            self.category_weights[cat_name] = cat_conf.get("weight", 0.25)

        # 归一化权重
        total = sum(self.category_weights.values())
        if total > 0:
            self.category_weights = {
                k: v / total for k, v in self.category_weights.items()
            }

        # 统计
        self.stats = {
            "total_generated": 0,
            "by_category": {},
            "by_difficulty": {"easy": 0, "medium": 0},
        }

    def generate(self) -> List[Dict]:
        """
        生成合成定理数据集

        返回:
            List[Dict]: 合成定理记录列表
        """
        logger.info(f"合成定理生成: 目标 {self.num_theorems} 条")

        # 按类别权重分配数量
        category_counts = self._distribute_categories()

        results = []
        theorem_counter = 0

        for category, count in category_counts.items():
            templates = SYNTHETIC_TEMPLATES.get(category, [])
            if not templates:
                logger.warning(f"类别 '{category}' 无可用模板，跳过")
                continue

            cat_results = []
            generated_in_cat = 0

            while generated_in_cat < count:
                # 循环使用模板
                template = templates[generated_in_cat % len(templates)]
                theorem_counter += 1

                # 生成基础定理
                record = self._generate_from_template(
                    template, category, theorem_counter
                )
                if record:
                    cat_results.append(record)
                    generated_in_cat += 1

                # 生成变体
                for variant in template.get("variants", []):
                    if generated_in_cat >= count:
                        break
                    theorem_counter += 1
                    variant_record = self._generate_variant(
                        template, variant, category, theorem_counter
                    )
                    if variant_record:
                        cat_results.append(variant_record)
                        generated_in_cat += 1

                # 生成参数重命名变体
                if generated_in_cat < count:
                    theorem_counter += 1
                    renamed = self._generate_renamed_variant(
                        template, category, theorem_counter
                    )
                    if renamed:
                        cat_results.append(renamed)
                        generated_in_cat += 1

            results.extend(cat_results)
            self.stats["by_category"][category] = len(cat_results)
            logger.info(f"  {category}: {len(cat_results)} 条")

        self.stats["total_generated"] = len(results)
        logger.info(f"合成定理生成完成: {len(results)} 条")

        return results

    def _distribute_categories(self) -> Dict[str, int]:
        """
        按权重分配各类别的生成数量

        返回:
            Dict[str, int]: 类别 → 数量映射
        """
        counts = {}
        remaining = self.num_theorems
        categories = list(self.category_weights.keys())

        for i, cat in enumerate(categories):
            if i == len(categories) - 1:
                # 最后一个类别取剩余数量，避免舍入误差
                counts[cat] = remaining
            else:
                n = int(self.num_theorems * self.category_weights[cat])
                counts[cat] = n
                remaining -= n

        return counts

    def _generate_from_template(self, template: Dict, category: str,
                                 counter: int) -> Optional[Dict]:
        """
        从模板生成一条合成定理记录

        参数:
            template: 模板字典
            category: 类别名称
            counter: 全局计数器

        返回:
            Dict: 合成定理记录
        """
        theorem_stmt = template["theorem"]
        proof_tactics = template["proof"]
        difficulty = template.get("difficulty", "easy")

        # 为定理名添加唯一后缀，避免重名
        unique_name = re.sub(
            r"theorem\s+(\w+)",
            f"theorem synth_{category}_{counter}",
            theorem_stmt,
            count=1
        )

        record = {
            "theorem_statement": unique_name,
            "category": category,
            "proof_tactics": proof_tactics,
            "difficulty": difficulty,
            "template_id": f"{category}_{counter}",
            "is_verified": False,
            "source": "synthetic",
        }

        self.stats["by_difficulty"][difficulty] = \
            self.stats["by_difficulty"].get(difficulty, 0) + 1

        return record

    def _generate_variant(self, template: Dict, variant: Tuple,
                           category: str, counter: int) -> Optional[Dict]:
        """
        从模板变体生成定理

        参数:
            template: 原始模板
            variant: (参数声明, 目标表达式) 元组
            category: 类别
            counter: 计数器

        返回:
            Dict: 合成定理记录
        """
        params, goal = variant
        proof_tactics = template["proof"]
        difficulty = template.get("difficulty", "easy")

        theorem_stmt = f"theorem synth_{category}_v{counter} ({params}) : {goal}"

        record = {
            "theorem_statement": theorem_stmt,
            "category": category,
            "proof_tactics": proof_tactics,
            "difficulty": difficulty,
            "template_id": f"{category}_v{counter}",
            "is_verified": False,
            "source": "synthetic_variant",
        }

        self.stats["by_difficulty"][difficulty] = \
            self.stats["by_difficulty"].get(difficulty, 0) + 1

        return record

    def _generate_renamed_variant(self, template: Dict, category: str,
                                   counter: int) -> Optional[Dict]:
        """
        通过参数重命名生成变体

        参数:
            template: 原始模板
            category: 类别
            counter: 计数器

        返回:
            Dict: 合成定理记录
        """
        theorem_stmt = template["theorem"]

        # 提取原始参数名（单字母变量）
        param_match = re.search(r"\(([^)]+)\)", theorem_stmt)
        if not param_match:
            return None

        param_str = param_match.group(1)
        # 提取变量名（排除类型声明）
        original_vars = re.findall(r"\b([a-z])\b(?=\s)", param_str)
        if not original_vars:
            return None

        # 随机选择替换变量名
        available = [n for n in PARAM_NAMES if n not in original_vars]
        if len(available) < len(original_vars):
            return None

        new_vars = random.sample(available, len(original_vars))
        rename_map = dict(zip(original_vars, new_vars))

        # 执行替换（仅替换独立的单字母变量）
        new_stmt = theorem_stmt
        for old_var, new_var in rename_map.items():
            new_stmt = re.sub(
                rf"\b{old_var}\b", new_var, new_stmt
            )

        # 更新定理名
        new_stmt = re.sub(
            r"theorem\s+(\w+)",
            f"theorem synth_{category}_r{counter}",
            new_stmt,
            count=1
        )

        record = {
            "theorem_statement": new_stmt,
            "category": category,
            "proof_tactics": template["proof"],
            "difficulty": template.get("difficulty", "easy"),
            "template_id": f"{category}_r{counter}",
            "is_verified": False,
            "source": "synthetic_renamed",
        }

        return record


# ================================================================
# 数据增强主类
# ================================================================

class DataAugmentation:
    """
    W8 数据增强主类，编排错误注入和合成定理生成。

    核心流程:
    1. 读取 W7 产出的 thought_dataset.jsonl
    2. 执行错误注入 → error_injection.jsonl
    3. 执行合成定理生成 → synthetic_theorems.jsonl
    4. 合并原始数据 + 错误注入 + 合成定理 → augmented_dataset.jsonl
    5. 数据集分割 (train/val/test)
    6. 去重与统计

    使用示例:
        augmentor = DataAugmentation(config)
        augmentor.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据增强模块

        参数:
            config: 完整配置字典（来自 data_pipeline.yaml）
        """
        self.config = config
        aug_config = config.get("augmentation", {})

        self.input_path = aug_config.get("input_path", "")
        self.output_path = aug_config.get("output_path", "")

        # 错误注入配置
        self.error_injection_enabled = aug_config.get(
            "error_injection",
        ).get("enable", True)
        self.error_injection_config = aug_config.get("error_injection", {})

        # 合成定理配置
        self.synthetic_enabled = aug_config.get(
            "synthetic_theorems", {}
        ).get("enable", True)
        self.synthetic_config = aug_config.get("synthetic_theorems", {})

        # 最终数据集配置
        final_config = config.get("final_dataset", {})
        self.final_output_path = final_config.get("output_path", self.output_path)
        self.split_config = final_config.get("split", {
            "train_ratio": 0.9, "val_ratio": 0.05, "test_ratio": 0.05
        })
        self.target_size = final_config.get("target_size", 100000)
        self.remove_duplicates = final_config.get(
            "validation", {}
        ).get("remove_duplicates", True)

        # 随机种子
        seed = config.get("global", {}).get("random_seed", 42)
        set_seed(seed)

        # 统计
        self.stats = {
            "original_count": 0,
            "error_injection_count": 0,
            "synthetic_count": 0,
            "total_before_dedup": 0,
            "total_after_dedup": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
        }

    def run(self) -> str:
        """
        执行完整的 W8 数据增强流程

        返回:
            str: 最终输出目录路径
        """
        logger.info("=" * 60)
        logger.info("W8 数据增强 (Data Augmentation) 开始")
        logger.info("=" * 60)

        start_time = time.time()
        ensure_dir(self.output_path)

        # 1. 加载 W7 产出数据
        thought_file = os.path.join(self.input_path, "thought_dataset.jsonl")
        if os.path.exists(thought_file):
            original_data = load_jsonl(thought_file)
            logger.info(f"加载原始数据: {len(original_data)} 条")
        else:
            logger.warning(f"W7 产出文件不存在: {thought_file}，使用空数据集")
            original_data = []
        self.stats["original_count"] = len(original_data)

        # 2. 错误注入
        error_records = []
        if self.error_injection_enabled and original_data:
            logger.info("-" * 40)
            logger.info("阶段 1: 错误注入")
            injector = ErrorInjector(self.error_injection_config)
            error_records = injector.inject(original_data)

            error_file = os.path.join(self.output_path, "error_injection.jsonl")
            save_jsonl(error_records, error_file)
            logger.info(f"错误注入数据已保存: {error_file}")
        self.stats["error_injection_count"] = len(error_records)

        # 3. 合成定理生成
        synthetic_records = []
        if self.synthetic_enabled:
            logger.info("-" * 40)
            logger.info("阶段 2: 合成定理生成")
            generator = SyntheticTheoremGenerator(self.synthetic_config)
            synthetic_records = generator.generate()

            synth_file = os.path.join(self.output_path, "synthetic_theorems.jsonl")
            save_jsonl(synthetic_records, synth_file)
            logger.info(f"合成定理数据已保存: {synth_file}")
        self.stats["synthetic_count"] = len(synthetic_records)

        # 4. 合并数据集
        logger.info("-" * 40)
        logger.info("阶段 3: 数据合并")
        all_data = self._merge_datasets(original_data, error_records, synthetic_records)
        self.stats["total_before_dedup"] = len(all_data)
        logger.info(f"合并后总量: {len(all_data)} 条")

        # 5. 去重
        if self.remove_duplicates:
            all_data = self._deduplicate(all_data)
        self.stats["total_after_dedup"] = len(all_data)
        logger.info(f"去重后总量: {len(all_data)} 条")

        # 6. 保存合并数据集
        ensure_dir(self.final_output_path)
        augmented_file = os.path.join(self.output_path, "augmented_dataset.jsonl")
        save_jsonl(all_data, augmented_file)
        logger.info(f"增强数据集已保存: {augmented_file}")

        # 7. 数据集分割
        logger.info("-" * 40)
        logger.info("阶段 4: 数据集分割")
        self._split_dataset(all_data)

        # 8. 保存统计信息
        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["timestamp"] = get_timestamp()

        stats_file = os.path.join(self.output_path, "augmentation_stats.json")
        save_json(self.stats, stats_file)

        logger.info("=" * 60)
        logger.info("W8 数据增强完成")
        logger.info(f"原始数据: {self.stats['original_count']}")
        logger.info(f"错误注入: {self.stats['error_injection_count']}")
        logger.info(f"合成定理: {self.stats['synthetic_count']}")
        logger.info(f"最终总量: {self.stats['total_after_dedup']}")
        logger.info(f"训练集: {self.stats['train_count']}")
        logger.info(f"验证集: {self.stats['val_count']}")
        logger.info(f"测试集: {self.stats['test_count']}")
        logger.info(f"耗时: {elapsed:.1f}s")
        logger.info("=" * 60)

        return self.output_path

    def _merge_datasets(self, original: List[Dict],
                         errors: List[Dict],
                         synthetic: List[Dict]) -> List[Dict]:
        """
        合并三类数据集，为每条记录添加来源标记

        参数:
            original: 原始 Thought 数据
            errors: 错误注入数据
            synthetic: 合成定理数据

        返回:
            List[Dict]: 合并后的数据集
        """
        merged = []

        # 原始数据标记来源
        for item in original:
            item_copy = copy.copy(item)
            if "source" not in item_copy:
                item_copy["source"] = "original"
            merged.append(item_copy)

        # 错误注入数据
        for item in errors:
            item_copy = copy.copy(item)
            if "source" not in item_copy:
                item_copy["source"] = "error_injection"
            merged.append(item_copy)

        # 合成定理数据
        for item in synthetic:
            item_copy = copy.copy(item)
            if "source" not in item_copy:
                item_copy["source"] = "synthetic"
            merged.append(item_copy)

        # 打乱顺序
        random.shuffle(merged)
        return merged

    def _deduplicate(self, data: List[Dict]) -> List[Dict]:
        """
        数据去重：基于核心字段的哈希值去重

        参数:
            data: 待去重数据

        返回:
            List[Dict]: 去重后的数据
        """
        seen_hashes = set()
        unique_data = []

        for item in data:
            # 构建去重键：基于定理名 + 策略 + 状态
            key_parts = [
                item.get("theorem_name", item.get("theorem_statement", "")),
                item.get("tactic", item.get("original_tactic", "")),
                item.get("state_before", ""),
            ]
            key_str = "||".join(str(p) for p in key_parts)
            h = compute_hash(key_str)

            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_data.append(item)

        removed = len(data) - len(unique_data)
        if removed > 0:
            logger.info(f"去重移除 {removed} 条重复记录")

        return unique_data

    def _split_dataset(self, data: List[Dict]):
        """
        将数据集分割为 train/val/test

        参数:
            data: 完整数据集
        """
        train_ratio = self.split_config.get("train_ratio", 0.9)
        val_ratio = self.split_config.get("val_ratio", 0.05)
        # test_ratio 为剩余部分

        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        self.stats["train_count"] = len(train_data)
        self.stats["val_count"] = len(val_data)
        self.stats["test_count"] = len(test_data)

        # 保存分割后的数据集
        output_dir = self.final_output_path
        ensure_dir(output_dir)

        train_file = os.path.join(output_dir, "train.jsonl")
        val_file = os.path.join(output_dir, "val.jsonl")
        test_file = os.path.join(output_dir, "test.jsonl")

        save_jsonl(train_data, train_file)
        save_jsonl(val_data, val_file)
        save_jsonl(test_data, test_file)

        logger.info(f"训练集: {len(train_data)} 条 → {train_file}")
        logger.info(f"验证集: {len(val_data)} 条 → {val_file}")
        logger.info(f"测试集: {len(test_data)} 条 → {test_file}")

    # ============================================================
    # 兼容旧接口
    # ============================================================

    def inject_errors(self, dataset: List[Dict]) -> List[Dict]:
        """
        错误注入（兼容旧接口）

        参数:
            dataset: 原始数据集

        返回:
            List[Dict]: 包含错误的数据集
        """
        if not self.error_injection_config:
            return []
        injector = ErrorInjector(self.error_injection_config)
        return injector.inject(dataset)

    def generate_synthetic_theorems(self, num_theorems: int = 1000) -> List[Dict]:
        """
        生成合成定理（兼容旧接口）

        参数:
            num_theorems: 生成定理数量

        返回:
            List[Dict]: 合成定理数据集
        """
        config = copy.copy(self.synthetic_config)
        config["num_theorems"] = num_theorems
        generator = SyntheticTheoremGenerator(config)
        return generator.generate()

    def create_error_correction_pairs(self, correct_tactic: str,
                                       state: str) -> Tuple[str, str]:
        """
        创建错误-修正对（兼容旧接口）

        参数:
            correct_tactic: 正确的策略
            state: 当前状态

        返回:
            Tuple[str, str]: (错误策略, 正确策略)
        """
        injector = ErrorInjector(self.error_injection_config)
        error_tactic, _ = injector._inject_typo(correct_tactic)
        if not error_tactic:
            error_tactic, _ = injector._inject_wrong_tactic(correct_tactic)
        return (error_tactic, correct_tactic)


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="W8 数据增强")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml",
                        help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/augmentation.log", encoding="utf-8"),
        ],
    )

    config = load_yaml(args.config)
    augmentor = DataAugmentation(config)
    output_path = augmentor.run()
    print(f"\n数据增强完成！输出目录: {output_path}")


if __name__ == "__main__":
    main()
