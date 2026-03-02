#!/usr/bin/env python3
"""
Task 2.3: Reflection 数据生成脚本
@author ygw
日期: 2026-03-01

为错误注入记录生成 <reflection> 文本，构建认知修复微调指令。
利用 Teacher Model (DeepSeek-V3 / GPT-4o) 或模板回退生成反思内容。

三阶段流水线:
  Phase 1: 数据过滤 — 筛选有效错误记录
  Phase 2: Reflection 生成 — Teacher Model API + 模板回退
  Phase 3: SFT 数据集构建 — 格式化 + train/val/test 分割

使用方式:
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --mode template
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --dry-run
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.utils import (
    load_yaml, load_jsonl, save_jsonl, save_json,
    ensure_dir, get_timestamp, batch_iter, ProgressTracker,
)

# aiohttp 可选导入
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# 日志器
logger = logging.getLogger("rtap.reflection")


# ================================================================
# 常量定义
# ================================================================

# 有效的 verification_status 值
# v2: 仅保留 Pantograph 真正验证通过的记录（错误策略确实触发了 Lean4 报错）
# @author ygw 2026-03-01 修正：从 replay_failed_at_step_ 改为 verified
VALID_STATUS_PREFIX = "verified"

# 默认的 Reflection Prompt 模板（v2: 增加 repair_hint + 抗误导指令）
# @author ygw 2026-03-02 优化：减少错误信息误导，增强正确策略关联
DEFAULT_REFLECTION_PROMPT = """You are a Lean4 proof error analyst. Given a proof step that failed, analyze the error and explain the correct fix.

[Proof State]
{state_before}

[Correct Tactic]
{original_tactic}

[Attempted (Wrong) Tactic]
{error_tactic}

[Error Type]
{error_type}

[Repair Hint]
{repair_hint}

[Lean4 Error Message]
{error_message}

[Critical Rules]
Write 2-4 sentences explaining the error and the fix. Follow this reasoning order:
1. FIRST, analyze the proof state and explain WHY the correct tactic `{original_tactic}` is the right choice for this goal
2. THEN, explain the semantic gap between the attempted tactic `{error_tactic}` and the correct tactic — what conceptual mistake does the wrong tactic represent?
3. Finally, briefly note the error type ({error_type}) and how the repair hint applies

IMPORTANT:
- The Lean4 error message may be MISLEADING or refer to internal elaboration details. Do NOT simply parrot the error message. Instead, focus on the structural mismatch between the attempted tactic and the proof goal.
- Always reference the correct tactic `{original_tactic}` explicitly in your explanation.
    - You MUST incorporate the repair hint text into your explanation: "{repair_hint}"
    - Reference the proof goal (the line after ⊢) when explaining why the correct tactic fits.
- Output the reflection text directly with no XML tags or prefixes
- Write in English"""

# 模板回退 — 多变体模板引擎 (v3: 三步推理对齐 + error_message 集成 + 假设提取 + 多变体)
# @author ygw 2026-03-02 v3 重构：
#   1. 每种错误类型 3 个结构变体，随机选择提升多样性
#   2. 集成 error_message（Lean4 真实报错），不再忽略
#   3. 提取假设上下文 {hypotheses}，引用具体变量名/类型
#   4. 与 API Prompt 三步推理结构对齐：正确策略分析 → 语义偏差 → 错误类型+hint
#   5. 消除泛化描述，使用 tactic 动词和 goal 具体结构

REFLECTION_TEMPLATES = {
    # ========== tactic_typo: 3 个变体 ==========
    "tactic_typo": [
        # 变体 A: 先分析正确策略 → 再解释拼写错误的影响
        (
            'The correct tactic `{original_tactic}` is the right choice for the goal '
            '{state_summary} because it references a valid Lean4 declaration that '
            'directly operates on the current proof structure{hypotheses_clause}. '
            'The attempted tactic `{error_tactic}` fails because the identifier is misspelled — '
            'Lean4\'s name resolution cannot find a matching declaration, '
            'producing the error: {error_message_short}. '
            'This is a tactic_typo error; {repair_hint}.'
        ),
        # 变体 B: 先抛出 Lean4 报错现象 → 再定位拼写根因
        (
            'Lean4 reports {error_message_short} when executing `{error_tactic}` '
            'against the goal {state_summary}. '
            'The root cause is a spelling error: the identifier in `{error_tactic}` '
            'does not match any known declaration in the current environment{hypotheses_clause}. '
            '{repair_hint}. The correct tactic `{original_tactic}` uses the properly '
            'spelled identifier and resolves the goal.'
        ),
        # 变体 C: 对比正确 vs 错误标识符
        (
            'Comparing `{error_tactic}` with the correct tactic `{original_tactic}`, '
            'the attempted name contains a typo that prevents Lean4 from resolving the identifier. '
            'Given the goal {state_summary}{hypotheses_clause}, '
            'Lean4 requires exact name matching for declarations, '
            'and the misspelling triggers: {error_message_short}. '
            '{repair_hint}.'
        ),
    ],
    # ========== wrong_tactic: 3 个变体 ==========
    "wrong_tactic": [
        # 变体 A: 先解释正确策略为何适合 → 再说错误策略的语义偏差
        (
            'The correct tactic `{original_tactic}` fits the goal {state_summary} '
            'because it applies the appropriate transformation to the proof structure'
            '{hypotheses_clause}. '
            'In contrast, `{error_tactic}` targets a fundamentally different operation — '
            'it does not match the logical connective or algebraic structure '
            'required by the goal, leading to: {error_message_short}. '
            'This is a wrong_tactic error; {repair_hint}.'
        ),
        # 变体 B: 先从 goal 结构出发 → 推导出需要什么策略 → 排除错误策略
        (
            'The goal {state_summary} demands a tactic that addresses its specific structure'
            '{hypotheses_clause}. '
            '`{original_tactic}` does exactly this by applying the matching lemma or rewrite. '
            'The attempted tactic `{error_tactic}` operates on a different mathematical '
            'structure, creating a semantic mismatch — Lean4 reports: {error_message_short}. '
            '{repair_hint}.'
        ),
        # 变体 C: 先描述 Lean4 报错 → 分析语义不匹配 → 给出修复
        (
            'Lean4 rejects `{error_tactic}` with: {error_message_short}. '
            'This happens because the goal {state_summary} requires `{original_tactic}`, '
            'which correctly matches the proof obligation{hypotheses_clause}. '
            'The wrong tactic attempts an incompatible operation '
            'that does not align with the goal\'s logical form. '
            '{repair_hint}.'
        ),
    ],
    # ========== argument_error: 3 个变体 ==========
    "argument_error": [
        # 变体 A: 正确参数分析 → 错误参数对比
        (
            'The correct invocation `{original_tactic}` provides arguments that '
            'match the types and terms in the proof context for goal {state_summary}'
            '{hypotheses_clause}. '
            'The attempted tactic `{error_tactic}` supplies arguments with wrong types, '
            'wrong order, or references variables not present in the current context, '
            'triggering: {error_message_short}. '
            'This is an argument_error; {repair_hint}.'
        ),
        # 变体 B: 从 Lean4 报错出发 → 追溯到参数问题
        (
            'Lean4 reports {error_message_short} when processing `{error_tactic}`. '
            'The goal {state_summary} requires specific arguments '
            'that match the available hypotheses{hypotheses_clause}. '
            'The attempted arguments are incompatible — they either have the wrong type, '
            'are in the wrong order, or reference unavailable terms. '
            '{repair_hint}. The correct form is `{original_tactic}`.'
        ),
        # 变体 C: 假设上下文驱动的参数分析
        (
            'Given the proof state with goal {state_summary}{hypotheses_clause}, '
            'the tactic arguments must reference terms available in this context. '
            '`{error_tactic}` fails because its arguments do not type-check '
            'against the current goal — {error_message_short}. '
            '`{original_tactic}` provides the correct arguments; {repair_hint}.'
        ),
    ],
    # ========== missing_step: 3 个变体 ==========
    "missing_step": [
        # 变体 A: 分析为何需要中间步骤 → 解释跳步失败原因
        (
            'The goal {state_summary} cannot be closed directly by `{error_tactic}` '
            'because there are intermediate conditions that must be established first'
            '{hypotheses_clause}. '
            'The proof requires `{original_tactic}` to introduce the necessary '
            'intermediate reasoning step. '
            'Lean4 confirms this gap: {error_message_short}. '
            'This is a missing_step error; {repair_hint}.'
        ),
        # 变体 B: 从 Lean4 报错推导出缺失步骤
        (
            'Lean4 reports {error_message_short} because `{error_tactic}` '
            'attempts to jump directly to a conclusion that requires intermediate work. '
            'The goal {state_summary} has a logical gap{hypotheses_clause} '
            'that `{original_tactic}` bridges by providing the missing reasoning step. '
            '{repair_hint}.'
        ),
        # 变体 C: 对比直接 vs 分步证明
        (
            'The attempted tactic `{error_tactic}` tries to close the goal {state_summary} '
            'in one step, but the proof structure requires an intermediate step'
            '{hypotheses_clause}. '
            'Without this step, Lean4 cannot verify the logical chain: {error_message_short}. '
            'The correct approach is `{original_tactic}`, which provides '
            'the necessary bridge; {repair_hint}.'
        ),
    ],
}

# 通用回退模板 — 多变体 (v3)
GENERIC_TEMPLATES = [
    (
        'The correct tactic `{original_tactic}` addresses the goal {state_summary} '
        'by applying the appropriate transformation{hypotheses_clause}. '
        'The attempted tactic `{error_tactic}` fails with: {error_message_short}. '
        'This is a {error_type} error; {repair_hint}.'
    ),
    (
        'Lean4 rejects `{error_tactic}` with {error_message_short} '
        'when applied to goal {state_summary}{hypotheses_clause}. '
        '{repair_hint}. The correct tactic is `{original_tactic}`, '
        'which matches the proof obligation.'
    ),
    (
        'Given the goal {state_summary}{hypotheses_clause}, '
        '`{error_tactic}` is incorrect — Lean4 reports: {error_message_short}. '
        'The fix is `{original_tactic}`; {repair_hint}.'
    ),
]


# Thought 增强后缀 — 按错误类型追加诊断性说明 (v1)
# @author ygw 2026-03-02
# 目的: 使 thought 字段从「纯正确策略描述」升级为「正确策略 + 错误类型导向」
# 训练时模型能学到：不同错误类型需要不同的纠错视角
ERROR_TYPE_THOUGHT_SUFFIX = {
    "tactic_typo": (
        " [Error Context] The error is purely syntactic — the intended tactic "
        "is clear from the proof structure, only the spelling needs correction. "
        "The key insight is recognizing the correct identifier name."
    ),
    "wrong_tactic": (
        " [Error Context] The error is strategic — a fundamentally different "
        "tactic must be chosen to match the goal structure. The key insight is "
        "understanding why this specific tactic fits the logical form of the goal."
    ),
    "missing_step": (
        " [Error Context] The error is a missing reasoning step — the placeholder "
        "`done` must be replaced with the actual required tactic. The key insight is "
        "identifying what intermediate step the proof requires to bridge the gap."
    ),
    "argument_error": (
        " [Error Context] The error is in the arguments — the tactic choice is correct "
        "but its parameters need adjustment to match the types and terms in the "
        "current proof context. The key insight is aligning argument types with the goal."
    ),
}


def _enhance_thought(thought: str, error_type: str) -> str:
    """
    按错误类型增强 thought 文本，在原始 thought 末尾追加错误导向后缀。

    目的：使 thought 从「纯正确策略描述」升级为「正确策略 + 错误类型诊断视角」，
    训练模型在纠错时具备更强的针对性。

    参数:
        thought: 原始 thought 文本（描述正确策略为何有效）
        error_type: 错误类型 (tactic_typo / wrong_tactic / missing_step / argument_error)

    返回:
        str: 增强后的 thought 文本

    @author ygw 2026-03-02
    """
    if not thought:
        return thought
    suffix = ERROR_TYPE_THOUGHT_SUFFIX.get(error_type, "")
    if suffix:
        # 确保原文末尾有句号，再追加后缀
        thought_trimmed = thought.rstrip()
        if thought_trimmed and thought_trimmed[-1] not in '.!?。':
            thought_trimmed += '.'
        return thought_trimmed + suffix
    return thought


def _safe_format(template: str, **kwargs) -> str:
    """
    安全的模板填充，避免 Lean4 证明状态中的花括号 {} 干扰 str.format()

    Lean4 数据中大量出现 Category.{v, u} C、{x : ℕ | x > 0} 等花括号，
    使用 str.format() 会触发 KeyError / ValueError。
    此方法改用逐项字符串替换，仅替换已知的占位符。

    参数:
        template: 含 {key} 占位符的模板字符串
        **kwargs: 键值对

    返回:
        str: 填充后的字符串

    @author ygw
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# ================================================================
# 数据结构
# ================================================================

@dataclass
class ReflectionRecord:
    """
    Reflection 标注记录

    属性:
        original_tactic: 原始正确策略
        error_tactic: 注入错误后的策略
        error_type: 错误类型
        state_before: 错误发生时的证明状态
        state_after: 正确策略执行后的状态
        theorem_name: 定理名称
        thought: 正确策略的 Thought 文本
        repair_hint: 修复提示
        error_message: Pantograph 返回的错误信息
        verification_status: 验证状态
        reflection: 生成的 Reflection 文本
        reflection_source: Reflection 来源 (api / template)
    """
    original_tactic: str = ""
    error_tactic: str = ""
    error_type: str = ""
    state_before: str = ""
    state_after: str = ""
    theorem_name: str = ""
    theorem_full_name: str = ""
    step_index: str = ""
    thought: str = ""
    repair_hint: str = ""
    source: str = ""
    error_message: str = ""
    verification_status: str = ""
    reflection: str = ""
    reflection_source: str = ""


# ================================================================
# Phase 1: 数据过滤器
# ================================================================

class ReflectionDataFilter:
    """
    过滤错误注入数据，筛选适合生成 Reflection 的记录。

    过滤条件:
    - verification_status 为 replay_failed_at_step_* 系列
    - 必须包含 error_message, state_before 等关键字段
    - 可选：按错误类型/最大数量限制

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据过滤器

        参数:
            config: 过滤配置字典 (data_filter 节)
        """
        self.input_path = config.get("input_path", "")
        self.valid_statuses = set(config.get("valid_statuses", []))
        self.required_fields = config.get("required_fields", [])
        self.max_samples = config.get("max_samples", 0)
        self.max_per_error_type = config.get("max_per_error_type", 0)

    def load_and_filter(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        加载数据并执行过滤

        返回:
            Tuple[List[Dict], Dict]: (过滤后的记录, 统计信息)
        """
        logger.info("=" * 60)
        logger.info("Phase 1: 数据过滤")
        logger.info("=" * 60)
        start_time = time.time()

        # 加载原始数据
        if not os.path.exists(self.input_path):
            logger.error(f"输入文件不存在: {self.input_path}")
            return [], {}

        logger.info(f"加载输入文件: {self.input_path}")
        records = load_jsonl(self.input_path)
        total = len(records)
        logger.info(f"原始记录数: {total:,}")

        # 统计原始分布
        original_status_dist = Counter(r.get("verification_status", "") for r in records)
        original_type_dist = Counter(r.get("error_type", "") for r in records)

        # 过滤 1: verification_status
        if self.valid_statuses:
            filtered = [r for r in records if r.get("verification_status", "") in self.valid_statuses]
        else:
            # 使用前缀匹配
            filtered = [
                r for r in records
                if r.get("verification_status", "").startswith(VALID_STATUS_PREFIX)
            ]
        after_status = len(filtered)
        logger.info(f"verification_status 过滤后: {after_status:,} "
                     f"(去除 {total - after_status:,} 条)")

        # 过滤 2: 必须字段
        if self.required_fields:
            filtered = [
                r for r in filtered
                if all(r.get(f, "").strip() for f in self.required_fields)
            ]
        after_fields = len(filtered)
        logger.info(f"必须字段过滤后: {after_fields:,} "
                     f"(去除 {after_status - after_fields:,} 条)")

        # 过滤 3: 按错误类型限制
        if self.max_per_error_type > 0:
            type_counts: Dict[str, int] = {}
            limited = []
            for r in filtered:
                et = r.get("error_type", "unknown")
                cnt = type_counts.get(et, 0)
                if cnt < self.max_per_error_type:
                    limited.append(r)
                    type_counts[et] = cnt + 1
            logger.info(f"按错误类型限制后: {len(limited):,} "
                         f"(每类最多 {self.max_per_error_type:,})")
            filtered = limited

        # 过滤 4: 全局采样限制
        if self.max_samples > 0 and len(filtered) > self.max_samples:
            random.shuffle(filtered)
            filtered = filtered[:self.max_samples]
            logger.info(f"全局采样限制后: {len(filtered):,}")

        # 过滤后分布统计
        filtered_type_dist = Counter(r.get("error_type", "") for r in filtered)
        filtered_status_dist = Counter(r.get("verification_status", "") for r in filtered)

        elapsed = time.time() - start_time
        stats = {
            "phase": "data_filter",
            "total_records": total,
            "after_status_filter": after_status,
            "after_field_filter": after_fields,
            "final_count": len(filtered),
            "elapsed_seconds": round(elapsed, 2),
            "original_status_distribution": dict(original_status_dist.most_common()),
            "original_type_distribution": dict(original_type_dist.most_common()),
            "filtered_type_distribution": dict(filtered_type_dist.most_common()),
            "filtered_status_distribution": dict(filtered_status_dist.most_common()),
        }

        logger.info(f"Phase 1 完成: {len(filtered):,} 条有效记录, 耗时 {elapsed:.1f}s")
        logger.info(f"  错误类型分布: {dict(filtered_type_dist.most_common())}")

        return filtered, stats


# ================================================================
# Phase 2: Reflection 生成器
# ================================================================

class ReflectionGenerator:
    """
    使用 Teacher Model 为错误记录生成 Reflection 文本。
    支持异步并发 API 调用 + 模板回退。

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Reflection 生成器

        参数:
            config: reflection_generation 配置节
        """
        # Teacher Model
        teacher_cfg = config.get("teacher_model", {})
        self.provider = teacher_cfg.get("provider", "deepseek-v3")
        self.model_name = teacher_cfg.get("model_name", "deepseek-chat")
        self.api_base = teacher_cfg.get("api_base", "https://api.deepseek.com/v1")
        self.api_key_env = teacher_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "")
        self.fallback_config = teacher_cfg.get("fallback", None)

        # 生成参数
        gen_cfg = config.get("generation", {})
        self.temperature = gen_cfg.get("temperature", 0.7)
        self.max_tokens = gen_cfg.get("max_tokens", 600)
        self.top_p = gen_cfg.get("top_p", 0.95)

        # Prompt
        self.prompt_template = config.get("prompt_template", DEFAULT_REFLECTION_PROMPT)

        # 批处理
        batch_cfg = config.get("batch", {})
        self.batch_size = batch_cfg.get("batch_size", 30)
        self.max_concurrent = batch_cfg.get("max_concurrent_requests", 25)
        self.rate_limit_rpm = batch_cfg.get("rate_limit_rpm", 500)
        self.save_interval = batch_cfg.get("save_interval", 2000)

        # 质量过滤
        quality_cfg = config.get("quality_filter", {})
        self.min_length = quality_cfg.get("min_length", 40)
        self.max_length = quality_cfg.get("max_length", 800)
        self.forbidden_patterns = quality_cfg.get("forbidden_patterns", [])
        self.require_keywords = quality_cfg.get("require_keywords", [])

        # 模板回退 (v3: 支持多变体列表格式)
        fallback_cfg = config.get("template_fallback", {})
        self.template_fallback_enabled = fallback_cfg.get("enable", True)
        custom_templates = fallback_cfg.get("templates", {})
        # 深拷贝内置多变体模板
        self.reflection_templates = {k: list(v) for k, v in REFLECTION_TEMPLATES.items()}
        if custom_templates:
            # YAML 覆盖: 若 YAML 值是字符串则包装为列表追加到变体池
            for etype, tpl in custom_templates.items():
                if isinstance(tpl, list):
                    self.reflection_templates[etype] = tpl
                elif isinstance(tpl, str):
                    if etype in self.reflection_templates:
                        self.reflection_templates[etype].append(tpl)
                    else:
                        self.reflection_templates[etype] = [tpl]

        # 统计
        self.stats = {
            "total_input": 0,
            "api_generated": 0,
            "template_generated": 0,
            "filtered": 0,
            "errors": 0,
        }

        if not self.api_key:
            logger.warning(f"API Key 环境变量 {self.api_key_env} 未设置，将使用模板回退方案")

    def run(self, records: List[Dict], output_dir: str,
            mode: str = "auto",
            batch_id: int = 0) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        执行 Reflection 生成

        参数:
            records: 过滤后的错误记录列表
            output_dir: 输出目录
            mode: 运行模式 ("auto" / "api" / "template")
            batch_id: 分批执行的批次 ID (0=不分批, 1-N=批次号)

        返回:
            Tuple[List[Dict], Dict]: (带 reflection 的记录, 统计信息)

        @author ygw 2026-03-02 新增 batch_id 参数支持分批执行
        """
        self._batch_id = batch_id
        logger.info("=" * 60)
        logger.info("Phase 2: Reflection 生成")
        logger.info("=" * 60)
        start_time = time.time()

        self.stats["total_input"] = len(records)
        ensure_dir(output_dir)

        # 确定运行模式
        if mode == "template":
            use_api = False
        elif mode == "api":
            if not self.api_key:
                logger.error("API 模式但 API Key 未设置，切换到模板模式")
                use_api = False
            elif not HAS_AIOHTTP:
                logger.error("API 模式但 aiohttp 未安装 (pip install aiohttp)，切换到模板模式")
                use_api = False
            else:
                use_api = True
        else:  # auto
            use_api = bool(self.api_key) and HAS_AIOHTTP

        if use_api:
            logger.info(f"使用 API 模式: {self.provider} / {self.model_name}")
            logger.info(f"  并发数: {self.max_concurrent}, RPM: {self.rate_limit_rpm}")
            results = asyncio.run(self._async_generate(records, output_dir))
        else:
            logger.info("使用模板回退模式")
            results = self._template_generate(records)

        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        speed = len(results) / elapsed if elapsed > 0 else 0
        self.stats["speed_per_sec"] = round(speed, 1)

        logger.info(f"Phase 2 完成: {len(results):,} 条 Reflection")
        logger.info(f"  API: {self.stats['api_generated']:,}, "
                     f"模板: {self.stats['template_generated']:,}, "
                     f"过滤: {self.stats['filtered']:,}, "
                     f"错误: {self.stats['errors']:,}")
        logger.info(f"  耗时: {elapsed:.1f}s, 速度: {speed:.1f} 条/s")

        return results, dict(self.stats)

    # --------------------------------------------------------
    # 异步 API 生成
    # --------------------------------------------------------

    async def _async_generate(self, records: List[Dict],
                               output_dir: str) -> List[Dict]:
        """
        异步并发调用 Teacher Model 生成 Reflection

        参数:
            records: 输入记录
            output_dir: 输出目录（存中间结果）

        返回:
            List[Dict]: 带 reflection 字段的记录
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        rps = self.rate_limit_rpm / 60.0
        interval = 1.0 / rps if rps > 0 else 0.5

        results: List[Dict] = []
        processed = 0
        checkpoint_saved = 0  # 已保存到 checkpoint 的记录数，用于增量保存
        last_print = time.time()
        start_time = time.time()

        # 分批执行时使用批次专用 checkpoint 路径，避免不同批次间冲突
        # @author ygw 2026-03-02 新增分批 checkpoint 支持
        if hasattr(self, '_batch_id') and self._batch_id > 0:
            checkpoint_path = os.path.join(
                output_dir, f".reflection_batch_{self._batch_id}_checkpoint.jsonl"
            )
        else:
            checkpoint_path = os.path.join(output_dir, ".reflection_checkpoint.jsonl")

        # 清除上次残留的 checkpoint 文件（非分批模式）
        # 分批模式下保留 checkpoint 以支持断点恢复
        if self._batch_id == 0 and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 5)
        timeout = aiohttp.ClientTimeout(total=150)  # v2.1: 90→150s 降低超时导致的回退

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def process_one(record: Dict) -> Optional[Dict]:
                """处理单条记录"""
                nonlocal processed
                async with semaphore:
                    prompt = self._build_prompt(record)
                    reflection = await self._async_api_call(session, prompt)
                    processed += 1

                    # v2.1: 后处理 API 输出 — 补充缺失的 hint/goal 引用
                    if reflection:
                        reflection = self._enrich_api_output(reflection, record)

                    if reflection and self._validate_reflection(reflection, record):
                        rec = dict(record)
                        rec["reflection"] = reflection
                        rec["reflection_source"] = "api"
                        # v1: 增强 thought — 追加错误类型导向后缀
                        # @author ygw 2026-03-02
                        rec["thought"] = _enhance_thought(
                            rec.get("thought", ""), rec.get("error_type", "")
                        )
                        self.stats["api_generated"] += 1
                        return rec
                    elif reflection:
                        # API 返回了但质量不合格，尝试模板
                        self.stats["filtered"] += 1
                        if self.template_fallback_enabled:
                            fallback = self._generate_template_reflection(record)
                            if fallback:
                                rec = dict(record)
                                rec["reflection"] = fallback
                                rec["reflection_source"] = "template_after_api"
                                # v1: 增强 thought @author ygw 2026-03-02
                                rec["thought"] = _enhance_thought(
                                    rec.get("thought", ""), rec.get("error_type", "")
                                )
                                self.stats["template_generated"] += 1
                                return rec
                        return None
                    else:
                        # API 失败，使用模板回退
                        self.stats["errors"] += 1
                        if self.template_fallback_enabled:
                            fallback = self._generate_template_reflection(record)
                            if fallback:
                                rec = dict(record)
                                rec["reflection"] = fallback
                                rec["reflection_source"] = "template_fallback"
                                # v1: 增强 thought @author ygw 2026-03-02
                                rec["thought"] = _enhance_thought(
                                    rec.get("thought", ""), rec.get("error_type", "")
                                )
                                self.stats["template_generated"] += 1
                                return rec
                        return None

            # 分块处理
            chunk_size = self.max_concurrent * 3
            for chunk_start in range(0, len(records), chunk_size):
                chunk = records[chunk_start:chunk_start + chunk_size]
                tasks = []
                for record in chunk:
                    tasks.append(asyncio.create_task(process_one(record)))
                    if interval > 0:
                        await asyncio.sleep(interval)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, Exception):
                        logger.warning(f"任务异常: {r}")
                        self.stats["errors"] += 1
                    elif r is not None:
                        results.append(r)

                # 增量中间保存（仅保存自上次 checkpoint 以来新增的记录）
                if len(results) - checkpoint_saved >= self.save_interval:
                    new_results = results[checkpoint_saved:]
                    save_jsonl(new_results, checkpoint_path, mode='a')
                    logger.info(f"中间保存: +{len(new_results)} 条 (累计 {len(results)})")
                    checkpoint_saved = len(results)

                # 进度报告
                now = time.time()
                if now - last_print >= 15:
                    elapsed = now - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    total_gen = self.stats["api_generated"] + self.stats["template_generated"]
                    remaining = (len(records) - processed) / speed if speed > 0 else 0
                    print(
                        f"[Reflection] 进度: {processed}/{len(records)} "
                        f"({processed/len(records)*100:.1f}%) | "
                        f"生成: {total_gen} | "
                        f"过滤: {self.stats['filtered']} | "
                        f"错误: {self.stats['errors']} | "
                        f"速度: {speed:.1f}/s | "
                        f"剩余: {remaining/3600:.1f}h",
                        flush=True,
                    )
                    last_print = now

        return results

    async def _async_api_call(self, session: 'aiohttp.ClientSession',
                               prompt: str) -> Optional[str]:
        """
        异步调用 Teacher Model API

        参数:
            session: aiohttp 客户端会话
            prompt: 构建好的 Prompt

        返回:
            Optional[str]: 生成的 Reflection 文本，失败返回 None
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        url = f"{self.api_base}/chat/completions"

        for attempt in range(4):
            try:
                async with session.post(url, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:  # v2.1: 60→120s
                    if resp.status == 429:
                        wait = min(2 ** attempt * 2, 30)
                        logger.warning(f"API 429 限速，等待 {wait}s 后重试")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    result = await resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        text = choices[0]["message"]["content"].strip()
                        return text
                    return None
            except asyncio.TimeoutError:
                logger.warning(f"API 超时 (attempt {attempt + 1}/4)")
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                logger.warning(f"API 异步调用失败 (attempt {attempt + 1}/4): {e}")
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        return None

    # --------------------------------------------------------
    # 模板生成模式
    # --------------------------------------------------------

    def _template_generate(self, records: List[Dict]) -> List[Dict]:
        """
        纯模板模式生成 Reflection

        参数:
            records: 输入记录

        返回:
            List[Dict]: 带 reflection 字段的记录
        """
        results = []
        tracker = ProgressTracker(
            total=len(records),
            desc="Reflection 模板生成",
            log_interval=5000,
        )

        for record in records:
            reflection = self._generate_template_reflection(record)
            if reflection and self._validate_reflection(reflection, record):
                rec = dict(record)
                rec["reflection"] = reflection
                rec["reflection_source"] = "template"
                results.append(rec)
                self.stats["template_generated"] += 1
            else:
                self.stats["filtered"] += 1
            tracker.update(success=True)

        tracker.finish()
        return results

    def _generate_template_reflection(self, record: Dict) -> Optional[str]:
        """
        使用多变体模板为单条记录生成 Reflection
        (v3: 三步推理对齐 + error_message 集成 + 假设提取 + 随机变体)

        v3 核心改进:
          1. 每种错误类型 3 个结构变体，随机选择 → 避免 SFT 过拟合
          2. 集成 error_message（Lean4 真实报错）→ 缩小与 API 质量差距
          3. 提取假设上下文（变量名/类型）→ 引用具体信息而非泛化描述
          4. 三步推理与 API Prompt 对齐: 正确策略 → 语义偏差 → 错误类型+hint

        参数:
            record: 错误记录

        返回:
            Optional[str]: 生成的 Reflection 文本

        @author ygw 2026-03-02 v3 重构
        """
        error_type = record.get("error_type", "unknown")
        templates = self.reflection_templates.get(error_type)

        # 选择模板变体: 按记录哈希确定性选择（保证相同记录 → 相同输出，利于调试）
        if templates and isinstance(templates, list):
            variant_idx = self._deterministic_variant_index(record, len(templates))
            template = templates[variant_idx]
        elif templates and isinstance(templates, str):
            # 兼容旧版单模板字符串 (YAML 覆盖场景)
            template = templates
        else:
            # 未知错误类型 → 通用模板
            variant_idx = self._deterministic_variant_index(record, len(GENERIC_TEMPLATES))
            template = GENERIC_TEMPLATES[variant_idx]

        # 提取 proof state 上下文
        state_before_full = record.get("state_before", "")
        state_summary = self._extract_goal_summary(state_before_full)
        hypotheses_clause = self._extract_hypotheses_clause(state_before_full)
        error_message_short = self._summarize_error_message(
            record.get("error_message", "")
        )

        # 截断过长的字段以避免模板膨胀
        state_before = state_before_full
        if len(state_before) > 300:
            state_before = state_before[:300] + "..."

        # 获取 repair_hint；若缺失则构造基本提示
        repair_hint = record.get("repair_hint", "")
        if not repair_hint.strip():
            repair_hint = f"the correct approach is to use `{record.get('original_tactic', '')}`"

        try:
            reflection = self._safe_format(
                template,
                error_tactic=record.get("error_tactic", ""),
                original_tactic=record.get("original_tactic", ""),
                error_type=error_type,
                error_message=record.get("error_message", "unknown error"),
                error_message_short=error_message_short,
                repair_hint=repair_hint,
                state_before=state_before,
                state_summary=state_summary,
                hypotheses_clause=hypotheses_clause,
            )
            return reflection.strip()
        except Exception as e:
            logger.debug(f"模板格式化失败: {e}")
            # 使用通用模板列表的第一个
            try:
                return self._safe_format(
                    GENERIC_TEMPLATES[0],
                    error_tactic=record.get("error_tactic", ""),
                    original_tactic=record.get("original_tactic", ""),
                    error_type=error_type,
                    error_message_short=error_message_short,
                    repair_hint=repair_hint,
                    state_summary=state_summary,
                    hypotheses_clause=hypotheses_clause,
                ).strip()
            except Exception:
                return None

    @staticmethod
    def _deterministic_variant_index(record: Dict, num_variants: int) -> int:
        """
        根据记录内容确定性地选择模板变体索引

        使用记录的关键字段哈希来选择变体，确保:
        - 相同记录始终选择相同变体（可复现）
        - 不同记录均匀分布到各变体（多样性）

        参数:
            record: 错误记录
            num_variants: 变体总数

        返回:
            int: 变体索引 [0, num_variants)

        @author ygw 2026-03-02 v3 新增
        """
        key = (
            record.get("theorem_name", "") +
            record.get("error_tactic", "") +
            str(record.get("step_index", 0))
        )
        h = hashlib.md5(key.encode()).hexdigest()
        return int(h[:8], 16) % num_variants

    @staticmethod
    def _extract_hypotheses_clause(state_before: str) -> str:
        """
        从 proof state 中提取假设变量的简洁从句

        解析 state_before 中 ⊢ 之前的假设行，提取变量名和类型，
        生成形如 " (with hypotheses h : P, n : ℕ)" 的从句。
        若无假设或解析失败，返回空字符串。

        参数:
            state_before: 证明状态文本

        返回:
            str: 假设从句（含前导空格）或空字符串

        @author ygw 2026-03-02 v3 新增
        """
        if not state_before or "⊢" not in state_before:
            return ""

        # ⊢ 之前的部分包含假设
        goal_pos = state_before.find("⊢")
        hyp_section = state_before[:goal_pos].strip()
        if not hyp_section:
            return ""

        # 解析假设行: 每行通常是 "varname : Type" 格式
        hypotheses = []
        for line in hyp_section.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 跳过 case/context 标记行
            if line.startswith("case ") or line.startswith("context"):
                continue
            # 提取 "name : Type" 模式
            if ":" in line:
                parts = line.split(":", 1)
                var_name = parts[0].strip()
                var_type = parts[1].strip()
                # 截断过长的类型
                if len(var_type) > 60:
                    var_type = var_type[:60] + "..."
                if var_name and len(var_name) < 30:
                    hypotheses.append(f"{var_name} : {var_type}")

        if not hypotheses:
            return ""

        # 最多展示 4 个假设，避免模板膨胀
        display = hypotheses[:4]
        if len(hypotheses) > 4:
            display.append(f"... ({len(hypotheses) - 4} more)")

        return f" (with hypotheses {', '.join(display)})"

    @staticmethod
    def _summarize_error_message(error_message: str) -> str:
        """
        将 Lean4 原始错误信息压缩为模板可用的摘要

        Lean4 错误信息可能非常长（含内部 elaboration 细节），
        提取第一行有意义的错误描述，截断到合理长度。

        参数:
            error_message: 原始错误信息

        返回:
            str: 错误摘要（≤120 字符）

        @author ygw 2026-03-02 v3 新增
        """
        if not error_message or error_message.strip() == "":
            return "an elaboration error"

        msg = error_message.strip()

        # 取第一行非空内容
        first_line = ""
        for line in msg.split("\n"):
            line = line.strip()
            if line and len(line) > 5:
                first_line = line
                break
        if not first_line:
            first_line = msg

        # 移除常见前缀噪声
        noise_prefixes = [
            "tactic '", "error: ", "Error: ",
            "unknown tactic", "unsolved goals\n",
        ]
        summary = first_line
        for prefix in noise_prefixes:
            if summary.startswith(prefix):
                # 保留有意义的部分
                break

        # 截断到合理长度
        if len(summary) > 120:
            summary = summary[:117] + "..."

        # 确保不以引号开头（避免与模板中的引号嵌套冲突）
        summary = summary.strip('"\'')

        return summary if summary else "an elaboration error"

    @staticmethod
    def _extract_goal_summary(state_before: str) -> str:
        """
        从 proof state 中提取目标摘要（⊢ 后面的部分）

        若 state_before 包含 ⊢ 符号，提取其后内容作为摘要；
        否则取前 150 字符。

        参数:
            state_before: 证明状态文本

        返回:
            str: 目标摘要

        @author ygw 2026-03-02 新增
        """
        if not state_before:
            return "(empty state)"

        # 查找 ⊢ 符号（Lean4 goal marker）
        goal_marker_pos = state_before.find("⊢")
        if goal_marker_pos >= 0:
            goal_text = state_before[goal_marker_pos:].strip()
            # 截取合理长度
            if len(goal_text) > 200:
                goal_text = goal_text[:200] + "..."
            return goal_text

        # 无 ⊢ 则返回前 150 字符
        summary = state_before[:150].strip()
        return summary if summary else "(empty state)"

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------

    @staticmethod
    def _safe_format(template: str, **kwargs) -> str:
        """类方法包装，调用模块级 _safe_format"""
        return _safe_format(template, **kwargs)

    def _build_prompt(self, record: Dict) -> str:
        """
        构建 API 调用的 Prompt（v2: 增加 repair_hint，引导模型关注正确策略）

        参数:
            record: 错误记录

        返回:
            str: 格式化后的 Prompt

        @author ygw 2026-03-02 优化：增加 repair_hint 减少误导
        """
        # 截断过长的 state_before 以控制 token 消耗
        state_before = record.get("state_before", "")
        if len(state_before) > 1500:
            state_before = state_before[:1500] + "\n... (truncated)"

        error_message = record.get("error_message", "")
        if len(error_message) > 500:
            error_message = error_message[:500] + "... (truncated)"

        # 获取 repair_hint；若缺失则构造一个基本提示
        repair_hint = record.get("repair_hint", "")
        if not repair_hint.strip():
            repair_hint = f"Use `{record.get('original_tactic', '')}` instead"

        return self._safe_format(
            self.prompt_template,
            state_before=state_before,
            original_tactic=record.get("original_tactic", ""),
            error_tactic=record.get("error_tactic", ""),
            error_type=record.get("error_type", ""),
            error_message=error_message,
            repair_hint=repair_hint,
        )

    def _enrich_api_output(self, reflection: str, record: Dict) -> str:
        """
        后处理 API 输出：补充缺失的 repair_hint 引用和证明目标引用

        当 API 生成的文本没有明确包含修复提示或证明目标时，
        追加补充句以提高信息密度和一致性。

        参数:
            reflection: API 生成的 Reflection 文本
            record: 原始错误记录

        返回:
            str: 补充后的 Reflection 文本

        @author ygw 2026-03-02 新增 v2.1
        """
        additions = []

        # 检查 1: repair_hint 相关内容是否出现在 reflection 中
        repair_hint = record.get("repair_hint", "").strip()
        if repair_hint:
            # 提取 hint 中的关键片段（前半段通常是错误描述，后半段是修复建议）
            hint_keywords = [w for w in repair_hint.split() if len(w) > 3]
            hint_coverage = sum(1 for kw in hint_keywords if kw.lower() in reflection.lower())
            # 如果少于 30% 的 hint 关键词出现在 reflection 中，追加
            if hint_keywords and hint_coverage / len(hint_keywords) < 0.3:
                additions.append(
                    f"The repair hint directly addresses this: {repair_hint}."
                )

        # 检查 2: 证明目标（⊢ 后的内容）是否被引用
        state_before = record.get("state_before", "")
        if "⊢" in state_before and "⊢" not in reflection:
            goal_summary = self._extract_goal_summary(state_before)
            if goal_summary and goal_summary != "(empty state)":
                # 截取目标摘要的前 100 字符
                short_goal = goal_summary[:100]
                if len(goal_summary) > 100:
                    short_goal += "..."
                additions.append(
                    f"The current proof goal \"{short_goal}\" requires "
                    f"the specific structure provided by the correct tactic."
                )

        if additions:
            reflection = reflection.rstrip()
            if not reflection.endswith("."):
                reflection += "."
            reflection += " " + " ".join(additions)

        return reflection

    def _validate_reflection(self, text: str, record: Optional[Dict] = None) -> bool:
        """
        验证 Reflection 文本质量（v2: 增加一致性校验）

        参数:
            text: Reflection 文本
            record: 原始错误记录（可选，用于一致性校验）

        返回:
            bool: 是否通过质量检查

        @author ygw 2026-03-02 优化：增加与正确策略的一致性校验
        """
        if not text:
            return False
        if len(text) < self.min_length:
            return False
        if len(text) > self.max_length:
            return False
        for pattern in self.forbidden_patterns:
            if pattern.lower() in text.lower():
                return False
        if self.require_keywords and not any(kw in text for kw in self.require_keywords):
            return False

        # 一致性校验：反思文本需提及正确策略的核心标识符
        if record:
            if not self._validate_consistency(text, record):
                logger.debug(
                    f"一致性校验失败: reflection 未提及正确策略 "
                    f"'{record.get('original_tactic', '')}'"
                )
                return False

        return True

    def _validate_consistency(self, reflection: str, record: Dict) -> bool:
        """
        验证 Reflection 文本与正确策略的一致性

        检查反思文本是否至少提及了正确策略中的核心标识符（策略名或关键词），
        以确保反思内容不会完全脱离正确答案。

        一致性判定规则（满足任一即通过）:
        1. 反思文本包含完整的 original_tactic 字符串
        2. 反思文本包含 original_tactic 中的主要标识符（如 Nat.add_comm, simp, rw 等）
        3. 反思文本提及了 error_tactic 和 error_type 关键词（至少还在讨论相关错误）

        参数:
            reflection: 反思文本
            record: 原始错误记录

        返回:
            bool: 是否通过一致性校验

        @author ygw 2026-03-02 新增
        """
        original_tactic = record.get("original_tactic", "").strip()
        error_tactic = record.get("error_tactic", "").strip()
        error_type = record.get("error_type", "").strip()

        if not original_tactic:
            return True  # 无正确策略信息，跳过校验

        reflection_lower = reflection.lower()

        # 规则 1: 完整匹配正确策略
        if original_tactic in reflection:
            return True

        # 规则 2: 提取正确策略中的核心标识符并检查匹配
        core_ids = self._extract_tactic_identifiers(original_tactic)
        if core_ids:
            # 至少匹配一个核心标识符即可
            matched = sum(1 for cid in core_ids if cid.lower() in reflection_lower)
            if matched > 0:
                return True

        # 规则 3: 至少讨论了错误策略和错误类型（表明反思在谈论相关问题）
        mentions_error_tactic = error_tactic and (
            error_tactic in reflection or
            any(eid.lower() in reflection_lower
                for eid in self._extract_tactic_identifiers(error_tactic))
        )
        error_type_keywords = {
            "tactic_typo": ["typo", "spelling", "misspell"],
            "wrong_tactic": ["wrong", "inappropriate", "mismatch", "incorrect tactic"],
            "argument_error": ["argument", "parameter", "type mismatch"],
            "missing_step": ["missing", "intermediate", "skip", "gap"],
        }
        type_kws = error_type_keywords.get(error_type, [error_type])
        mentions_error_type = any(kw in reflection_lower for kw in type_kws)

        if mentions_error_tactic and mentions_error_type:
            return True

        return False

    @staticmethod
    def _extract_tactic_identifiers(tactic: str) -> List[str]:
        """
        从策略字符串中提取核心标识符

        例:
        - "exact Nat.add_comm n m" → ["Nat.add_comm"]
        - "rw [mul_comm]" → ["mul_comm"]
        - "simp [List.length_append]" → ["List.length_append"]
        - "apply Nat.lt_trans" → ["Nat.lt_trans"]
        - "sorry" → ["sorry"]

        参数:
            tactic: 策略字符串

        返回:
            List[str]: 核心标识符列表

        @author ygw 2026-03-02 新增
        """
        if not tactic:
            return []

        identifiers = []
        # 去掉常见策略关键字前缀，提取标识符
        tactic_keywords = {
            "exact", "apply", "rw", "simp", "intro", "intros",
            "have", "let", "show", "calc", "constructor", "cases",
            "induction", "refl", "ring", "omega", "norm_num",
            "ext", "funext", "congr", "trivial", "assumption",
            "contradiction", "exfalso", "by_contra", "push_neg",
            "linarith", "nlinarith", "field_simp", "ring_nf",
        }

        # 提取方括号内的标识符 [xxx, yyy]
        import re
        bracket_match = re.findall(r'\[([^\]]+)\]', tactic)
        for content in bracket_match:
            for part in content.split(','):
                part = part.strip().lstrip('←').lstrip('↔').strip()
                if part and len(part) > 2:
                    identifiers.append(part)

        # 按空格分词，跳过关键字和短 token
        tokens = tactic.replace('[', ' ').replace(']', ' ').split()
        for token in tokens:
            token_stripped = token.strip('(),;')
            if (token_stripped.lower() not in tactic_keywords
                    and len(token_stripped) > 2
                    and not token_stripped.startswith('-')
                    and '.' in token_stripped):
                # 包含 . 的通常是 qualified name (e.g., Nat.add_comm)
                identifiers.append(token_stripped)

        # 如果没有找到带 . 的标识符，取第一个非关键字 token
        if not identifiers:
            for token in tokens:
                token_stripped = token.strip('(),;[]')
                if (token_stripped.lower() not in tactic_keywords
                        and len(token_stripped) > 2):
                    identifiers.append(token_stripped)
                    break

        # 如果策略本身就是一个简单词（如 sorry, ring），直接用它
        if not identifiers and tactic.strip():
            identifiers.append(tactic.strip())

        return identifiers


# ================================================================
# Phase 3: SFT 数据集构建器
# ================================================================

class ReflectionDatasetBuilder:
    """
    将 Reflection 记录格式化为 SFT 微调数据集，并执行 train/val/test 分割。

    输出格式 (每条记录):
    {
        "instruction": "<错误上下文指令>",
        "input": "",
        "output": "<reflection 文本>",
        "metadata": {
            "type": "reflection",
            "error_type": "...",
            "theorem_name": "...",
            "correct_tactic": "...",
            "original_thought": "...",
            "reflection_source": "api / template"
        }
    }

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据集构建器

        参数:
            config: sft_dataset 配置节
        """
        self.output_path = config.get("output_path", "")
        split_cfg = config.get("split", {})
        self.train_ratio = split_cfg.get("train_ratio", 0.90)
        self.val_ratio = split_cfg.get("val_ratio", 0.05)
        self.test_ratio = split_cfg.get("test_ratio", 0.05)
        filenames_cfg = config.get("filenames", {})
        self.full_filename = filenames_cfg.get("full", "reflection_dataset.jsonl")
        self.train_filename = filenames_cfg.get("train", "reflection_train.jsonl")
        self.val_filename = filenames_cfg.get("val", "reflection_val.jsonl")
        self.test_filename = filenames_cfg.get("test", "reflection_test.jsonl")
        self.stats_filename = filenames_cfg.get("stats", "reflection_stats.json")
        self.instruction_template = config.get("instruction_template", "")

    def build(self, records: List[Dict],
              filter_stats: Dict, gen_stats: Dict,
              seed: int = 42) -> Dict[str, Any]:
        """
        构建 SFT 数据集

        参数:
            records: 带 reflection 的记录列表
            filter_stats: Phase 1 统计
            gen_stats: Phase 2 统计
            seed: 随机种子

        返回:
            Dict: 完整统计信息
        """
        logger.info("=" * 60)
        logger.info("Phase 3: SFT 数据集构建")
        logger.info("=" * 60)
        start_time = time.time()
        ensure_dir(self.output_path)

        # 格式化为 SFT 格式
        sft_records = []
        for rec in records:
            sft_rec = self._format_sft_record(rec)
            if sft_rec:
                sft_records.append(sft_rec)

        logger.info(f"SFT 格式化: {len(sft_records):,} 条")

        # 去重（基于 instruction + output 的 hash）
        unique_records = self._deduplicate(sft_records)
        logger.info(f"去重后: {len(unique_records):,} 条 "
                     f"(去除 {len(sft_records) - len(unique_records):,} 条重复)")

        # 按定理名称分割（防止同一定理出现在不同集合）
        train, val, test = self._split_by_theorem(unique_records, seed)
        logger.info(f"数据集分割: train={len(train):,}, "
                     f"val={len(val):,}, test={len(test):,}")

        # 保存
        full_path = os.path.join(self.output_path, self.full_filename)
        train_path = os.path.join(self.output_path, self.train_filename)
        val_path = os.path.join(self.output_path, self.val_filename)
        test_path = os.path.join(self.output_path, self.test_filename)

        save_jsonl(unique_records, full_path)
        save_jsonl(train, train_path)
        save_jsonl(val, val_path)
        save_jsonl(test, test_path)
        logger.info(f"保存完成: {full_path}")

        # 统计
        elapsed = time.time() - start_time
        type_dist = Counter(r["metadata"]["error_type"] for r in unique_records)
        source_dist = Counter(r["metadata"]["reflection_source"] for r in unique_records)

        stats = {
            "timestamp": get_timestamp(),
            "phase1_filter": filter_stats,
            "phase2_generation": gen_stats,
            "phase3_dataset": {
                "sft_formatted": len(sft_records),
                "after_dedup": len(unique_records),
                "train_count": len(train),
                "val_count": len(val),
                "test_count": len(test),
                "error_type_distribution": dict(type_dist.most_common()),
                "reflection_source_distribution": dict(source_dist.most_common()),
                "elapsed_seconds": round(elapsed, 2),
            },
            "output_files": {
                "full": full_path,
                "train": train_path,
                "val": val_path,
                "test": test_path,
            },
        }

        # 保存统计
        stats_path = os.path.join(self.output_path, self.stats_filename)
        save_json(stats, stats_path)
        logger.info(f"统计信息已保存: {stats_path}")
        logger.info(f"Phase 3 完成, 耗时 {elapsed:.1f}s")

        return stats

    def _format_sft_record(self, record: Dict) -> Optional[Dict]:
        """
        将原始记录格式化为 SFT 指令格式

        参数:
            record: 带 reflection 的错误记录

        返回:
            Optional[Dict]: SFT 格式的记录
        """
        reflection = record.get("reflection", "")
        if not reflection:
            return None

        # 构建 instruction
        state_before = record.get("state_before", "")
        if len(state_before) > 2000:
            state_before = state_before[:2000] + "\n... (truncated)"

        error_message = record.get("error_message", "")
        if len(error_message) > 500:
            error_message = error_message[:500] + "..."

        if self.instruction_template:
            instruction = _safe_format(
                self.instruction_template,
                state_before=state_before,
                error_tactic=record.get("error_tactic", ""),
                error_message=error_message,
            )
        else:
            instruction = (
                f"You are proving a Lean4 theorem. The following tactic was "
                f"attempted but failed. Analyze the error and explain what "
                f"went wrong and how to fix it.\n\n"
                f"[Proof State]\n{state_before}\n\n"
                f"[Attempted Tactic]\n{record.get('error_tactic', '')}\n\n"
                f"[Error Message]\n{error_message}"
            )

        return {
            "instruction": instruction.strip(),
            "input": "",
            "output": reflection,
            "metadata": {
                "type": "reflection",
                "error_type": record.get("error_type", ""),
                "theorem_name": record.get("theorem_name", ""),
                "theorem_full_name": record.get("theorem_full_name", ""),
                "correct_tactic": record.get("original_tactic", ""),
                "original_thought": record.get("thought", ""),
                "reflection_source": record.get("reflection_source", ""),
                "repair_hint": record.get("repair_hint", ""),
            },
        }

    def _deduplicate(self, records: List[Dict]) -> List[Dict]:
        """
        基于 instruction + output 内容哈希去重

        参数:
            records: SFT 记录列表

        返回:
            List[Dict]: 去重后的记录
        """
        seen: Set[str] = set()
        unique = []
        for rec in records:
            content = rec["instruction"] + rec["output"]
            h = hashlib.md5(content.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(rec)
        return unique

    def _split_by_theorem(self, records: List[Dict],
                           seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        按定理名称分割数据集（防止数据泄漏）

        参数:
            records: SFT 记录列表
            seed: 随机种子

        返回:
            Tuple: (train, val, test)
        """
        # 收集所有定理名称
        theorem_records: Dict[str, List[Dict]] = {}
        for rec in records:
            thm = rec["metadata"].get("theorem_name", "unknown")
            if thm not in theorem_records:
                theorem_records[thm] = []
            theorem_records[thm].append(rec)

        # 按定理名称排序后打乱（确保可复现）
        theorem_names = sorted(theorem_records.keys())
        rng = random.Random(seed)
        rng.shuffle(theorem_names)

        # 按定理数量分割
        n_total = len(theorem_names)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_theorems = set(theorem_names[:n_train])
        val_theorems = set(theorem_names[n_train:n_train + n_val])
        test_theorems = set(theorem_names[n_train + n_val:])

        train = []
        val = []
        test = []
        for thm, recs in theorem_records.items():
            if thm in train_theorems:
                train.extend(recs)
            elif thm in val_theorems:
                val.extend(recs)
            else:
                test.extend(recs)

        # 打乱顺序
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        return train, val, test


# ================================================================
# 分批执行辅助函数
# @author ygw 2026-03-02 新增分批执行（分期付款模式）
# ================================================================

def _load_batch_progress(output_dir: str) -> Dict[str, Any]:
    """
    加载分批执行进度文件

    参数:
        output_dir: 输出目录

    返回:
        Dict: 进度信息 {"1": {"status": "completed", "count": 19845, ...}, ...}

    @author ygw 2026-03-02
    """
    progress_file = os.path.join(output_dir, ".reflection_batch_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_batch_progress(output_dir: str, progress: Dict[str, Any]) -> None:
    """
    保存分批执行进度文件

    参数:
        output_dir: 输出目录
        progress: 进度信息

    @author ygw 2026-03-02
    """
    ensure_dir(output_dir)
    progress_file = os.path.join(output_dir, ".reflection_batch_progress.json")
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def _update_batch_progress(output_dir: str, batch_id: int,
                           total_batches: int, count: int,
                           status: str) -> None:
    """
    更新指定批次的进度

    参数:
        output_dir: 输出目录
        batch_id: 批次 ID
        total_batches: 总批次数
        count: 该批次生成的记录数
        status: 状态 ("completed" / "partial")

    @author ygw 2026-03-02
    """
    progress = _load_batch_progress(output_dir)
    progress[str(batch_id)] = {
        "status": status,
        "count": count,
        "total_batches": total_batches,
        "timestamp": get_timestamp(),
    }
    _save_batch_progress(output_dir, progress)


def _get_batch_slice(total: int, total_batches: int,
                     batch_id: int) -> Tuple[int, int]:
    """
    计算指定批次的记录范围 [start, end)

    使用均匀分配：前 remainder 个批次多分 1 条，保证所有记录都被覆盖。

    参数:
        total: 总记录数
        total_batches: 总批次数
        batch_id: 批次 ID (1-based)

    返回:
        Tuple[int, int]: (start, end) 左闭右开区间

    @author ygw 2026-03-02
    """
    base_size = total // total_batches
    remainder = total % total_batches

    # 前 remainder 个批次每个分 base_size + 1 条
    if batch_id <= remainder:
        start = (batch_id - 1) * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (batch_id - 1 - remainder) * base_size
        end = start + base_size

    return start, end


def _show_batch_status(output_dir: str, total_batches: int,
                       total_records: int = 0) -> None:
    """
    显示分批执行进度面板

    参数:
        output_dir: 输出目录
        total_batches: 总批次数
        total_records: 总记录数（0=不显示费用估算）

    @author ygw 2026-03-02
    """
    progress = _load_batch_progress(output_dir)
    completed_count = 0
    completed_records = 0

    print("\n" + "=" * 60, flush=True)
    print("分批执行进度", flush=True)
    print("=" * 60, flush=True)

    for bid in range(1, total_batches + 1):
        info = progress.get(str(bid), {})
        status = info.get("status", "")
        count = info.get("count", 0)
        ts = info.get("timestamp", "")

        if status == "completed":
            icon = "✅"
            completed_count += 1
            completed_records += count
            detail = f"{count:,} 条, {ts}"
        elif status == "partial":
            icon = "⏳"
            detail = f"进行中 ({count:,} 条已完成, {ts})"
        else:
            # 检查是否有 checkpoint 文件
            ckpt = os.path.join(output_dir, f".reflection_batch_{bid}_checkpoint.jsonl")
            if os.path.exists(ckpt):
                try:
                    ckpt_lines = sum(1 for _ in open(ckpt, 'r', encoding='utf-8'))
                except Exception:
                    ckpt_lines = 0
                icon = "⏳"
                detail = f"中断 (checkpoint: {ckpt_lines:,} 条)"
            else:
                icon = "⬜"
                detail = "未开始"

        print(f"  Batch {bid}/{total_batches}: {icon} {detail}", flush=True)

    print(f"\n  已完成批次: {completed_count}/{total_batches}", flush=True)
    print(f"  已生成记录: {completed_records:,}", flush=True)

    # 费用估算（基于 119K 记录 × 94% API × ~0.003 元/条）
    if total_records > 0:
        cost_per_record = 0.003  # 约 0.003 元/条（DeepSeek-V3 估算）
        total_cost = total_records * 0.94 * cost_per_record
        done_cost = completed_records * cost_per_record
        remaining_cost = total_cost - done_cost
        print(f"  预计总费用: ~{total_cost:.0f} 元", flush=True)
        print(f"  已花费: ~{done_cost:.0f} 元", flush=True)
        print(f"  剩余: ~{remaining_cost:.0f} 元 ({total_batches - completed_count} 个批次)", flush=True)

    if completed_count == total_batches:
        print(f"\n  🎉 所有批次已完成! 运行 --merge 合并数据集", flush=True)
    elif completed_count > 0:
        next_batch = min(
            bid for bid in range(1, total_batches + 1)
            if progress.get(str(bid), {}).get("status") != "completed"
        )
        print(f"\n  下一步: --batch-id {next_batch}", flush=True)

    print("=" * 60, flush=True)


def _merge_batches(config: Dict, output_dir: str,
                   total_batches: int, seed: int) -> None:
    """
    合并所有已完成的批次，执行 Phase 3 (SFT 数据集构建)

    参数:
        config: 完整配置
        output_dir: 输出目录
        total_batches: 总批次数
        seed: 随机种子

    @author ygw 2026-03-02
    """
    print("=" * 60, flush=True)
    print("合并分批结果 → Phase 3 SFT 数据集构建", flush=True)
    print("=" * 60, flush=True)

    progress = _load_batch_progress(output_dir)
    all_records: List[Dict] = []
    missing_batches = []

    for bid in range(1, total_batches + 1):
        batch_file = os.path.join(output_dir, f"reflection_batch_{bid}.jsonl")
        if os.path.exists(batch_file):
            batch_records = load_jsonl(batch_file)
            all_records.extend(batch_records)
            print(f"  Batch {bid}: {len(batch_records):,} 条 ✅", flush=True)
        else:
            status = progress.get(str(bid), {}).get("status", "")
            if status != "completed":
                missing_batches.append(bid)
                print(f"  Batch {bid}: ⬜ 未找到输出文件", flush=True)
            else:
                missing_batches.append(bid)
                print(f"  Batch {bid}: ⚠️  进度标记完成但文件缺失", flush=True)

    if missing_batches:
        print(f"\n⚠️  以下批次未完成: {missing_batches}", flush=True)
        print(f"继续合并已有的 {len(all_records):,} 条记录...\n", flush=True)
    else:
        print(f"\n全部 {total_batches} 个批次已加载, 共 {len(all_records):,} 条\n",
              flush=True)

    if not all_records:
        print("[Merge] 无可合并的记录，退出", flush=True)
        return

    # 合并统计
    merge_stats = {
        "merge_total_records": len(all_records),
        "merge_total_batches": total_batches,
        "merge_completed_batches": total_batches - len(missing_batches),
        "merge_missing_batches": missing_batches,
        "merge_timestamp": get_timestamp(),
    }

    # 来源分布
    source_dist = Counter(r.get("reflection_source", "unknown") for r in all_records)
    type_dist = Counter(r.get("error_type", "unknown") for r in all_records)
    print(f"  来源分布: {dict(source_dist.most_common())}", flush=True)
    print(f"  错误类型分布: {dict(type_dist.most_common())}", flush=True)

    # Phase 3: SFT 数据集构建
    dataset_cfg = config.get("sft_dataset", {})
    builder = ReflectionDatasetBuilder(dataset_cfg)
    final_stats = builder.build(
        all_records,
        {"phase": "merge", **merge_stats},
        {"phase": "merge", "total_from_batches": len(all_records)},
        seed=seed,
    )

    # 最终报告
    phase3 = final_stats.get("phase3_dataset", {})
    print("\n" + "=" * 60, flush=True)
    print("分批合并 + SFT 数据集构建完成!", flush=True)
    print("=" * 60, flush=True)
    print(f"  SFT 记录总数: {phase3.get('after_dedup', 0):,}", flush=True)
    print(f"  Train: {phase3.get('train_count', 0):,}", flush=True)
    print(f"  Val:   {phase3.get('val_count', 0):,}", flush=True)
    print(f"  Test:  {phase3.get('test_count', 0):,}", flush=True)
    print(f"  错误类型分布: {phase3.get('error_type_distribution', {})}", flush=True)
    print(f"  来源分布:     {phase3.get('reflection_source_distribution', {})}", flush=True)

    output_files = final_stats.get("output_files", {})
    for name, path in output_files.items():
        print(f"  {name}: {path}", flush=True)
    print("=" * 60, flush=True)


def _build_record_key(record: Dict) -> str:
    """
    构建记录唯一键（用于断点恢复时识别已处理记录）

    参数:
        record: 错误记录

    返回:
        str: 唯一键

    @author ygw 2026-03-02
    """
    return (
        f"{record.get('theorem_name', '')}|"
        f"{record.get('error_tactic', '')}|"
        f"{record.get('step_index', '')}|"
        f"{record.get('error_type', '')}"
    )


# ================================================================
# 主函数
# ================================================================

def main():
    """
    命令行入口

    支持三种运行方式:
      1. 一次性处理: python build_reflection_data.py --config ...
      2. 分批处理:   python build_reflection_data.py --config ... --batch-id 1 --total-batches 6
      3. 合并批次:   python build_reflection_data.py --config ... --merge
      4. 查看进度:   python build_reflection_data.py --config ... --batch-status

    @author ygw 2026-03-02 新增分批执行支持
    """
    parser = argparse.ArgumentParser(
        description="Task 2.3: Reflection 数据生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # ====== 一次性模式 ======
  # 自动模式（有 API Key 用 API，否则用模板）
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml

  # 强制模板模式（无需 API Key）
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --mode template

  # Dry-run 仅过滤不生成
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --dry-run

  # 限制数量（测试用）
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --max-samples 1000

  # ====== 分批模式（分期付款）======
  # 分 6 批执行，每批约 20K 条，每次花费 ~50 元
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --batch-id 1 --total-batches 6
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --batch-id 2 --total-batches 6
  # ...隔天继续...
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --batch-id 6 --total-batches 6

  # 查看分批进度
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --batch-status

  # 所有批次完成后，合并 → Phase 3 SFT 数据集
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --merge

  # 强制重新处理某个批次
  python scripts/build_reflection_data.py --config configs/reflection_generation.yaml --batch-id 3 --force
        """,
    )
    parser.add_argument("--config", type=str,
                        default="configs/reflection_generation.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["auto", "api", "template"],
                        default="auto",
                        help="运行模式: auto / api / template")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="限制处理的最大记录数（0=不限制）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅执行 Phase 1 过滤，不生成 Reflection")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别")

    # 分批执行参数组
    batch_group = parser.add_argument_group("分批执行（分期付款模式）")
    batch_group.add_argument(
        "--batch-id", type=int, default=0,
        help="批次 ID (1-based, 0=不分批, 一次性处理所有记录)",
    )
    batch_group.add_argument(
        "--total-batches", type=int, default=6,
        help="总批次数 (默认 6, 即分 6 次执行)",
    )
    batch_group.add_argument(
        "--merge", action="store_true",
        help="合并所有已完成的批次并构建 SFT 数据集",
    )
    batch_group.add_argument(
        "--batch-status", action="store_true",
        help="查看分批执行进度面板",
    )
    batch_group.add_argument(
        "--force", action="store_true",
        help="强制重新处理已完成的批次",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_handlers = [
        logging.StreamHandler(),
    ]
    # 确保 logs 目录存在
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_handlers.append(
        logging.FileHandler(os.path.join(logs_dir, "reflection_generation.log"),
                            encoding="utf-8")
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=log_handlers,
    )
    # 配置 rtap 命名空间
    rtap_logger = logging.getLogger("rtap")
    rtap_logger.setLevel(log_level)

    print("=" * 60, flush=True)
    print("Task 2.3: Reflection 数据生成", flush=True)
    if args.batch_id > 0:
        print(f"  分批模式: 批次 {args.batch_id}/{args.total_batches}", flush=True)
    elif args.merge:
        print("  合并模式: 汇总所有批次 → SFT 数据集", flush=True)
    print("=" * 60, flush=True)

    # 加载配置
    print(f"[Reflection] 加载配置: {args.config}", flush=True)
    config = load_yaml(args.config)

    # 命令行覆盖
    if args.max_samples > 0:
        config.setdefault("data_filter", {})["max_samples"] = args.max_samples
        print(f"[Reflection] 命令行覆盖: max_samples = {args.max_samples}", flush=True)

    global_cfg = config.get("global", {})
    seed = global_cfg.get("random_seed", 42)
    random.seed(seed)

    output_dir = config.get("sft_dataset", {}).get("output_path", "")

    # ========================================
    # 快捷操作: 查看进度 / 合并
    # ========================================
    if args.batch_status:
        _show_batch_status(output_dir, args.total_batches)
        return

    if args.merge:
        _merge_batches(config, output_dir, args.total_batches, seed)
        return

    # ========================================
    # Phase 1: 数据过滤（每次运行都执行，成本极低）
    # ========================================
    filter_cfg = config.get("data_filter", {})
    data_filter = ReflectionDataFilter(filter_cfg)
    filtered_records, filter_stats = data_filter.load_and_filter()

    if not filtered_records:
        print("[Reflection] 过滤后无有效记录，退出", flush=True)
        return

    print(f"\n[Reflection] Phase 1 完成: {len(filtered_records):,} 条有效记录\n",
          flush=True)

    if args.dry_run:
        print("[Reflection] Dry-run 模式，跳过 Phase 2 & 3", flush=True)
        print(f"  过滤统计: {json.dumps(filter_stats, indent=2, ensure_ascii=False)}",
              flush=True)
        if args.total_batches > 0:
            print(f"\n[Dry-run] 分批预览 ({args.total_batches} 批次):", flush=True)
            for bid in range(1, args.total_batches + 1):
                s, e = _get_batch_slice(len(filtered_records), args.total_batches, bid)
                est_cost = (e - s) * 0.94 * 0.003
                print(f"  Batch {bid}: [{s:,}, {e:,}) = {e - s:,} 条, "
                      f"预计费用 ~{est_cost:.0f} 元", flush=True)
        return

    # ========================================
    # 分批模式: 切片 + 断点恢复
    # ========================================
    already_done: List[Dict] = []  # 断点恢复的已完成记录
    batch_checkpoint_path = ""

    if args.batch_id > 0:
        # 参数校验
        if args.batch_id > args.total_batches:
            print(f"[Error] batch-id ({args.batch_id}) 超过 "
                  f"total-batches ({args.total_batches})", flush=True)
            return

        # 确定性排序（所有批次使用相同排序，确保切片不重叠）
        filtered_records.sort(key=lambda r: (
            r.get("theorem_name", ""),
            str(r.get("step_index", 0)).zfill(10),
            r.get("error_tactic", ""),
        ))

        # 计算当前批次的切片范围
        total_count = len(filtered_records)
        start, end = _get_batch_slice(total_count, args.total_batches, args.batch_id)
        batch_records = filtered_records[start:end]

        print(f"[Batch {args.batch_id}/{args.total_batches}] "
              f"记录范围: [{start:,}, {end:,}), 共 {len(batch_records):,} 条",
              flush=True)

        # 检查是否已完成
        batch_output_file = os.path.join(
            output_dir, f"reflection_batch_{args.batch_id}.jsonl"
        )
        if os.path.exists(batch_output_file) and not args.force:
            existing = load_jsonl(batch_output_file)
            print(f"[Batch] 批次 {args.batch_id} 已完成 "
                  f"({len(existing):,} 条), 使用 --force 强制重新处理",
                  flush=True)
            _update_batch_progress(
                output_dir, args.batch_id, args.total_batches,
                len(existing), "completed"
            )
            _show_batch_status(output_dir, args.total_batches, total_count)
            return

        # 断点恢复: 从 checkpoint 加载已处理的记录
        batch_checkpoint_path = os.path.join(
            output_dir, f".reflection_batch_{args.batch_id}_checkpoint.jsonl"
        )
        if os.path.exists(batch_checkpoint_path) and not args.force:
            already_done = load_jsonl(batch_checkpoint_path)
            done_keys = set(_build_record_key(r) for r in already_done)
            remaining = [
                r for r in batch_records
                if _build_record_key(r) not in done_keys
            ]
            print(f"[Batch] 🔄 断点恢复: 已完成 {len(already_done):,}, "
                  f"剩余 {len(remaining):,}", flush=True)
            batch_records = remaining

            if not batch_records:
                # checkpoint 已包含所有记录，直接保存为最终输出
                print(f"[Batch] 所有记录已在 checkpoint 中，保存最终输出", flush=True)
                save_jsonl(already_done, batch_output_file)
                _update_batch_progress(
                    output_dir, args.batch_id, args.total_batches,
                    len(already_done), "completed"
                )
                if os.path.exists(batch_checkpoint_path):
                    os.remove(batch_checkpoint_path)
                _show_batch_status(output_dir, args.total_batches, total_count)
                return
        elif args.force and os.path.exists(batch_checkpoint_path):
            os.remove(batch_checkpoint_path)
            already_done = []

        # 更新 filtered_records 为当前批次（可能是断点恢复后的剩余部分）
        filtered_records = batch_records

        # 实时进度提示
        est_hours = len(filtered_records) / (500 / 60.0 * 0.8) / 3600
        est_cost = len(filtered_records) * 0.94 * 0.003
        print(f"[Batch] 本批次预计: ~{est_hours:.1f} 小时, ~{est_cost:.0f} 元",
              flush=True)

    # ========================================
    # Phase 2: Reflection 生成
    # ========================================
    gen_cfg = config.get("reflection_generation", {})
    generator = ReflectionGenerator(gen_cfg)
    reflection_records, gen_stats = generator.run(
        filtered_records, output_dir, mode=args.mode,
        batch_id=args.batch_id,
    )

    if args.batch_id > 0:
        # ====== 分批模式: 保存批次输出 ======
        # 合并断点恢复的记录 + 本次生成的记录
        all_batch_results = already_done + reflection_records
        batch_output_file = os.path.join(
            output_dir, f"reflection_batch_{args.batch_id}.jsonl"
        )
        save_jsonl(all_batch_results, batch_output_file)
        print(f"\n[Batch {args.batch_id}] 批次完成: {len(all_batch_results):,} 条 "
              f"→ {batch_output_file}", flush=True)
        print(f"  本次 API: {gen_stats.get('api_generated', 0):,}, "
              f"模板: {gen_stats.get('template_generated', 0):,}, "
              f"断点恢复: {len(already_done):,}", flush=True)

        # 更新进度
        _update_batch_progress(
            output_dir, args.batch_id, args.total_batches,
            len(all_batch_results), "completed"
        )

        # 清理 checkpoint
        if batch_checkpoint_path and os.path.exists(batch_checkpoint_path):
            os.remove(batch_checkpoint_path)

        # 显示总体进度
        total_count = filter_stats.get("final_count", 0)
        _show_batch_status(output_dir, args.total_batches, total_count)

        # 检查是否所有批次都已完成
        progress = _load_batch_progress(output_dir)
        completed = sum(
            1 for bid in range(1, args.total_batches + 1)
            if progress.get(str(bid), {}).get("status") == "completed"
        )
        if completed == args.total_batches:
            print(f"\n🎉 所有 {args.total_batches} 个批次已完成!", flush=True)
            print(f"运行以下命令合并并构建 SFT 数据集:", flush=True)
            print(f"  python scripts/build_reflection_data.py "
                  f"--config {args.config} --merge", flush=True)
        else:
            next_batch = min(
                bid for bid in range(1, args.total_batches + 1)
                if progress.get(str(bid), {}).get("status") != "completed"
            )
            print(f"\n下一步: python scripts/build_reflection_data.py "
                  f"--config {args.config} --batch-id {next_batch} "
                  f"--total-batches {args.total_batches}", flush=True)
        return

    # ====== 非分批模式: 常规流程 ======
    if not reflection_records:
        print("[Reflection] 未生成任何 Reflection，退出", flush=True)
        return

    print(f"\n[Reflection] Phase 2 完成: {len(reflection_records):,} 条 Reflection\n",
          flush=True)

    # ========================================
    # Phase 3: SFT 数据集构建
    # ========================================
    dataset_cfg = config.get("sft_dataset", {})
    builder = ReflectionDatasetBuilder(dataset_cfg)
    final_stats = builder.build(
        reflection_records,
        filter_stats, gen_stats,
        seed=seed,
    )

    # ========================================
    # 最终报告
    # ========================================
    phase3 = final_stats.get("phase3_dataset", {})
    print("\n" + "=" * 60, flush=True)
    print("Reflection 数据生成完成!", flush=True)
    print("=" * 60, flush=True)
    print(f"  SFT 记录总数: {phase3.get('after_dedup', 0):,}", flush=True)
    print(f"  Train: {phase3.get('train_count', 0):,}", flush=True)
    print(f"  Val:   {phase3.get('val_count', 0):,}", flush=True)
    print(f"  Test:  {phase3.get('test_count', 0):,}", flush=True)
    print(f"  错误类型分布: {phase3.get('error_type_distribution', {})}", flush=True)
    print(f"  来源分布:     {phase3.get('reflection_source_distribution', {})}", flush=True)

    output_files = final_stats.get("output_files", {})
    for name, path in output_files.items():
        print(f"  {name}: {path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
