#!/usr/bin/env python3
"""
模板回退 v3 快速验证脚本
@author ygw
日期: 2026-03-02

验证项:
  1. 多变体选择：不同记录 → 不同变体索引 → 多样性
  2. 假设提取: _extract_hypotheses_clause 正确解析
  3. 错误摘要: _summarize_error_message 截断与清洁
  4. 确定性: 相同记录 → 相同输出
  5. 4 种错误类型 × 3+ 个变体 → 生成结果质量
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.utils import load_yaml

# 加载配置
config_path = PROJECT_ROOT / "configs" / "reflection_generation.yaml"
config = load_yaml(str(config_path))
rg_config = config.get("reflection_generation", {})

# 导入被测模块
from scripts.build_reflection_data import ReflectionGenerator, REFLECTION_TEMPLATES, GENERIC_TEMPLATES

print("=" * 70)
print("模板回退 v3 验证测试")
print("=" * 70)

pass_count = 0
fail_count = 0

def check(desc, condition):
    global pass_count, fail_count
    if condition:
        print(f"  PASS: {desc}")
        pass_count += 1
    else:
        print(f"  FAIL: {desc}")
        fail_count += 1

# ============================================================
# Test 1: 常量结构验证
# ============================================================
print("\nTest 1: 常量结构验证")
check("REFLECTION_TEMPLATES 是 dict", isinstance(REFLECTION_TEMPLATES, dict))
check("4 种错误类型", len(REFLECTION_TEMPLATES) == 4)
for etype, variants in REFLECTION_TEMPLATES.items():
    check(f"{etype} 是列表", isinstance(variants, list))
    check(f"{etype} 至少 3 个变体", len(variants) >= 3)
    for i, v in enumerate(variants):
        check(f"{etype}[{i}] 是字符串", isinstance(v, str))
        check(f"{etype}[{i}] 含 {{original_tactic}}", "{original_tactic}" in v)
        check(f"{etype}[{i}] 含 {{error_tactic}}", "{error_tactic}" in v)
        check(f"{etype}[{i}] 含 {{error_message_short}}", "{error_message_short}" in v)
        check(f"{etype}[{i}] 含 {{repair_hint}}", "{repair_hint}" in v)
        check(f"{etype}[{i}] 含 {{state_summary}}", "{state_summary}" in v)

check("GENERIC_TEMPLATES 是列表", isinstance(GENERIC_TEMPLATES, list))
check("GENERIC_TEMPLATES 至少 3 个变体", len(GENERIC_TEMPLATES) >= 3)

# ============================================================
# Test 2: _deterministic_variant_index
# ============================================================
print("\nTest 2: 确定性变体选择")
gen = ReflectionGenerator.__new__(ReflectionGenerator)

# 相同记录 → 相同索引
rec1 = {"theorem_name": "add_comm", "error_tactic": "sipm", "step_index": 2}
idx1a = gen._deterministic_variant_index(rec1, 3)
idx1b = gen._deterministic_variant_index(rec1, 3)
check("相同记录 → 相同索引", idx1a == idx1b)

# 不同记录 → 通常不同索引（哈希分布）
records = [
    {"theorem_name": f"thm_{i}", "error_tactic": f"tac_{i}", "step_index": i}
    for i in range(30)
]
indices = [gen._deterministic_variant_index(r, 3) for r in records]
unique_indices = set(indices)
check("30 条记录覆盖所有 3 个变体", len(unique_indices) == 3)
from collections import Counter
dist = Counter(indices)
check("分布相对均匀 (每个变体 >= 3)", all(c >= 3 for c in dist.values()))
print(f"    分布: {dict(dist)}")

# ============================================================
# Test 3: _extract_hypotheses_clause
# ============================================================
print("\nTest 3: 假设提取")

# 有假设的 state
state1 = "n : ℕ\nh : n > 0\n⊢ n + 1 > 1"
clause1 = gen._extract_hypotheses_clause(state1)
check("提取到假设从句", len(clause1) > 0)
check("含 'with hypotheses'", "with hypotheses" in clause1)
check("含变量名 n", "n : " in clause1)
check("含变量名 h", "h : " in clause1)
print(f"    结果: {clause1}")

# 无假设
state2 = "⊢ True"
clause2 = gen._extract_hypotheses_clause(state2)
check("纯 goal 状态返回空", clause2 == "")

# 空字符串
clause3 = gen._extract_hypotheses_clause("")
check("空状态返回空", clause3 == "")

# 多假设截断
state4 = "a : ℕ\nb : ℕ\nc : ℕ\nd : ℕ\ne : ℕ\nf : ℕ\n⊢ a + b = c"
clause4 = gen._extract_hypotheses_clause(state4)
check("多假设截断到 4 个 + more 提示", "more" in clause4)
print(f"    结果: {clause4}")

# ============================================================
# Test 4: _summarize_error_message
# ============================================================
print("\nTest 4: 错误摘要")

msg1 = "tactic 'simp' failed, there are unsolved goals\ncase blah\n⊢ something"
summary1 = gen._summarize_error_message(msg1)
check("取第一行", "\n" not in summary1)
check("结果不为空", len(summary1) > 0)
check("长度 ≤ 120", len(summary1) <= 120)
print(f"    结果: {summary1}")

msg2 = ""
summary2 = gen._summarize_error_message(msg2)
check("空消息回退", summary2 == "an elaboration error")

msg3 = "x" * 200
summary3 = gen._summarize_error_message(msg3)
check("超长消息截断 ≤ 120", len(summary3) <= 120)

# ============================================================
# Test 5: 完整模板生成 — 4 种错误类型
# ============================================================
print("\nTest 5: 完整模板生成")

# 构造 ReflectionGenerator (最小化初始化)
gen2 = ReflectionGenerator.__new__(ReflectionGenerator)
gen2.reflection_templates = {k: list(v) for k, v in REFLECTION_TEMPLATES.items()}

test_records = [
    {
        "theorem_name": "Nat.add_comm",
        "error_type": "tactic_typo",
        "error_tactic": "sipm",
        "original_tactic": "simp",
        "error_message": "unknown tactic 'sipm'",
        "state_before": "n m : ℕ\n⊢ n + m = m + n",
        "repair_hint": "fix the spelling: 'sipm' → 'simp'",
        "step_index": 0,
    },
    {
        "theorem_name": "List.length_append",
        "error_type": "wrong_tactic",
        "error_tactic": "ring",
        "original_tactic": "simp [List.length_append]",
        "error_message": "tactic 'ring' failed, the goal is not an algebraic expression",
        "state_before": "l₁ l₂ : List α\n⊢ (l₁ ++ l₂).length = l₁.length + l₂.length",
        "repair_hint": "use simp with the List.length_append lemma instead of ring",
        "step_index": 1,
    },
    {
        "theorem_name": "Int.neg_neg",
        "error_type": "argument_error",
        "error_tactic": "exact Int.neg_neg_of_neg",
        "original_tactic": "exact Int.neg_neg n",
        "error_message": "function expected at 'Int.neg_neg_of_neg'",
        "state_before": "n : ℤ\n⊢ -(-n) = n",
        "repair_hint": "use the correct lemma name Int.neg_neg with argument n",
        "step_index": 0,
    },
    {
        "theorem_name": "Set.finite_union",
        "error_type": "missing_step",
        "error_tactic": "exact Set.Finite.union",
        "original_tactic": "apply Set.Finite.union",
        "error_message": "type mismatch, expected Set.Finite (s ∪ t), got forall ...",
        "state_before": "s t : Set α\nhs : s.Finite\nht : t.Finite\n⊢ (s ∪ t).Finite",
        "repair_hint": "use 'apply' instead of 'exact' since the goal needs further arguments",
        "step_index": 2,
    },
]

for rec in test_records:
    etype = rec["error_type"]
    reflection = gen2._generate_template_reflection(rec)
    check(f"{etype}: 生成非空", reflection is not None and len(reflection) > 0)
    if reflection:
        check(f"{etype}: 包含正确策略", rec["original_tactic"] in reflection)
        check(f"{etype}: 包含错误策略", rec["error_tactic"] in reflection)
        check(f"{etype}: 包含 repair_hint", rec["repair_hint"][:30] in reflection)
        check(f"{etype}: 长度 > 50", len(reflection) > 50)
        check(f"{etype}: 无未替换占位符", "{" not in reflection or "⊢" in reflection)
        print(f"    [{etype}] 长度={len(reflection)}")
        print(f"    前 200 字符: {reflection[:200]}...")

# ============================================================
# Test 6: 确定性复现
# ============================================================
print("\nTest 6: 确定性复现")
for rec in test_records:
    r1 = gen2._generate_template_reflection(rec)
    r2 = gen2._generate_template_reflection(rec)
    check(f"{rec['error_type']}: 两次输出完全一致", r1 == r2)

# ============================================================
# Test 7: 多样性验证（同类型不同记录）
# ============================================================
print("\nTest 7: 多样性验证")
typo_records = [
    {
        "theorem_name": f"thm_typo_{i}",
        "error_type": "tactic_typo",
        "error_tactic": f"sipm_{i}",
        "original_tactic": f"simp_{i}",
        "error_message": f"unknown tactic 'sipm_{i}'",
        "state_before": f"n : ℕ\n⊢ n = {i}",
        "repair_hint": f"fix spelling to simp_{i}",
        "step_index": i,
    }
    for i in range(20)
]
reflections = [gen2._generate_template_reflection(r) for r in typo_records]
unique_prefixes = set(r[:80] if r else "" for r in reflections)
check(f"20 条 tactic_typo 记录 → ≥3 种不同前缀", len(unique_prefixes) >= 3)
print(f"    不同前缀数: {len(unique_prefixes)}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print(f"测试汇总: {pass_count} PASS, {fail_count} FAIL")
print("=" * 70)

if fail_count > 0:
    sys.exit(1)
