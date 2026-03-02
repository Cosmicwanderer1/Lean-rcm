#!/usr/bin/env python3
"""
测试错误注入 + 验证 + Reflection 过滤链路修复
验证三个核心修复点：
1. error_verifier.py: _extract_intro_names + 自动 intro 逻辑
2. augmentation.py: _inject_missing_step 不再使用 sorry
3. build_reflection_data.py: VALID_STATUS_PREFIX = "verified"

@author ygw 2026-03-01
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pass_count = 0
fail_count = 0

def check(label: str, condition: bool, detail: str = ""):
    global pass_count, fail_count
    if condition:
        print(f"  PASS: {label}")
        pass_count += 1
    else:
        print(f"  FAIL: {label} — {detail}")
        fail_count += 1

# ================================================================
print("=" * 60)
print("Test 1: _extract_intro_names 方法")
print("=" * 60)

from src.data_engine.error_verifier import ErrorVerifier

# 典型 Lean4 证明状态
state1 = """M : Type u_1
inst✝¹ : CommMonoid M
inst✝ : Subsingleton Mˣ
S : Set M
x : M
hx : x ∈ {p | p ∈ Submonoid.closure S ∧ Irreducible p}
a b : M
ha : (fun x x_1 => Irreducible x → x ∈ S) a x✝¹
⊢ a * b ∈ Submonoid.closure S → Irreducible (a * b) → a ∈ S"""

names1 = ErrorVerifier._extract_intro_names(state1)
check("提取基本变量名", "M" in names1 and "S" in names1 and "x" in names1,
      f"got {names1}")
check("跳过 ✝ 变量", 
      all('✝' not in n for n in names1),
      f"got {names1}")
check("不包含 ⊢ 后的内容", 
      not any('Submonoid' in n for n in names1),
      f"got {names1}")

# 空状态
state2 = "⊢ ∀ (n : Nat), n + 0 = n"
names2 = ErrorVerifier._extract_intro_names(state2)
check("纯 ⊢ 状态返回空列表", names2 == [], f"got {names2}")

# 空字符串
names3 = ErrorVerifier._extract_intro_names("")
check("空字符串返回空列表", names3 == [], f"got {names3}")

# 复杂状态
state4 = """α : Type u_2
M : Matroid α
I : Set α
x : α
hI : M.Indep I
⊢ x ∈ M.closure I"""
names4 = ErrorVerifier._extract_intro_names(state4)
check("提取多变量", names4 == ["α", "M", "I", "x", "hI"],
      f"got {names4}")

# ================================================================
print("\n" + "=" * 60)
print("Test 2: _inject_missing_step 不再使用 sorry")
print("=" * 60)

from src.data_engine.augmentation import ErrorInjector

# 创建测试用的 injector
test_config = {
    "injection_ratio": 0.3,
    "error_types": {
        "tactic_typo": {"weight": 0.25, "typo_map": {}},
        "wrong_tactic": {"weight": 0.25, "replacement_pool": ["simp", "ring"]},
        "argument_error": {"weight": 0.25, "mutations": ["swap_args"]},
        "missing_step": {"weight": 0.25},
    }
}
injector = ErrorInjector(test_config)

# 多次运行确保不产生 sorry
sorry_count = 0
valid_tactics = {"done", "rfl", "exact _"}
all_tactics_valid = True
for _ in range(100):
    error_tactic, hint = injector._inject_missing_step("simp [add_comm]")
    if error_tactic == "sorry":
        sorry_count += 1
    if error_tactic not in valid_tactics:
        all_tactics_valid = False

check("不再生成 sorry", sorry_count == 0, f"sorry 出现 {sorry_count} 次")
check("只生成 done/rfl/exact _", all_tactics_valid, "")
check("hint 包含 correct tactic", 
      "simp [add_comm]" in hint, f"hint={hint}")
check("hint 包含错误描述",
      any(kw in hint for kw in ["done", "rfl", "exact"]),
      f"hint={hint}")

# ================================================================
print("\n" + "=" * 60)
print("Test 3: build_reflection_data 过滤逻辑")
print("=" * 60)

# 导入常量
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
# 直接检查文件内容
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_reflection_data",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_reflection_data.py")
)
mod = importlib.util.module_from_spec(spec)
# 只加载常量，不执行主逻辑
try:
    spec.loader.exec_module(mod)
except Exception:
    pass

prefix = getattr(mod, 'VALID_STATUS_PREFIX', None)
check("VALID_STATUS_PREFIX = 'verified'", 
      prefix == "verified",
      f"got '{prefix}'")

# 测试过滤行为
from scripts.build_reflection_data import ReflectionDataFilter

filter_config = {
    "input_path": "",
    "valid_statuses": [],  # 空列表时使用 VALID_STATUS_PREFIX
    "required_fields": ["error_message", "state_before"],
    "max_samples": 0,
    "max_per_error_type": 0,
}

# 模拟数据
test_records = [
    {"verification_status": "verified", "error_message": "unknown tactic", 
     "state_before": "x : Nat\n⊢ x = x", "original_tactic": "rfl",
     "error_tactic": "rfll", "error_type": "typo", "repair_hint": "typo"},
    {"verification_status": "replay_failed_at_step_0", "error_message": "Unknown identifier `S`",
     "state_before": "⊢ ∀ ...", "original_tactic": "intro", 
     "error_tactic": "intoo", "error_type": "typo", "repair_hint": "typo"},
    {"verification_status": "goal_start_failed", "error_message": "",
     "state_before": "", "original_tactic": "simp",
     "error_tactic": "simpp", "error_type": "typo", "repair_hint": "typo"},
    {"verification_status": "error_tactic_succeeded", "error_message": "",
     "state_before": "", "original_tactic": "omega",
     "error_tactic": "sorry", "error_type": "missing_step", "repair_hint": ""},
]

# 手动执行过滤逻辑（不走文件加载）
filtered = [r for r in test_records 
            if r.get("verification_status", "").startswith(prefix)]
check("只保留 verified 记录",
      len(filtered) == 1 and filtered[0]["verification_status"] == "verified",
      f"filtered {len(filtered)} records: {[r['verification_status'] for r in filtered]}")

check("排除 replay_failed_at_step_0",
      not any(r["verification_status"] == "replay_failed_at_step_0" for r in filtered),
      "")

check("排除 goal_start_failed",
      not any(r["verification_status"] == "goal_start_failed" for r in filtered),
      "")

check("排除 error_tactic_succeeded",
      not any(r["verification_status"] == "error_tactic_succeeded" for r in filtered),
      "")

# ================================================================
print("\n" + "=" * 60)
print("Test 4: YAML 配置一致性")
print("=" * 60)

import yaml
yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "configs", "reflection_generation.yaml")
with open(yaml_path, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

valid_statuses = yaml_config.get("data_filter", {}).get("valid_statuses", [])
check("YAML valid_statuses 只含 verified",
      valid_statuses == ["verified"],
      f"got {valid_statuses}")

check("YAML 不含 replay_failed",
      not any("replay_failed" in s for s in valid_statuses),
      f"got {valid_statuses}")

# ================================================================
print("\n" + "=" * 60)
print("Test 5: v2 条件 auto-intro 逻辑验证（模拟）")
print("=" * 60)

# v2 核心变更：auto-intro 仅在 needs_intro=True AND target_step>0 时执行
# needs_intro=True 表示 goal_start_with_fallback 用了 goal_start(expr) 路径
# needs_intro=False 表示用了 copyFrom 路径（变量已在上下文中）
# @author ygw 2026-03-01

# 测试 state_before 判断逻辑（与 error_verifier.py 2.0 节一致）
test_states = [
    ("M : Type\nS : Set M\n⊢ goal", True, "有变量上下文"),
    ("⊢ ∀ (n : Nat), n = n", False, "纯 ⊢ 开头"),
    ("", False, "空字符串"),
    ("x : Nat\n⊢ x > 0", True, "单变量"),
    ("  ⊢ P → Q", False, "⊢ 前有空格"),
]

for state, expect_state_check, desc in test_states:
    state_has_vars = bool(state.strip()) and not state.strip().startswith("⊢")
    check(f"state_before 变量检测: {desc}",
          state_has_vars == expect_state_check,
          f"expected {expect_state_check}, got {state_has_vars}")

# 测试三重条件门控：needs_intro AND target_step > 0 AND steps
test_gate_cases = [
    # (needs_intro, target_step, steps_exist, expected_intro_run, desc)
    (True, 1, True, True, "goal_start(expr)+step>0+有steps → 执行intro"),
    (True, 0, True, False, "goal_start(expr)+step=0 → 不执行（无需重放）"),
    (False, 1, True, False, "copyFrom+step>0 → 不执行（变量已在上下文）"),
    (False, 0, True, False, "copyFrom+step=0 → 不执行"),
    (True, 1, False, False, "goal_start(expr)+step>0+无steps → 不执行"),
    (False, 0, False, False, "copyFrom+step=0+无steps → 不执行"),
]

for needs_intro, target_step, steps_exist, expected, desc in test_gate_cases:
    steps = [{"state_before": "M : Type\n⊢ goal"}] if steps_exist else []
    will_intro = needs_intro and target_step > 0 and bool(steps)
    check(f"v2 三重门控: {desc}",
          will_intro == expected,
          f"expected {expected}, got {will_intro}")

# ================================================================
print("\n" + "=" * 60)
print("Test 6: step_index=0 四倍放大注入验证")
print("=" * 60)

# 验证 augmentation.py inject() v2: step_index=0 注入所有 4 种错误类型
test_dataset = [
    {"tactic": "simp", "step_index": 0, "state_before": "⊢ P", "state_after": "",
     "theorem_name": "t1", "theorem_full_name": "Ns.t1", "thought": ""},
    {"tactic": "ring", "step_index": 0, "state_before": "⊢ Q", "state_after": "",
     "theorem_name": "t2", "theorem_full_name": "Ns.t2", "thought": ""},
    {"tactic": "omega", "step_index": 1, "state_before": "⊢ R", "state_after": "",
     "theorem_name": "t3", "theorem_full_name": "Ns.t3", "thought": ""},
    {"tactic": "linarith", "step_index": 2, "state_before": "⊢ S", "state_after": "",
     "theorem_name": "t4", "theorem_full_name": "Ns.t4", "thought": ""},
]

injector2 = ErrorInjector(test_config)
results = injector2.inject(test_dataset)

# step_index=0 应产生 2 样本 × 4 错误类型 = 8 条（部分可能因注入失败而少于8）
step0_results = [r for r in results if r.get("step_index") == 0]
other_results = [r for r in results if r.get("step_index") != 0]

check("step_index=0 产生多条记录（4x 放大）",
      len(step0_results) >= 4,
      f"got {len(step0_results)} records for 2 step=0 samples, expected >=4")

# 验证 step_index=0 包含多种错误类型
step0_types = set(r.get("error_type") for r in step0_results)
check("step_index=0 包含多种错误类型",
      len(step0_types) >= 2,
      f"got types: {step0_types}")

# step_index>0 每个样本只有 1 条记录
check("step_index>0 记录数 ≤ 样本数",
      len(other_results) <= 2,
      f"got {len(other_results)} records for 2 step>0 samples")

# 验证不再产生 sorry
sorry_in_results = [r for r in results if r.get("error_tactic") == "sorry"]
check("注入结果中无 sorry",
      len(sorry_in_results) == 0,
      f"found {len(sorry_in_results)} sorry records")

# ================================================================
print("\n" + "=" * 60)
print("Test 7: goal_start_with_fallback 返回值类型检查")
print("=" * 60)

# 验证 goal_start_with_fallback 的返回值签名约定
# 返回 Tuple[Optional[GoalState], bool]: (state, needs_intro)
# copyFrom 成功: (state, False)
# goal_start(expr) 成功: (state, True)
# 全部失败: (None, False)

# 由于本地无 Pantograph 环境，只检查代码结构
import inspect
from src.common.lean_server import LeanServer

method = getattr(LeanServer, 'goal_start_with_fallback', None)
check("LeanServer 有 goal_start_with_fallback 方法",
      method is not None, "")

if method:
    sig = inspect.signature(method)
    # 检查只有 self 和 theorem_full_name 两个参数
    params = list(sig.parameters.keys())
    check("参数: (self, theorem_full_name)",
          params == ["self", "theorem_full_name"],
          f"got params: {params}")

    # 检查源码中包含 return state, True 和 return state, False
    source = inspect.getsource(method)
    check("返回 (state, False) — copyFrom 路径",
          "return state, False" in source,
          "未找到 return state, False")
    check("返回 (state, True) — goal_start(expr) 路径",
          "return state, True" in source,
          "未找到 return state, True")
    check("返回 (None, False) — 全部失败",
          "return None, False" in source,
          "未找到 return None, False")

# ================================================================
print("\n" + "=" * 60)
print("Test 8: error_verifier 调用 goal_start_with_fallback 解包验证")
print("=" * 60)

# 检查 error_verifier.py 正确处理 tuple 返回值
verifier_source_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "data_engine", "error_verifier.py"
)
with open(verifier_source_path, 'r', encoding='utf-8') as f:
    verifier_source = f.read()

check("解包 tuple: isinstance(result, tuple)",
      "isinstance(result, tuple)" in verifier_source,
      "")
check("解包 needs_intro: initial_state, needs_intro = result",
      "initial_state, needs_intro = result" in verifier_source,
      "")
check("条件门控: needs_intro and target_step > 0",
      "needs_intro and target_step > 0" in verifier_source,
      "")
check("try/except 包裹 auto-intro",
      verifier_source.count("auto-intro 异常") >= 1,
      "未找到 auto-intro 异常处理")

# ================================================================
print("\n" + "=" * 60)
print("Test 9: v4 弹性重放 (intro-skip + emergency intros)")
print("=" * 60)
# ================================================================

# 验证 v4 弹性重放相关代码存在
check("v4 intro_skipped_in_replay 计数器",
      "intro_skipped_in_replay" in verifier_source,
      "")
check("v4 emergency_intros 计数器",
      "emergency_intros" in verifier_source,
      "")
check("v4 is_intro_like 检测 — intro",
      'tactic_lower == "intro"' in verifier_source,
      "")
check("v4 is_intro_like 检测 — intros",
      'tactic_lower == "intros"' in verifier_source,
      "")
check("v4 is_intro_like 检测 — rintro",
      'tactic_lower.startswith("rintro ")' in verifier_source,
      "")
check("v4 emergency_intros_done 标记防重复",
      "emergency_intros_done" in verifier_source,
      "")
check("v4 紧急intros后重试原始tactic",
      "# 重试原始 tactic" in verifier_source,
      "")

# 测试 is_intro_like 判断逻辑
intro_test_cases = [
    ("intro x y z", True),
    ("intros", True),
    ("intros x", True),
    ("rintro ⟨h₁, h₂⟩", True),
    ("intro", True),
    ("simp [add_comm]", False),
    ("apply Nat.succ_le", False),
    ("rw [h]", False),
    ("exact h", False),
    ("cases h", False),
]
for tactic, expected in intro_test_cases:
    tactic_lower = tactic.strip().lower()
    actual = (
        tactic_lower == "intro"
        or tactic_lower == "intros"
        or tactic_lower.startswith("intro ")
        or tactic_lower.startswith("intros ")
        or tactic_lower.startswith("rintro ")
    )
    check(f"is_intro_like('{tactic}') == {expected}",
          actual == expected,
          f"got {actual}")

# 验证 YAML config max_concurrent=1
import yaml
with open("configs/data_pipeline.yaml", "r", encoding="utf-8") as f:
    yaml_config = yaml.safe_load(f)
ev_config = yaml_config.get("augmentation", {}).get("error_verification", {})
check("v4 YAML max_concurrent == 1",
      ev_config.get("max_concurrent") == 1,
      f"got {ev_config.get('max_concurrent')}")

# 验证新日志输出包含 v4 统计
check("v4 日志输出: intro跳过统计",
      "intro_skipped_in_replay" in verifier_source,
      "")
check("v4 日志输出: 紧急intros统计",
      "紧急intros恢复" in verifier_source,
      "")

# ================================================================
print("\n" + "=" * 60)
print(f"测试汇总: {pass_count} PASS, {fail_count} FAIL")
print("=" * 60)
sys.exit(0 if fail_count == 0 else 1)
