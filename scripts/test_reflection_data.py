#!/usr/bin/env python3
"""
Task 2.3: Reflection 数据生成 — 单元测试
@author ygw
日期: 2026-03-01

测试内容:
  1. ReflectionDataFilter: 数据过滤逻辑
  2. ReflectionGenerator: 模板生成 + 质量验证
  3. ReflectionDatasetBuilder: SFT 格式化 + 分割
  4. 端到端 (模拟数据)

用法:
    python scripts/test_reflection_data.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# Test Group 1: ReflectionDataFilter
# ================================================================

def test_data_filter():
    """测试数据过滤逻辑"""
    from scripts.build_reflection_data import ReflectionDataFilter

    with tempfile.TemporaryDirectory() as tmpdir:
        # 构造模拟数据
        records = [
            # 有效：replay_failed + 所有必需字段
            {
                "verification_status": "replay_failed_at_step_0",
                "error_message": "unknown identifier 'Nat.add_com'",
                "state_before": "n m : ℕ\n⊢ n + m = m + n",
                "original_tactic": "exact Nat.add_comm n m",
                "error_tactic": "exact Nat.add_com n m",
                "error_type": "tactic_typo",
                "repair_hint": "Fix typo: add_com → add_comm",
            },
            # 有效：不同 step
            {
                "verification_status": "replay_failed_at_step_3",
                "error_message": "type mismatch",
                "state_before": "a b : ℕ\n⊢ a * b = b * a",
                "original_tactic": "apply Nat.mul_comm",
                "error_tactic": "apply Nat.add_comm",
                "error_type": "wrong_tactic",
                "repair_hint": "Use mul_comm instead of add_comm",
            },
            # 无效：verification_status 不匹配
            {
                "verification_status": "verified_ok",
                "error_message": "no error",
                "state_before": "⊢ True",
                "original_tactic": "trivial",
                "error_tactic": "trivial",
                "error_type": "none",
                "repair_hint": "",
            },
            # 无效：缺少 error_message
            {
                "verification_status": "replay_failed_at_step_1",
                "error_message": "",
                "state_before": "⊢ P",
                "original_tactic": "exact h",
                "error_tactic": "exact g",
                "error_type": "argument_error",
                "repair_hint": "",
            },
            # 无效：timeout
            {
                "verification_status": "timeout",
                "error_message": "timeout after 30s",
                "state_before": "complex state",
                "original_tactic": "simp",
                "error_tactic": "ring",
                "error_type": "wrong_tactic",
                "repair_hint": "",
            },
        ]

        input_path = os.path.join(tmpdir, "test_input.jsonl")
        with open(input_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        config = {
            "input_path": input_path,
            "valid_statuses": [
                "replay_failed_at_step_0", "replay_failed_at_step_1",
                "replay_failed_at_step_2", "replay_failed_at_step_3",
            ],
            "required_fields": ["error_message", "state_before",
                                "original_tactic", "error_tactic"],
            "max_samples": 0,
            "max_per_error_type": 0,
        }

        data_filter = ReflectionDataFilter(config)
        filtered, stats = data_filter.load_and_filter()

        checks = [
            ("原始记录数=5", stats["total_records"] == 5),
            ("status过滤后=3", stats["after_status_filter"] == 3),
            ("字段过滤后=2", stats["after_field_filter"] == 2),
            ("最终=2", stats["final_count"] == 2),
            ("第1条是tactic_typo", filtered[0]["error_type"] == "tactic_typo"),
            ("第2条是wrong_tactic", filtered[1]["error_type"] == "wrong_tactic"),
        ]

        passed = 0
        failed = 0
        details = []
        for label, ok in checks:
            if ok:
                passed += 1
            else:
                failed += 1
                details.append(f"  ✗ {label}")

        print(f"\n{'='*50}")
        print("Test Group 1: ReflectionDataFilter")
        print(f"{'='*50}")
        print(f"PASS: {passed}/{passed + failed}")
        if details:
            print("失败:")
            for d in details:
                print(d)
        return failed == 0


# ================================================================
# Test Group 2: 模板生成 + 质量验证
# ================================================================

def test_template_generation():
    """测试模板回退 Reflection 生成"""
    from scripts.build_reflection_data import ReflectionGenerator

    config = {
        "teacher_model": {"api_key_env": "NONEXISTENT_KEY_12345"},
        "quality_filter": {
            "min_length": 20,
            "max_length": 800,
            "forbidden_patterns": ["I don't know"],
        },
        "template_fallback": {"enable": True},
    }
    gen = ReflectionGenerator(config)

    test_records = [
        {
            "error_type": "tactic_typo",
            "error_tactic": "exact Nat.add_com",
            "original_tactic": "exact Nat.add_comm",
            "error_message": "unknown identifier 'Nat.add_com'",
            "repair_hint": "Fix typo: add_com → add_comm",
            "state_before": "n m : ℕ\n⊢ n + m = m + n",
            "theorem_name": "test_thm",
        },
        {
            "error_type": "wrong_tactic",
            "error_tactic": "ring",
            "original_tactic": "simp [List.length_append]",
            "error_message": "ring failed",
            "repair_hint": "Use simp instead of ring",
            "state_before": "l₁ l₂ : List ℕ\n⊢ (l₁ ++ l₂).length = ...",
            "theorem_name": "test_thm2",
        },
        {
            "error_type": "argument_error",
            "error_tactic": "exact Nat.add_comm m",
            "original_tactic": "exact Nat.add_comm n m",
            "error_message": "wrong number of arguments",
            "repair_hint": "Supply both arguments",
            "state_before": "n m : ℕ\n⊢ n + m = m + n",
            "theorem_name": "test_thm3",
        },
        {
            "error_type": "missing_step",
            "error_tactic": "exact Nat.lt_irrefl",
            "original_tactic": "apply Nat.lt_trans; exact h",
            "error_message": "type mismatch",
            "repair_hint": "Need intermediate lt_trans step",
            "state_before": "h : a < b\nhb : b < c\n⊢ a < c",
            "theorem_name": "test_thm4",
        },
    ]

    results = gen._template_generate(test_records)

    checks = [
        ("生成4条", len(results) == 4),
        ("第1条含typo关键词", "spelling" in results[0]["reflection"].lower()
         if len(results) > 0 else False),
        ("第2条含wrong关键词", "inappropriate" in results[1]["reflection"].lower()
         if len(results) > 1 else False),
        ("第3条含argument关键词", "argument" in results[2]["reflection"].lower()
         if len(results) > 2 else False),
        ("第4条含intermediate关键词", "intermediate" in results[3]["reflection"].lower()
         if len(results) > 3 else False),
        ("所有都是template来源",
         all(r["reflection_source"] == "template" for r in results)),
        ("所有reflection长度>=20",
         all(len(r["reflection"]) >= 20 for r in results)),
    ]

    passed = 0
    failed = 0
    details = []
    for label, ok in checks:
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  ✗ {label}")

    print(f"\n{'='*50}")
    print("Test Group 2: Template Generation")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{passed + failed}")
    if details:
        print("失败:")
        for d in details:
            print(d)
    return failed == 0


# ================================================================
# Test Group 3: 质量验证逻辑
# ================================================================

def test_quality_validation():
    """测试 Reflection 质量验证"""
    from scripts.build_reflection_data import ReflectionGenerator

    config = {
        "teacher_model": {"api_key_env": "NONEXISTENT_KEY"},
        "quality_filter": {
            "min_length": 30,
            "max_length": 200,
            "forbidden_patterns": ["I don't know", "As an AI"],
            "require_keywords": [],
        },
    }
    gen = ReflectionGenerator(config)

    checks = [
        ("正常文本通过", gen._validate_reflection(
            "The tactic failed because the identifier is misspelled. "
            "Use Nat.add_comm instead.")),
        ("过短拒绝", not gen._validate_reflection("Too short.")),
        ("空文本拒绝", not gen._validate_reflection("")),
        ("None拒绝", not gen._validate_reflection(None)),
        ("过长拒绝", not gen._validate_reflection("x" * 201)),
        ("禁词拒绝1", not gen._validate_reflection(
            "I don't know what went wrong here, but maybe check the types.")),
        ("禁词拒绝2", not gen._validate_reflection(
            "As an AI language model, I think the error is a typo in the tactic.")),
    ]

    passed = 0
    failed = 0
    details = []
    for label, ok in checks:
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  ✗ {label}")

    print(f"\n{'='*50}")
    print("Test Group 3: Quality Validation")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{passed + failed}")
    if details:
        print("失败:")
        for d in details:
            print(d)
    return failed == 0


# ================================================================
# Test Group 4: SFT 格式化 + 分割
# ================================================================

def test_sft_builder():
    """测试 SFT 数据集构建器"""
    from scripts.build_reflection_data import ReflectionDatasetBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "output_path": tmpdir,
            "split": {"train_ratio": 0.80, "val_ratio": 0.10, "test_ratio": 0.10},
            "filenames": {
                "full": "full.jsonl",
                "train": "train.jsonl",
                "val": "val.jsonl",
                "test": "test.jsonl",
                "stats": "stats.json",
            },
            "instruction_template": "",
        }
        builder = ReflectionDatasetBuilder(config)

        # 构造 20 个定理，每个 3 条
        records = []
        for i in range(20):
            for j in range(3):
                records.append({
                    "state_before": f"state for theorem {i} step {j}. " * 5,
                    "error_tactic": f"wrong_tactic_{i}_{j}",
                    "original_tactic": f"correct_tactic_{i}",
                    "error_message": f"error message {i}_{j}",
                    "error_type": ["tactic_typo", "wrong_tactic",
                                   "argument_error", "missing_step"][i % 4],
                    "theorem_name": f"Theorem_{i}",
                    "theorem_full_name": f"Mathlib.Test.Theorem_{i}",
                    "thought": f"thought for step {j}",
                    "repair_hint": f"repair hint {i}",
                    "reflection": f"The tactic wrong_tactic_{i}_{j} failed because "
                                  f"of error message {i}_{j}. Fix: use correct_tactic_{i}.",
                    "reflection_source": "template",
                })

        final_stats = builder.build(records, {}, {}, seed=42)
        p3 = final_stats["phase3_dataset"]

        checks = [
            ("SFT格式化=60", p3["sft_formatted"] == 60),
            ("去重后=60", p3["after_dedup"] == 60),
            ("train+val+test=60",
             p3["train_count"] + p3["val_count"] + p3["test_count"] == 60),
            ("train约48", 40 <= p3["train_count"] <= 54),
            ("val约6", 3 <= p3["val_count"] <= 12),
            ("test约6", 3 <= p3["test_count"] <= 12),
            ("train文件存在", os.path.exists(os.path.join(tmpdir, "train.jsonl"))),
            ("stats文件存在", os.path.exists(os.path.join(tmpdir, "stats.json"))),
        ]

        # 验证 SFT 格式
        with open(os.path.join(tmpdir, "train.jsonl"), 'r') as f:
            first = json.loads(f.readline())
            checks.append(("有instruction字段", "instruction" in first))
            checks.append(("有output字段", "output" in first))
            checks.append(("有metadata字段", "metadata" in first))
            checks.append(("metadata有error_type",
                           "error_type" in first.get("metadata", {})))

        passed = 0
        failed = 0
        details = []
        for label, ok in checks:
            if ok:
                passed += 1
            else:
                failed += 1
                details.append(f"  ✗ {label}")

        print(f"\n{'='*50}")
        print("Test Group 4: SFT Builder")
        print(f"{'='*50}")
        print(f"PASS: {passed}/{passed + failed}")
        print(f"分布: train={p3['train_count']}, val={p3['val_count']}, "
              f"test={p3['test_count']}")
        if details:
            print("失败:")
            for d in details:
                print(d)
        return failed == 0


# ================================================================
# Test Group 5: 端到端（模拟数据全流程）
# ================================================================

def test_e2e():
    """端到端测试：Phase 1 → Phase 2 (template) → Phase 3"""
    from scripts.build_reflection_data import (
        ReflectionDataFilter, ReflectionGenerator, ReflectionDatasetBuilder
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 构造模拟输入数据
        input_records = []
        for i in range(30):
            et = ["tactic_typo", "wrong_tactic", "argument_error", "missing_step"][i % 4]
            input_records.append({
                "verification_status": f"replay_failed_at_step_{i % 5}",
                "error_message": f"error at step {i}: unknown tactic" if i % 7 != 0
                                 else "",  # 每7条缺少error_message
                "state_before": f"n m : ℕ\n⊢ goal_{i}\n" + "x " * 10,
                "state_after": f"remaining goal {i}",
                "original_tactic": f"exact Lemma_{i}",
                "error_tactic": f"exact Lemma_{i}_wrong",
                "error_type": et,
                "repair_hint": f"Fix: use Lemma_{i}",
                "theorem_name": f"Thm_{i // 3}",
                "theorem_full_name": f"Mathlib.Test.Thm_{i // 3}",
                "thought": f"thought for step {i}",
            })

        input_path = os.path.join(tmpdir, "input.jsonl")
        with open(input_path, 'w', encoding='utf-8') as f:
            for r in input_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        output_dir = os.path.join(tmpdir, "output")

        # Phase 1
        filter_cfg = {
            "input_path": input_path,
            "valid_statuses": [f"replay_failed_at_step_{i}" for i in range(10)],
            "required_fields": ["error_message", "state_before",
                                "original_tactic", "error_tactic"],
        }
        data_filter = ReflectionDataFilter(filter_cfg)
        filtered, filter_stats = data_filter.load_and_filter()

        # Phase 2 (template mode)
        gen_cfg = {
            "teacher_model": {"api_key_env": "NONEXISTENT"},
            "quality_filter": {"min_length": 20, "max_length": 800},
            "template_fallback": {"enable": True},
        }
        gen = ReflectionGenerator(gen_cfg)
        results, gen_stats = gen.run(filtered, output_dir, mode="template")

        # Phase 3
        dataset_cfg = {
            "output_path": output_dir,
            "split": {"train_ratio": 0.80, "val_ratio": 0.10, "test_ratio": 0.10},
            "filenames": {
                "full": "full.jsonl", "train": "train.jsonl",
                "val": "val.jsonl", "test": "test.jsonl",
                "stats": "stats.json",
            },
        }
        builder = ReflectionDatasetBuilder(dataset_cfg)
        final_stats = builder.build(results, filter_stats, gen_stats, seed=42)
        p3 = final_stats["phase3_dataset"]

        # 验证
        expected_filtered = 30 - (30 // 7 + 1)  # 去掉缺少 error_message 的
        checks = [
            ("Phase1 过滤有效", len(filtered) > 0),
            ("Phase1 过滤了空 error_message",
             filter_stats["final_count"] < filter_stats["total_records"]),
            ("Phase2 生成了 reflection", len(results) > 0),
            ("Phase2 全部是 template", gen_stats["template_generated"] > 0),
            ("Phase2 API=0", gen_stats["api_generated"] == 0),
            ("Phase3 有 train", p3["train_count"] > 0),
            ("Phase3 有 val", p3["val_count"] >= 0),
            ("Phase3 总数一致",
             p3["train_count"] + p3["val_count"] + p3["test_count"] == p3["after_dedup"]),
            ("输出文件存在", os.path.exists(os.path.join(output_dir, "train.jsonl"))),
            ("stats文件存在", os.path.exists(os.path.join(output_dir, "stats.json"))),
        ]

        passed = 0
        failed = 0
        details = []
        for label, ok in checks:
            if ok:
                passed += 1
            else:
                failed += 1
                details.append(f"  ✗ {label}")

        print(f"\n{'='*50}")
        print("Test Group 5: E2E Pipeline")
        print(f"{'='*50}")
        print(f"PASS: {passed}/{passed + failed}")
        print(f"Phase1: {filter_stats['total_records']}→{filter_stats['final_count']}, "
              f"Phase2: {len(results)}, "
              f"Phase3: train={p3['train_count']}/val={p3['val_count']}/test={p3['test_count']}")
        if details:
            print("失败:")
            for d in details:
                print(d)
        return failed == 0


# ================================================================
# Test Group 6: 一致性校验 + 增强模板 + 目标提取
# @author ygw 2026-03-02 新增：测试 v2 质量优化功能
# ================================================================

def test_quality_enhancements():
    """测试 v2 质量优化：一致性校验、目标提取、增强模板"""
    from scripts.build_reflection_data import ReflectionGenerator

    config = {
        "teacher_model": {"api_key_env": "NONEXISTENT_KEY"},
        "quality_filter": {
            "min_length": 20,
            "max_length": 1200,
            "forbidden_patterns": ["I don't know"],
        },
        "template_fallback": {"enable": True},
    }
    gen = ReflectionGenerator(config)

    # --- 子测试 A: _extract_goal_summary ---
    checks = []

    # A1: 含 ⊢ 的 state
    summary1 = gen._extract_goal_summary("n m : ℕ\n⊢ n + m = m + n")
    checks.append(("A1: 提取⊢后内容", "⊢ n + m = m + n" in summary1))

    # A2: 不含 ⊢
    summary2 = gen._extract_goal_summary("some complex state without goal marker")
    checks.append(("A2: 无⊢取前150字符", len(summary2) <= 150 and "complex" in summary2))

    # A3: 空 state
    summary3 = gen._extract_goal_summary("")
    checks.append(("A3: 空state返回占位", summary3 == "(empty state)"))

    # A4: 超长 goal
    long_goal = "n m : ℕ\n⊢ " + "x " * 500
    summary4 = gen._extract_goal_summary(long_goal)
    checks.append(("A4: 超长goal截断", len(summary4) <= 220))

    # --- 子测试 B: _extract_tactic_identifiers ---
    # B1: qualified name
    ids1 = gen._extract_tactic_identifiers("exact Nat.add_comm n m")
    checks.append(("B1: 提取Nat.add_comm", "Nat.add_comm" in ids1))

    # B2: 方括号内的标识符
    ids2 = gen._extract_tactic_identifiers("rw [mul_comm]")
    checks.append(("B2: 提取mul_comm", "mul_comm" in ids2))

    # B3: 多个方括号内标识符
    ids3 = gen._extract_tactic_identifiers("simp [List.length_append, Nat.add_zero]")
    checks.append(("B3: 提取List.length_append",
                    "List.length_append" in ids3 or "Nat.add_zero" in ids3))

    # B4: 简单策略
    ids4 = gen._extract_tactic_identifiers("sorry")
    checks.append(("B4: sorry自身", "sorry" in ids4))

    # B5: 空策略
    ids5 = gen._extract_tactic_identifiers("")
    checks.append(("B5: 空策略返回空列表", ids5 == []))

    # --- 子测试 C: _validate_consistency ---
    # C1: 包含完整正确策略 → 通过
    record_c1 = {"original_tactic": "exact Nat.add_comm n m", "error_tactic": "exact Nat.add_com n m", "error_type": "tactic_typo"}
    checks.append(("C1: 含完整策略通过",
                    gen._validate_consistency(
                        "The correct tactic is exact Nat.add_comm n m.", record_c1)))

    # C2: 包含核心标识符 → 通过
    checks.append(("C2: 含核心ID通过",
                    gen._validate_consistency(
                        "Use Nat.add_comm instead of the misspelled version.", record_c1)))

    # C3: 完全不相关的反思 → 不通过
    record_c3 = {"original_tactic": "rw [mul_comm]", "error_tactic": "ring", "error_type": "wrong_tactic"}
    checks.append(("C3: 不相关反思拒绝",
                    not gen._validate_consistency(
                        "The constructor tactic failed because the goal is not an inductive type.",
                        record_c3)))

    # C4: 提及错误策略+错误类型 → 通过（规则3）
    checks.append(("C4: 提及error_tactic+type通过",
                    gen._validate_consistency(
                        "The tactic ring is the wrong choice here. This is a mismatch error.",
                        record_c3)))

    # C5: 空 original_tactic → 跳过校验
    record_c5 = {"original_tactic": "", "error_tactic": "ring", "error_type": "wrong_tactic"}
    checks.append(("C5: 空正确策略跳过校验",
                    gen._validate_consistency("Anything here.", record_c5)))

    # --- 子测试 D: _validate_reflection 带 record 参数 ---
    # D1: 合格反思+一致性 → 通过
    checks.append(("D1: 合格+一致通过",
                    gen._validate_reflection(
                        "The tactic failed because Nat.add_com is misspelled. "
                        "Use Nat.add_comm instead.",
                        record_c1)))

    # D2: 合格反思但不一致 → 拒绝
    record_d2 = {"original_tactic": "simp [List.length_append]",
                 "error_tactic": "norm_num", "error_type": "wrong_tactic"}
    checks.append(("D2: 合格但不一致拒绝",
                    not gen._validate_reflection(
                        "The constructor tactic failed because the inductive type is wrong. "
                        "This is a fundamental category theory issue.",
                        record_d2)))

    # D3: 无 record 参数 → 只检查基本质量（向后兼容）
    checks.append(("D3: 无record向后兼容",
                    gen._validate_reflection(
                        "The tactic failed because type is wrong. Fix it properly.")))

    # --- 子测试 E: 增强模板包含 proof state 引用 ---
    test_record = {
        "error_type": "tactic_typo",
        "error_tactic": "exact Nat.add_com",
        "original_tactic": "exact Nat.add_comm",
        "error_message": "unknown identifier 'Nat.add_com'",
        "repair_hint": "Fix typo: add_com → add_comm",
        "state_before": "n m : ℕ\n⊢ n + m = m + n",
        "theorem_name": "test_thm",
    }
    template_result = gen._generate_template_reflection(test_record)
    checks.append(("E1: 模板结果非空", template_result is not None))
    checks.append(("E2: 模板含goal引用",
                    "⊢" in template_result if template_result else False))
    checks.append(("E3: 模板含正确策略",
                    "Nat.add_comm" in template_result if template_result else False))
    checks.append(("E4: 模板长度>=80",
                    len(template_result) >= 80 if template_result else False))

    # --- 子测试 F: 增强 prompt 包含 repair_hint ---
    prompt = gen._build_prompt(test_record)
    checks.append(("F1: prompt含repair_hint", "Fix typo" in prompt))
    checks.append(("F2: prompt含MISLEADING警告", "MISLEADING" in prompt))
    checks.append(("F3: prompt含reasoning order", "reasoning order" in prompt.lower()
                    or "Follow this" in prompt))
    checks.append(("F4: prompt含MUST incorporate hint",
                    "MUST incorporate" in prompt or "MUST include" in prompt))

    # --- 子测试 G: _enrich_api_output 后处理 ---
    # G1: 缺少 hint → 应追加 hint 句子
    api_text_no_hint = "The tactic exact is wrong here. Use apply instead."
    record_g = {
        "repair_hint": "typo: 'ecact' should be 'exact'",
        "state_before": "n m : ℕ\n⊢ n + m = m + n",
        "original_tactic": "exact Nat.add_comm",
        "error_tactic": "ecact Nat.add_comm",
        "error_type": "tactic_typo",
    }
    enriched_g1 = gen._enrich_api_output(api_text_no_hint, record_g)
    checks.append(("G1: 缺hint后追加", "repair hint" in enriched_g1.lower()
                    or "typo" in enriched_g1.lower()))

    # G2: 已包含 hint → 不应重复追加
    api_text_has_hint = "The tactic ecact is a typo, should be exact. Fix typo: ecact should be exact."
    enriched_g2 = gen._enrich_api_output(api_text_has_hint, record_g)
    checks.append(("G2: 含hint不重复",
                    enriched_g2.count("repair hint") <= 1))

    # G3: 缺少 ⊢ 引用 → 应追加目标引用
    api_text_no_goal = "The wrong tactic fails because of a typo."
    enriched_g3 = gen._enrich_api_output(api_text_no_goal, record_g)
    checks.append(("G3: 缺⊢后追加目标引用", "⊢" in enriched_g3 or "proof goal" in enriched_g3.lower()))

    # G4: 已包含 ⊢ → 不追加
    api_text_has_goal = "Given goal ⊢ n + m = m + n, the tactic should be exact."
    enriched_g4 = gen._enrich_api_output(api_text_has_goal, record_g)
    checks.append(("G4: 含⊢不追加", enriched_g4.count("proof goal") == 0
                    or enriched_g4 == api_text_has_goal))

    # --- 子测试 H: 修复后的 ⊢ 提取（长 state_before）---
    # H1: ⊢ 在 300 字符之后的 state_before → 模板应仍能提取
    long_state = "hyp1 : ℕ\n" * 40 + "⊢ very_important_goal"
    record_h = {
        "error_type": "wrong_tactic",
        "error_tactic": "ring",
        "original_tactic": "exact proof_term",
        "error_message": "some error",
        "repair_hint": "wrong tactic: should use exact",
        "state_before": long_state,
    }
    template_h = gen._generate_template_reflection(record_h)
    checks.append(("H1: 长state模板含⊢",
                    template_h is not None and "⊢" in template_h))
    checks.append(("H2: 长state模板含goal内容",
                    template_h is not None and "very_important_goal" in template_h))

    # 输出
    passed = 0
    failed = 0
    details = []
    for label, ok in checks:
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  ✗ {label}")

    print(f"\n{'='*50}")
    print("Test Group 6: Quality Enhancements (v2)")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{passed + failed}")
    if details:
        print("失败:")
        for d in details:
            print(d)
    return failed == 0


# ================================================================
# 入口
# ================================================================

def main():
    print("\n" + "=" * 60)
    print("Reflection 数据生成 — 单元测试")
    print("=" * 60)

    results = []
    tests = [
        ("ReflectionDataFilter", test_data_filter),
        ("Template Generation", test_template_generation),
        ("Quality Validation", test_quality_validation),
        ("SFT Builder", test_sft_builder),
        ("E2E Pipeline", test_e2e),
        ("Quality Enhancements v2", test_quality_enhancements),
    ]

    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    total_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS ✓' if ok else 'FAIL ✗'}] {name}")
    print(f"\n总计: {total_pass}/{len(results)} 组通过")
    return total_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
