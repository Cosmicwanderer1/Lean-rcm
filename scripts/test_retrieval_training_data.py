#!/usr/bin/env python3
"""
检索训练数据构建 — 单元测试
@author ygw
创建日期: 2026-03-01

测试内容:
  1. TacticPremiseParser: 正则解析覆盖率验证
  2. PositivePairExtractor: 名称解析逻辑验证
  3. TrainingDatasetBuilder: 格式化 & 分割逻辑验证

用法:
    python scripts/test_retrieval_training_data.py
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# Test Group 1: TacticPremiseParser
# ================================================================

def test_tactic_parser():
    """测试 tactic 前提解析器"""
    from scripts.build_retrieval_training_data import TacticPremiseParser

    test_cases = [
        # (tactic_text, expected_premises)
        # --- exact ---
        ("exact Nat.add_comm a b", ["Nat.add_comm"]),
        ("exact @Nat.succ_lt_succ h", ["Nat.succ_lt_succ"]),
        # --- apply ---
        ("apply Nat.lt_of_lt_of_le", ["Nat.lt_of_lt_of_le"]),
        ("apply @Function.Injective.comp", ["Function.Injective.comp"]),
        # --- rw ---
        ("rw [Nat.add_comm]", ["Nat.add_comm"]),
        ("rw [Nat.add_comm, Nat.mul_comm]", ["Nat.add_comm", "Nat.mul_comm"]),
        ("rw [←Nat.add_assoc]", ["Nat.add_assoc"]),
        ("rewrite [Finset.sum_empty]", ["Finset.sum_empty"]),
        # --- simp ---
        ("simp [Nat.add_zero]", ["Nat.add_zero"]),
        ("simp only [Nat.succ_eq_add_one, Nat.add_comm]",
         ["Nat.add_comm", "Nat.succ_eq_add_one"]),
        # --- refine ---
        ("refine Nat.le_antisymm ?_ ?_", ["Nat.le_antisymm"]),
        ("refine' Exists.intro _ ?_", ["Exists.intro"]),
        # --- have ---
        ("have := Nat.lt_irrefl n", ["Nat.lt_irrefl"]),
        # --- convert ---
        ("convert Finset.card_filter_le _ _", ["Finset.card_filter_le"]),
        # --- linarith ---
        ("linarith [Nat.zero_le n, Nat.succ_pos k]",
         ["Nat.succ_pos", "Nat.zero_le"]),
        # --- norm_num ---
        ("norm_num [Nat.factorial]", ["Nat.factorial"]),
        # --- 无前提的 tactic ---
        ("intro h", []),
        ("intros a b c", []),
        ("constructor", []),
        ("ring", []),
        ("omega", []),
        ("decide", []),
        ("ext x", []),
        ("trivial", []),
        ("assumption", []),
        # --- 局部变量（应被过滤） ---
        ("exact h", []),
        ("apply this", []),
        ("rw [h]", []),
    ]

    passed = 0
    failed = 0
    details = []

    for tactic, expected in test_cases:
        result = TacticPremiseParser.parse(tactic)
        ok = sorted(result) == sorted(expected)
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  {status} '{tactic}'\n"
                           f"     期望: {expected}\n"
                           f"     实际: {result}")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Test Group 1: TacticPremiseParser")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{total}")

    if details:
        print("\n失败用例:")
        for d in details:
            print(d)

    return failed == 0


# ================================================================
# Test Group 2: PositivePairExtractor._resolve_premise
# ================================================================

def test_premise_resolution():
    """测试前提名称解析逻辑"""
    from scripts.build_retrieval_training_data import PositivePairExtractor

    # 创建模拟配置
    config = {
        "retrieval_training": {
            "cos_flat_path": "/dev/null",
            "corpus_path": "/dev/null",
            "positive_pairs_path": "/dev/null",
        }
    }
    extractor = PositivePairExtractor(config)

    # 模拟语料库
    mock_corpus = [
        {"name": "Nat.add_comm", "type_expr": "∀ (n m : ℕ), n + m = m + n"},
        {"name": "Nat.mul_comm", "type_expr": "∀ (n m : ℕ), n * m = m * n"},
        {"name": "Nat.lt_of_lt_of_le", "type_expr": "∀ {a b c : ℕ}, a < b → b ≤ c → a < c"},
        {"name": "List.length_append", "type_expr": "∀ {α}, (l₁ l₂ : List α), ..."},
        {"name": "Finset.sum_empty", "type_expr": "∀ {β}, ..."},
    ]

    for doc in mock_corpus:
        extractor.corpus_index[doc["name"]] = doc
        parts = doc["name"].split(".")
        if len(parts) >= 2:
            extractor.short_name_index[parts[-1]].append(doc["name"])

    test_cases = [
        # (input_name, expected_resolved)
        ("Nat.add_comm", "Nat.add_comm"),          # 精确匹配
        ("Nat.mul_comm", "Nat.mul_comm"),           # 精确匹配
        ("add_comm", "Nat.add_comm"),                # 短名称匹配
        ("sum_empty", "Finset.sum_empty"),           # 短名称匹配
        ("NonExistent.theorem", None),               # 无匹配
    ]

    passed = 0
    failed = 0
    details = []

    for name, expected in test_cases:
        result = extractor._resolve_premise(name)
        ok = result == expected
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  {status} '{name}' → 期望: {expected}, 实际: {result}")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Test Group 2: Premise Resolution")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{total}")

    if details:
        print("\n失败用例:")
        for d in details:
            print(d)

    return failed == 0


# ================================================================
# Test Group 3: TrainingDatasetBuilder._format_sample
# ================================================================

def test_format_sample():
    """测试训练样本格式化"""
    from scripts.build_retrieval_training_data import TrainingDatasetBuilder

    config = {
        "retrieval_training": {
            "triplets_path": "/dev/null",
            "output_dir": "/tmp/test_rt",
            "max_query_length": 100,
            "max_passage_length": 50,
            "train_ratio": 0.9,
            "val_ratio": 0.05,
            "test_ratio": 0.05,
            "seed": 42,
        }
    }
    builder = TrainingDatasetBuilder(config)

    sample = {
        "query_state": "⊢ n + m = m + n" * 20,  # 长查询
        "positive": {
            "name": "Nat.add_comm",
            "type_expr": "∀ (n m : ℕ), n + m = m + n",
            "text": "Nat.add_comm : ∀ (n m : ℕ), n + m = m + n",
        },
        "negatives": [
            {"name": "Nat.mul_comm", "type_expr": "∀ ...", "text": "Nat.mul_comm : ∀ ..."},
            {"name": "Nat.sub_self", "type_expr": "∀ ...", "text": "Nat.sub_self : ∀ ..."},
        ],
        "tactic": "exact Nat.add_comm n m",
        "theorem_name": "test_theorem",
        "step_index": 2,
    }

    result = builder._format_sample(sample)

    checks = [
        ("query 存在", "query" in result),
        ("query 截断", len(result["query"]) <= 100),
        ("positive 存在", "positive" in result),
        ("positive 截断", len(result["positive"]) <= 50),
        ("negatives 数量", len(result["negatives"]) == 2),
        ("positive_name 存在", result["positive_name"] == "Nat.add_comm"),
        ("metadata 存在", "metadata" in result),
        ("metadata.tactic", result["metadata"]["tactic"] == "exact Nat.add_comm n m"),
    ]

    passed = 0
    failed = 0
    details = []

    for label, ok in checks:
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
            details.append(f"  {status} {label}")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Test Group 3: Format Sample")
    print(f"{'='*50}")
    print(f"PASS: {passed}/{total}")

    if details:
        print("\n失败检查:")
        for d in details:
            print(d)

    return failed == 0


# ================================================================
# Test Group 4: 端到端 Phase 1 小数据测试
# ================================================================

def test_phase1_e2e():
    """端到端测试 Phase 1（使用模拟数据）"""
    from scripts.build_retrieval_training_data import PositivePairExtractor

    # 创建临时测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟 cos_flat
        cos_data = [
            {
                "theorem_name": "test_thm_1",
                "theorem_full_name": "Mathlib.Test.thm_1",
                "theorem_type": "theorem",
                "state_before": "n m : ℕ\n⊢ n + m = m + n",
                "tactic": "exact Nat.add_comm n m",
                "state_after": "",
                "step_index": 0,
                "total_steps": 1,
                "file_path": "test.lean",
                "hash": "abc123",
            },
            {
                "theorem_name": "test_thm_2",
                "theorem_full_name": "Mathlib.Test.thm_2",
                "theorem_type": "theorem",
                "state_before": "l₁ l₂ : List ℕ\n⊢ (l₁ ++ l₂).length = l₁.length + l₂.length",
                "tactic": "simp [List.length_append]",
                "state_after": "",
                "step_index": 0,
                "total_steps": 1,
                "file_path": "test.lean",
                "hash": "def456",
            },
            {
                "theorem_name": "test_thm_3",
                "theorem_full_name": "Mathlib.Test.thm_3",
                "theorem_type": "theorem",
                "state_before": "h : P\n⊢ P",
                "tactic": "exact h",  # 局部变量，不应匹配
                "state_after": "",
                "step_index": 0,
                "total_steps": 1,
                "file_path": "test.lean",
                "hash": "ghi789",
            },
            {
                "theorem_name": "test_thm_4",
                "theorem_full_name": "Mathlib.Test.thm_4",
                "theorem_type": "theorem",
                "state_before": "⊢ True",     # 太短
                "tactic": "exact Nat.add_comm n m",
                "state_after": "",
                "step_index": 0,
                "total_steps": 1,
                "file_path": "test.lean",
                "hash": "jkl012",
            },
        ]

        cos_path = os.path.join(tmpdir, "cos_flat.jsonl")
        with open(cos_path, 'w', encoding='utf-8') as f:
            for item in cos_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 模拟 corpus
        corpus_data = [
            {"name": "Nat.add_comm", "type_expr": "∀ (n m : ℕ), n + m = m + n"},
            {"name": "Nat.mul_comm", "type_expr": "∀ (n m : ℕ), n * m = m * n"},
            {"name": "List.length_append", "type_expr": "∀ {α}, (l₁ l₂ : List α), ..."},
        ]

        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for item in corpus_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        output_path = os.path.join(tmpdir, "positive_pairs.jsonl")

        config = {
            "retrieval_training": {
                "cos_flat_path": cos_path,
                "corpus_path": corpus_path,
                "positive_pairs_path": output_path,
                "min_state_length": 20,
                "max_state_length": 2048,
            }
        }

        extractor = PositivePairExtractor(config)
        stats = extractor.run()

        # 验证结果
        pairs = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                pairs.append(json.loads(line))

        checks = [
            ("总记录数=4", stats["total_cos_records"] == 4),
            ("有前提的记录数=2", stats["records_with_premises"] == 2),
            ("正样本对数=2", stats["total_positive_pairs"] == 2),
            # test_thm_3 的 "exact h" 应被过滤（h 是局部变量）
            # test_thm_4 的 state_before 太短应被过滤
            ("第1对前提=Nat.add_comm",
             any(p["positive_name"] == "Nat.add_comm" for p in pairs)),
            ("第2对前提=List.length_append",
             any(p["positive_name"] == "List.length_append" for p in pairs)),
            ("state_too_short>=1", stats["state_too_short"] >= 1),
        ]

        passed = 0
        failed = 0
        details = []

        for label, ok in checks:
            status = "✓" if ok else "✗"
            if ok:
                passed += 1
            else:
                failed += 1
                details.append(f"  {status} {label}")

        total = passed + failed
        print(f"\n{'='*50}")
        print(f"Test Group 4: Phase 1 E2E (Mock Data)")
        print(f"{'='*50}")
        print(f"PASS: {passed}/{total}")
        if stats:
            print(f"统计: total={stats['total_cos_records']}, "
                  f"has_premises={stats['records_with_premises']}, "
                  f"pairs={stats['total_positive_pairs']}, "
                  f"short={stats['state_too_short']}")

        if details:
            print("\n失败检查:")
            for d in details:
                print(d)

        return failed == 0


# ================================================================
# Test Group 5: 端到端 Phase 3 分割测试
# ================================================================

def test_phase3_split():
    """测试 Phase 3 数据集分割"""
    from scripts.build_retrieval_training_data import TrainingDatasetBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟三元组（20个定理，每个3条样本 = 60条）
        triplets = []
        for i in range(20):
            for j in range(3):
                triplets.append({
                    "query_state": f"state for theorem {i} step {j}. " * 5,
                    "positive": {
                        "name": f"Premise.{i}",
                        "type_expr": f"type_{i}",
                        "text": f"Premise.{i} : type_{i}",
                    },
                    "negatives": [
                        {"name": f"Neg.{k}", "type_expr": f"neg_type_{k}",
                         "text": f"Neg.{k} : neg_type_{k}"}
                        for k in range(3)
                    ],
                    "tactic": f"apply Premise.{i}",
                    "theorem_name": f"Theorem.{i}",
                    "step_index": j,
                })

        triplets_path = os.path.join(tmpdir, "triplets.jsonl")
        with open(triplets_path, 'w', encoding='utf-8') as f:
            for t in triplets:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        config = {
            "retrieval_training": {
                "triplets_path": triplets_path,
                "output_dir": tmpdir,
                "max_query_length": 1024,
                "max_passage_length": 256,
                "train_ratio": 0.80,
                "val_ratio": 0.10,
                "test_ratio": 0.10,
                "seed": 42,
            }
        }

        builder = TrainingDatasetBuilder(config)
        stats = builder.run()

        checks = [
            ("总定理数=20", stats["total_theorems"] == 20),
            ("总样本=60", stats["total_triplets"] == 60),
            ("train+val+test=60",
             stats["train_samples"] + stats["val_samples"] + stats["test_samples"] == 60),
            ("train_theorems=16", stats["train_theorems"] == 16),
            ("val_theorems=2", stats["val_theorems"] == 2),
            ("test_theorems=2", stats["test_theorems"] == 2),
            ("train 文件存在", os.path.exists(os.path.join(tmpdir, "train.jsonl"))),
            ("val 文件存在", os.path.exists(os.path.join(tmpdir, "val.jsonl"))),
            ("test 文件存在", os.path.exists(os.path.join(tmpdir, "test.jsonl"))),
        ]

        passed = 0
        failed = 0
        details = []

        for label, ok in checks:
            status = "✓" if ok else "✗"
            if ok:
                passed += 1
            else:
                failed += 1
                details.append(f"  {status} {label} (train_thm={stats['train_theorems']}, "
                               f"val_thm={stats['val_theorems']}, test_thm={stats['test_theorems']})")

        total = passed + failed
        print(f"\n{'='*50}")
        print(f"Test Group 5: Phase 3 Split")
        print(f"{'='*50}")
        print(f"PASS: {passed}/{total}")
        print(f"统计: train={stats['train_samples']}, "
              f"val={stats['val_samples']}, test={stats['test_samples']}")

        if details:
            print("\n失败检查:")
            for d in details:
                print(d)

        return failed == 0


# ================================================================
# 入口
# ================================================================

def main():
    """运行所有测试组"""
    print("\n" + "=" * 60)
    print("检索训练数据构建 — 单元测试")
    print("=" * 60)

    results = []
    tests = [
        ("TacticPremiseParser", test_tactic_parser),
        ("Premise Resolution", test_premise_resolution),
        ("Format Sample", test_format_sample),
        ("Phase 1 E2E", test_phase1_e2e),
        ("Phase 3 Split", test_phase3_split),
    ]

    for name, test_fn in tests:
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{status}] {name}")

    print(f"\n总计: {total_pass}/{total} 组通过")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
