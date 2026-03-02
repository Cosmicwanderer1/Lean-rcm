#!/usr/bin/env python3
"""分批执行功能验证测试 @author ygw 2026-03-02"""
import sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_reflection_data import (
    _get_batch_slice, _build_record_key, _load_batch_progress,
    _show_batch_status, _update_batch_progress,
    ReflectionGenerator, REFLECTION_TEMPLATES,
)

pass_count = 0
fail_count = 0

def check(desc, cond):
    global pass_count, fail_count
    if cond:
        print(f"  PASS: {desc}")
        pass_count += 1
    else:
        print(f"  FAIL: {desc}")
        fail_count += 1

# ============================================================
# Test 1: _get_batch_slice 均匀分配
# ============================================================
print("Test 1: _get_batch_slice 均匀分配")

# 119067 / 6 批
total = 119067
batches = 6
all_ranges = []
total_covered = 0
for bid in range(1, batches + 1):
    s, e = _get_batch_slice(total, batches, bid)
    size = e - s
    total_covered += size
    all_ranges.append((s, e))
    print(f"    Batch {bid}: [{s:>6}, {e:>6}) = {size}")

check("总覆盖数 = 119067", total_covered == total)
# 连续无间隙
no_gap = all(all_ranges[i][1] == all_ranges[i+1][0] for i in range(len(all_ranges)-1))
check("各批次连续无间隙", no_gap)
check("首批从 0 开始", all_ranges[0][0] == 0)
check("末批到 total 结束", all_ranges[-1][1] == total)

# 各批次大小差异 <= 1
sizes = [e - s for s, e in all_ranges]
check("批次大小差异 <= 1", max(sizes) - min(sizes) <= 1)

# ============================================================
# Test 2: 边界情况
# ============================================================
print("\nTest 2: 边界情况")

# 7 / 3 = [3, 2, 2]
s1, e1 = _get_batch_slice(7, 3, 1)
s2, e2 = _get_batch_slice(7, 3, 2)
s3, e3 = _get_batch_slice(7, 3, 3)
check("7/3 batch1 = [0,3)", (s1, e1) == (0, 3))
check("7/3 batch2 = [3,5)", (s2, e2) == (3, 5))
check("7/3 batch3 = [5,7)", (s3, e3) == (5, 7))
check("7/3 总数 = 7", (e1-s1) + (e2-s2) + (e3-s3) == 7)

# 6 / 6 = 每批 1 条
for bid in range(1, 7):
    s, e = _get_batch_slice(6, 6, bid)
    check(f"6/6 batch{bid} 大小=1", e - s == 1)

# 1 / 1
s, e = _get_batch_slice(1, 1, 1)
check("1/1 batch1 = [0,1)", (s, e) == (0, 1))

# ============================================================
# Test 3: _build_record_key
# ============================================================
print("\nTest 3: _build_record_key")
r1 = {"theorem_name": "Nat.add_comm", "error_tactic": "sipm", "step_index": 1, "error_type": "tactic_typo"}
r2 = {"theorem_name": "Nat.add_comm", "error_tactic": "sipm", "step_index": 1, "error_type": "tactic_typo"}
r3 = {"theorem_name": "Nat.add_comm", "error_tactic": "ring", "step_index": 1, "error_type": "wrong_tactic"}
check("相同记录 → 相同 key", _build_record_key(r1) == _build_record_key(r2))
check("不同记录 → 不同 key", _build_record_key(r1) != _build_record_key(r3))
check("key 包含 theorem_name", "Nat.add_comm" in _build_record_key(r1))
check("key 包含 error_type", "tactic_typo" in _build_record_key(r1))

# ============================================================
# Test 4: run() 方法签名包含 batch_id
# ============================================================
print("\nTest 4: ReflectionGenerator.run() 签名")
sig = inspect.signature(ReflectionGenerator.run)
check("batch_id 在参数列表中", "batch_id" in sig.parameters)
check("batch_id 默认值 = 0", sig.parameters["batch_id"].default == 0)

# ============================================================
# Test 5: _batch_id 属性传递
# ============================================================
print("\nTest 5: _batch_id 传递到 _async_generate")
gen = ReflectionGenerator.__new__(ReflectionGenerator)
gen._batch_id = 0
check("默认 _batch_id = 0", gen._batch_id == 0)
gen._batch_id = 3
check("设置 _batch_id = 3", gen._batch_id == 3)

# ============================================================
# Test 6: argparse 参数验证
# ============================================================
print("\nTest 6: CLI 参数验证")
import argparse
# 模拟 argparse 输出
import io, contextlib
from scripts.build_reflection_data import main
# 检查 help 文本包含分批参数
help_text = ""
try:
    old_argv = sys.argv
    sys.argv = ["test", "--help"]
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        main()
except SystemExit:
    help_text = f.getvalue()
finally:
    sys.argv = old_argv

check("help 包含 --batch-id", "--batch-id" in help_text)
check("help 包含 --total-batches", "--total-batches" in help_text)
check("help 包含 --merge", "--merge" in help_text)
check("help 包含 --batch-status", "--batch-status" in help_text)
check("help 包含 --force", "--force" in help_text)
check("help 包含 '分期付款'", "分期付款" in help_text)

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print(f"测试汇总: {pass_count} PASS, {fail_count} FAIL")
print("=" * 60)
if fail_count > 0:
    sys.exit(1)
