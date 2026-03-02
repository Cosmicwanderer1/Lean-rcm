#!/usr/bin/env python3
"""
Batch-1 Thought 增强补丁

对已生成的 reflection_batch_1.jsonl 进行 thought 后缀追加，
使其与后续批次（batch 2-6，代码中已内置增强逻辑）保持一致。

用法:
    python scripts/patch_batch1_thought.py [--dry-run]

@author ygw 2026-03-02
"""
import json
import sys
import shutil
from pathlib import Path
from collections import Counter

# ── 常量 ──────────────────────────────────────────────
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

INPUT_FILE = Path(r"D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl")
OUTPUT_FILE = Path(r"D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl")
BACKUP_FILE = Path(r"D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl.bak")


def enhance_thought(thought: str, error_type: str) -> str:
    """按错误类型追加 thought 后缀"""
    if not thought:
        return thought
    suffix = ERROR_TYPE_THOUGHT_SUFFIX.get(error_type, "")
    if suffix:
        thought_trimmed = thought.rstrip()
        if thought_trimmed and thought_trimmed[-1] not in '.!?。':
            thought_trimmed += '.'
        return thought_trimmed + suffix
    return thought


def main():
    dry_run = "--dry-run" in sys.argv

    if not INPUT_FILE.exists():
        print(f"错误: 文件不存在 {INPUT_FILE}")
        sys.exit(1)

    print(f"读取: {INPUT_FILE}")
    lines = INPUT_FILE.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(l) for l in lines]
    N = len(records)

    # 检查是否已经打过补丁（通过检测 [Error Context] 标记）
    already_patched = sum(1 for r in records if "[Error Context]" in r.get("thought", ""))
    if already_patched > 0:
        print(f"⚠ 检测到 {already_patched}/{N} 条已含 [Error Context]，疑似重复打补丁")
        if already_patched > N * 0.5:
            print("已有超过 50% 的记录包含增强后缀，跳过。")
            sys.exit(0)

    # 统计修改情况
    modified = 0
    skipped = 0
    type_stats = Counter()

    for r in records:
        et = r.get("error_type", "")
        old_thought = r.get("thought", "")
        if et in ERROR_TYPE_THOUGHT_SUFFIX and old_thought:
            new_thought = enhance_thought(old_thought, et)
            if new_thought != old_thought:
                r["thought"] = new_thought
                modified += 1
                type_stats[et] += 1
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"\n修改统计:")
    print(f"  总记录: {N}")
    print(f"  已增强: {modified}")
    print(f"  跳过:   {skipped}")
    print(f"\n按错误类型:")
    for et, cnt in type_stats.most_common():
        print(f"  {et}: {cnt}")

    # 抽样展示
    print(f"\n增强前后对比 (各类型1条):")
    shown = set()
    for r in records:
        et = r.get("error_type", "")
        if et not in shown and "[Error Context]" in r.get("thought", ""):
            shown.add(et)
            thought = r["thought"]
            # 找到 [Error Context] 边界
            idx = thought.find(" [Error Context]")
            orig = thought[:idx] if idx > 0 else thought
            suffix = thought[idx:] if idx > 0 else ""
            print(f"\n  [{et}] {r['theorem_name']}")
            print(f"  原始末尾: ...{orig[-60:]}")
            print(f"  增强后缀: {suffix[:100]}...")
        if len(shown) >= 4:
            break

    if dry_run:
        print(f"\n[DRY RUN] 不写入文件。去掉 --dry-run 参数以执行实际写入。")
        return

    # 备份原文件
    print(f"\n备份原文件: {BACKUP_FILE}")
    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    # 写入
    print(f"写入: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 校验
    written_lines = OUTPUT_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(written_lines) == N, f"行数不匹配: {len(written_lines)} vs {N}"
    print(f"\n✓ 写入完成: {N} 条记录，文件大小 {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"✓ 备份保存在: {BACKUP_FILE}")


if __name__ == "__main__":
    main()
