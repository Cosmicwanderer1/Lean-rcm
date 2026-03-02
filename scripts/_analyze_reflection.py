#!/usr/bin/env python3
"""分析 reflection_dataset.jsonl 的测试效果 @author ygw 2026-03-02"""
import json, sys
from collections import Counter
from pathlib import Path

fp = Path(r"D:\RTAP\data\processed\error_correction\reflection_dataset.jsonl")
records = [json.loads(l) for l in fp.read_text(encoding="utf-8").strip().splitlines()]

N = len(records)
print("=" * 65)
print(f"Reflection 数据集分析 — 共 {N} 条")
print("=" * 65)

# 1. 来源分布
src_dist = Counter(r["metadata"]["reflection_source"] for r in records)
print(f"\n1. 来源分布:")
for src, cnt in src_dist.most_common():
    print(f"   {src}: {cnt} ({cnt/N*100:.1f}%)")

# 2. 错误类型分布
type_dist = Counter(r["metadata"]["error_type"] for r in records)
print(f"\n2. 错误类型分布:")
for et, cnt in type_dist.most_common():
    print(f"   {et}: {cnt} ({cnt/N*100:.1f}%)")

# 3. output 长度统计
lengths = [len(r["output"]) for r in records]
print(f"\n3. Reflection 长度统计:")
print(f"   最短: {min(lengths)}")
print(f"   最长: {max(lengths)}")
print(f"   平均: {sum(lengths)/len(lengths):.0f}")
print(f"   中位数: {sorted(lengths)[len(lengths)//2]}")

# 4. 按来源分开统计长度
print(f"\n4. 按来源的长度分布:")
for src in src_dist:
    src_lengths = [len(r["output"]) for r in records if r["metadata"]["reflection_source"] == src]
    if src_lengths:
        print(f"   {src}: avg={sum(src_lengths)/len(src_lengths):.0f}, "
              f"min={min(src_lengths)}, max={max(src_lengths)}")

# 5. 质量指标检查
print(f"\n5. 质量指标:")
# 5a. 是否包含正确策略引用
has_correct = sum(
    1 for r in records
    if r["metadata"]["correct_tactic"][:20] in r["output"]
)
print(f"   包含正确策略引用: {has_correct}/{N} ({has_correct/N*100:.1f}%)")

# 5b. 是否包含错误策略引用
has_error_tactic = 0
for r in records:
    inst = r["instruction"]
    marker = "[Attempted Tactic]\n"
    idx = inst.find(marker)
    if idx >= 0:
        tactic_start = idx + len(marker)
        tactic_end = inst.find("\n\n", tactic_start)
        if tactic_end < 0:
            tactic_end = len(inst)
        attempted = inst[tactic_start:tactic_end].strip()
        if attempted[:15] in r["output"]:
            has_error_tactic += 1
print(f"   包含错误策略引用: {has_error_tactic}/{N} ({has_error_tactic/N*100:.1f}%)")

# 5c. 是否包含 repair_hint
has_hint = sum(
    1 for r in records
    if r["metadata"]["repair_hint"]
    and r["metadata"]["repair_hint"][:25] in r["output"]
)
print(f"   包含 repair_hint: {has_hint}/{N} ({has_hint/N*100:.1f}%)")

# 5d. forbidden patterns
forbidden = ["I don't know", "I'm not sure", "I cannot determine", "As an AI"]
has_forbidden = sum(
    1 for r in records
    if any(fp in r["output"] for fp in forbidden)
)
print(f"   含禁止短语: {has_forbidden}/{N}")

# 5e. 是否引用 error_message 内容
has_error_msg_ref = 0
for r in records:
    inst = r["instruction"]
    marker = "[Error Message]\n"
    idx = inst.find(marker)
    if idx >= 0:
        err_text = inst[idx+len(marker):].strip()[:50]
        if err_text[:20] in r["output"]:
            has_error_msg_ref += 1
print(f"   引用 error_message: {has_error_msg_ref}/{N} ({has_error_msg_ref/N*100:.1f}%)")

# 6. 抽样展示（每种来源各 1 条）
print(f"\n6. 抽样展示:")
shown_sources = set()
for r in records:
    src = r["metadata"]["reflection_source"]
    if src not in shown_sources:
        shown_sources.add(src)
        print(f"\n   --- [{src}] {r['metadata']['error_type']} | {r['metadata']['theorem_name']} ---")
        output = r["output"]
        if len(output) > 500:
            output = output[:500] + "..."
        print(f"   {output}")
        print(f"   [长度: {len(r['output'])}]")

# 7. 模板 vs API 质量对比
print(f"\n7. 模板 vs API 质量对比:")
api_records = [r for r in records if r["metadata"]["reflection_source"] == "api"]
tpl_records = [r for r in records if "template" in r["metadata"]["reflection_source"]]
if api_records and tpl_records:
    api_avg = sum(len(r["output"]) for r in api_records) / len(api_records)
    tpl_avg = sum(len(r["output"]) for r in tpl_records) / len(tpl_records)
    print(f"   API:      {len(api_records)} 条, 平均长度 {api_avg:.0f}")
    print(f"   Template: {len(tpl_records)} 条, 平均长度 {tpl_avg:.0f}")
    print(f"   长度比:   Template/API = {tpl_avg/api_avg:.2f}")
elif api_records:
    api_avg = sum(len(r["output"]) for r in api_records) / len(api_records)
    print(f"   API:      {len(api_records)} 条, 平均长度 {api_avg:.0f}")
    print(f"   Template: 0 条")
elif tpl_records:
    tpl_avg = sum(len(r["output"]) for r in tpl_records) / len(tpl_records)
    print(f"   API:      0 条")
    print(f"   Template: {len(tpl_records)} 条, 平均长度 {tpl_avg:.0f}")

# 8. 去重检查
outputs = [r["output"] for r in records]
unique_outputs = len(set(outputs))
print(f"\n8. 去重检查: {unique_outputs} 唯一 / {N} 总计 (重复率 {(N-unique_outputs)/N*100:.1f}%)")

# 9. 模板多样性（模板来源的前 80 字符去重）
if tpl_records:
    tpl_prefixes = set(r["output"][:80] for r in tpl_records)
    print(f"\n9. 模板多样性: {len(tpl_prefixes)} 种不同前缀 / {len(tpl_records)} 条模板记录")

# 10. 按错误类型 × 来源交叉统计
print(f"\n10. 错误类型 × 来源 交叉表:")
cross = Counter((r["metadata"]["error_type"], r["metadata"]["reflection_source"]) for r in records)
all_types = sorted(set(r["metadata"]["error_type"] for r in records))
all_sources = sorted(set(r["metadata"]["reflection_source"] for r in records))
header = f"   {'type':<20}" + "".join(f"{s:<22}" for s in all_sources) + "total"
print(header)
for et in all_types:
    row = f"   {et:<20}"
    row_total = 0
    for src in all_sources:
        c = cross.get((et, src), 0)
        row += f"{c:<22}"
        row_total += c
    row += str(row_total)
    print(row)

print("\n" + "=" * 65)
