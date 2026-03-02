"""
@author ygw
反思数据质量分析脚本 - v2 优化后评估
"""
import json
import statistics

JSONL_PATH = "data/processed/error_correction/reflection_dataset.jsonl"

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"总记录数: {len(records)}")

# ── 1. 生成来源分布 ───────────────────────────────────────────────────────────
sources = {}
for r in records:
    src = r["metadata"].get("reflection_source", "unknown")
    sources[src] = sources.get(src, 0) + 1
print(f"\n生成来源分布:")
for k, v in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(records)*100:.1f}%)")

# ── 2. 输出长度统计 ───────────────────────────────────────────────────────────
lengths = [len(r["output"]) for r in records]
print(f"\n输出长度统计:")
print(f"  最短: {min(lengths)} 字符")
print(f"  最长: {max(lengths)} 字符")
print(f"  均值: {statistics.mean(lengths):.0f} 字符")
print(f"  中位数: {statistics.median(lengths):.0f} 字符")
too_short = sum(1 for l in lengths if l < 100)
too_long = sum(1 for l in lengths if l > 800)
print(f"  过短(<100): {too_short}  过长(>800): {too_long}")

# ── 3. 正确战术出现率 ──────────────────────────────────────────────────────────
contain_tactic = 0
for r in records:
    correct = r["metadata"].get("correct_tactic", "")
    output = r["output"]
    if correct and correct in output:
        contain_tactic += 1
# 也统计 partial match（至少有首个非平凡 token）
partial_match = 0
for r in records:
    correct = r["metadata"].get("correct_tactic", "")
    output = r["output"]
    tokens = [t for t in correct.split() if len(t) > 3]
    if tokens and any(t in output for t in tokens[:2]):
        partial_match += 1
print(f"\n正确战术(exact)出现在输出中: {contain_tactic}/{len(records)} ({contain_tactic/len(records)*100:.1f}%)")
print(f"正确战术(partial)出现在输出中: {partial_match}/{len(records)} ({partial_match/len(records)*100:.1f}%)")

# ── 4. repair_hint 被使用率 ────────────────────────────────────────────────────
hint_present = 0
for r in records:
    hint = r["metadata"].get("repair_hint", "")
    output = r["output"]
    if hint and hint[:20] in output:
        hint_present += 1
print(f"\nrepair_hint 内容出现在输出中: {hint_present}/{len(records)} ({hint_present/len(records)*100:.1f}%)")

# ── 5. 错误类型分布 ───────────────────────────────────────────────────────────
error_types = {}
for r in records:
    et = r["metadata"].get("error_type", "unknown")
    error_types[et] = error_types.get(et, 0) + 1
print(f"\n错误类型分布:")
for k, v in sorted(error_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(records)*100:.1f}%)")

# ── 6. 输出中包含 ⊢ 符号（说明引用了证明目标）──────────────────────────────────
has_goal_ref = sum(1 for r in records if "⊢" in r["output"])
print(f"\n输出引用证明目标(⊢): {has_goal_ref}/{len(records)} ({has_goal_ref/len(records)*100:.1f}%)")

# ── 7. 链式推理质量：包含 "because" / "therefore" / "since" / "thus" ──────────
reasoning_words = ["because", "therefore", "since", "thus", "implies", "which means"]
reasoning_count = sum(1 for r in records if any(w in r["output"].lower() for w in reasoning_words))
print(f"输出包含推理连词: {reasoning_count}/{len(records)} ({reasoning_count/len(records)*100:.1f}%)")

# ── 8. 抽样展示(template_after_api 类型) ──────────────────────────────────────
print("\n── template_after_api 样本展示 ──")
tmpl_samples = [r for r in records if r["metadata"].get("reflection_source") == "template_after_api"]
if tmpl_samples:
    sample = tmpl_samples[0]
    print(f"  error_type: {sample['metadata']['error_type']}")
    print(f"  correct_tactic: {sample['metadata']['correct_tactic'][:60]}")
    print(f"  输出长度: {len(sample['output'])} 字符")
    print(f"  输出前200字符:\n    {sample['output'][:200]}")

print("\n── API 样本展示 ──")
api_samples = [r for r in records if r["metadata"].get("reflection_source") == "api"]
if api_samples:
    sample = api_samples[0]
    print(f"  error_type: {sample['metadata']['error_type']}")
    print(f"  correct_tactic: {sample['metadata']['correct_tactic'][:60]}")
    print(f"  输出长度: {len(sample['output'])} 字符")
    print(f"  输出前200字符:\n    {sample['output'][:200]}")
