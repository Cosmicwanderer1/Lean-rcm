"""分析 thought 字段的重复情况和内容模式 @author ygw 2026-03-02"""
import json
from collections import Counter, defaultdict
from pathlib import Path

FP = Path(r"D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl")
records = [json.loads(l) for l in FP.read_text(encoding="utf-8").strip().splitlines()]
N = len(records)

print(f"总记录: {N}")

# 1. thought 唯一性统计
thoughts = [r["thought"] for r in records]
unique_thoughts = len(set(thoughts))
print(f"\n1. thought 唯一性: {unique_thoughts} 唯一 / {N} 总计 ({unique_thoughts/N*100:.1f}%)")

# 2. 同定理+同步骤 的 thought 重复率
by_thm_step = defaultdict(list)
for r in records:
    key = (r["theorem_name"], r.get("step_index", 0))
    by_thm_step[key].append(r)

groups_with_multi = {k: v for k, v in by_thm_step.items() if len(v) > 1}
print(f"\n2. 同定理+步骤有多条记录的组: {len(groups_with_multi)} 组")

# 检查这些组内 thought 是否完全一致
thought_same_count = 0
thought_diff_count = 0
for key, recs in groups_with_multi.items():
    unique_t = len(set(r["thought"] for r in recs))
    if unique_t == 1:
        thought_same_count += 1
    else:
        thought_diff_count += 1

print(f"   thought 完全一致: {thought_same_count} 组 ({thought_same_count/(thought_same_count+thought_diff_count)*100:.1f}%)")
print(f"   thought 有差异:  {thought_diff_count} 组")

# 3. 展示同定理同步骤不同错误类型的 thought 对比（取3个例子）
print(f"\n3. 同定理不同错误类型的 thought 对比:")
shown = 0
for key, recs in groups_with_multi.items():
    types_present = set(r["error_type"] for r in recs)
    if len(types_present) >= 3 and shown < 3:
        shown += 1
        thm, step = key
        print(f"\n   ═══ {thm} (step={step}, {len(recs)}条) ═══")
        for r in recs[:4]:
            print(f"   [{r['error_type']}] thought({len(r['thought'])}字):")
            print(f"     {r['thought'][:150]}...")
            print(f"   [{r['error_type']}] reflection前80字: {r['reflection'][:80]}...")
            print()

# 4. thought 长度统计
thought_lens = [len(r["thought"]) for r in records]
print(f"\n4. thought 长度统计:")
print(f"   avg={sum(thought_lens)/N:.0f}, min={min(thought_lens)}, max={max(thought_lens)}, "
      f"med={sorted(thought_lens)[N//2]}")

# 5. thought 是否已经包含与错误类型相关的描述
error_type_keywords = {
    "tactic_typo": ["typo", "spelling", "拼写", "spell", "syntactic", "identifier name"],
    "wrong_tactic": ["wrong", "incorrect", "不正确", "错误策略", "strategic", "fundamentally different"],
    "missing_step": ["missing", "placeholder", "占位", "done", "missing reasoning step", "bridge the gap"],
    "argument_error": ["argument", "parameter", "参数", "type-check", "parameters need adjustment", "aligning argument"],
}
print(f"\n5. thought 中是否已包含错误类型关键词:")
for et, keywords in error_type_keywords.items():
    sub = [r for r in records if r["error_type"] == et]
    has_kw = sum(1 for r in sub if any(kw.lower() in r["thought"].lower() for kw in keywords))
    print(f"   {et}: {has_kw}/{len(sub)} ({has_kw/len(sub)*100:.1f}%) thought 包含相关关键词")

# 6. thought 来源分析：是哪个阶段生成的？
# 检查 thought 内容模式
print(f"\n6. thought 内容模式（前5条不同的 thought 摘要）:")
seen = set()
count = 0
for r in records:
    t = r["thought"]
    if t not in seen and count < 5:
        seen.add(t)
        count += 1
        print(f"   [{r['error_type']}] {r['theorem_name']}")
        print(f"     {t[:200]}...")
        print()
