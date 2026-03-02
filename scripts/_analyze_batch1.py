#!/usr/bin/env python3
"""
批次1 reflection 数据全面质量分析
适配扁平字段结构: reflection, reflection_source, original_tactic, error_tactic, error_type, etc.
@author ygw
2026-03-02
"""
import json, sys, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path
import statistics

# ─── 配置 ───────────────────────────────────────────────
FP = Path(r"D:\RTAP\data\processed\error_correction\reflection_batch_1.jsonl")
# ─────────────────────────────────────────────────────────

print("正在加载数据...")
records = [json.loads(l) for l in FP.read_text(encoding="utf-8").strip().splitlines()]
N = len(records)

print("=" * 72)
print(f"  Reflection Batch-1 质量分析报告 — 共 {N:,} 条")
print(f"  文件: {FP.name}  大小: {FP.stat().st_size / 1024 / 1024:.1f} MB")
print("=" * 72)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 来源分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
src_dist = Counter(r["reflection_source"] for r in records)
print(f"\n{'━'*72}")
print("1. 来源分布")
print(f"{'━'*72}")
for src, cnt in src_dist.most_common():
    bar = "█" * int(cnt / N * 50)
    print(f"   {src:<25} {cnt:>6} ({cnt/N*100:5.1f}%) {bar}")

api_count = sum(v for k, v in src_dist.items() if k == "api")
tpl_count = sum(v for k, v in src_dist.items() if "template" in k)
print(f"   ── API 成功率: {api_count}/{N} = {api_count/N*100:.2f}%")
print(f"   ── 模板回退率: {tpl_count}/{N} = {tpl_count/N*100:.2f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 错误类型分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
type_dist = Counter(r["error_type"] for r in records)
print(f"\n{'━'*72}")
print("2. 错误类型分布")
print(f"{'━'*72}")
for et, cnt in type_dist.most_common():
    bar = "█" * int(cnt / N * 50)
    print(f"   {et:<25} {cnt:>6} ({cnt/N*100:5.1f}%) {bar}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Reflection 长度统计 (字符)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lengths = [len(r["reflection"]) for r in records]
sorted_lengths = sorted(lengths)
print(f"\n{'━'*72}")
print("3. Reflection 长度统计 (字符)")
print(f"{'━'*72}")
print(f"   最短:   {min(lengths):>8}")
print(f"   最长:   {max(lengths):>8}")
print(f"   平均:   {statistics.mean(lengths):>8.0f}")
print(f"   中位数: {statistics.median(lengths):>8.0f}")
print(f"   标准差: {statistics.stdev(lengths):>8.0f}")
print(f"   P10:    {sorted_lengths[int(N*0.1)]:>8}")
print(f"   P25:    {sorted_lengths[int(N*0.25)]:>8}")
print(f"   P75:    {sorted_lengths[int(N*0.75)]:>8}")
print(f"   P90:    {sorted_lengths[int(N*0.9)]:>8}")
print(f"   P99:    {sorted_lengths[int(N*0.99)]:>8}")

# 长度分布直方图（文本版）
print(f"\n   长度分布直方图:")
bins = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 5000, 10000]
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    cnt = sum(1 for l in lengths if lo <= l < hi)
    bar = "█" * int(cnt / N * 80)
    print(f"   [{lo:>5}-{hi:>5}) {cnt:>6} ({cnt/N*100:5.1f}%) {bar}")
over = sum(1 for l in lengths if l >= bins[-1])
if over:
    print(f"   [{bins[-1]:>5}+)     {over:>6} ({over/N*100:5.1f}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 按来源的长度对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("4. 按来源的长度对比")
print(f"{'━'*72}")
api_lengths = [len(r["reflection"]) for r in records if r["reflection_source"] == "api"]
tpl_lengths = [len(r["reflection"]) for r in records if "template" in r["reflection_source"]]

if api_lengths:
    print(f"   API ({len(api_lengths):,}条):")
    print(f"     avg={statistics.mean(api_lengths):.0f}, med={statistics.median(api_lengths):.0f}, "
          f"min={min(api_lengths)}, max={max(api_lengths)}, std={statistics.stdev(api_lengths):.0f}")
if tpl_lengths:
    print(f"   Template ({len(tpl_lengths):,}条):")
    print(f"     avg={statistics.mean(tpl_lengths):.0f}, med={statistics.median(tpl_lengths):.0f}, "
          f"min={min(tpl_lengths)}, max={max(tpl_lengths)}, std={statistics.stdev(tpl_lengths):.0f}")
if api_lengths and tpl_lengths:
    ratio = statistics.mean(tpl_lengths) / statistics.mean(api_lengths)
    print(f"   长度比 (Template/API): {ratio:.3f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 核心质量指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("5. 核心质量指标")
print(f"{'━'*72}")

# 5a. 正确策略引用 (original_tactic)
has_correct = 0
for r in records:
    ct = r.get("original_tactic", "")
    if ct and len(ct) >= 10 and ct[:20] in r["reflection"]:
        has_correct += 1
    elif ct and 0 < len(ct) < 10 and ct in r["reflection"]:
        has_correct += 1
pct_correct = has_correct / N * 100
print(f"   ✦ 正确策略引用 (original_tactic): {has_correct:>6}/{N} ({pct_correct:5.1f}%)")

# 5b. 错误策略引用 (error_tactic)
has_error_tactic = 0
for r in records:
    et = r.get("error_tactic", "")
    if et and len(et) >= 10 and et[:15] in r["reflection"]:
        has_error_tactic += 1
    elif et and 0 < len(et) < 10 and et in r["reflection"]:
        has_error_tactic += 1
pct_error = has_error_tactic / N * 100
print(f"   ✦ 错误策略引用 (error_tactic):    {has_error_tactic:>6}/{N} ({pct_error:5.1f}%)")

# 5c. repair_hint 引用
has_hint = 0
for r in records:
    hint = r.get("repair_hint", "")
    if hint and len(hint) >= 15 and hint[:25] in r["reflection"]:
        has_hint += 1
    elif hint and 0 < len(hint) < 15 and hint in r["reflection"]:
        has_hint += 1
pct_hint = has_hint / N * 100
print(f"   ✦ 修复提示引用 (repair_hint):     {has_hint:>6}/{N} ({pct_hint:5.1f}%)")

# 5d. 禁止短语检查
forbidden_patterns = [
    "I don't know", "I'm not sure", "I cannot determine",
    "As an AI", "I apologize", "I'm sorry", "I can't",
    "不确定", "无法判断", "抱歉"
]
has_forbidden = 0
forbidden_detail = Counter()
for r in records:
    for fp in forbidden_patterns:
        if fp.lower() in r["reflection"].lower():
            has_forbidden += 1
            forbidden_detail[fp] += 1
            break  # 每条只计一次
print(f"   ✦ 含禁止短语:                     {has_forbidden:>6}/{N} ({has_forbidden/N*100:5.1f}%)")
if forbidden_detail:
    for fp, cnt in forbidden_detail.most_common(5):
        print(f"     └─ \"{fp}\": {cnt}次")

# 5e. error_message 引用
has_error_msg = 0
for r in records:
    em = r.get("error_message", "")
    if em and len(em) >= 20:
        # 取 error_message 中的关键片段（跳过 Message(data=' 前缀）
        clean = em
        if "data='" in clean:
            start = clean.find("data='") + 6
            end = clean.find("'", start)
            if end > start:
                clean = clean[start:end]
        snippet = clean[:40]
        if len(snippet) >= 10 and snippet[:20] in r["reflection"]:
            has_error_msg += 1
pct_emsg = has_error_msg / N * 100
print(f"   ✦ 错误消息引用 (error_message):   {has_error_msg:>6}/{N} ({pct_emsg:5.1f}%)")

# 5f. thought 引用（检查 reflection 是否整合了 thought 的分析）
has_thought = 0
for r in records:
    th = r.get("thought", "")
    if th and len(th) >= 20 and th[:30] in r["reflection"]:
        has_thought += 1
pct_thought = has_thought / N * 100
print(f"   ✦ thought 整合:                   {has_thought:>6}/{N} ({pct_thought:5.1f}%)")

# 5g. 空输出 / 极短输出检查
empty_output = sum(1 for r in records if len(r["reflection"].strip()) == 0)
short_output = sum(1 for r in records if 0 < len(r["reflection"].strip()) < 100)
print(f"   ✦ 空输出:                         {empty_output:>6}/{N}")
print(f"   ✦ 极短(<100字符):                 {short_output:>6}/{N}")

# 5h. 超长输出检查
very_long = sum(1 for r in records if len(r["reflection"]) > 5000)
print(f"   ✦ 超长(>5000字符):                {very_long:>6}/{N}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 按来源分别统计质量指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("6. 按来源分别统计质量指标")
print(f"{'━'*72}")
for src_name in sorted(src_dist.keys()):
    sub = [r for r in records if r["reflection_source"] == src_name]
    sn = len(sub)
    if sn == 0:
        continue
    
    sc_correct = 0
    for r in sub:
        ct = r.get("original_tactic", "")
        if ct and len(ct) >= 10 and ct[:20] in r["reflection"]:
            sc_correct += 1
        elif ct and 0 < len(ct) < 10 and ct in r["reflection"]:
            sc_correct += 1
    
    sc_hint = 0
    for r in sub:
        hint = r.get("repair_hint", "")
        if hint and len(hint) >= 15 and hint[:25] in r["reflection"]:
            sc_hint += 1
        elif hint and 0 < len(hint) < 15 and hint in r["reflection"]:
            sc_hint += 1
    
    sc_forbid = sum(1 for r in sub if any(fp.lower() in r["reflection"].lower() for fp in forbidden_patterns))
    
    sc_error_t = 0
    for r in sub:
        et = r.get("error_tactic", "")
        if et and len(et) >= 10 and et[:15] in r["reflection"]:
            sc_error_t += 1
        elif et and 0 < len(et) < 10 and et in r["reflection"]:
            sc_error_t += 1
    
    sub_lengths = [len(r["reflection"]) for r in sub]
    avg_len = statistics.mean(sub_lengths)
    
    print(f"   [{src_name}] ({sn:,}条, 平均长度{avg_len:.0f})")
    print(f"     正确策略: {sc_correct/sn*100:.1f}% | 错误策略: {sc_error_t/sn*100:.1f}% | "
          f"修复提示: {sc_hint/sn*100:.1f}% | 禁止短语: {sc_forbid/sn*100:.1f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 去重分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("7. 去重分析")
print(f"{'━'*72}")

# 7a. 完全重复 (reflection 完全相同)
output_hashes = [hashlib.md5(r["reflection"].encode()).hexdigest() for r in records]
unique_outputs = len(set(output_hashes))
dup_count = N - unique_outputs
print(f"   完全重复 (reflection): {dup_count}/{N} ({dup_count/N*100:.2f}%)")

# 7b. 近似重复 (前200字符相同)
prefix_hashes = set()
near_dup = 0
for r in records:
    prefix = r["reflection"][:200]
    h = hashlib.md5(prefix.encode()).hexdigest()
    if h in prefix_hashes:
        near_dup += 1
    prefix_hashes.add(h)
print(f"   近似重复 (前200字符): {near_dup}/{N} ({near_dup/N*100:.2f}%)")

# 7c. 同定理重复
theorem_counts = Counter(r["theorem_name"] for r in records)
dup_theorems = sum(1 for k, v in theorem_counts.items() if v > 1)
total_with_dup = sum(v for k, v in theorem_counts.items() if v > 1)
print(f"   同定理名多条: {dup_theorems} 个定理有重复条目 (涉及 {total_with_dup} 条)")
if dup_theorems > 0:
    top_dups = theorem_counts.most_common(5)
    for thm, cnt in top_dups:
        if cnt > 1:
            print(f"     └─ {thm}: {cnt}条")

# 7d. 相同 (theorem_name + step_index + error_type) 组合
combo_counts = Counter(
    (r["theorem_name"], r.get("step_index", 0), r["error_type"]) for r in records
)
unique_combos = len(combo_counts)
dup_combos = sum(1 for k, v in combo_counts.items() if v > 1)
print(f"   唯一 (theorem+step+type) 组合: {unique_combos}/{N}")
print(f"   重复组合: {dup_combos}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 模板多样性分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tpl_records = [r for r in records if "template" in r["reflection_source"]]
if tpl_records:
    print(f"\n{'━'*72}")
    print(f"8. 模板多样性分析 ({len(tpl_records):,} 条模板)")
    print(f"{'━'*72}")
    
    # 前80字符变体
    tpl_prefixes = Counter(r["reflection"][:80] for r in tpl_records)
    print(f"   前80字符变体数: {len(tpl_prefixes)}")
    print(f"   最常见前缀 TOP-5:")
    for prefix, cnt in tpl_prefixes.most_common(5):
        print(f"     [{cnt:>4}次] {prefix[:60]}...")
    
    # 按错误类型分布
    tpl_type_dist = Counter(r["error_type"] for r in tpl_records)
    print(f"\n   模板按错误类型分布:")
    for et, cnt in tpl_type_dist.most_common():
        pct_of_type = cnt / type_dist[et] * 100 if type_dist[et] > 0 else 0
        print(f"     {et:<20} {cnt:>5}条 (占该类型 {pct_of_type:.1f}%)")
else:
    print(f"\n{'━'*72}")
    print("8. 模板多样性分析: 无模板记录")
    print(f"{'━'*72}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. 错误类型 × 来源 交叉表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("9. 错误类型 × 来源 交叉表")
print(f"{'━'*72}")
cross = Counter((r["error_type"], r["reflection_source"]) for r in records)
all_types = sorted(set(r["error_type"] for r in records))
all_sources = sorted(set(r["reflection_source"] for r in records))
header = f"   {'type':<20}" + "".join(f"{s:<22}" for s in all_sources) + "  total"
print(header)
print("   " + "─" * (20 + 22 * len(all_sources) + 7))
for et in all_types:
    row = f"   {et:<20}"
    row_total = 0
    for src in all_sources:
        c = cross.get((et, src), 0)
        row += f"{c:<22}"
        row_total += c
    row += f"  {row_total}"
    print(row)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. 字段完整性检查
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("10. 字段完整性检查")
print(f"{'━'*72}")
required_fields = ["theorem_name", "error_type", "original_tactic", "error_tactic",
                    "repair_hint", "reflection_source", "reflection", "state_before",
                    "verification_status"]
optional_fields = ["error_message", "thought", "theorem_full_name", "step_index", "source"]
for field in required_fields:
    missing = sum(1 for r in records if field not in r or not r[field])
    status = "✓" if missing == 0 else "✗"
    print(f"   {status} {field:<25} 缺失: {missing}/{N}")

for field in optional_fields:
    present = sum(1 for r in records if field in r and r[field])
    print(f"   ○ {field:<25} 存在: {present}/{N} ({present/N*100:.1f}%)")

# verification_status 分布
vs_dist = Counter(r.get("verification_status", "unknown") for r in records)
print(f"\n   verification_status 分布:")
for vs, cnt in vs_dist.most_common():
    print(f"     {vs}: {cnt} ({cnt/N*100:.1f}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. 按错误类型的质量详细分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("11. 按错误类型的质量详细分析")
print(f"{'━'*72}")
for et_name in sorted(type_dist.keys()):
    sub = [r for r in records if r["error_type"] == et_name]
    sn = len(sub)
    sub_api = sum(1 for r in sub if r["reflection_source"] == "api")
    sub_tpl = sn - sub_api
    sub_lens = [len(r["reflection"]) for r in sub]
    
    sc_correct = sum(1 for r in sub 
                     if r.get("original_tactic","") and 
                     r["original_tactic"][:20] in r["reflection"])
    sc_error = sum(1 for r in sub
                   if r.get("error_tactic","") and
                   r["error_tactic"][:15] in r["reflection"])
    
    print(f"   [{et_name}] {sn:,}条 (API:{sub_api} TPL:{sub_tpl})")
    print(f"     长度 avg={statistics.mean(sub_lens):.0f} med={statistics.median(sub_lens):.0f} | "
          f"正确引用:{sc_correct/sn*100:.0f}% 错误引用:{sc_error/sn*100:.0f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. 抽样展示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("12. 抽样展示 (每个来源+错误类型组合各1条)")
print(f"{'━'*72}")
shown_combos = set()
sample_count = 0
for r in records:
    combo = (r["reflection_source"], r["error_type"])
    if combo not in shown_combos and sample_count < 10:
        shown_combos.add(combo)
        sample_count += 1
        src = r["reflection_source"]
        et = r["error_type"]
        thm = r["theorem_name"]
        output = r["reflection"]
        if len(output) > 400:
            output = output[:400] + "..."
        print(f"\n   ── [{src}|{et}] {thm} (长度:{len(r['reflection'])})")
        print(f"   {output}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. 综合评分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'━'*72}")
print("13. 综合质量评分")
print(f"{'━'*72}")

scores = {}
# API成功率 (权重 20%)  
scores["API成功率"] = (api_count / N * 100, 20)
# 正确策略引用 (权重 20%)
scores["正确策略引用"] = (pct_correct, 20)
# 错误策略引用 (权重 15%)
scores["错误策略引用"] = (pct_error, 15)
# 修复提示 (权重 15%)
scores["修复提示引用"] = (pct_hint, 15)
# 无禁止短语 (权重 10%)
scores["无禁止短语"] = (100 - has_forbidden / N * 100, 10)
# 去重率 (权重 10%)
scores["唯一性"] = (unique_outputs / N * 100, 10)
# 适当长度 (300-3000字符)
reasonable = sum(1 for l in lengths if 300 <= l <= 3000)
scores["合理长度"] = (reasonable / N * 100, 10)

total_weighted = sum(score * weight / 100 for score, weight in scores.values())
total_weight = sum(w for _, w in scores.values())

print(f"   {'指标':<18} {'得分':>8} {'权重':>6} {'加权':>8}")
print(f"   {'─'*44}")
for name, (score, weight) in scores.items():
    weighted = score * weight / 100
    print(f"   {name:<18} {score:>7.1f}% {weight:>5}% {weighted:>7.2f}")
print(f"   {'─'*44}")
print(f"   {'综合加权评分':<18} {'':>8} {total_weight:>5}% {total_weighted:>7.2f}")
final = total_weighted / total_weight * 100
print(f"\n   ★ 最终评分: {final:.1f} / 100")

# 评级
if final >= 90:
    grade = "A (优秀)"
elif final >= 80:
    grade = "B (良好)"
elif final >= 70:
    grade = "C (合格)"
elif final >= 60:
    grade = "D (需改善)"
else:
    grade = "F (不合格)"
print(f"   ★ 评级: {grade}")

# 与 100 条测试的对比
print(f"\n   📊 与 100 条测试对比:")
print(f"   {'指标':<18} {'100条测试':>10} {'批次1({N:,})':>12}")
print(f"   {'─'*42}")
test100 = {"API成功率": 92.0, "正确策略引用": 80.0, "错误策略引用": 82.0,
           "修复提示引用": 84.0, "禁止短语率": 0.0, "模板/API长度比": 0.93}
batch1 = {"API成功率": api_count/N*100, "正确策略引用": pct_correct,
          "错误策略引用": pct_error, "修复提示引用": pct_hint,
          "禁止短语率": has_forbidden/N*100}
if api_lengths and tpl_lengths:
    batch1["模板/API长度比"] = statistics.mean(tpl_lengths) / statistics.mean(api_lengths)
for k in test100:
    v100 = test100[k]
    v1 = batch1.get(k, 0)
    diff = v1 - v100
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"   {k:<18} {v100:>9.1f}% {v1:>11.1f}% {arrow}{abs(diff):.1f}")

print(f"\n{'='*72}")
print("分析完成。")
