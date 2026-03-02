"""检查 CoS 数据集样本质量 @author ygw"""
import json
import os

cos_path = os.path.join("data", "processed", "cos_dataset", "cos_dataset.jsonl")
stats_path = os.path.join("data", "processed", "cos_dataset", "cos_stats.json")
flat_path = os.path.join("data", "processed", "cos_dataset", "cos_flat.jsonl")

# 1. 读取统计信息
if os.path.exists(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    print("=== cos_stats.json ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()

# 2. 统计 cos_dataset
total = 0
real_states = 0
placeholder_states = 0
has_trace_true = 0
with open(cos_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        chain = record.get("full_cos_chain", [])
        meta = record.get("metadata", {})
        if meta.get("has_leandojo_trace"):
            has_trace_true += 1
        if chain and not chain[0].get("state_before", "").startswith("[state_"):
            real_states += 1
        else:
            placeholder_states += 1

print("=== cos_dataset.jsonl ===")
print(f"total records: {total}")
print(f"real states (LeanDojo): {real_states}")
print(f"placeholder states: {placeholder_states}")
print(f"has_leandojo_trace=true: {has_trace_true}")
if total > 0:
    print(f"real states ratio: {real_states/total*100:.1f}%")
print()

# 3. 统计 cos_flat
flat_total = 0
if os.path.exists(flat_path):
    with open(flat_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                flat_total += 1
print(f"=== cos_flat.jsonl ===")
print(f"total flat records: {flat_total}")
print()

# 4. 输出一条有真实状态的样本
print("=== SAMPLE (real states) ===")
with open(cos_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        chain = record.get("full_cos_chain", [])
        if chain and not chain[0].get("state_before", "").startswith("[state_"):
            # 只打印前2个 chain step 避免过长
            sample = {
                "theorem_name": record.get("theorem_name"),
                "theorem_full_name": record.get("theorem_full_name"),
                "proof_steps": record.get("proof_steps"),
                "key_steps": record.get("key_steps"),
                "compression_ratio": record.get("metadata", {}).get("compression_ratio"),
                "full_cos_chain_first_2": chain[:2],
                "key_cos_chain_count": len(record.get("key_cos_chain", [])),
            }
            for k, v in sample.items():
                print(f"  {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}")
            break
