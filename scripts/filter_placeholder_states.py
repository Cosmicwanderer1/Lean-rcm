"""过滤掉占位符状态的 cos_flat 记录，只保留有真实 Lean4 状态的数据 @author ygw"""
import json
import os

input_path = os.path.join("data", "processed", "cos_dataset", "cos_flat.jsonl")
output_path = os.path.join("data", "processed", "cos_dataset", "cos_flat_real.jsonl")

total = 0
kept = 0
removed = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        total += 1
        rec = json.loads(line)
        sb = rec.get("state_before", "")
        # 占位符格式: [state_0], [state_1], ...
        if sb.startswith("[state_"):
            removed += 1
            continue
        kept += 1
        fout.write(line)

print(f"Total: {total}")
print(f"Kept (real states): {kept} ({kept/total*100:.1f}%)")
print(f"Removed (placeholder): {removed} ({removed/total*100:.1f}%)")
print(f"Output: {output_path}")

# 备份原文件，用过滤后的替换
backup_path = input_path + ".bak"
os.rename(input_path, backup_path)
os.rename(output_path, input_path)
print(f"Backup: {backup_path}")
print(f"Replaced: {input_path}")
