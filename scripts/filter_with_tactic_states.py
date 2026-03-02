"""
过滤脚本：从 traces_filtered.jsonl 中只保留有 tactic_states 的定理
@author ygw

用法:
    python scripts/filter_with_tactic_states.py

输入: data/processed/traces/traces_filtered.jsonl
输出: data/processed/traces/traces_with_states.jsonl
"""

import json
import os
import sys

# 路径配置
INPUT_FILE = "data/processed/traces/traces_filtered.jsonl"
OUTPUT_FILE = "data/processed/traces/traces_with_states.jsonl"


def main():
    """
    从 traces_filtered.jsonl 中过滤出有 tactic_states 的记录

    过滤条件: tactic_states 列表非空
    """
    if not os.path.exists(INPUT_FILE):
        print(f"输入文件不存在: {INPUT_FILE}")
        sys.exit(1)

    total = 0
    kept = 0
    removed = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            # 只保留有 tactic_states 的记录
            if record.get("tactic_states"):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1

    print(f"过滤完成:")
    print(f"  输入: {total} 条")
    print(f"  保留: {kept} 条 (有 tactic_states)")
    print(f"  移除: {removed} 条 (无 tactic_states)")
    print(f"  输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
