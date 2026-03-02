"""临时脚本：检查数据文件字段结构"""
import json

def inspect(path, label, n=2):
    print(f"\n{'='*60}")
    print(f"{label}: {path}")
    print('='*60)
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(f"\n--- record {i} ---")
            print(f"Keys: {list(obj.keys())}")
            for k, v in obj.items():
                if isinstance(v, str):
                    print(f"  {k}: {repr(v[:150])}")
                elif isinstance(v, list):
                    print(f"  {k}: list len={len(v)}")
                    if v and isinstance(v[0], dict):
                        print(f"    [0] keys: {list(v[0].keys())}")
                        for sk, sv in list(v[0].items())[:6]:
                            print(f"      {sk}: {repr(str(sv)[:120])}")
                    elif v:
                        print(f"    [0]: {repr(str(v[0])[:120])}")
                else:
                    print(f"  {k}: {repr(v)}")

inspect("data/processed/traces/traces_with_states.jsonl", "traces_with_states", 1)
inspect("data/processed/cos_dataset/cos_flat.jsonl", "cos_flat", 2)
inspect("data/processed/cos_dataset/thought_dataset_cleaned.jsonl", "thought_cleaned", 1)
inspect("data/processed/error_correction/error_injection_verified.jsonl", "error_verified", 1)
