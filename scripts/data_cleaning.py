"""
数据清洗脚本 - 剔除不合格样本，验证 CoS 连通性
@author ygw
更新日期: 2026-02-12

清洗规则:
1. 去重: 同一定理同一 step_index 的重复记录，保留首条
2. step_index 完整性: 剔除 step_index 不从 0 开始的定理的所有步骤
3. step_index 连续性: 剔除 step_index 有间隔的定理的所有步骤
4. 空字段: 剔除 thought/tactic/state_before/state_after 为空的记录
5. sorry tactic: 剔除 tactic 为 sorry 的记录

用法:
    python scripts/data_cleaning.py --input data/processed/cos_dataset/thought_dataset.jsonl
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    数据清洗器

    对 thought_dataset.jsonl 执行多轮清洗，输出清洗后的数据集和统计报告。
    @author ygw
    """

    def __init__(self, input_path: str, output_dir: str = None):
        """
        初始化清洗器

        参数:
            input_path: thought_dataset.jsonl 路径
            output_dir: 输出目录，默认与输入文件同目录
        """
        self.input_path = input_path
        self.output_dir = output_dir or os.path.dirname(input_path)
        self.stats = {
            "total_input": 0,
            "removed_duplicate": 0,
            "removed_incomplete_theorem": 0,
            "removed_gap_theorem": 0,
            "removed_empty_field": 0,
            "removed_sorry": 0,
            "total_output": 0,
            "theorems_input": 0,
            "theorems_output": 0,
        }

    def run(self) -> str:
        """
        执行完整清洗流程

        返回:
            str: 输出文件路径
        """
        logger.info("=" * 60)
        logger.info("数据清洗开始")
        logger.info(f"输入: {self.input_path}")
        logger.info("=" * 60)

        # 1. 加载数据，按定理分组
        records, theorem_map = self._load_and_group()
        self.stats["total_input"] = len(records)
        self.stats["theorems_input"] = len(theorem_map)
        logger.info(f"加载: {len(records)} 条记录, {len(theorem_map)} 个定理")

        # 2. 去重
        records, theorem_map = self._deduplicate(records, theorem_map)

        # 3. 剔除 step_index 不完整的定理
        bad_theorems = self._find_bad_theorems(theorem_map)
        records = self._remove_theorems(records, bad_theorems)
        # 重建 theorem_map
        theorem_map = self._build_theorem_map(records)

        # 4. 剔除空字段和 sorry
        records = self._remove_bad_records(records)
        theorem_map = self._build_theorem_map(records)

        # 5. 输出
        self.stats["total_output"] = len(records)
        self.stats["theorems_output"] = len(theorem_map)

        output_file = os.path.join(self.output_dir, "thought_dataset_cleaned.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # 6. 统计报告
        self._print_report()

        # 7. 保存统计
        stats_file = os.path.join(self.output_dir, "cleaning_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        logger.info(f"清洗后数据: {output_file}")
        logger.info(f"统计报告: {stats_file}")

        return output_file

    def _load_and_group(self) -> Tuple[List[Dict], Dict[str, Dict[int, Dict]]]:
        """
        加载数据并按定理分组

        返回:
            (记录列表, {theorem_full_name: {step_index: record}})
        """
        records = []
        theorem_map = defaultdict(dict)

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                thm = rec.get("theorem_full_name", "")
                si = rec.get("step_index", -1)
                theorem_map[thm][si] = rec

        return records, dict(theorem_map)

    def _deduplicate(self, records: List[Dict],
                     theorem_map: Dict) -> Tuple[List[Dict], Dict]:
        """
        去重: 同一定理同一 step_index 保留首条

        返回:
            去重后的 (records, theorem_map)
        """
        seen = set()
        deduped = []
        removed = 0

        for rec in records:
            key = (rec.get("theorem_full_name", ""), rec.get("step_index", -1))
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            deduped.append(rec)

        self.stats["removed_duplicate"] = removed
        if removed > 0:
            logger.info(f"去重: 移除 {removed} 条重复记录")

        new_map = self._build_theorem_map(deduped)
        return deduped, new_map

    def _find_bad_theorems(self, theorem_map: Dict) -> Set[str]:
        """
        找出 step_index 不完整或有间隔的定理

        返回:
            需要剔除的定理名称集合
        """
        bad = set()
        incomplete_count = 0
        gap_count = 0

        for thm, steps in theorem_map.items():
            indices = sorted(steps.keys())

            # step_index 不从 0 开始
            if indices[0] != 0:
                bad.add(thm)
                incomplete_count += 1
                continue

            # step_index 有间隔
            has_gap = False
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] > 1:
                    has_gap = True
                    break
            if has_gap:
                bad.add(thm)
                gap_count += 1

        self.stats["removed_incomplete_theorem"] = incomplete_count
        self.stats["removed_gap_theorem"] = gap_count

        if bad:
            # 计算受影响的记录数
            affected = sum(len(theorem_map[t]) for t in bad)
            logger.info(f"定理完整性: 剔除 {len(bad)} 个定理 "
                        f"({incomplete_count} 不从0开始, {gap_count} 有间隔), "
                        f"影响 {affected} 条记录")

        return bad

    def _remove_theorems(self, records: List[Dict],
                         bad_theorems: Set[str]) -> List[Dict]:
        """
        移除指定定理的所有记录

        返回:
            过滤后的记录列表
        """
        if not bad_theorems:
            return records
        return [r for r in records
                if r.get("theorem_full_name", "") not in bad_theorems]

    def _remove_bad_records(self, records: List[Dict]) -> List[Dict]:
        """
        移除空字段和 sorry tactic 的记录

        返回:
            过滤后的记录列表
        """
        clean = []
        empty_count = 0
        sorry_count = 0

        for rec in records:
            thought = rec.get("thought", "")
            tactic = rec.get("tactic", "")
            sb = rec.get("state_before", "")
            sa = rec.get("state_after", "")

            # 空字段检查
            if not thought or not tactic or not sb or not sa:
                empty_count += 1
                continue

            # sorry 检查
            if tactic.strip() == "sorry":
                sorry_count += 1
                continue

            clean.append(rec)

        self.stats["removed_empty_field"] = empty_count
        self.stats["removed_sorry"] = sorry_count

        if empty_count > 0:
            logger.info(f"空字段: 移除 {empty_count} 条")
        if sorry_count > 0:
            logger.info(f"sorry tactic: 移除 {sorry_count} 条")

        return clean

    def _build_theorem_map(self, records: List[Dict]) -> Dict[str, Dict[int, Dict]]:
        """
        从记录列表重建定理映射

        返回:
            {theorem_full_name: {step_index: record}}
        """
        theorem_map = defaultdict(dict)
        for rec in records:
            thm = rec.get("theorem_full_name", "")
            si = rec.get("step_index", -1)
            theorem_map[thm][si] = rec
        return dict(theorem_map)

    def _print_report(self):
        """打印清洗统计报告"""
        s = self.stats
        total_removed = (s["total_input"] - s["total_output"])

        logger.info("=" * 60)
        logger.info("数据清洗完成")
        logger.info(f"输入: {s['total_input']} 条 ({s['theorems_input']} 个定理)")
        logger.info(f"输出: {s['total_output']} 条 ({s['theorems_output']} 个定理)")
        logger.info(f"移除: {total_removed} 条 ({total_removed/s['total_input']*100:.2f}%)")
        logger.info(f"  - 重复记录: {s['removed_duplicate']}")
        logger.info(f"  - step不从0开始的定理: {s['removed_incomplete_theorem']} 个定理")
        logger.info(f"  - step有间隔的定理: {s['removed_gap_theorem']} 个定理")
        logger.info(f"  - 空字段: {s['removed_empty_field']}")
        logger.info(f"  - sorry tactic: {s['removed_sorry']}")
        logger.info("=" * 60)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="数据清洗")
    parser.add_argument("--input", type=str,
                        default="data/processed/cos_dataset/thought_dataset.jsonl",
                        help="输入文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认与输入同目录）")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    cleaner = DataCleaner(args.input, args.output_dir)
    output = cleaner.run()
    print(f"\n清洗完成！输出: {output}")


if __name__ == "__main__":
    main()
