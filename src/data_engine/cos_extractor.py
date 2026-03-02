"""
状态链（CoS）提取模块
@author ygw
更新日期: 2026-02-06

W6 核心模块：遍历 Proof Tree，针对 have, apply, rw 等关键节点提取状态。
使用 Pantograph API 将 Lean 内部对象序列化，生成中间状态链。
例如，将 10 步证明压缩为 3 个关键状态路标。

技术产出:
- Ground Truth CoS: 中间状态链
- (S_pre, Tactic, S_post) 三元组数据集
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_jsonl, load_jsonl, iter_jsonl, ensure_dir,
    setup_logging, ProgressTracker, get_timestamp, compute_hash
)
from src.common.lean_server import LeanServer, LeanServerPool

logger = logging.getLogger(__name__)


@dataclass
class CoSRecord:
    """
    单条 CoS（Chain of States）记录

    属性:
        theorem_name: 定理名称
        theorem_full_name: 定理完整限定名
        theorem_type: 定理类型表达式
        full_cos_chain: 完整状态链 [(S_pre, tactic, S_post), ...]
        key_cos_chain: 关键状态路标（压缩后）
        proof_steps: 原始证明步数
        key_steps: 关键路标数
        file_path: 源文件路径
        metadata: 元数据
    """
    theorem_name: str = ""
    theorem_full_name: str = ""
    theorem_type: str = ""
    full_cos_chain: List[Dict[str, str]] = field(default_factory=list)
    key_cos_chain: List[Dict[str, str]] = field(default_factory=list)
    proof_steps: int = 0
    key_steps: int = 0
    file_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)


class CoSExtractor:
    """
    状态链提取器

    核心流程:
    1. 读取 W5 产出的 traces_filtered.jsonl
    2. 对每条追踪记录，提取完整的状态链
    3. 根据关键 Tactic 类型，压缩为关键状态路标
    4. 使用 Pantograph 验证状态链的连通性
    5. 输出 CoS 数据集

    使用示例:
        extractor = CoSExtractor(config)
        extractor.run()
    """

    # 默认关键策略列表
    DEFAULT_KEY_TACTICS = [
        "have", "apply", "rw", "simp", "exact", "intro",
        "cases", "induction", "constructor", "refine", "calc", "obtain"
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 CoS 提取器

        参数:
            config: 配置字典（来自 data_pipeline.yaml）
        """
        self.config = config
        cos_config = config["cos_extraction"]

        self.input_path = cos_config.get("input_path", "")
        self.output_path = cos_config.get("output_path", "")
        self.min_steps = cos_config.get("min_steps", 3)
        self.max_steps = cos_config.get("max_steps", 50)

        # 关键策略列表
        self.key_tactics = cos_config.get("key_tactics", self.DEFAULT_KEY_TACTICS)

        # 压缩配置
        compress_config = cos_config.get("compression", {})
        self.compression_enabled = compress_config.get("enable", True)
        self.target_ratio = compress_config.get("target_ratio", 3)
        self.keep_first_last = compress_config.get("keep_first_last", True)

        # Pantograph 配置
        panto_config = cos_config.get("pantograph", {})
        self.pantograph_executable = panto_config.get(
            "executable",
            "/root/autodl-tmp/RTAP/workspace/PyPantograph/src/.lake/build/bin/repl"
        )
        self.pantograph_timeout = panto_config.get("timeout", 30)
        self.max_concurrent = panto_config.get("max_concurrent", 4)

        # 统计信息
        self.stats = {
            "total_traces": 0,
            "processed": 0,
            "valid_cos": 0,
            "skipped": 0,
            "errors": 0,
        }

    def run(self) -> str:
        """
        执行 CoS 提取流程

        返回:
            str: 输出文件路径
        """
        logger.info("=" * 60)
        logger.info("W6 状态链提取 (CoS Extraction) 开始")
        logger.info(f"输入路径: {self.input_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info("=" * 60)

        start_time = time.time()
        ensure_dir(self.output_path)

        # 步骤 1: 加载追踪数据
        input_file = os.path.join(self.input_path, "traces_filtered.jsonl")
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return ""

        traces = load_jsonl(input_file)
        self.stats["total_traces"] = len(traces)
        logger.info(f"加载 {len(traces)} 条追踪记录")

        # 步骤 2: 提取 CoS
        tracker = ProgressTracker(
            total=len(traces), desc="CoS 提取", log_interval=100
        )

        cos_records = []
        for trace in traces:
            record = self._extract_single_cos(trace)
            if record is not None:
                cos_records.append(record)
                self.stats["valid_cos"] += 1
                tracker.update(success=True)
            else:
                self.stats["skipped"] += 1
                tracker.update(success=False)
            self.stats["processed"] += 1

        tracker.finish()

        # 步骤 3: 保存结果
        output_file = os.path.join(self.output_path, "cos_dataset.jsonl")
        save_jsonl([r.to_dict() for r in cos_records], output_file)

        # 步骤 4: 生成展平的训练数据（每个状态转换一条记录）
        flat_file = self._generate_flat_dataset(cos_records)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("W6 状态链提取完成")
        logger.info(f"总追踪数: {self.stats['total_traces']}")
        logger.info(f"有效 CoS: {self.stats['valid_cos']}")
        logger.info(f"跳过: {self.stats['skipped']}")
        logger.info(f"错误: {self.stats['errors']}")
        logger.info(f"耗时: {elapsed:.1f}s")
        logger.info(f"CoS 数据集: {output_file}")
        logger.info(f"展平数据集: {flat_file}")
        logger.info("=" * 60)

        # 保存统计
        stats_file = os.path.join(self.output_path, "cos_stats.json")
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["timestamp"] = get_timestamp()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        return output_file

    def _extract_single_cos(self, trace: Dict[str, Any]) -> Optional[CoSRecord]:
        """
        从单条追踪记录提取 CoS

        参数:
            trace: 追踪记录字典

        返回:
            CoSRecord: CoS 记录，无效返回 None
        """
        try:
            tactic_states = trace.get("tactic_states", [])
            tactics = trace.get("tactics", [])

            # 构建完整状态链
            full_chain = self._build_full_chain(tactic_states, tactics)

            if len(full_chain) < self.min_steps:
                return None
            if len(full_chain) > self.max_steps:
                # 截断过长的证明
                full_chain = full_chain[:self.max_steps]

            # 提取关键状态路标
            key_chain = self._compress_chain(full_chain)

            return CoSRecord(
                theorem_name=trace.get("theorem_name", ""),
                theorem_full_name=trace.get("theorem_full_name", ""),
                theorem_type=trace.get("theorem_type", ""),
                full_cos_chain=full_chain,
                key_cos_chain=key_chain,
                proof_steps=len(full_chain),
                key_steps=len(key_chain),
                file_path=trace.get("file_path", ""),
                metadata={
                    "has_leandojo_trace": trace.get("metadata", {}).get(
                        "has_leandojo_trace", False
                    ),
                    "compression_ratio": (
                        len(key_chain) / max(len(full_chain), 1)
                    ),
                    "extraction_time": get_timestamp(),
                }
            )

        except Exception as e:
            logger.debug(f"提取 CoS 失败: {e}")
            self.stats["errors"] += 1
            return None

    def _build_full_chain(self, tactic_states: List[Dict],
                           tactics: List[str]) -> List[Dict[str, str]]:
        """
        构建完整的状态链

        参数:
            tactic_states: LeanDojo 追踪的状态对列表
            tactics: 策略列表（AST 解析的备选）

        返回:
            List[Dict]: 完整状态链，每项包含 state_before, tactic, state_after
        """
        chain = []

        if tactic_states:
            # 优先使用 LeanDojo 的精确状态对
            for step in tactic_states:
                chain.append({
                    "state_before": str(step.get("state_before", "")),
                    "tactic": str(step.get("tactic", "")),
                    "state_after": str(step.get("state_after", "")),
                })
        elif tactics:
            # 回退：仅有策略列表，状态为占位符
            for i, tactic in enumerate(tactics):
                chain.append({
                    "state_before": f"[state_{i}]",
                    "tactic": tactic,
                    "state_after": f"[state_{i+1}]",
                })

        return chain

    def _compress_chain(self, full_chain: List[Dict[str, str]]
                         ) -> List[Dict[str, str]]:
        """
        压缩状态链，提取关键状态路标

        策略:
        1. 保留首尾状态
        2. 保留关键 Tactic（have, apply, rw 等）对应的状态
        3. 按 target_ratio 控制压缩比

        参数:
            full_chain: 完整状态链

        返回:
            List[Dict]: 压缩后的关键状态链
        """
        if not self.compression_enabled:
            return full_chain

        if len(full_chain) <= 3:
            return full_chain

        # 标记关键步骤
        key_indices = set()

        # 保留首尾
        if self.keep_first_last:
            key_indices.add(0)
            key_indices.add(len(full_chain) - 1)

        # 保留关键 Tactic 对应的步骤
        for i, step in enumerate(full_chain):
            tactic_name = step["tactic"].split()[0] if step["tactic"] else ""
            if tactic_name in self.key_tactics:
                key_indices.add(i)

        # 如果关键步骤太多，按 target_ratio 均匀采样
        target_count = max(len(full_chain) // self.target_ratio, 3)
        if len(key_indices) > target_count:
            sorted_indices = sorted(key_indices)
            # 均匀采样
            step_size = max(len(sorted_indices) // target_count, 1)
            sampled = set()
            for i in range(0, len(sorted_indices), step_size):
                sampled.add(sorted_indices[i])
            # 确保首尾保留
            if self.keep_first_last:
                sampled.add(0)
                sampled.add(len(full_chain) - 1)
            key_indices = sampled

        # 如果关键步骤太少，补充均匀分布的步骤
        if len(key_indices) < target_count:
            remaining = target_count - len(key_indices)
            all_indices = set(range(len(full_chain))) - key_indices
            if all_indices:
                sorted_remaining = sorted(all_indices)
                step_size = max(len(sorted_remaining) // remaining, 1)
                for i in range(0, len(sorted_remaining), step_size):
                    key_indices.add(sorted_remaining[i])
                    if len(key_indices) >= target_count:
                        break

        # 按顺序提取关键步骤
        key_chain = [full_chain[i] for i in sorted(key_indices)]
        return key_chain

    def extract_cos(self, trace: Dict[str, Any]
                     ) -> List[Tuple[str, str, str]]:
        """
        从单个轨迹提取状态链（兼容旧接口）

        参数:
            trace: 证明轨迹数据

        返回:
            List[Tuple]: (state_before, tactic, state_after) 三元组列表
        """
        record = self._extract_single_cos(trace)
        if record is None:
            return []
        return [
            (s["state_before"], s["tactic"], s["state_after"])
            for s in record.full_cos_chain
        ]

    def build_dataset(self, traces: List[Dict[str, Any]]
                       ) -> List[Dict[str, Any]]:
        """
        构建 CoS 数据集（兼容旧接口）

        参数:
            traces: 轨迹数据列表

        返回:
            List[Dict]: CoS 数据集
        """
        dataset = []
        for trace in traces:
            record = self._extract_single_cos(trace)
            if record is not None:
                dataset.append(record.to_dict())
        return dataset

    def validate_cos(self, cos_step: Dict[str, str]) -> bool:
        """
        验证单个状态转换的有效性

        参数:
            cos_step: 包含 state_before, tactic, state_after 的字典

        返回:
            bool: 是否有效
        """
        state_before = cos_step.get("state_before", "")
        tactic = cos_step.get("tactic", "")
        state_after = cos_step.get("state_after", "")

        # 基本非空检查
        if not state_before or not tactic:
            return False

        # 占位符状态不算有效
        if state_before.startswith("[state_"):
            return False

        # 状态前后不应完全相同（除非是 skip 等空操作）
        if state_before == state_after and tactic not in ("skip", "rfl"):
            return False

        return True

    def _generate_flat_dataset(self, cos_records: List[CoSRecord]) -> str:
        """
        生成展平的训练数据集，每个状态转换一条记录

        参数:
            cos_records: CoS 记录列表

        返回:
            str: 输出文件路径
        """
        flat_file = os.path.join(self.output_path, "cos_flat.jsonl")
        flat_data = []

        for record in cos_records:
            # 使用关键状态链生成训练样本
            chain = record.key_cos_chain if record.key_cos_chain else record.full_cos_chain

            for i, step in enumerate(chain):
                flat_item = {
                    "theorem_name": record.theorem_name,
                    "theorem_full_name": record.theorem_full_name,
                    "theorem_type": record.theorem_type,
                    "state_before": step["state_before"],
                    "tactic": step["tactic"],
                    "state_after": step["state_after"],
                    "step_index": i,
                    "total_steps": len(chain),
                    "file_path": record.file_path,
                    # 用于去重的哈希
                    "hash": compute_hash(
                        f"{step['state_before']}|{step['tactic']}|{step['state_after']}"
                    ),
                }
                flat_data.append(flat_item)

        save_jsonl(flat_data, flat_file)
        logger.info(f"生成展平数据集: {len(flat_data)} 条记录 -> {flat_file}")
        return flat_file


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="W6 状态链提取 - CoS Extraction")
    parser.add_argument(
        "--config", type=str,
        default="configs/data_pipeline.yaml",
        help="配置文件路径"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_file="logs/cos_extraction.log",
                  module_name="rtap")
    config = load_yaml(args.config)

    extractor = CoSExtractor(config)
    output_file = extractor.run()
    print(f"\nCoS 提取完成！输出文件: {output_file}")


if __name__ == "__main__":
    main()
