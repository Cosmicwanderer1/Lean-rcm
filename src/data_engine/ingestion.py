"""
数据摄取模块 - 全量追踪 (Ingestion & Tracing)
@author ygw
更新日期: 2026-02-06

W5 核心模块：利用 LeanDojo-v2 运行 Mathlib4 中的每一个证明脚本，
提取 AST、状态对 (S_pre, S_post)，记录每一行代码执行前后的形式化状态变化。

技术产出:
- AST: 抓取定理的语法树
- 状态对: 记录每一行代码执行前后的形式化状态变化
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_jsonl, load_jsonl, ensure_dir,
    setup_logging, ProgressTracker, CheckpointManager,
    get_timestamp, batch_iter, retry_with_backoff
)
from src.common.ast_parser import ASTParser, TheoremInfo

logger = logging.getLogger(__name__)


@dataclass
class TraceRecord:
    """
    单条追踪记录数据类

    属性:
        theorem_name: 定理名称
        theorem_full_name: 定理完整限定名
        theorem_type: 定理类型表达式
        file_path: 源文件路径
        proof_mode: 证明模式 (tactic / term / mixed)
        tactics: 策略列表
        tactic_states: 每步策略的状态对列表
        proof_steps: 证明步数
        imports: 依赖列表
        namespace: 命名空间
        attributes: 属性标签
        metadata: 元数据
    """
    theorem_name: str = ""
    theorem_full_name: str = ""
    theorem_type: str = ""
    file_path: str = ""
    proof_mode: str = ""
    tactics: List[str] = field(default_factory=list)
    tactic_states: List[Dict[str, Any]] = field(default_factory=list)
    proof_steps: int = 0
    imports: List[str] = field(default_factory=list)
    namespace: str = ""
    attributes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)


class DataIngestion:
    """
    数据摄取类，负责从 Mathlib4 提取原始 Trace 数据。

    核心流程:
    1. 扫描 Mathlib4 目录，收集所有 .lean 文件
    2. 使用 ASTParser 解析每个文件，提取定理声明
    3. 使用 LeanDojo 追踪每个定理的证明过程
    4. 记录每步策略的 (S_pre, Tactic, S_post) 三元组
    5. 过滤无效证明，保存为 JSONL 格式

    使用示例:
        ingestion = DataIngestion(config)
        ingestion.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据摄取模块

        参数:
            config: 配置字典（来自 data_pipeline.yaml）
        """
        self.config = config
        self.mathlib_path = config["data_source"]["mathlib4_path"]
        self.output_path = config["trace_extraction"]["output_path"]
        self.exclude_dirs = config["data_source"].get("exclude_dirs", [])
        self.file_extensions = config["data_source"].get("file_extensions", [".lean"])

        # LeanDojo 追踪参数
        trace_config = config["trace_extraction"]
        self.timeout_per_file = trace_config["leandojo"].get("timeout_per_file", 300)
        self.max_retries = trace_config["leandojo"].get("max_retries", 3)
        self.retry_delay = trace_config["leandojo"].get("retry_delay", 5)
        self.batch_size = trace_config["leandojo"].get("batch_size", 50)

        # 过滤条件
        filter_config = trace_config.get("filter", {})
        self.min_proof_steps = filter_config.get("min_proof_steps", 2)
        self.max_proof_steps = filter_config.get("max_proof_steps", 200)
        self.exclude_sorry = filter_config.get("exclude_sorry", True)
        self.exclude_native_decide = filter_config.get("exclude_native_decide", True)
        self.require_tactic_proof = filter_config.get("require_tactic_proof", True)

        # 断点续传
        ckpt_config = trace_config.get("checkpoint", {})
        self.checkpoint_enabled = ckpt_config.get("enable", True)
        self.checkpoint_path = ckpt_config.get("checkpoint_path", "")
        self.checkpoint_interval = ckpt_config.get("save_interval", 100)

        # 初始化组件
        self.ast_parser = ASTParser()
        self.checkpoint: Optional[CheckpointManager] = None

        # 统计信息
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_theorems": 0,
            "valid_theorems": 0,
            "filtered_theorems": 0,
            "errors": 0,
        }

    def run(self) -> str:
        """
        执行全量追踪流程

        返回:
            str: 输出文件路径
        """
        logger.info("=" * 60)
        logger.info("W5 全量追踪 (Ingestion & Tracing) 开始")
        logger.info(f"Mathlib4 路径: {self.mathlib_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info("=" * 60)

        start_time = time.time()

        # 步骤 1: 扫描文件
        lean_files = self._scan_lean_files()
        self.stats["total_files"] = len(lean_files)
        logger.info(f"扫描到 {len(lean_files)} 个 .lean 文件")

        # 步骤 2: 初始化断点续传
        if self.checkpoint_enabled and self.checkpoint_path:
            self.checkpoint = CheckpointManager(
                self.checkpoint_path,
                save_interval=self.checkpoint_interval
            )
            logger.info(f"断点续传已启用，已处理 {self.checkpoint.processed_count} 个文件")

        # 步骤 3: 确保输出目录存在
        ensure_dir(self.output_path)
        output_file = os.path.join(self.output_path, "traces_raw.jsonl")

        # 步骤 3.5: 一次性执行 LeanDojo 仓库级追踪
        self.traced_repo = self._trace_repo_once()

        # 步骤 3.6: 构建 traced_file 路径索引，加速按文件查找
        self.traced_file_index: Dict[str, Any] = {}
        if self.traced_repo is not None:
            for tf in self.traced_repo.traced_files:
                # TracedFile.path 返回相对于仓库根目录的路径
                self.traced_file_index[str(tf.path)] = tf
            logger.info(f"LeanDojo 追踪成功，共 {len(self.traced_file_index)} 个文件")

        # 步骤 4: 逐文件处理
        tracker = ProgressTracker(
            total=len(lean_files),
            desc="全量追踪",
            log_interval=50
        )

        all_traces = []

        for batch in batch_iter(lean_files, self.batch_size):
            batch_traces = self._process_file_batch(batch)
            all_traces.extend(batch_traces)

            for _ in batch:
                tracker.update(success=True)

            # 定期保存中间结果
            if len(all_traces) >= 1000:
                save_jsonl(
                    [t.to_dict() for t in all_traces],
                    output_file,
                    mode='a'
                )
                logger.info(f"已保存 {len(all_traces)} 条追踪记录")
                all_traces.clear()

        # 保存剩余数据
        if all_traces:
            save_jsonl(
                [t.to_dict() for t in all_traces],
                output_file,
                mode='a'
            )

        # 保存检查点
        if self.checkpoint:
            self.checkpoint.save()

        # 步骤 5: 过滤有效证明
        filtered_file = self._filter_valid_proofs(output_file)

        # 完成统计
        tracker_stats = tracker.finish()
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("W5 全量追踪完成")
        logger.info(f"总文件数: {self.stats['total_files']}")
        logger.info(f"总定理数: {self.stats['total_theorems']}")
        logger.info(f"有效定理数: {self.stats['valid_theorems']}")
        logger.info(f"过滤定理数: {self.stats['filtered_theorems']}")
        logger.info(f"错误数: {self.stats['errors']}")
        logger.info(f"总耗时: {elapsed:.1f}s")
        logger.info(f"输出文件: {filtered_file}")
        logger.info("=" * 60)

        # 保存统计信息
        stats_file = os.path.join(self.output_path, "ingestion_stats.json")
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["timestamp"] = get_timestamp()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        return filtered_file

    def _scan_lean_files(self) -> List[str]:
        """
        扫描 Mathlib4 目录，收集所有 .lean 文件

        返回:
            List[str]: .lean 文件路径列表
        """
        lean_files = []
        mathlib_path = Path(self.mathlib_path)

        if not mathlib_path.exists():
            logger.error(f"Mathlib4 目录不存在: {self.mathlib_path}")
            return lean_files

        for root, dirs, files in os.walk(mathlib_path):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file_name in files:
                if any(file_name.endswith(ext) for ext in self.file_extensions):
                    file_path = os.path.join(root, file_name)
                    lean_files.append(file_path)

        # 按文件路径排序，确保可复现
        lean_files.sort()
        return lean_files

    def _process_file_batch(self, file_batch: List[str]) -> List[TraceRecord]:
        """
        处理一批文件

        参数:
            file_batch: 文件路径列表

        返回:
            List[TraceRecord]: 追踪记录列表
        """
        batch_traces = []

        for file_path in file_batch:
            # 检查断点
            if self.checkpoint and self.checkpoint.is_done(file_path):
                continue

            try:
                traces = self._process_single_file(file_path)
                batch_traces.extend(traces)
                self.stats["processed_files"] += 1

                # 标记完成
                if self.checkpoint:
                    self.checkpoint.mark_done(file_path)

            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                self.stats["errors"] += 1

        return batch_traces

    def _process_single_file(self, file_path: str) -> List[TraceRecord]:
        """
        处理单个 .lean 文件，提取所有定理的追踪记录

        参数:
            file_path: .lean 文件路径

        返回:
            List[TraceRecord]: 追踪记录列表
        """
        traces = []

        # 步骤 1: 使用 AST 解析器提取定理
        theorems = self.ast_parser.parse_file(file_path)
        self.stats["total_theorems"] += len(theorems)

        if not theorems:
            return traces

        # 步骤 2: 从已有的 traced_repo 中提取该文件的追踪数据
        leandojo_traces = self._extract_leandojo_traces(file_path)

        # 步骤 3: 合并 AST 解析结果和 LeanDojo 追踪结果
        for thm in theorems:
            trace = self._build_trace_record(thm, leandojo_traces)
            if trace is not None:
                traces.append(trace)

        return traces

    def _trace_repo_once(self) -> Optional[Any]:
        """
        一次性对整个 Mathlib4 仓库执行 LeanDojo 追踪

        LeanDojo 的 trace() 是仓库级别操作，分两阶段执行：
        阶段1 - 提取：编译项目 + 运行 ExtractData.lean 生成 AST JSON（耗时4+小时，结果会缓存）
        阶段2 - 解析：解析 AST JSON 文件构建 TracedRepo 对象

        本方法将两阶段分离处理：
        - 提取阶段：利用 LeanDojo 缓存机制，已缓存则跳过
        - 解析阶段：失败时不重新提取，而是优雅降级

        返回:
            TracedRepo 对象，失败返回 None
        @author ygw
        """
        try:
            from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
            from lean_dojo_v2.lean_dojo.data_extraction.trace import (
                get_traced_repo_path,
            )
            from lean_dojo_v2.lean_dojo.data_extraction.traced_data import TracedRepo

            repo = LeanGitRepo.from_path(self.mathlib_path)

            # === 阶段1: 获取缓存路径（提取阶段，已缓存则秒级返回） ===
            logger.info("阶段1: 检查 LeanDojo 缓存 / 执行仓库级提取...")
            try:
                cached_path = get_traced_repo_path(repo, build_deps=True)
                logger.info(f"阶段1 完成，缓存路径: {cached_path}")
            except Exception as e:
                logger.error(f"阶段1 提取失败: {e}")
                logger.error("提取阶段失败无法恢复，将仅使用 AST 解析器")
                return None

            # === 阶段2: 从缓存加载并解析（解析阶段） ===
            logger.info(f"阶段2: 从缓存加载并解析 TracedRepo...")
            try:
                traced_repo = TracedRepo.load_from_disk(
                    cached_path, build_deps=True
                )
                # check_sanity 可能因依赖缺失而失败，不阻断流程
                try:
                    traced_repo.check_sanity()
                except Exception as sanity_err:
                    logger.warning(f"check_sanity 警告（不影响使用）: {sanity_err}")
                logger.info(f"阶段2 完成，TracedRepo 加载成功，共 {len(traced_repo.traced_files)} 个文件")
                return traced_repo
            except Exception as e:
                logger.warning(f"阶段2 load_from_disk 失败: {e}")
                logger.info("尝试使用 from_traced_files 方式加载...")

            # === 阶段2 备选: 直接从 AST JSON 解析（跳过 XML） ===
            try:
                traced_repo = TracedRepo.from_traced_files(
                    cached_path, build_deps=True
                )
                traced_repo.save_to_disk()  # 保存 XML 以便下次快速加载
                logger.info("阶段2 备选方式完成，TracedRepo 构建成功")
                return traced_repo
            except Exception as e:
                logger.error(f"阶段2 from_traced_files 也失败: {e}")
                logger.error("解析阶段失败，将仅使用 AST 解析器提取数据")
                return None

        except ImportError:
            logger.warning("LeanDojo 未安装，将仅使用 AST 解析器提取数据")
            return None
        except Exception as e:
            logger.error(f"LeanDojo 仓库级追踪失败: {e}")
            return None

    def _extract_leandojo_traces(self, file_path: str) -> Dict[str, Any]:
        """
        从已有的 traced_repo 中提取指定文件的追踪结果

        返回字典包含三个索引层级，供 _build_trace_record 多级回退匹配：
        - "by_full_name": 以 LeanDojo full_name 为键（精确匹配）
        - "by_short_name": 以短名（最后一段）为键（回退匹配）
        - "by_suffix": 以 "namespace.short_name" 后缀为键（中间匹配）

        参数:
            file_path: .lean 文件绝对路径

        返回:
            Dict: 包含多级索引的追踪结果
        @author ygw
        """
        leandojo_results = {
            "by_full_name": {},
            "by_short_name": {},
            "by_suffix": {},
        }

        if self.traced_repo is None:
            return leandojo_results

        # 计算相对路径（相对于 Mathlib4 根目录）
        rel_path = os.path.relpath(file_path, self.mathlib_path)
        # 统一路径分隔符为 /
        rel_path = rel_path.replace(os.sep, "/")

        traced_file = self.traced_file_index.get(rel_path)
        if traced_file is None:
            return leandojo_results

        try:
            for traced_thm in traced_file.get_traced_theorems():
                full_name = traced_thm.theorem.full_name
                tactic_states = []

                for step in traced_thm.get_traced_tactics():
                    state_record = {
                        "tactic": step.tactic,
                        "state_before": step.state_before,
                        "state_after": step.state_after,
                    }
                    tactic_states.append(state_record)

                trace_data = {
                    "tactic_states": tactic_states,
                    "success": True,
                    "leandojo_full_name": full_name,
                }

                # 索引1: 完整限定名
                leandojo_results["by_full_name"][full_name] = trace_data

                # 索引2: 短名（最后一个 . 之后的部分）
                short_name = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
                # 短名可能重复，只保留第一个（避免覆盖）
                if short_name not in leandojo_results["by_short_name"]:
                    leandojo_results["by_short_name"][short_name] = trace_data

                # 索引3: 后缀匹配（倒数两段，如 "Set.vsub_left_cancel"）
                parts = full_name.rsplit(".", 2)
                if len(parts) >= 2:
                    suffix = f"{parts[-2]}.{parts[-1]}"
                    if suffix not in leandojo_results["by_suffix"]:
                        leandojo_results["by_suffix"][suffix] = trace_data

        except Exception as e:
            logger.debug(f"提取文件追踪数据失败 {rel_path}: {e}")

        return leandojo_results

    def _build_trace_record(self, thm: TheoremInfo,
                             leandojo_traces: Dict[str, Any]
                             ) -> Optional[TraceRecord]:
        """
        构建单条追踪记录，使用多级回退匹配 LeanDojo 追踪结果

        匹配优先级：
        1. 精确匹配 full_name（AST 解析的完整限定名）
        2. 后缀匹配（AST 的 namespace.name 与 LeanDojo 后缀索引）
        3. 短名匹配（仅定理名，无命名空间）

        参数:
            thm: 定理信息（来自 AST 解析）
            leandojo_traces: LeanDojo 追踪结果（含多级索引）

        返回:
            TraceRecord: 追踪记录，不满足条件返回 None
        @author ygw
        """
        by_full = leandojo_traces.get("by_full_name", {})
        by_suffix = leandojo_traces.get("by_suffix", {})
        by_short = leandojo_traces.get("by_short_name", {})

        # 多级回退匹配
        leandojo_data = (
            by_full.get(thm.full_name)
            or by_suffix.get(thm.full_name)
            or by_short.get(thm.name)
            or {}
        )
        tactic_states = leandojo_data.get("tactic_states", [])

        # 如果 LeanDojo 匹配成功，用其权威 full_name 覆盖 AST 的不准确名称
        effective_full_name = leandojo_data.get("leandojo_full_name", thm.full_name)

        # 如果 LeanDojo 没有追踪到，使用 AST 解析的策略列表
        tactics = thm.tactics
        if tactic_states:
            tactics = [s["tactic"] for s in tactic_states]

        # 构建记录
        record = TraceRecord(
            theorem_name=thm.name,
            theorem_full_name=effective_full_name,
            theorem_type=thm.type_expr,
            file_path=thm.file_path,
            proof_mode=thm.proof_mode,
            tactics=tactics,
            tactic_states=tactic_states,
            proof_steps=len(tactics),
            imports=thm.imports,
            namespace=thm.namespace,
            attributes=thm.attributes,
            metadata={
                "line_start": thm.line_start,
                "line_end": thm.line_end,
                "has_leandojo_trace": bool(tactic_states),
                "extraction_time": get_timestamp(),
            }
        )

        return record

    def _filter_valid_proofs(self, raw_file: str) -> str:
        """
        过滤有效的证明，生成清洗后的数据文件

        参数:
            raw_file: 原始追踪文件路径

        返回:
            str: 过滤后的文件路径
        """
        filtered_file = os.path.join(self.output_path, "traces_filtered.jsonl")
        valid_count = 0
        filtered_count = 0

        if not os.path.exists(raw_file):
            logger.warning(f"原始追踪文件不存在: {raw_file}")
            return filtered_file

        with open(raw_file, 'r', encoding='utf-8') as fin, \
             open(filtered_file, 'w', encoding='utf-8') as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                # 应用过滤条件
                if not self._is_valid_proof(record):
                    filtered_count += 1
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                valid_count += 1

        self.stats["valid_theorems"] = valid_count
        self.stats["filtered_theorems"] = filtered_count

        logger.info(f"过滤完成: {valid_count} 有效 / {filtered_count} 过滤")
        return filtered_file

    def _is_valid_proof(self, record: Dict[str, Any]) -> bool:
        """
        判断证明记录是否有效

        参数:
            record: 追踪记录字典

        返回:
            bool: 是否有效
        """
        tactics = record.get("tactics", [])
        proof_mode = record.get("proof_mode", "")
        proof_steps = record.get("proof_steps", 0)

        # 检查证明步数
        if proof_steps < self.min_proof_steps:
            return False
        if proof_steps > self.max_proof_steps:
            return False

        # 检查证明模式
        if self.require_tactic_proof and proof_mode not in ("tactic", "mixed"):
            return False

        # 检查是否包含 sorry
        if self.exclude_sorry:
            for tactic in tactics:
                if "sorry" in tactic:
                    return False

        # 检查是否包含 native_decide
        if self.exclude_native_decide:
            for tactic in tactics:
                if "native_decide" in tactic:
                    return False

        return True

    def extract_traces(self) -> List[Dict[str, Any]]:
        """
        提取所有定理的证明轨迹（兼容旧接口）

        返回:
            List[Dict]: 轨迹数据列表
        """
        output_file = self.run()
        if os.path.exists(output_file):
            return load_jsonl(output_file)
        return []

    def filter_valid_proofs(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤有效的证明（兼容旧接口）

        参数:
            traces: 原始轨迹数据

        返回:
            List[Dict]: 过滤后的有效证明
        """
        return [t for t in traces if self._is_valid_proof(t)]

    def save_traces(self, traces: List[Dict[str, Any]]):
        """
        保存轨迹数据（兼容旧接口）

        参数:
            traces: 轨迹数据列表
        """
        output_file = os.path.join(self.output_path, "traces_raw.jsonl")
        save_jsonl(traces, output_file)
        logger.info(f"已保存 {len(traces)} 条轨迹到 {output_file}")


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="W5 全量追踪 - Mathlib4 证明轨迹提取")
    parser.add_argument(
        "--config", type=str,
        default="configs/data_pipeline.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="日志级别"
    )
    args = parser.parse_args()

    # 配置日志
    setup_logging(
        log_level=args.log_level,
        log_file="logs/ingestion.log",
        module_name="rtap"
    )

    # 加载配置
    config = load_yaml(args.config)

    # 执行追踪
    ingestion = DataIngestion(config)
    output_file = ingestion.run()

    print(f"\n追踪完成！输出文件: {output_file}")


if __name__ == "__main__":
    main()
