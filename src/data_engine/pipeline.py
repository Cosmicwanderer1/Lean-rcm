"""
Reverse-DSP++ 数据流水线编排器
@author ygw
更新日期: 2026-02-06

W5-W8 全流程编排入口，按顺序执行：
  W5: 全量追踪 (Ingestion & Tracing) → traces_raw.jsonl / traces_filtered.jsonl
  W6: 状态链提取 (CoS Extraction) → cos_dataset.jsonl / cos_flat.jsonl
  W7: 思维回溯 (Thought Back-translation) → thought_dataset.jsonl
  W8: 数据增强 (Data Augmentation) → augmented_dataset.jsonl + train/val/test.jsonl

支持:
- 单步执行（--step w5 / w6 / w7 / w8）
- 全流程执行（--step all）
- 从指定步骤恢复（--resume-from w6）
- 空跑模式（--dry-run）

使用示例:
    python -m src.data_engine.pipeline --config configs/data_pipeline.yaml --step all
    python -m src.data_engine.pipeline --config configs/data_pipeline.yaml --step w6
    python -m src.data_engine.pipeline --resume-from w7
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_json, ensure_dir, setup_logging, get_timestamp
)

logger = logging.getLogger(__name__)


# ================================================================
# 步骤定义
# ================================================================

# 步骤名称 → 顺序编号
STEP_ORDER = {
    "w5": 1,
    "w6": 2,
    "w7": 3,
    "w8": 4,
}

# 步骤名称 → 中文描述
STEP_NAMES = {
    "w5": "全量追踪 (Ingestion & Tracing)",
    "w6": "状态链提取 (CoS Extraction)",
    "w7": "思维回溯 (Thought Back-translation)",
    "w8": "数据增强 (Data Augmentation)",
}


# ================================================================
# 流水线状态管理
# ================================================================

class PipelineState:
    """
    流水线状态管理器，记录各步骤的执行状态。

    状态文件存储在输出目录下的 pipeline_state.json，
    用于支持断点恢复和执行历史追踪。
    """

    def __init__(self, state_file: str):
        """
        初始化状态管理器

        参数:
            state_file: 状态文件路径
        """
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        """加载状态文件，不存在则返回初始状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"状态文件读取失败: {e}，使用初始状态")
        return {
            "steps": {},
            "last_completed_step": None,
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
        }

    def save(self):
        """保存状态到文件"""
        self.state["updated_at"] = get_timestamp()
        # 确保父目录存在（注意：ensure_dir 接收目录路径，不是文件路径）
        parent_dir = os.path.dirname(self.state_file)
        if parent_dir:
            ensure_dir(parent_dir)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def mark_started(self, step: str):
        """
        标记步骤开始

        参数:
            step: 步骤名称 (w5/w6/w7/w8)
        """
        self.state["steps"][step] = {
            "status": "running",
            "started_at": get_timestamp(),
            "completed_at": None,
            "output_path": None,
            "error": None,
        }
        self.save()

    def mark_completed(self, step: str, output_path: str = ""):
        """
        标记步骤完成

        参数:
            step: 步骤名称
            output_path: 输出路径
        """
        if step not in self.state["steps"]:
            self.state["steps"][step] = {}
        self.state["steps"][step]["status"] = "completed"
        self.state["steps"][step]["completed_at"] = get_timestamp()
        self.state["steps"][step]["output_path"] = output_path
        self.state["last_completed_step"] = step
        self.save()

    def mark_failed(self, step: str, error: str):
        """
        标记步骤失败

        参数:
            step: 步骤名称
            error: 错误信息
        """
        if step not in self.state["steps"]:
            self.state["steps"][step] = {}
        self.state["steps"][step]["status"] = "failed"
        self.state["steps"][step]["error"] = error
        self.save()

    def is_completed(self, step: str) -> bool:
        """
        检查步骤是否已完成

        参数:
            step: 步骤名称

        返回:
            bool: 是否已完成
        """
        step_info = self.state["steps"].get(step, {})
        return step_info.get("status") == "completed"

    def get_last_completed(self) -> Optional[str]:
        """
        获取最后完成的步骤

        返回:
            Optional[str]: 步骤名称，无则返回 None
        """
        return self.state.get("last_completed_step")


# ================================================================
# 数据流水线主类
# ================================================================

class DataPipeline:
    """
    W5-W8 数据流水线编排器。

    负责按顺序调度各阶段模块，管理数据流转和状态持久化。

    数据流:
        .lean 文件 → [W5] → traces_filtered.jsonl
                   → [W6] → cos_flat.jsonl
                   → [W7] → thought_dataset.jsonl
                   → [W8] → train.jsonl / val.jsonl / test.jsonl

    使用示例:
        pipeline = DataPipeline(config)
        pipeline.run(step="all")
        pipeline.run(step="w6")
        pipeline.run(resume_from="w7")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化流水线

        参数:
            config: 完整配置字典（来自 data_pipeline.yaml）
        """
        self.config = config
        global_config = config.get("global", {})

        self.project_root = global_config.get("project_root", "")
        self.log_level = global_config.get("log_level", "INFO")

        # 状态管理
        state_dir = os.path.join(self.project_root, "data", "processed")
        state_file = os.path.join(state_dir, "pipeline_state.json")
        self.state = PipelineState(state_file)

        # 全局统计
        self.stats = {
            "pipeline_start": "",
            "pipeline_end": "",
            "steps_executed": [],
            "steps_skipped": [],
            "steps_failed": [],
        }

    def run(self, step: str = "all", resume_from: Optional[str] = None,
            dry_run: bool = False) -> Dict[str, Any]:
        """
        执行流水线

        参数:
            step: 执行步骤 ("all" / "w5" / "w6" / "w7" / "w8")
            resume_from: 从指定步骤恢复执行（含该步骤）
            dry_run: 空跑模式，仅打印执行计划不实际执行

        返回:
            Dict: 执行统计信息
        """
        self.stats["pipeline_start"] = get_timestamp()

        logger.info("=" * 70)
        logger.info("  Reverse-DSP++ 数据流水线")
        logger.info("=" * 70)

        # 确定要执行的步骤
        steps_to_run = self._resolve_steps(step, resume_from)
        logger.info(f"执行计划: {' → '.join(steps_to_run)}")

        if dry_run:
            logger.info("[空跑模式] 以下步骤将被执行:")
            for s in steps_to_run:
                enabled = self._is_step_enabled(s)
                status = "启用" if enabled else "禁用(配置)"
                logger.info(f"  {s}: {STEP_NAMES.get(s, s)} [{status}]")
            return self.stats

        # 逐步执行
        for s in steps_to_run:
            if not self._is_step_enabled(s):
                logger.info(f"步骤 {s} 已在配置中禁用，跳过")
                self.stats["steps_skipped"].append(s)
                continue

            success = self._execute_step(s)
            if not success:
                logger.error(f"步骤 {s} 执行失败，流水线中止")
                self.stats["steps_failed"].append(s)
                break

        self.stats["pipeline_end"] = get_timestamp()

        # 保存流水线统计
        self._save_pipeline_stats()

        logger.info("=" * 70)
        logger.info("  流水线执行完毕")
        logger.info(f"  成功: {self.stats['steps_executed']}")
        logger.info(f"  跳过: {self.stats['steps_skipped']}")
        logger.info(f"  失败: {self.stats['steps_failed']}")
        logger.info("=" * 70)

        return self.stats

    def _resolve_steps(self, step: str, resume_from: Optional[str]) -> List[str]:
        """
        解析要执行的步骤列表

        参数:
            step: 步骤参数
            resume_from: 恢复起点

        返回:
            List[str]: 有序步骤列表
        """
        all_steps = ["w5", "w6", "w7", "w8"]

        if resume_from:
            # 从指定步骤开始（含该步骤）
            if resume_from not in STEP_ORDER:
                logger.error(f"无效的恢复步骤: {resume_from}")
                return []
            start_idx = all_steps.index(resume_from)
            return all_steps[start_idx:]

        if step == "all":
            return all_steps

        if step in STEP_ORDER:
            return [step]

        logger.error(f"无效的步骤参数: {step}，可选: all / w5 / w6 / w7 / w8")
        return []

    def _is_step_enabled(self, step: str) -> bool:
        """
        检查步骤是否在配置中启用

        参数:
            step: 步骤名称

        返回:
            bool: 是否启用
        """
        config_map = {
            "w5": "trace_extraction",
            "w6": "cos_extraction",
            "w7": "thought_generation",
            "w8": "augmentation",
        }
        section = config_map.get(step, "")
        if section and section in self.config:
            return self.config[section].get("enable", True)
        return True

    def _execute_step(self, step: str) -> bool:
        """
        执行单个步骤

        参数:
            step: 步骤名称

        返回:
            bool: 是否成功
        """
        step_name = STEP_NAMES.get(step, step)
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  {step.upper()}: {step_name}")
        logger.info("=" * 60)

        self.state.mark_started(step)
        start_time = time.time()

        try:
            output_path = ""

            if step == "w5":
                output_path = self._run_w5()
            elif step == "w6":
                output_path = self._run_w6()
            elif step == "w7":
                output_path = self._run_w7()
            elif step == "w8":
                output_path = self._run_w8()

            elapsed = time.time() - start_time
            self.state.mark_completed(step, output_path)
            self.stats["steps_executed"].append(step)

            logger.info(f"步骤 {step.upper()} 完成，耗时 {elapsed:.1f}s")
            logger.info(f"输出路径: {output_path}")
            return True

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.state.mark_failed(step, error_msg)
            logger.error(f"步骤 {step.upper()} 失败 (耗时 {elapsed:.1f}s): {error_msg}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    # ============================================================
    # W5: 全量追踪
    # ============================================================

    def _run_w5(self) -> str:
        """
        执行 W5 全量追踪

        返回:
            str: 输出路径
        """
        from src.data_engine.ingestion import DataIngestion

        ingestion = DataIngestion(self.config)
        output_file = ingestion.run()
        return output_file

    # ============================================================
    # W6: 状态链提取
    # ============================================================

    def _run_w6(self) -> str:
        """
        执行 W6 状态链提取

        返回:
            str: 输出路径
        """
        from src.data_engine.cos_extractor import CoSExtractor

        extractor = CoSExtractor(self.config)
        output_file = extractor.run()
        return output_file

    # ============================================================
    # W7: 思维回溯
    # ============================================================

    def _run_w7(self) -> str:
        """
        执行 W7 思维回溯

        返回:
            str: 输出路径
        """
        from src.data_engine.thought_backtrans import ThoughtBacktranslator

        translator = ThoughtBacktranslator(self.config)
        output_file = translator.run()
        return output_file

    # ============================================================
    # W8: 数据增强
    # ============================================================

    def _run_w8(self) -> str:
        """
        执行 W8 数据增强

        返回:
            str: 输出路径
        """
        from src.data_engine.augmentation import DataAugmentation

        augmentor = DataAugmentation(self.config)
        output_path = augmentor.run()
        return output_path

    # ============================================================
    # 统计与报告
    # ============================================================

    def _save_pipeline_stats(self):
        """保存流水线执行统计"""
        stats_dir = os.path.join(self.project_root, "data", "processed")
        ensure_dir(stats_dir)
        stats_file = os.path.join(stats_dir, "pipeline_stats.json")
        save_json(self.stats, stats_file)
        logger.info(f"流水线统计已保存: {stats_file}")

    def get_status(self) -> Dict[str, Any]:
        """
        获取流水线当前状态

        返回:
            Dict: 各步骤状态信息
        """
        status = {}
        for step in ["w5", "w6", "w7", "w8"]:
            step_info = self.state.state.get("steps", {}).get(step, {})
            status[step] = {
                "name": STEP_NAMES.get(step, step),
                "status": step_info.get("status", "not_started"),
                "enabled": self._is_step_enabled(step),
            }
        status["last_completed"] = self.state.get_last_completed()
        return status

    def print_status(self):
        """打印流水线状态摘要"""
        status = self.get_status()
        logger.info("流水线状态:")
        for step in ["w5", "w6", "w7", "w8"]:
            info = status[step]
            icon = "✓" if info["status"] == "completed" else \
                   "✗" if info["status"] == "failed" else \
                   "→" if info["status"] == "running" else "○"
            enabled = "" if info["enabled"] else " [禁用]"
            logger.info(f"  {icon} {step.upper()}: {info['name']} "
                        f"[{info['status']}]{enabled}")


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reverse-DSP++ 数据流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 执行全流程
  python -m src.data_engine.pipeline --config configs/data_pipeline.yaml --step all

  # 仅执行 W6
  python -m src.data_engine.pipeline --step w6

  # 从 W7 恢复执行
  python -m src.data_engine.pipeline --resume-from w7

  # 空跑模式（仅打印计划）
  python -m src.data_engine.pipeline --step all --dry-run

  # 查看状态
  python -m src.data_engine.pipeline --status
        """
    )
    parser.add_argument(
        "--config", type=str, default="configs/data_pipeline.yaml",
        help="配置文件路径 (默认: configs/data_pipeline.yaml)"
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "w5", "w6", "w7", "w8"],
        help="执行步骤 (默认: all)"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        choices=["w5", "w6", "w7", "w8"],
        help="从指定步骤恢复执行"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="空跑模式，仅打印执行计划"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="查看流水线状态"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )

    args = parser.parse_args()

    # 初始化日志
    setup_logging(
        log_level=args.log_level,
        log_file="logs/pipeline.log",
        module_name="rtap"
    )

    # 加载配置
    config = load_yaml(args.config)

    # 创建流水线
    pipeline = DataPipeline(config)

    # 查看状态
    if args.status:
        pipeline.print_status()
        return

    # 执行流水线
    stats = pipeline.run(
        step=args.step,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )

    # 打印最终状态
    pipeline.print_status()

    # 输出摘要
    print("\n" + "=" * 50)
    print("  Reverse-DSP++ 数据流水线执行完毕")
    print("=" * 50)
    print(f"  成功步骤: {stats.get('steps_executed', [])}")
    print(f"  跳过步骤: {stats.get('steps_skipped', [])}")
    print(f"  失败步骤: {stats.get('steps_failed', [])}")
    print("=" * 50)


if __name__ == "__main__":
    main()
