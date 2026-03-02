"""
错误验证模块 - 通过 Pantograph 获取真实 Lean4 编译器报错
@author ygw
更新日期: 2026-03-01 (v4: 单worker + 弹性重放 intro跳过 + 紧急intros恢复)

W8 第二阶段：对错误注入数据执行 Pantograph 验证，获取真实报错信息。python -m src.data_engine.error_verifier --config configs/data_pipeline.yaml

流程:
1. 读取 error_injection.jsonl（已有 error_tactic）
2. 读取 cos_flat.jsonl（获取同一定理的完整策略序列）
3. 按定理分组，用 Pantograph 重放前置策略到达 state_before
4. 执行 error_tactic，捕获 Lean4 真实报错
5. 回填 error_message 字段，输出 error_injection_verified.jsonl

技术产出:
- error_injection_verified.jsonl: 包含真实报错信息的错误注入数据集
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_json, save_jsonl, load_jsonl, ensure_dir, get_timestamp
)
from src.common.lean_server import LeanServer

logger = logging.getLogger(__name__)


class ErrorVerifier:
    """
    错误验证器：通过 Pantograph 获取真实 Lean4 编译器报错信息。

    对每条错误注入记录:
    1. 根据 theorem_type 创建 Pantograph 证明目标
    2. 重放前置策略到达 state_before 对应的状态
    3. 执行 error_tactic，捕获 Lean4 返回的真实报错
    4. 将报错信息写入 error_message 字段

    使用示例:
        verifier = ErrorVerifier(config)
        verifier.run()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化错误验证器

        参数:
            config: 完整配置字典
        """
        self.config = config
        aug_config = config.get("augmentation", {})

        # 路径配置
        self.cos_input_path = aug_config.get("input_path", "data/processed/cos_dataset")
        self.error_input_path = aug_config.get("output_path", "data/processed/error_correction")
        self.output_path = aug_config.get("output_path", "data/processed/error_correction")

        # Pantograph 配置
        panto_config = config.get("cos_extraction", {}).get("pantograph", {})
        self.executable_path = panto_config.get(
            "executable",
            "/root/autodl-tmp/RTAP/workspace/PyPantograph/src/.lake/build/bin/repl"
        )
        self.pantograph_timeout = panto_config.get("timeout", 30)

        # Mathlib 项目路径（Pantograph 需要在此目录下启动以加载 Mathlib 环境）
        data_source = config.get("data_source", {})
        self.project_path = data_source.get(
            "mathlib4_path",
            "/root/autodl-tmp/RTAP/data/raw/mathlib4"
        )

        # 并发配置
        error_verify_config = aug_config.get("error_verification", {})
        self.max_concurrent = error_verify_config.get("max_concurrent", 4)
        self.batch_save_interval = error_verify_config.get("batch_save_interval", 500)

        # 统计
        self.stats = {
            "total_errors": 0,
            "verified": 0,
            "replay_failed": 0,
            "pantograph_error": 0,
            "skipped": 0,
            "unique_theorems": 0,
        }

    @staticmethod
    def _split_tactic_block(tactic: str) -> List[str]:
        """
        将多行 tactic 块拆分为可独立执行的原子 tactic 列表。

        Lean4 中多行 tactic 有两种情况:
        - 顶层独立 tactic: 行首无缩进，如 "classical\nobtain ...\nconvert ..."
        - 不可分割的多行 tactic: 后续行有缩进（续行/子策略），如 "calc\n  y ≤ ..."

        拆分规则: 以无缩进的行作为新 tactic 的起点，缩进行归属上一个 tactic。
        特殊处理: 以 '·' 开头的 focused goal 行视为新 tactic 的起点。

        参数:
            tactic: 可能包含换行符的 tactic 字符串

        返回:
            List[str]: 原子 tactic 列表，每个元素可直接发送给 Pantograph
        @author ygw
        """
        if '\n' not in tactic:
            return [tactic]

        lines = tactic.split('\n')
        blocks = []
        current_block = []

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                # 空行归属当前块
                if current_block:
                    current_block.append(line)
                continue

            # 判断是否为新 tactic 的起点:
            # 1. 行首无缩进（不以空格/tab开头）
            # 2. 以 '·' 开头的 focused goal 标记
            is_new_tactic = (not line[0].isspace()) or stripped.startswith('·')

            if is_new_tactic and current_block:
                # 保存上一个块
                block_text = '\n'.join(current_block).rstrip()
                if block_text:
                    blocks.append(block_text)
                current_block = [line]
            else:
                current_block.append(line)

        # 保存最后一个块
        if current_block:
            block_text = '\n'.join(current_block).rstrip()
            if block_text:
                blocks.append(block_text)

        return blocks if blocks else [tactic]

    @staticmethod
    def _fix_double_prefix(name: str) -> Optional[str]:
        """
        修正定理名称中的连续重复命名空间段。

        例如: "Ordinal.Ordinal.not_bddAbove" → "Ordinal.not_bddAbove"
              "PreQuasiregular.Unitization.Unitization.xxx" → "PreQuasiregular.Unitization.xxx"

        参数:
            name: 定理全名

        返回:
            Optional[str]: 修正后的名称，如果没有双前缀则返回 None
        @author ygw
        """
        parts = name.split('.')
        has_dup = False
        for i in range(len(parts) - 1):
            if parts[i] == parts[i + 1]:
                has_dup = True
                break

        if not has_dup:
            return None

        # 去除连续重复段
        fixed = [parts[0]]
        for p in parts[1:]:
            if p != fixed[-1]:
                fixed.append(p)
        return '.'.join(fixed)

    @staticmethod
    def _extract_intro_names(state_before: str) -> List[str]:
        """
        从 cos_flat 的 state_before 中提取变量名，用于构建 intro 命令。

        Lean4 证明状态格式:
            M : Type u_1
            inst✝¹ : CommMonoid M
            S : Set M
            ⊢ goal_expression

        提取 ⊢ 之前所有 "name : type" 行中的 name。
        跳过含 ✝ 的不可访问变量（Lean4 不允许 intro 这类名字）。

        参数:
            state_before: step 0 的证明状态文本

        返回:
            List[str]: 可用于 intro 的变量名列表
        @author ygw 2026-03-01
        """
        names = []
        for line in state_before.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('⊢'):
                break
            # 解析 "name : type" 格式
            if ' : ' in line:
                name_part = line.split(' : ', 1)[0].strip()
                # 跳过含有 ✝ 的不可访问变量和含有 x✝ 的自动命名
                if '✝' in name_part:
                    continue
                # 验证名字是合法标识符（字母、数字、下划线、引号）
                if name_part and not name_part[0].isdigit():
                    names.append(name_part)
        return names

    def run(self) -> str:
        """
        执行错误验证流程

        返回:
            str: 输出文件路径
        """
        logger.info("=" * 60)
        logger.info("W8-B 错误验证 (Error Verification via Pantograph) 开始")
        logger.info("=" * 60)
        start_time = time.time()

        # 1. 加载错误注入数据
        error_file = os.path.join(self.error_input_path, "error_injection.jsonl")
        if not os.path.exists(error_file):
            logger.error(f"错误注入文件不存在: {error_file}")
            return ""
        error_records = load_jsonl(error_file)
        self.stats["total_errors"] = len(error_records)
        logger.info(f"加载错误注入数据: {len(error_records)} 条")

        # 2. 加载 cos_flat 数据，构建定理 → 步骤映射
        cos_file = os.path.join(self.cos_input_path, "cos_flat.jsonl")
        if not os.path.exists(cos_file):
            logger.error(f"CoS 数据文件不存在: {cos_file}")
            return ""
        theorem_steps = self._build_theorem_steps_map(cos_file)
        self.stats["unique_theorems"] = len(theorem_steps)
        logger.info(f"构建定理步骤映射: {len(theorem_steps)} 个定理")

        # 2.5 构建短名 → 全名的反向映射（兼容旧版 error_injection.jsonl 只有 theorem_name）
        short_to_full = {}
        for full_name in theorem_steps:
            # 短名 = 全名最后一段（如 "Nat.add_comm" → "add_comm"）
            short_name = full_name.rsplit(".", 1)[-1] if "." in full_name else full_name
            # 如果短名有冲突，保留第一个（不完美但可接受）
            if short_name not in short_to_full:
                short_to_full[short_name] = full_name
        logger.info(f"短名→全名映射: {len(short_to_full)} 条")

        # 3. 按定理分组错误记录
        grouped_errors = self._group_by_theorem(error_records)
        logger.info(f"错误记录涉及 {len(grouped_errors)} 个定理（分组后）")

        # 3.5 修正分组键：将短名映射到全名，使其与 theorem_steps 的键一致
        remapped_errors = {}
        unmatched = 0
        for name, records in grouped_errors.items():
            if name in theorem_steps:
                # 已经是全名，直接使用
                remapped_errors.setdefault(name, []).extend(records)
            elif name in short_to_full:
                # 短名，映射到全名
                full_name = short_to_full[name]
                remapped_errors.setdefault(full_name, []).extend(records)
            else:
                # 无法匹配
                unmatched += len(records)
                self.stats["skipped"] += len(records)
        grouped_errors = remapped_errors
        if unmatched > 0:
            logger.warning(f"无法匹配到 cos_flat 的错误记录: {unmatched} 条")
        logger.info(f"名称修正后涉及 {len(grouped_errors)} 个定理")

        # 4. 启动 Pantograph 服务器，并发验证
        # 策略：每个工作线程持有一个独立的 LeanServer 实例，串行处理分配给它的定理
        # 避免频繁 acquire/release 导致的池耗尽问题
        output_file = os.path.join(self.output_path, "error_injection_verified.jsonl")
        if os.path.exists(output_file):
            os.remove(output_file)

        verified_records = []
        processed_theorems = 0
        total_theorems = len(grouped_errors)

        # 将定理列表均匀分片，每个工作线程处理一个分片
        theorem_items = list(grouped_errors.items())
        num_workers = min(self.max_concurrent, len(theorem_items))
        if num_workers == 0:
            logger.warning("无定理需要验证")
            return output_file

        # 分片：将定理列表切分为 num_workers 份
        chunks = [[] for _ in range(num_workers)]
        for i, item in enumerate(theorem_items):
            chunks[i % num_workers].append(item)

        logger.info(f"分配 {len(theorem_items)} 个定理到 {num_workers} 个工作线程")

        # 线程安全的结果收集和进度统计
        results_lock = threading.Lock()

        def worker_fn(worker_id: int, chunk: List):
            """
            工作线程：启动独立的 LeanServer，串行处理分配的定理

            参数:
                worker_id: 工作线程编号
                chunk: 分配给该线程的 (theorem_name, records) 列表
            """
            nonlocal processed_theorems, verified_records

            # v3: 错峰启动 — 避免多个 Lean4 进程同时争抢资源导致启动失败
            # 每个 worker 延迟 worker_id * 20 秒再启动
            # @author ygw 2026-03-01
            if worker_id > 0:
                stagger_delay = worker_id * 20
                logger.info(f"工作线程 {worker_id}: 错峰启动，等待 {stagger_delay}s")
                time.sleep(stagger_delay)

            # v3: 重试机制 — 启动失败时最多重试 3 次，每次间隔 15s
            max_retries = 3
            server = None
            for attempt in range(max_retries):
                server = LeanServer(
                    executable_path=self.executable_path,
                    timeout=self.pantograph_timeout,
                    project_path=self.project_path,
                    imports=["Mathlib"],
                )
                if server.start():
                    logger.info(f"工作线程 {worker_id}: LeanServer 启动成功 (尝试 {attempt + 1}/{max_retries})")
                    break
                logger.warning(f"工作线程 {worker_id}: LeanServer 启动失败 (尝试 {attempt + 1}/{max_retries})")
                server = None
                if attempt < max_retries - 1:
                    time.sleep(15)  # 等待 15s 后重试

            if server is None:
                logger.error(f"工作线程 {worker_id}: LeanServer {max_retries} 次启动均失败，放弃")
                # 标记所有记录为服务器错误
                with results_lock:
                    for theorem_name, records in chunk:
                        for rec in records:
                            rec["error_message"] = ""
                            rec["verification_status"] = "server_start_failed"
                        verified_records.extend(records)
                        self.stats["pantograph_error"] += len(records)
                        processed_theorems += 1
                return

            try:
                for theorem_name, records in chunk:
                    steps = theorem_steps.get(theorem_name, [])
                    try:
                        # 检查服务器是否仍然存活
                        if not server.is_running():
                            logger.warning(f"工作线程 {worker_id}: 服务器已崩溃，尝试重启")
                            # v3: 崩溃重启也使用重试机制
                            restarted = False
                            for retry in range(3):
                                if server.start():
                                    logger.info(f"工作线程 {worker_id}: 服务器重启成功 (尝试 {retry + 1}/3)")
                                    restarted = True
                                    break
                                time.sleep(10)
                            if not restarted:
                                logger.error(f"工作线程 {worker_id}: 服务器重启失败")
                                # 标记剩余记录为错误
                                for rec in records:
                                    rec["error_message"] = ""
                                    rec["verification_status"] = "server_crashed"
                                with results_lock:
                                    verified_records.extend(records)
                                    self.stats["pantograph_error"] += len(records)
                                    processed_theorems += 1
                                continue

                        results = self._verify_theorem_errors_with_server(
                            server, theorem_name, records, steps
                        )
                    except Exception as e:
                        logger.warning(f"工作线程 {worker_id}: 定理 {theorem_name} 异常: {e}")
                        results = records
                        for rec in results:
                            rec["error_message"] = ""
                            rec["verification_status"] = "server_error"
                        self.stats["pantograph_error"] += len(records)

                    with results_lock:
                        verified_records.extend(results)
                        processed_theorems += 1

                        # 定期保存
                        if len(verified_records) >= self.batch_save_interval:
                            save_jsonl(verified_records, output_file, mode='a')
                            verified_records.clear()

                        # 进度输出
                        if processed_theorems % 100 == 0 or processed_theorems == total_theorems:
                            elapsed = time.time() - start_time
                            speed = processed_theorems / elapsed if elapsed > 0 else 0
                            print(
                                f"[W8-B] 进度: {processed_theorems}/{total_theorems} 定理 "
                                f"({processed_theorems/total_theorems*100:.1f}%) | "
                                f"已验证: {self.stats['verified']} | "
                                f"错误策略成功: {self.stats.get('error_tactic_succeeded', 0)} | "
                                f"重放失败: {self.stats['replay_failed']} | "
                                f"goal_start失败: {self.stats.get('goal_start_failed', 0)} | "
                                f"速度: {speed:.1f} 定理/s",
                                flush=True,
                            )
            finally:
                server.stop()

        # 启动工作线程
        threads = []
        for i, chunk in enumerate(chunks):
            if chunk:  # 跳过空分片
                t = threading.Thread(target=worker_fn, args=(i, chunk), daemon=True)
                threads.append(t)
                t.start()

        # 等待所有工作线程完成
        for t in threads:
            t.join()

        # 保存剩余记录
        if verified_records:
            save_jsonl(verified_records, output_file, mode='a')

        # 5. 保存统计
        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["timestamp"] = get_timestamp()

        stats_file = os.path.join(self.output_path, "error_verification_stats.json")
        save_json(self.stats, stats_file)

        logger.info("=" * 60)
        logger.info("W8-B 错误验证完成")
        logger.info(f"总错误记录: {self.stats['total_errors']}")
        logger.info(f"成功验证: {self.stats['verified']}")
        logger.info(f"错误策略意外成功: {self.stats.get('error_tactic_succeeded', 0)}")
        logger.info(f"重放失败: {self.stats['replay_failed']}")
        logger.info(f"goal_start失败: {self.stats.get('goal_start_failed', 0)}")
        logger.info(f"Pantograph 错误: {self.stats['pantograph_error']}")
        logger.info(f"跳过: {self.stats['skipped']}")
        logger.info(f"intro跳过(弹性重放): {self.stats.get('intro_skipped_in_replay', 0)}")
        logger.info(f"紧急intros恢复: {self.stats.get('emergency_intros', 0)}")
        logger.info(f"auto_intro应用: {self.stats.get('auto_intro_applied', 0)}")
        logger.info(f"耗时: {elapsed:.1f}s")
        logger.info("=" * 60)

        return output_file

    def _build_theorem_steps_map(self, cos_file: str) -> Dict[str, List[Dict]]:
        """
        构建定理 → 有序步骤列表的映射。

        优先使用 cos_dataset.jsonl（每条记录含 full_cos_chain 列表），
        回退到 cos_flat.jsonl（展平格式）。

        参数:
            cos_file: cos_flat.jsonl 文件路径（用于推导 cos_dataset.jsonl 路径）

        返回:
            Dict[str, List[Dict]]: theorem_full_name → 按 step_index 排序的步骤列表
        @author ygw
        """
        # 优先尝试 cos_dataset.jsonl
        dataset_file = os.path.join(os.path.dirname(cos_file), "cos_dataset.jsonl")
        if os.path.exists(dataset_file):
            return self._build_steps_from_dataset(dataset_file)

        # 回退到 cos_flat.jsonl
        logger.info("cos_dataset.jsonl 不存在，回退到 cos_flat.jsonl")
        return self._build_steps_from_flat(cos_file)

    def _build_steps_from_dataset(self, dataset_file: str) -> Dict[str, List[Dict]]:
        """
        从 cos_dataset.jsonl 构建步骤映射。
        每条记录的 full_cos_chain 已经是有序的原子步骤列表。

        参数:
            dataset_file: cos_dataset.jsonl 文件路径

        返回:
            Dict[str, List[Dict]]: theorem_full_name → 步骤列表
        @author ygw
        """
        theorem_steps = {}
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = rec.get("theorem_full_name", "")
                if not name:
                    continue
                chain = rec.get("full_cos_chain", [])
                # 转换为与 cos_flat 兼容的格式，添加 step_index
                steps = []
                for idx, step in enumerate(chain):
                    steps.append({
                        "theorem_full_name": name,
                        "step_index": idx,
                        "tactic": step.get("tactic", ""),
                        "state_before": step.get("state_before", ""),
                        "state_after": step.get("state_after", ""),
                    })
                theorem_steps[name] = steps
        logger.info(f"从 cos_dataset.jsonl 加载 {len(theorem_steps)} 个定理")
        return theorem_steps

    def _build_steps_from_flat(self, cos_file: str) -> Dict[str, List[Dict]]:
        """
        从 cos_flat.jsonl 构建步骤映射（原有逻辑）。

        参数:
            cos_file: cos_flat.jsonl 文件路径

        返回:
            Dict[str, List[Dict]]: theorem_full_name → 按 step_index 排序的步骤列表
        @author ygw
        """
        theorem_steps = defaultdict(list)
        with open(cos_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = rec.get("theorem_full_name", "")
                if name:
                    theorem_steps[name].append(rec)

        # 按 step_index 排序
        for name in theorem_steps:
            theorem_steps[name].sort(key=lambda x: x.get("step_index", 0))

        return dict(theorem_steps)

    def _group_by_theorem(self, error_records: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按定理全名分组错误记录

        优先使用 theorem_full_name（与 cos_flat.jsonl 一致），
        回退到 theorem_name。

        参数:
            error_records: 错误注入记录列表

        返回:
            Dict[str, List[Dict]]: theorem_full_name → 错误记录列表
        """
        grouped = defaultdict(list)
        for rec in error_records:
            # 优先使用 theorem_full_name，与 cos_flat.jsonl 的键一致
            name = rec.get("theorem_full_name", "") or rec.get("theorem_name", "")
            if name:
                grouped[name].append(rec)
            else:
                self.stats["skipped"] += 1
        return dict(grouped)

    def _verify_theorem_errors_with_server(self, server: LeanServer,
                                            theorem_name: str,
                                            error_records: List[Dict],
                                            steps: List[Dict]) -> List[Dict]:
        """
        使用指定的 LeanServer 实例验证同一定理下的所有错误记录

        优化策略：按 target_step 分组，同一 step 的错误记录共享前置重放，
        避免重复 goal_start + 重放前置策略。每条记录独立 try/except，
        一条失败不影响其他记录。

        参数:
            server: Pantograph 服务器实例（由工作线程持有）
            theorem_name: 定理全名
            error_records: 该定理的错误注入记录
            steps: 该定理的完整步骤序列（按 step_index 排序）

        返回:
            List[Dict]: 验证后的错误记录（含 error_message）
        """
        if not steps:
            for rec in error_records:
                rec["error_message"] = ""
                rec["verification_status"] = "no_steps"
            self.stats["skipped"] += len(error_records)
            return error_records

        # 构建 step_index → tactic 映射
        tactic_by_step = {}
        for s in steps:
            idx = s.get("step_index", -1)
            tactic_by_step[idx] = s.get("tactic", "")

        # 构建 (original_tactic, state_before 前 100 字符) → step_index 映射（回退用）
        step_lookup = {}
        for s in steps:
            key = (s.get("tactic", ""), s.get("state_before", "")[:100])
            step_lookup[key] = s.get("step_index", -1)

        # 为每条记录解析 target_step，按 step 分组
        step_to_records = defaultdict(list)
        results = []
        for rec in error_records:
            target_step = rec.get("step_index", -1)
            if not (isinstance(target_step, int) and target_step >= 0):
                # 回退：字符串匹配
                orig = rec.get("original_tactic", "")
                prefix = rec.get("state_before", "")[:100]
                target_step = step_lookup.get((orig, prefix), -1)

            if target_step < 0:
                rec["error_message"] = ""
                rec["verification_status"] = "step_not_found"
                self.stats["skipped"] += 1
                results.append(rec)
            else:
                rec["_target_step"] = target_step
                step_to_records[target_step].append(rec)

        # 收集候选名称：theorem_name（分组键）+ 记录中的 theorem_name（短名）
        # 同时加入双前缀修正后的名称作为候选 @author ygw
        candidate_names = [theorem_name]
        # 尝试双前缀修正
        fixed_name = self._fix_double_prefix(theorem_name)
        if fixed_name and fixed_name not in candidate_names:
            candidate_names.append(fixed_name)
        for rec in error_records:
            tn = rec.get("theorem_name", "")
            if tn and tn not in candidate_names:
                candidate_names.append(tn)
            # 对记录中的名称也尝试双前缀修正
            if tn:
                tn_fixed = self._fix_double_prefix(tn)
                if tn_fixed and tn_fixed not in candidate_names:
                    candidate_names.append(tn_fixed)

        # 按 target_step 升序处理，每组共享一次 goal_start + 前置重放
        for target_step in sorted(step_to_records.keys()):
            recs_at_step = step_to_records[target_step]
            try:
                # 1. 创建证明目标（多名称回退）
                # v2: goal_start_with_fallback 现在返回 (state, needs_intro)
                # needs_intro=True 表示用了 goal_start(expr) 路径，变量在 ∀ 中
                # needs_intro=False 表示用了 copyFrom 路径，变量已在上下文中
                # @author ygw 2026-03-01
                initial_state = None
                needs_intro = False
                for name_candidate in candidate_names:
                    result = server.goal_start_with_fallback(name_candidate)
                    if isinstance(result, tuple):
                        initial_state, needs_intro = result
                    else:
                        initial_state = result
                        needs_intro = False
                    if initial_state is not None:
                        break
                if initial_state is None:
                    for rec in recs_at_step:
                        rec["error_message"] = ""
                        rec["verification_status"] = "goal_start_failed"
                        rec.pop("_target_step", None)
                    self.stats.setdefault("goal_start_failed", 0)
                    self.stats["goal_start_failed"] += len(recs_at_step)
                    self.stats["replay_failed"] += len(recs_at_step)
                    results.extend(recs_at_step)
                    continue

                current_state_id = initial_state.state_id

                # 2.0 条件自动引入（Conditional Auto-intro）
                # 仅当 goal_start_with_fallback 使用了 goal_start(expr) 路径时才需要
                # copyFrom 路径已自动将变量放入上下文，intros 会破坏状态
                # @author ygw 2026-03-01 v2: 修复 v1 导致的 pantograph_error 爆炸
                if needs_intro and target_step > 0 and steps:
                    step0_state = steps[0].get("state_before", "")
                    if step0_state.strip() and not step0_state.strip().startswith("⊢"):
                        try:
                            intro_result = server.goal_tactic(
                                current_state_id, "intros"
                            )
                            if intro_result.success and intro_result.state_after:
                                current_state_id = intro_result.state_after.state_id
                                self.stats.setdefault("auto_intro_applied", 0)
                                self.stats["auto_intro_applied"] += 1
                            else:
                                # intros 失败，尝试具名 intro
                                var_names = self._extract_intro_names(step0_state)
                                if var_names:
                                    named_intro = "intro " + " ".join(var_names)
                                    named_result = server.goal_tactic(
                                        current_state_id, named_intro
                                    )
                                    if (named_result.success
                                            and named_result.state_after):
                                        current_state_id = (
                                            named_result.state_after.state_id
                                        )
                                        self.stats.setdefault(
                                            "auto_intro_named", 0
                                        )
                                        self.stats["auto_intro_named"] += 1
                        except Exception as e:
                            # auto-intro 异常不应影响后续验证
                            logger.debug(
                                f"auto-intro 异常: {theorem_name}: {e}"
                            )

                # 2. 重放前置策略（step 0 到 target_step - 1）
                #    v4: 弹性重放 — intro-like 步骤失败时自动跳过
                #    根因: copyFrom/auto_intro 已引入变量，原始 intro 步骤冗余
                #    @author ygw 2026-03-01
                replay_ok = True
                failed_step = -1
                emergency_intros_done = False  # 是否已执行过紧急 intros
                for step_idx in range(target_step):
                    tactic = tactic_by_step.get(step_idx, "")
                    if not tactic:
                        replay_ok = False
                        failed_step = step_idx
                        break
                    # 拆分多行 tactic 为原子 tactic 列表
                    sub_tactics = self._split_tactic_block(tactic)
                    sub_ok = True
                    for sub_tac in sub_tactics:
                        tac_result = server.goal_tactic(current_state_id, sub_tac)
                        if not tac_result.success:
                            sub_ok = False
                            break
                        if tac_result.state_after:
                            current_state_id = tac_result.state_after.state_id
                    if not sub_ok:
                        # 拆分后仍失败，尝试整体发送（兼容不可拆分的情况）
                        if len(sub_tactics) > 1:
                            tac_result = server.goal_tactic(
                                current_state_id, tactic
                            )
                            if tac_result.success and tac_result.state_after:
                                current_state_id = (
                                    tac_result.state_after.state_id
                                )
                                sub_ok = True

                    if not sub_ok:
                        # v4 弹性恢复策略
                        tactic_stripped = tactic.strip()
                        tactic_lower = tactic_stripped.lower()
                        is_intro_like = (
                            tactic_lower == "intro"
                            or tactic_lower == "intros"
                            or tactic_lower.startswith("intro ")
                            or tactic_lower.startswith("intros ")
                            or tactic_lower.startswith("rintro ")
                        )

                        if is_intro_like:
                            # 策略A: intro-like 步骤失败 → 跳过
                            # 原因: copyFrom/auto_intro 已将变量引入上下文
                            self.stats.setdefault(
                                "intro_skipped_in_replay", 0
                            )
                            self.stats["intro_skipped_in_replay"] += 1
                            continue  # 跳过此步，保持当前 state_id

                        if step_idx == 0 and not emergency_intros_done:
                            # 策略B: 非 intro 的 step 0 失败 → 尝试紧急 intros
                            # 可能目标含 ∀ 但 copyFrom 未自动引入变量
                            emergency_intros_done = True
                            try:
                                emer_result = server.goal_tactic(
                                    current_state_id, "intros"
                                )
                                if (emer_result.success
                                        and emer_result.state_after):
                                    current_state_id = (
                                        emer_result.state_after.state_id
                                    )
                                    self.stats.setdefault(
                                        "emergency_intros", 0
                                    )
                                    self.stats["emergency_intros"] += 1
                                    # 重试原始 tactic
                                    retry = server.goal_tactic(
                                        current_state_id, tactic
                                    )
                                    if retry.success and retry.state_after:
                                        current_state_id = (
                                            retry.state_after.state_id
                                        )
                                        continue  # 恢复成功
                            except Exception:
                                pass

                        # 所有恢复策略均失败
                        replay_ok = False
                        failed_step = step_idx
                        break

                if not replay_ok:
                    # 前置重放失败，标记该 step 的所有记录
                    for rec in recs_at_step:
                        rec["error_message"] = ""
                        rec["verification_status"] = f"replay_failed_at_step_{failed_step}"
                        rec.pop("_target_step", None)
                    self.stats["replay_failed"] += len(recs_at_step)
                    results.extend(recs_at_step)
                    # 清理状态
                    try:
                        server.goal_delete(initial_state.state_id)
                    except Exception:
                        pass
                    continue

                # 3. 前置重放成功，逐条执行错误策略
                for rec in recs_at_step:
                    try:
                        error_tactic = rec.get("error_tactic", "")
                        error_result = server.goal_tactic(current_state_id, error_tactic)

                        if error_result.success:
                            rec["error_message"] = ""
                            rec["verification_status"] = "error_tactic_succeeded"
                            self.stats.setdefault("error_tactic_succeeded", 0)
                            self.stats["error_tactic_succeeded"] += 1
                        else:
                            rec["error_message"] = error_result.error_message
                            rec["verification_status"] = "verified"
                            self.stats["verified"] += 1
                    except Exception as e:
                        rec["error_message"] = ""
                        rec["verification_status"] = f"exception: {str(e)[:100]}"
                        self.stats["pantograph_error"] += 1
                    finally:
                        rec.pop("_target_step", None)
                    results.append(rec)

                # 4. 清理状态
                try:
                    server.goal_delete(initial_state.state_id)
                except Exception:
                    pass

            except Exception as e:
                # 整个 step 组的异常（如服务器崩溃）
                logger.debug(f"定理 {theorem_name} step {target_step} 异常: {e}")
                for rec in recs_at_step:
                    rec["error_message"] = ""
                    rec["verification_status"] = f"exception: {str(e)[:100]}"
                    rec.pop("_target_step", None)
                self.stats["pantograph_error"] += len(recs_at_step)
                results.extend(recs_at_step)

        return results


# ================================================================
# 命令行入口
# ================================================================

def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="W8-B 错误验证 (Pantograph)")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml",
                        help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="覆盖配置中的并发数")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/error_verifier.log", encoding="utf-8"),
        ],
    )

    config = load_yaml(args.config)

    # 命令行参数覆盖
    if args.max_concurrent is not None:
        config.setdefault("augmentation", {}).setdefault(
            "error_verification", {}
        )["max_concurrent"] = args.max_concurrent

    verifier = ErrorVerifier(config)
    output_file = verifier.run()
    print(f"\n错误验证完成！输出文件: {output_file}")


if __name__ == "__main__":
    main()
