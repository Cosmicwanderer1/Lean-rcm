"""
通用工具函数
@author ygw
更新日期: 2026-02-06

提供 W5-W8 数据流水线所需的通用工具函数，包括：
文件 I/O、日志配置、进度追踪、断点续传、数据校验等。
"""

import os
import sys
import json
import yaml
import time
import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Iterator, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


# ================================================================
# 文件 I/O 工具
# ================================================================

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    参数:
        file_path: YAML 文件路径

    返回:
        Dict: 配置字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    保存数据为 JSON 文件

    参数:
        data: 要保存的数据
        file_path: 保存路径
        indent: 缩进空格数
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    加载 JSON 文件

    参数:
        file_path: JSON 文件路径

    返回:
        Any: 加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data_list: List[Dict], file_path: str, mode: str = 'w'):
    """
    保存数据为 JSONL 文件（每行一个 JSON 对象）

    参数:
        data_list: 数据字典列表
        file_path: 保存路径
        mode: 写入模式，'w' 覆盖，'a' 追加
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载 JSONL 文件

    参数:
        file_path: JSONL 文件路径

    返回:
        List[Dict]: 数据字典列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def iter_jsonl(file_path: str) -> Iterator[Dict]:
    """
    迭代读取 JSONL 文件（内存友好，适合大文件）

    参数:
        file_path: JSONL 文件路径

    返回:
        Iterator[Dict]: 数据字典迭代器
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dir(dir_path: str):
    """
    确保目录存在，不存在则创建

    参数:
        dir_path: 目录路径
    """
    if dir_path:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> str:
    """
    获取项目根目录

    返回:
        str: 项目根目录路径
    """
    return str(Path(__file__).parent.parent.parent)


def count_lines(file_path: str) -> int:
    """
    统计文件行数

    参数:
        file_path: 文件路径

    返回:
        int: 行数
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


# ================================================================
# 日志配置
# ================================================================

def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None,
                  module_name: str = "rtap") -> logging.Logger:
    """
    配置日志系统

    参数:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        log_file: 日志文件路径（可选，不指定则仅输出到控制台）
        module_name: 模块名称

    返回:
        Logger: 配置好的日志器
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除已有的 handler，避免重复
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ================================================================
# 进度追踪
# ================================================================

class ProgressTracker:
    """
    进度追踪器，用于长时间运行的任务

    使用示例:
        tracker = ProgressTracker(total=1000, desc="提取轨迹")
        for item in data:
            process(item)
            tracker.update()
        tracker.finish()
    """

    def __init__(self, total: int, desc: str = "处理中",
                 log_interval: int = 100):
        """
        初始化进度追踪器

        参数:
            total: 总任务数
            desc: 任务描述
            log_interval: 每处理 N 个任务输出一次日志
        """
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time = time.time()
        self.logger = logging.getLogger("progress")

    def update(self, success: bool = True):
        """
        更新进度

        参数:
            success: 当前任务是否成功
        """
        self.current += 1
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            speed = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / speed if speed > 0 else 0

            self.logger.info(
                f"[{self.desc}] {self.current}/{self.total} "
                f"({self.current * 100 / self.total:.1f}%) | "
                f"成功: {self.success_count} | 失败: {self.fail_count} | "
                f"速度: {speed:.1f}/s | 剩余: {remaining:.0f}s"
            )

    def finish(self) -> Dict[str, Any]:
        """
        完成追踪，返回统计信息

        返回:
            Dict: 统计信息
        """
        elapsed = time.time() - self.start_time
        stats = {
            "desc": self.desc,
            "total": self.total,
            "processed": self.current,
            "success": self.success_count,
            "failed": self.fail_count,
            "success_rate": self.success_count / max(self.current, 1),
            "elapsed_seconds": round(elapsed, 2),
            "speed_per_second": round(self.current / max(elapsed, 0.001), 2),
        }
        self.logger.info(
            f"[{self.desc}] 完成! "
            f"总计: {self.current}/{self.total} | "
            f"成功率: {stats['success_rate']:.1%} | "
            f"耗时: {elapsed:.1f}s"
        )
        return stats


# ================================================================
# 断点续传
# ================================================================

class CheckpointManager:
    """
    断点续传管理器，支持长时间任务的中断恢复

    使用示例:
        ckpt = CheckpointManager("traces/.checkpoint")
        processed = ckpt.load()  # 加载已处理的文件集合
        for file in all_files:
            if file in processed:
                continue
            process(file)
            ckpt.mark_done(file)
    """

    def __init__(self, checkpoint_path: str, save_interval: int = 100):
        """
        初始化断点管理器

        参数:
            checkpoint_path: 检查点文件路径
            save_interval: 每处理 N 个任务自动保存
        """
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self._processed: set = set()
        self._counter = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        """加载已有的检查点"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._processed.add(line)
            logging.getLogger("checkpoint").info(
                f"加载检查点: {len(self._processed)} 个已处理项"
            )

    def is_done(self, item_id: str) -> bool:
        """
        检查某项是否已处理

        参数:
            item_id: 项目标识

        返回:
            bool: 是否已处理
        """
        return item_id in self._processed

    def mark_done(self, item_id: str):
        """
        标记某项为已处理

        参数:
            item_id: 项目标识
        """
        self._processed.add(item_id)
        self._counter += 1

        # 定期保存
        if self._counter % self.save_interval == 0:
            self.save()

    def save(self):
        """保存检查点到文件"""
        ensure_dir(os.path.dirname(self.checkpoint_path))
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            for item_id in sorted(self._processed):
                f.write(item_id + '\n')

    def load(self) -> set:
        """
        获取已处理项集合

        返回:
            set: 已处理项的集合
        """
        return self._processed.copy()

    @property
    def processed_count(self) -> int:
        """已处理项数量"""
        return len(self._processed)


# ================================================================
# 数据校验与哈希
# ================================================================

def compute_hash(text: str) -> str:
    """
    计算文本的 SHA256 哈希值

    参数:
        text: 输入文本

    返回:
        str: 哈希值（16进制字符串）
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def deduplicate_by_key(data_list: List[Dict], key: str) -> List[Dict]:
    """
    按指定键去重

    参数:
        data_list: 数据列表
        key: 去重键名

    返回:
        List[Dict]: 去重后的数据列表
    """
    seen = set()
    result = []
    for item in data_list:
        val = item.get(key, "")
        if val not in seen:
            seen.add(val)
            result.append(item)
    return result


def deduplicate_by_hash(data_list: List[Dict],
                        hash_fields: List[str]) -> List[Dict]:
    """
    按多字段组合哈希去重

    参数:
        data_list: 数据列表
        hash_fields: 用于计算哈希的字段列表

    返回:
        List[Dict]: 去重后的数据列表
    """
    seen = set()
    result = []
    for item in data_list:
        combined = "|".join(str(item.get(f, "")) for f in hash_fields)
        h = compute_hash(combined)
        if h not in seen:
            seen.add(h)
            result.append(item)
    return result


# ================================================================
# 随机种子与可复现性
# ================================================================

def set_seed(seed: int = 42):
    """
    设置全局随机种子，确保可复现

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ================================================================
# 时间戳工具
# ================================================================

def get_timestamp() -> str:
    """
    获取当前时间戳字符串

    返回:
        str: 格式为 yyyy-MM-dd HH:mm:ss 的时间戳
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_date_str() -> str:
    """
    获取当前日期字符串

    返回:
        str: 格式为 yyyyMMdd 的日期
    """
    return datetime.now().strftime('%Y%m%d')


# ================================================================
# 批处理工具
# ================================================================

def batch_iter(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    将列表分批迭代

    参数:
        data: 数据列表
        batch_size: 每批大小

    返回:
        Iterator: 批次迭代器
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def retry_with_backoff(func: Callable,
                       max_retries: int = 3,
                       base_delay: float = 1.0,
                       max_delay: float = 60.0,
                       exceptions: tuple = (Exception,)) -> Any:
    """
    带指数退避的重试机制

    参数:
        func: 要执行的函数（无参数）
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exceptions: 需要捕获的异常类型

    返回:
        Any: 函数返回值
    """
    logger = logging.getLogger("retry")

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                logger.error(f"重试 {max_retries} 次后仍然失败: {e}")
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            # 添加随机抖动
            delay = delay * (0.5 + random.random())
            logger.warning(
                f"第 {attempt + 1} 次尝试失败: {e}, "
                f"{delay:.1f}秒后重试..."
            )
            time.sleep(delay)
