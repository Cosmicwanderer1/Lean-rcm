#!/usr/bin/env python3
"""
检索增强训练数据构建器 (Task 2.2)
@author ygw
创建日期: 2026-03-01

从已有的 CoS 数据中提取证明状态→前提定理的正样本对，
利用 FAISS 索引挖掘硬负样本，构建对比学习三元组训练集。

三阶段流水线:
  Phase 1: 从 cos_flat.jsonl 解析 tactic 中引用的前提名称，
           与 corpus 交叉匹配构建正样本对 (state, premise_pos)
  Phase 2: 用 FAISS 索引为每个 state 检索 top-K 近邻，
           排除正样本后作为硬负样本 (hard negatives)
  Phase 3: 合并为最终对比学习训练集，train/val/test 分割

用法:
    python scripts/build_retrieval_training_data.py
    python scripts/build_retrieval_training_data.py --phase 1
    python scripts/build_retrieval_training_data.py --phase 2 --num-negatives 7
    python scripts/build_retrieval_training_data.py --config configs/retrieval_training.yaml

技术产出:
    - data/retrieval_training/positive_pairs.jsonl
    - data/retrieval_training/training_triplets.jsonl
    - data/retrieval_training/train.jsonl / val.jsonl / test.jsonl
    - data/retrieval_training/stats.json
"""

import os
import sys
import re
import json
import time
import random
import logging
import argparse
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.utils import (
    load_yaml, save_jsonl, load_jsonl, ensure_dir,
    setup_logging, get_timestamp, set_seed
)

# 使用根 logger 确保输出可见（setup_logging 配置的是 "rtap" 命名空间）
logger = logging.getLogger("rtap.retrieval_training")


# ================================================================
# Phase 1: 正样本对提取
# ================================================================

class TacticPremiseParser:
    """
    从 Lean4 Tactic 文本中解析引用的前提（引理/定理）名称

    支持的 Tactic 模式:
        - exact / exact? NAME
        - apply / apply? NAME
        - rw / rewrite [NAME, ←NAME, ...]
        - simp [NAME, ...] / simp only [NAME, ...]
        - refine / refine' NAME
        - have := NAME / have : T := NAME
        - calc ... NAME
        - linarith [NAME, ...] / nlinarith [NAME, ...]
        - norm_num [NAME, ...]
        - cases NAME with ... (变量名不算前提)
        - induction NAME (变量名不算前提)
        - constructor / intro / ext / ring / omega / decide (无前提引用)

    @author ygw
    """

    # Lean4 合法标识符模式（含模块路径前缀如 Nat.add_comm, @Nat.add_comm）
    _IDENT_PATTERN = r'@?[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'

    # 策略中引用前提的正则模式
    _PATTERNS = [
        # exact / exact? NAME_EXPR
        (r'(?:exact|Exact\?)\s+(' + _IDENT_PATTERN + r')', 'single'),
        # apply / apply? NAME_EXPR
        (r'(?:apply)\s+(' + _IDENT_PATTERN + r')', 'single'),
        # refine / refine' NAME_EXPR
        (r'(?:refine\'?)\s+(' + _IDENT_PATTERN + r')', 'single'),
        # rw / rewrite [LIST]
        (r'(?:rw|rewrite)\s+\[([^\]]+)\]', 'bracket_list'),
        # simp [LIST] / simp only [LIST]
        (r'simp\s+(?:only\s+)?\[([^\]]+)\]', 'bracket_list'),
        # linarith [LIST] / nlinarith [LIST]
        (r'(?:n?linarith)\s+\[([^\]]+)\]', 'bracket_list'),
        # norm_num [LIST]
        (r'norm_num\s+\[([^\]]+)\]', 'bracket_list'),
        # have ... := NAME
        (r'have\s+[^:=]*:=\s+(' + _IDENT_PATTERN + r')', 'single'),
        # obtain ... using NAME
        (r'(?:obtain|rcases)\s+.*(?:using|with)\s+(' + _IDENT_PATTERN + r')', 'single'),
        # convert NAME / convert ← NAME
        (r'convert\s+(?:←\s*)?(' + _IDENT_PATTERN + r')', 'single'),
        # trans NAME
        (r'trans\s+(' + _IDENT_PATTERN + r')', 'single'),
        # gcongr ... with NAME
        (r'gcongr\s+(?:.*\s+)?(' + _IDENT_PATTERN + r')', 'single'),
    ]

    # 不应被视为前提的关键词/变量名
    _EXCLUDED_NAMES = {
        # Lean4 关键词
        'this', 'self', 'fun', 'by', 'with', 'at', 'using', 'True', 'False',
        'And', 'Or', 'Not', 'Iff', 'Exists', 'show', 'from', 'have', 'let',
        'match', 'do', 'return', 'where', 'if', 'then', 'else', 'for', 'in',
        'Type', 'Prop', 'Sort', 'Set',
        # 常见变量名模板
        'H', 'H1', 'H2', 'H3', 'IH', 'IHn', 'IHm',
    }

    # 编译正则
    _compiled = [(re.compile(p, re.IGNORECASE), mode) for p, mode in _PATTERNS]

    @classmethod
    def parse(cls, tactic: str) -> List[str]:
        """
        解析 tactic 字符串中引用的所有前提名称

        参数:
            tactic: Lean4 策略文本

        返回:
            List[str]: 前提全限定名列表（已去重、去排除项）
        """
        premises: Set[str] = set()

        for pattern, mode in cls._compiled:
            for match in pattern.finditer(tactic):
                if mode == 'single':
                    name = match.group(1).lstrip('@')
                    if cls._is_valid_premise(name):
                        premises.add(name)
                elif mode == 'bracket_list':
                    # 解析方括号内的逗号分隔列表
                    list_text = match.group(1)
                    for item in cls._split_bracket_list(list_text):
                        name = item.lstrip('←').lstrip('-').strip().lstrip('@')
                        # 提取纯标识符部分（可能带参数）
                        ident_match = re.match(
                            r'(' + cls._IDENT_PATTERN.lstrip('@?') + r')',
                            name
                        )
                        if ident_match:
                            clean_name = ident_match.group(1)
                            if cls._is_valid_premise(clean_name):
                                premises.add(clean_name)

        return sorted(premises)

    @classmethod
    def _split_bracket_list(cls, text: str) -> List[str]:
        """
        分割方括号内的项（处理嵌套括号）

        参数:
            text: 方括号内的文本

        返回:
            List[str]: 各项文本
        """
        items = []
        depth = 0
        current = ""
        for ch in text:
            if ch in '([{':
                depth += 1
                current += ch
            elif ch in ')]}':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                if current.strip():
                    items.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            items.append(current.strip())
        return items

    @classmethod
    def _is_valid_premise(cls, name: str) -> bool:
        """
        判断名称是否为合法前提（非关键词、非纯小写变量、含模块路径前缀）

        参数:
            name: 候选前提名称

        返回:
            bool: 是否合法
        """
        if not name or len(name) < 2:
            return False
        if name in cls._EXCLUDED_NAMES:
            return False
        # 纯小写 + 无点号 → 大概率是局部变量（h, n, a, this 等）
        if name.islower() and '.' not in name:
            return False
        # 纯数字
        if name.isdigit():
            return False
        # 单字符
        if len(name) == 1:
            return False
        return True


class PositivePairExtractor:
    """
    Phase 1: 从 CoS 展平数据中提取正样本对

    对每条 (state_before, tactic, state_after) 记录:
    1. 用 TacticPremiseParser 提取 tactic 中引用的前提名称
    2. 在 corpus 中查找匹配的前提
    3. 构建 (state_before, premise) 正样本对

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化正样本对提取器

        参数:
            config: 配置字典
        """
        self.config = config
        rt_cfg = config.get("retrieval_training", config)

        self.cos_flat_path = rt_cfg.get("cos_flat_path",
                                         "data/processed/cos_dataset/cos_flat.jsonl")
        self.corpus_path = rt_cfg.get("corpus_path",
                                       "data/vector_db/corpus.jsonl")
        self.output_path = rt_cfg.get("positive_pairs_path",
                                       "data/retrieval_training/positive_pairs.jsonl")
        self.min_state_length = rt_cfg.get("min_state_length", 20)
        self.max_state_length = rt_cfg.get("max_state_length", 2048)

        # 语料库索引: name → 文档
        self.corpus_index: Dict[str, Dict[str, Any]] = {}
        # 模糊匹配索引: 短名称 → [全限定名列表]
        self.short_name_index: Dict[str, List[str]] = defaultdict(list)

    def _load_corpus(self):
        """加载语料库并构建名称索引"""
        logger.info(f"加载语料库: {self.corpus_path}")
        corpus = load_jsonl(self.corpus_path)
        logger.info(f"语料库: {len(corpus)} 条文档")

        for doc in corpus:
            full_name = doc.get("name", "")
            if not full_name:
                continue
            self.corpus_index[full_name] = doc
            # 构建短名称索引（Nat.add_comm → add_comm）
            parts = full_name.split(".")
            if len(parts) >= 2:
                short = parts[-1]
                self.short_name_index[short].append(full_name)
            # 也索引无最后模块前缀的名称
            if len(parts) >= 3:
                mid = ".".join(parts[-2:])
                self.short_name_index[mid].append(full_name)

        logger.info(f"名称索引: {len(self.corpus_index)} 条精确, "
                     f"{len(self.short_name_index)} 条模糊")

    def _resolve_premise(self, name: str) -> Optional[str]:
        """
        将解析出的前提名称解析为语料库中的全限定名

        优先精确匹配，然后尝试模糊匹配（短名称查找）。

        参数:
            name: 从 tactic 中提取的前提名称

        返回:
            Optional[str]: 匹配到的全限定名，或 None
        """
        # 精确匹配
        if name in self.corpus_index:
            return name

        # 模糊匹配：尝试短名称
        if name in self.short_name_index:
            candidates = self.short_name_index[name]
            if len(candidates) == 1:
                return candidates[0]
            # 多候选时返回第一个（后续可优化为上下文消歧）
            return candidates[0]

        # 尝试去掉前缀
        parts = name.split(".")
        for i in range(1, len(parts)):
            suffix = ".".join(parts[i:])
            if suffix in self.short_name_index:
                candidates = self.short_name_index[suffix]
                if candidates:
                    return candidates[0]

        return None

    def run(self) -> Dict[str, Any]:
        """
        执行 Phase 1: 提取正样本对

        返回:
            Dict: 提取统计信息
        """
        self._load_corpus()

        logger.info(f"读取 CoS 展平数据: {self.cos_flat_path}")
        ensure_dir(os.path.dirname(self.output_path))

        stats = {
            "total_records": 0,
            "has_premises": 0,
            "total_pairs": 0,
            "unique_premises": set(),
            "unique_states": set(),
            "tactic_coverage": Counter(),
            "parse_failures": 0,
            "state_too_short": 0,
            "state_too_long": 0,
            "unresolved_premises": Counter(),
        }

        pairs = []
        batch_size = 10000

        with open(self.cos_flat_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["parse_failures"] += 1
                    continue

                stats["total_records"] += 1

                state = record.get("state_before", "")
                tactic = record.get("tactic", "")
                state_after = record.get("state_after", "")

                # 过滤过短/过长状态
                if len(state) < self.min_state_length:
                    stats["state_too_short"] += 1
                    continue
                if len(state) > self.max_state_length:
                    stats["state_too_long"] += 1
                    continue

                # 解析 tactic 中的前提名称
                raw_premises = TacticPremiseParser.parse(tactic)
                if not raw_premises:
                    continue

                # 解析并匹配到语料库
                resolved = []
                for rp in raw_premises:
                    full_name = self._resolve_premise(rp)
                    if full_name:
                        resolved.append(full_name)
                    else:
                        stats["unresolved_premises"][rp] += 1

                if not resolved:
                    continue

                stats["has_premises"] += 1

                # 为每个解析成功的前提生成一条正样本对
                for premise_name in resolved:
                    premise_doc = self.corpus_index[premise_name]
                    pair = {
                        "query_state": state,
                        "positive_name": premise_name,
                        "positive_type": premise_doc.get("type_expr", ""),
                        "positive_text": f"{premise_name} : {premise_doc.get('type_expr', '')}",
                        "tactic": tactic,
                        "state_after": state_after,
                        "theorem_name": record.get("theorem_full_name", ""),
                        "step_index": record.get("step_index", -1),
                        "hash": hashlib.md5(
                            f"{state}|{premise_name}".encode()
                        ).hexdigest(),
                    }
                    pairs.append(pair)
                    stats["total_pairs"] += 1
                    stats["unique_premises"].add(premise_name)
                    stats["unique_states"].add(
                        hashlib.md5(state.encode()).hexdigest()[:16]
                    )

                    # 记录 tactic 类型覆盖
                    tactic_type = tactic.split()[0] if tactic else "unknown"
                    stats["tactic_coverage"][tactic_type] += 1

                # 定期日志
                if (line_idx + 1) % batch_size == 0:
                    logger.info(f"  已处理 {line_idx + 1} 行, "
                                 f"提取 {stats['total_pairs']} 对, "
                                 f"{stats['has_premises']} 条含前提")

        # 去重（按 hash）
        seen_hashes = set()
        unique_pairs = []
        for pair in pairs:
            h = pair["hash"]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_pairs.append(pair)

        dedup_count = len(pairs) - len(unique_pairs)
        logger.info(f"去重: {len(pairs)} → {len(unique_pairs)} ({dedup_count} 重复)")

        # 保存
        save_jsonl(unique_pairs, self.output_path)
        logger.info(f"正样本对已保存: {self.output_path} ({len(unique_pairs)} 条)")

        # 整理统计
        final_stats = {
            "total_cos_records": stats["total_records"],
            "records_with_premises": stats["has_premises"],
            "total_positive_pairs": len(unique_pairs),
            "dedup_removed": dedup_count,
            "unique_premises_count": len(stats["unique_premises"]),
            "unique_states_count": len(stats["unique_states"]),
            "parse_failures": stats["parse_failures"],
            "state_too_short": stats["state_too_short"],
            "state_too_long": stats["state_too_long"],
            "tactic_coverage_top20": dict(stats["tactic_coverage"].most_common(20)),
            "unresolved_top20": dict(stats["unresolved_premises"].most_common(20)),
            "timestamp": get_timestamp(),
        }

        stats_path = self.output_path.replace(".jsonl", "_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"统计信息已保存: {stats_path}")

        return final_stats


# ================================================================
# Phase 2: 硬负样本挖掘
# ================================================================

class HardNegativeMiner:
    """
    Phase 2: 利用 FAISS 索引挖掘硬负样本

    对每条正样本对 (state, premise_pos):
    1. 用编码器将 state 编码为向量
    2. 在 FAISS 索引中检索 top-K 近邻
    3. 排除正样本 → 剩余为硬负样本
    4. 选取最难的 N 个负样本构建三元组

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化硬负样本挖掘器

        参数:
            config: 配置字典
        """
        self.config = config
        rt_cfg = config.get("retrieval_training", config)

        self.positive_pairs_path = rt_cfg.get("positive_pairs_path",
                                                "data/retrieval_training/positive_pairs.jsonl")
        self.output_path = rt_cfg.get("triplets_path",
                                       "data/retrieval_training/training_triplets.jsonl")
        self.num_negatives = rt_cfg.get("num_negatives", 7)
        self.retrieval_top_k = rt_cfg.get("retrieval_top_k", 50)
        self.batch_size = rt_cfg.get("mining_batch_size", 64)

        # 检索配置路径
        self.retrieval_config_path = rt_cfg.get("retrieval_config",
                                                  "configs/retrieval.yaml")

        self.retriever = None
        self.corpus_meta: Dict[str, Dict[str, Any]] = {}

    def _load_retriever(self):
        """加载稠密检索器（使用 Task 1.2 构建的索引）"""
        from src.models.retriever import DenseRetriever

        retrieval_config = load_yaml(self.retrieval_config_path)
        project_root = retrieval_config.get("global", {}).get(
            "project_root", str(Path(__file__).parent.parent)
        )

        self.retriever = DenseRetriever.from_config(retrieval_config, project_root)
        success = self.retriever.load()
        if not success:
            raise RuntimeError("稠密检索器加载失败，请先运行 build_faiss_index.py")

        # 构建 corpus_meta 名称索引
        for idx, doc in enumerate(self.retriever.corpus):
            name = doc.get("name", "")
            if name:
                self.corpus_meta[name] = doc

        logger.info(f"检索器加载完成: {self.retriever.index.ntotal} 个向量, "
                     f"{len(self.corpus_meta)} 条元数据")

    def run(self) -> Dict[str, Any]:
        """
        执行 Phase 2: 硬负样本挖掘

        返回:
            Dict: 挖掘统计信息
        """
        self._load_retriever()

        logger.info(f"读取正样本对: {self.positive_pairs_path}")
        positive_pairs = load_jsonl(self.positive_pairs_path)
        logger.info(f"正样本对: {len(positive_pairs)} 条")

        ensure_dir(os.path.dirname(self.output_path))

        stats = {
            "total_pairs": len(positive_pairs),
            "successful_mining": 0,
            "insufficient_negatives": 0,
            "retriever_failures": 0,
            "avg_negatives": 0.0,
        }

        triplets = []
        total_neg_count = 0

        # 分批处理
        total_batches = (len(positive_pairs) + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(positive_pairs))
            batch = positive_pairs[batch_start:batch_end]

            # 批量编码查询状态
            query_texts = [p["query_state"] for p in batch]
            try:
                query_vecs = self.retriever.encode(query_texts, is_query=True)
            except Exception as e:
                logger.warning(f"批次 {batch_idx} 编码失败: {e}")
                stats["retriever_failures"] += len(batch)
                continue

            # 批量检索
            scores_batch, indices_batch = self.retriever.index.search(
                query_vecs, self.retrieval_top_k
            )

            # 为每条正样本构建三元组
            for i, pair in enumerate(batch):
                pos_name = pair["positive_name"]
                scores = scores_batch[i]
                indices = indices_batch[i]

                # 收集负样本（排除正样本）
                negatives = []
                for score, idx in zip(scores, indices):
                    if idx < 0 or idx >= len(self.retriever.corpus):
                        continue
                    doc = self.retriever.corpus[idx]
                    neg_name = doc.get("name", "")
                    # 排除正样本自身
                    if neg_name == pos_name:
                        continue
                    # 排除同定理的其他步骤引用的前提（可选，更严格）
                    negatives.append({
                        "name": neg_name,
                        "type_expr": doc.get("type_expr", ""),
                        "text": f"{neg_name} : {doc.get('type_expr', '')}",
                        "score": float(score),
                    })

                    if len(negatives) >= self.num_negatives:
                        break

                if len(negatives) < 1:
                    stats["insufficient_negatives"] += 1
                    continue

                triplet = {
                    "query_state": pair["query_state"],
                    "positive": {
                        "name": pair["positive_name"],
                        "type_expr": pair["positive_type"],
                        "text": pair["positive_text"],
                    },
                    "negatives": negatives[:self.num_negatives],
                    "tactic": pair["tactic"],
                    "theorem_name": pair["theorem_name"],
                    "step_index": pair["step_index"],
                    "num_negatives": len(negatives[:self.num_negatives]),
                }
                triplets.append(triplet)
                stats["successful_mining"] += 1
                total_neg_count += len(negatives[:self.num_negatives])

            # 进度日志
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / total_batches * 100
                speed = (batch_end) / elapsed if elapsed > 0 else 0
                logger.info(f"  挖掘: {progress:.1f}% ({batch_end}/{len(positive_pairs)}), "
                             f"速率 {speed:.0f} pairs/s, "
                             f"成功 {stats['successful_mining']}")

        # 统计
        stats["avg_negatives"] = (total_neg_count / stats["successful_mining"]
                                  if stats["successful_mining"] > 0 else 0)

        # 保存
        save_jsonl(triplets, self.output_path)
        logger.info(f"三元组已保存: {self.output_path} ({len(triplets)} 条)")

        stats["total_triplets"] = len(triplets)
        stats["mining_time_seconds"] = round(time.time() - start_time, 1)
        stats["timestamp"] = get_timestamp()

        stats_path = self.output_path.replace(".jsonl", "_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"统计信息已保存: {stats_path}")

        return stats


# ================================================================
# Phase 3: 最终训练集构建
# ================================================================

class TrainingDatasetBuilder:
    """
    Phase 3: 构建最终的对比学习训练数据集

    功能:
    1. 加载三元组数据
    2. 添加 in-batch negatives 标记（训练时动态使用）
    3. 按定理名称分割 train/val/test（确保同一定理的样本不跨集）
    4. 数据增强: 状态截断、前缀变化等（可选）

    @author ygw
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练数据集构建器

        参数:
            config: 配置字典
        """
        self.config = config
        rt_cfg = config.get("retrieval_training", config)

        self.triplets_path = rt_cfg.get("triplets_path",
                                         "data/retrieval_training/training_triplets.jsonl")
        self.output_dir = rt_cfg.get("output_dir",
                                      "data/retrieval_training")
        self.train_ratio = rt_cfg.get("train_ratio", 0.90)
        self.val_ratio = rt_cfg.get("val_ratio", 0.05)
        self.test_ratio = rt_cfg.get("test_ratio", 0.05)
        self.seed = rt_cfg.get("seed", 42)
        self.max_query_length = rt_cfg.get("max_query_length", 1024)
        self.max_passage_length = rt_cfg.get("max_passage_length", 256)

    def run(self) -> Dict[str, Any]:
        """
        执行 Phase 3: 构建训练 / 验证 / 测试集

        按定理名分割（novel-premise split 策略），确保泛化评估的公正性。

        返回:
            Dict: 构建统计信息
        """
        set_seed(self.seed)
        ensure_dir(self.output_dir)

        logger.info(f"加载三元组: {self.triplets_path}")
        triplets = load_jsonl(self.triplets_path)
        logger.info(f"三元组: {len(triplets)} 条")

        # 按定理名称分组
        theorem_groups: Dict[str, List[Dict]] = defaultdict(list)
        for t in triplets:
            thm = t.get("theorem_name", "unknown")
            theorem_groups[thm].append(t)

        theorem_names = sorted(theorem_groups.keys())
        random.shuffle(theorem_names)

        # 按定理名分割
        n_total = len(theorem_names)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_theorems = set(theorem_names[:n_train])
        val_theorems = set(theorem_names[n_train:n_train + n_val])
        test_theorems = set(theorem_names[n_train + n_val:])

        train_data = []
        val_data = []
        test_data = []

        for thm, samples in theorem_groups.items():
            # 格式化为统一训练格式
            formatted = [self._format_sample(s) for s in samples]
            if thm in train_theorems:
                train_data.extend(formatted)
            elif thm in val_theorems:
                val_data.extend(formatted)
            else:
                test_data.extend(formatted)

        # 打乱训练集
        random.shuffle(train_data)

        # 保存
        train_path = os.path.join(self.output_dir, "train.jsonl")
        val_path = os.path.join(self.output_dir, "val.jsonl")
        test_path = os.path.join(self.output_dir, "test.jsonl")

        save_jsonl(train_data, train_path)
        save_jsonl(val_data, val_path)
        save_jsonl(test_data, test_path)

        logger.info(f"训练集: {len(train_data)} 条 → {train_path}")
        logger.info(f"验证集: {len(val_data)} 条 → {val_path}")
        logger.info(f"测试集: {len(test_data)} 条 → {test_path}")

        # 统计
        stats = {
            "total_triplets": len(triplets),
            "total_theorems": n_total,
            "train_theorems": len(train_theorems),
            "val_theorems": len(val_theorems),
            "test_theorems": len(test_theorems),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "seed": self.seed,
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "timestamp": get_timestamp(),
        }

        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"数据集统计已保存: {stats_path}")

        return stats

    def _format_sample(self, triplet: Dict) -> Dict:
        """
        将三元组格式化为标准训练样本

        格式:
            {
                "query": 截断后的 state 文本,
                "positive": "name : type_expr",
                "negatives": ["name1 : type1", "name2 : type2", ...],
                "metadata": {theorem_name, tactic, step_index}
            }

        参数:
            triplet: 原始三元组

        返回:
            Dict: 格式化的训练样本
        """
        query = triplet["query_state"][:self.max_query_length]
        pos = triplet["positive"]
        pos_text = pos.get("text", f"{pos['name']} : {pos.get('type_expr', '')}")
        pos_text = pos_text[:self.max_passage_length]

        neg_texts = []
        for neg in triplet.get("negatives", []):
            neg_text = neg.get("text", f"{neg['name']} : {neg.get('type_expr', '')}")
            neg_texts.append(neg_text[:self.max_passage_length])

        return {
            "query": query,
            "positive": pos_text,
            "positive_name": pos.get("name", ""),
            "negatives": neg_texts,
            "num_negatives": len(neg_texts),
            "metadata": {
                "theorem_name": triplet.get("theorem_name", ""),
                "tactic": triplet.get("tactic", ""),
                "step_index": triplet.get("step_index", -1),
            },
        }


# ================================================================
# CLI 入口
# ================================================================

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    加载配置文件

    参数:
        config_path: YAML 配置文件路径（可选）

    返回:
        Dict: 配置字典
    """
    if config_path and os.path.exists(config_path):
        config = load_yaml(config_path)
        logger.info(f"已加载配置: {config_path}")
        return config

    # 默认配置
    project_root = str(Path(__file__).parent.parent)
    return {
        "retrieval_training": {
            "cos_flat_path": os.path.join(project_root, "data/processed/cos_dataset/cos_flat.jsonl"),
            "corpus_path": os.path.join(project_root, "data/vector_db/corpus.jsonl"),
            "positive_pairs_path": os.path.join(project_root, "data/retrieval_training/positive_pairs.jsonl"),
            "triplets_path": os.path.join(project_root, "data/retrieval_training/training_triplets.jsonl"),
            "output_dir": os.path.join(project_root, "data/retrieval_training"),
            "retrieval_config": os.path.join(project_root, "configs/retrieval.yaml"),
            "min_state_length": 20,
            "max_state_length": 2048,
            "max_query_length": 1024,
            "max_passage_length": 256,
            "num_negatives": 7,
            "retrieval_top_k": 50,
            "mining_batch_size": 64,
            "train_ratio": 0.90,
            "val_ratio": 0.05,
            "test_ratio": 0.05,
            "seed": 42,
        }
    }


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="检索增强训练数据构建器 (Task 2.2)")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (YAML)")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="执行阶段: 0=全部, 1=正样本对, 2=硬负样本, 3=训练集构建")
    parser.add_argument("--num-negatives", type=int, default=None,
                        help="每条正样本的硬负样本数量 (default: 7)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = load_config(args.config)
    rt_cfg = config.get("retrieval_training", config)

    if args.num_negatives is not None:
        rt_cfg["num_negatives"] = args.num_negatives
    if args.seed is not None:
        rt_cfg["seed"] = args.seed

    run_all = args.phase == 0

    logger.info("=" * 60)
    logger.info("检索增强训练数据构建器 (Task 2.2)")
    logger.info("=" * 60)

    total_start = time.time()

    # Phase 1: 正样本对提取
    if run_all or args.phase == 1:
        logger.info("\n" + "=" * 40)
        logger.info("Phase 1: 正样本对提取")
        logger.info("=" * 40)
        extractor = PositivePairExtractor(config)
        phase1_stats = extractor.run()
        logger.info(f"Phase 1 完成: {phase1_stats['total_positive_pairs']} 条正样本对, "
                     f"{phase1_stats['unique_premises_count']} 个唯一前提")

    # Phase 2: 硬负样本挖掘
    if run_all or args.phase == 2:
        logger.info("\n" + "=" * 40)
        logger.info("Phase 2: 硬负样本挖掘")
        logger.info("=" * 40)
        miner = HardNegativeMiner(config)
        phase2_stats = miner.run()
        logger.info(f"Phase 2 完成: {phase2_stats['total_triplets']} 条三元组, "
                     f"平均 {phase2_stats['avg_negatives']:.1f} 个负样本")

    # Phase 3: 训练集构建
    if run_all or args.phase == 3:
        logger.info("\n" + "=" * 40)
        logger.info("Phase 3: 训练集构建")
        logger.info("=" * 40)
        builder = TrainingDatasetBuilder(config)
        phase3_stats = builder.run()
        logger.info(f"Phase 3 完成: train={phase3_stats['train_samples']}, "
                     f"val={phase3_stats['val_samples']}, "
                     f"test={phase3_stats['test_samples']}")

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"全部完成! 总耗时 {total_time:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
