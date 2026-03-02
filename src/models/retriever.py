"""
双粒度检索增强语义状态规划 (DG-RASP)
@author ygw
更新日期: 2026-02-28

创新点一：Dual-Grained Retrieval-Augmented State Planning
实现宏观层（Macro-Level）和微观层（Micro-Level）的双层检索架构：
    - 宏观层：在将非形式化步骤映射为形式化中间状态时，通过全局语义检索
      在 Mathlib 中锚定相关引理，确保状态链在数学库拓扑可达空间内。
    - 微观层：在生成连接相邻锚定状态的具体 Tactics 时，执行细粒度的
      局部前提检索，提高战术生成准确率。

技术栈:
    - 稠密检索: sentence-transformers（E5/BGE 系列模型）
    - 向量索引: FAISS (IVF-Flat / IVF-PQ 量化索引)
    - 符号检索: Lean 类型签名精确匹配
    - 融合策略: Reciprocal Rank Fusion (RRF)

配置驱动: 支持通过 configs/retrieval.yaml 初始化
"""

import os
import json
import logging
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("retriever")


# ================================================================
# 数据结构定义
# ================================================================

@dataclass
class RetrievalResult:
    """
    单条检索结果

    属性:
        name: 定理/引理名称（如 Nat.add_comm）
        type_expr: 类型表达式（Lean4 格式）
        statement: 完整声明文本
        score: 相关性得分（0~1，越高越相关）
        source: 检索来源 ('dense' / 'symbolic' / 'fused')
        metadata: 附加元数据（所属模块、证明长度等）
    """
    name: str = ""
    type_expr: str = ""
    statement: str = ""
    score: float = 0.0
    source: str = "dense"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "type_expr": self.type_expr,
            "statement": self.statement,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class PremiseDocument:
    """
    Mathlib 前提文档（用于索引构建）

    属性:
        name: 定理全限定名
        type_expr: 类型表达式
        module_path: 所属模块路径
        docstring: 文档字符串
        proof_length: 证明步数
    """
    name: str = ""
    type_expr: str = ""
    module_path: str = ""
    docstring: str = ""
    proof_length: int = 0


# ================================================================
# 稠密检索器 (Dense Retriever)
# ================================================================

class DenseRetriever:
    """
    基于 Sentence-Transformers 的稠密语义检索器

    使用对比学习训练的编码模型，将证明状态和定理语句编码为稠密向量，
    通过 FAISS 索引进行高效近邻搜索。

    对应论文: LeanSearch-PS 的检索后端

    参数:
        model_name: 编码模型名称（推荐 E5-mistral-7b-instruct 或 bge-large-en-v1.5）
        index_path: FAISS 索引文件路径
        corpus_path: 语料库 JSON 路径（name → type_expr 映射）
        device: 推理设备
    """

    def __init__(self,
                 model_name: str = "intfloat/e5-large-v2",
                 index_path: Optional[str] = None,
                 corpus_path: Optional[str] = None,
                 device: str = "cpu",
                 cache_dir: Optional[str] = None):
        """
        初始化稠密检索器

        参数:
            model_name: sentence-transformers 模型名称或本地路径
            index_path: 预构建的 FAISS 索引路径（.bin 或 .index 文件）
            corpus_path: 语料库元数据路径（JSON 或 JSONL 格式）
            device: 运行设备 ('cpu' / 'cuda')
            cache_dir: 模型缓存目录（可选）
        """
        self.model_name = model_name
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.device = device
        self.cache_dir = cache_dir
        self.encoder = None
        self.index = None
        self.corpus: List[Dict[str, str]] = []
        self._embedding_dim: int = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any],
                    project_root: str = "") -> "DenseRetriever":
        """
        从配置字典创建稠密检索器

        参数:
            config: 配置字典（包含 encoder, faiss_index 等键）
            project_root: 项目根目录（用于解析相对路径）

        返回:
            DenseRetriever: 配置好的检索器实例
        """
        def resolve(p):
            if p and not os.path.isabs(p) and project_root:
                return os.path.join(project_root, p)
            return p

        encoder_cfg = config.get("encoder", {})
        faiss_cfg = config.get("faiss_index", {})

        return cls(
            model_name=encoder_cfg.get("model_name", "intfloat/e5-large-v2"),
            index_path=resolve(faiss_cfg.get("index_path", "data/vector_db/faiss_index.bin")),
            corpus_path=resolve(faiss_cfg.get("meta_path", "data/vector_db/corpus_meta.json")),
            device=encoder_cfg.get("device", "cuda"),
            cache_dir=resolve(encoder_cfg.get("cache_dir")),
        )

    def load(self) -> bool:
        """
        加载编码模型和 FAISS 索引

        支持 JSON 和 JSONL 两种格式的语料库文件。
        编码器加载失败时降级为仅索引模式（用于纯搜索）。

        返回:
            bool: 是否加载成功（至少编码器可用）
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载稠密检索编码器: {self.model_name}")
            start = time.time()
            kwargs = {"device": self.device}
            if self.cache_dir:
                kwargs["cache_folder"] = self.cache_dir
            self.encoder = SentenceTransformer(self.model_name, **kwargs)
            self._embedding_dim = self.encoder.get_sentence_embedding_dimension()
            elapsed = time.time() - start
            logger.info(f"编码器加载完成 (dim={self._embedding_dim}, "
                        f"耗时 {elapsed:.1f}s)")
        except ImportError:
            logger.error("sentence-transformers 未安装，稠密检索不可用")
            return False
        except Exception as e:
            logger.error(f"编码器加载失败: {e}")
            return False

        # 加载 FAISS 索引
        if self.index_path and os.path.exists(self.index_path):
            try:
                import faiss
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS 索引加载完成: {self.index.ntotal} 个向量")
            except ImportError:
                logger.error("faiss 未安装，无法加载索引")
                return False
            except Exception as e:
                logger.error(f"FAISS 索引加载失败: {e}")
                return False
        else:
            logger.info("无预构建索引，可通过 build_index() 构建")

        # 加载语料库元数据（支持 JSON 和 JSONL 格式）
        if self.corpus_path and os.path.exists(self.corpus_path):
            self.corpus = self._load_corpus_file(self.corpus_path)
            logger.info(f"语料库加载完成: {len(self.corpus)} 条文档")
        else:
            logger.info("无语料库元数据文件")

        return True

    def _load_corpus_file(self, path: str) -> List[Dict[str, str]]:
        """
        加载语料库文件（自动识别 JSON / JSONL 格式）

        参数:
            path: 文件路径

        返回:
            List[Dict]: 文档列表
        """
        with open(path, 'r', encoding='utf-8') as f:
            first_char = f.read(1).strip()
            f.seek(0)

            if first_char == '[':
                # JSON 数组格式
                return json.load(f)
            else:
                # JSONL 格式
                corpus = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            corpus.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return corpus

    def encode(self, texts: List[str], batch_size: int = 32,
               is_query: bool = True) -> np.ndarray:
        """
        将文本编码为稠密向量

        参数:
            texts: 待编码文本列表
            batch_size: 编码批次大小
            is_query: True 添加 "query: " 前缀, False 添加 "passage: "

        返回:
            np.ndarray: 编码向量矩阵 (N, D), L2 归一化
        """
        if self.encoder is None:
            raise RuntimeError("编码器未加载，请先调用 load()")
        # E5 模型需要添加 "query: " 或 "passage: " 前缀
        prefix = "query: " if is_query else "passage: "
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self.encoder.encode(
            prefixed, batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, documents: List["PremiseDocument"],
                    batch_size: int = 64, nlist: int = 256,
                    nprobe: int = 32):
        """
        从前提文档列表构建 FAISS 索引

        参数:
            documents: 前提文档列表
            batch_size: 编码批次大小
            nlist: IVF 聚类中心数（索引质量与速度的平衡点）
            nprobe: 搜索时探测的聚类数（越大精度越高但越慢）
        """
        import faiss

        # 准备编码文本：使用 "passage: name : type_expr" 格式
        texts = []
        self.corpus = []
        for doc in documents:
            text = f"{doc.name} : {doc.type_expr}"
            texts.append(text)
            self.corpus.append({
                "name": doc.name,
                "type_expr": doc.type_expr,
                "module_path": doc.module_path,
                "docstring": doc.docstring,
            })

        logger.info(f"开始编码 {len(texts)} 个前提文档...")
        start_time = time.time()

        # 分批编码（带进度日志）
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]

            # 使用 passage 前缀
            emb = self.encode(batch, batch_size=batch_size, is_query=False)
            all_embeddings.append(emb)

            if (batch_idx + 1) % 20 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / total_batches * 100
                logger.info(f"  编码: {progress:.1f}% ({batch_end}/{len(texts)}), "
                            f"已耗时 {elapsed:.1f}s")

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        dim = embeddings.shape[1]
        encode_time = time.time() - start_time
        logger.info(f"编码完成: {embeddings.shape[0]} 个向量, 维度 {dim}, "
                    f"耗时 {encode_time:.1f}s")

        # 构建索引（自适应选择策略）
        build_start = time.time()
        if len(documents) > 10000:
            # IVF-Flat 索引（适合万级及以上语料库）
            effective_nlist = min(nlist, max(1, len(documents) // 39))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(nprobe, effective_nlist)
            logger.info(f"  IVF-Flat 索引: nlist={effective_nlist}, nprobe={self.index.nprobe}")
        else:
            # 小规模语料直接用 Flat 索引
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            logger.info("  Flat 索引（小规模精确搜索）")

        build_time = time.time() - build_start
        logger.info(f"FAISS 索引构建完成: {self.index.ntotal} 个向量, "
                    f"构建耗时 {build_time:.1f}s")

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        执行稠密检索

        参数:
            query: 查询文本（证明状态或自然语言描述）
            top_k: 返回结果数量

        返回:
            List[RetrievalResult]: 检索结果列表（按相关性降序）
        """
        if self.index is None or self.encoder is None:
            logger.warning("稠密检索器未就绪（索引或编码器未加载）")
            return []

        query_vec = self.encode([query])
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.corpus):
                continue
            doc = self.corpus[idx]
            results.append(RetrievalResult(
                name=doc.get("name", ""),
                type_expr=doc.get("type_expr", ""),
                statement=f"{doc.get('name', '')} : {doc.get('type_expr', '')}",
                score=float(score),
                source="dense",
                metadata={"module_path": doc.get("module_path", "")},
            ))

        return results

    def save_index(self, index_path: str, corpus_path: str):
        """
        保存 FAISS 索引和语料库到磁盘

        参数:
            index_path: 索引文件保存路径 (.index)
            corpus_path: 语料库 JSON 保存路径
        """
        import faiss
        if self.index is not None:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS 索引已保存: {index_path}")
        if self.corpus:
            Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False)
            logger.info(f"语料库已保存: {corpus_path}")


# ================================================================
# 符号检索器 (Symbolic Retriever)
# ================================================================

class SymbolicRetriever:
    """
    基于 Lean4 类型签名的符号检索器

    通过解析证明状态中的类型信息，在 Mathlib 定理库中进行精确的
    类型模式匹配，查找签名兼容的引理。

    匹配策略:
        1. 结论类型匹配：目标类型 ≈ 引理结论类型
        2. 假设类型匹配：上下文假设 ⊆ 引理前提
        3. 关键词匹配：提取类型中的核心标识符进行集合交集计算
    """

    def __init__(self, corpus_path: Optional[str] = None):
        """
        初始化符号检索器

        参数:
            corpus_path: 定理类型签名数据库路径（JSON 或 JSONL 格式）
        """
        self.corpus_path = corpus_path
        self.theorems: List[Dict[str, str]] = []
        self._type_index: Dict[str, List[int]] = {}  # 关键词 → 定理索引列表

    @classmethod
    def from_config(cls, config: Dict[str, Any],
                    project_root: str = "") -> "SymbolicRetriever":
        """
        从配置字典创建符号检索器

        参数:
            config: 配置字典（包含 symbolic 键）
            project_root: 项目根目录

        返回:
            SymbolicRetriever: 配置好的检索器实例
        """
        def resolve(p):
            if p and not os.path.isabs(p) and project_root:
                return os.path.join(project_root, p)
            return p

        sym_cfg = config.get("symbolic", {})
        # 符号检索共享语料库元数据
        faiss_cfg = config.get("faiss_index", {})
        corpus_path = resolve(
            sym_cfg.get("corpus_path") or
            faiss_cfg.get("meta_path", "data/vector_db/corpus_meta.json")
        )
        return cls(corpus_path=corpus_path)

    def load(self) -> bool:
        """
        加载定理类型签名数据库并构建倒排索引

        支持 JSON 数组和 JSONL 两种格式。

        返回:
            bool: 是否加载成功
        """
        if not self.corpus_path or not os.path.exists(self.corpus_path):
            logger.warning(f"符号检索语料库不存在: {self.corpus_path}")
            return False

        # 自动识别格式
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1).strip()
            f.seek(0)
            if first_char == '[':
                self.theorems = json.load(f)
            else:
                self.theorems = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.theorems.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # 构建关键词倒排索引
        self._build_type_index()
        logger.info(f"符号检索器加载完成: {len(self.theorems)} 个定理, "
                     f"{len(self._type_index)} 个关键词")
        return True

    def _build_type_index(self):
        """构建类型关键词倒排索引"""
        import re
        self._type_index = {}
        for idx, thm in enumerate(self.theorems):
            type_expr = thm.get("type_expr", "")
            # 提取类型表达式中的标识符作为关键词
            tokens = set(re.findall(r'[A-Za-z_][\w.]*', type_expr))
            for token in tokens:
                if token not in self._type_index:
                    self._type_index[token] = []
                self._type_index[token].append(idx)

    def search(self, query_state: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        基于类型签名的符号检索

        参数:
            query_state: 当前证明状态文本
            top_k: 返回结果数量

        返回:
            List[RetrievalResult]: 检索结果列表
        """
        import re
        if not self.theorems:
            return []

        # 从证明状态提取关键标识符
        query_tokens = set(re.findall(r'[A-Za-z_][\w.]*', query_state))

        # 计算每个定理的匹配度（Jaccard 相似度）
        candidate_scores: Dict[int, float] = {}
        for token in query_tokens:
            if token in self._type_index:
                for idx in self._type_index[token]:
                    candidate_scores[idx] = candidate_scores.get(idx, 0) + 1

        # 按匹配关键词数排序
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k * 3]  # 取较多候选再精排

        results = []
        for idx, raw_score in sorted_candidates[:top_k]:
            thm = self.theorems[idx]
            thm_tokens = set(re.findall(r'[A-Za-z_][\w.]*', thm.get("type_expr", "")))
            # Jaccard 相似度
            union = len(query_tokens | thm_tokens)
            jaccard = raw_score / union if union > 0 else 0.0

            results.append(RetrievalResult(
                name=thm.get("name", ""),
                type_expr=thm.get("type_expr", ""),
                statement=f"{thm.get('name', '')} : {thm.get('type_expr', '')}",
                score=jaccard,
                source="symbolic",
                metadata={"module_path": thm.get("module_path", "")},
            ))

        return results


# ================================================================
# 双粒度检索增强器 (DG-RASP)
# ================================================================

class DualGrainedRetriever:
    """
    双粒度检索增强语义状态规划器 (DG-RASP)

    创新点一的核心实现，整合宏观层和微观层的双层检索架构。

    架构:
        ┌─────────────────────────────────┐
        │     非形式化证明骨架 (DSP+)       │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │  宏观层检索 (Macro-Level)         │
        │  全局语义检索 → 状态锚定           │
        │  输入: 自然语言语义描述            │
        │  输出: 锚定在 Mathlib 上的中间状态  │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────▼──────────────────┐
        │  微观层检索 (Micro-Level)         │
        │  局部前提检索 → 战术生成增强       │
        │  输入: 局部证明状态 (S_i → S_{i+1})│
        │  输出: 相关前提列表供 Tactic 生成   │
        └─────────────────────────────────┘

    融合策略: Reciprocal Rank Fusion (RRF)

    参数:
        dense_retriever: 稠密检索器实例
        symbolic_retriever: 符号检索器实例
        rrf_k: RRF 参数 (默认 60)
        macro_top_k: 宏观层检索返回数量
        micro_top_k: 微观层检索返回数量
    """

    def __init__(self,
                 dense_retriever: Optional[DenseRetriever] = None,
                 symbolic_retriever: Optional[SymbolicRetriever] = None,
                 rrf_k: int = 60,
                 macro_top_k: int = 5,
                 micro_top_k: int = 10):
        """
        初始化双粒度检索器

        参数:
            dense_retriever: 稠密检索器（未提供则自动创建）
            symbolic_retriever: 符号检索器（未提供则自动创建）
            rrf_k: RRF 融合参数
            macro_top_k: 宏观检索返回数量
            micro_top_k: 微观检索返回数量
        """
        self.dense = dense_retriever or DenseRetriever()
        self.symbolic = symbolic_retriever or SymbolicRetriever()
        self.rrf_k = rrf_k
        self.macro_top_k = macro_top_k
        self.micro_top_k = micro_top_k

        # 检索缓存（避免重复检索相同状态）
        self._cache: Dict[str, List[RetrievalResult]] = {}
        self._cache_max_size = 5000

    @classmethod
    def from_config(cls, config: Dict[str, Any],
                    project_root: str = "") -> "DualGrainedRetriever":
        """
        从配置字典创建双粒度检索器

        会同时创建并配置好内部的稠密检索器和符号检索器。

        参数:
            config: 完整配置字典（通常从 configs/retrieval.yaml 加载）
            project_root: 项目根目录

        返回:
            DualGrainedRetriever: 完整配置的检索器
        """
        dg_cfg = config.get("dual_grained", {})

        dense = DenseRetriever.from_config(config, project_root)
        symbolic = SymbolicRetriever.from_config(config, project_root)

        return cls(
            dense_retriever=dense,
            symbolic_retriever=symbolic,
            rrf_k=dg_cfg.get("rrf_k", 60),
            macro_top_k=dg_cfg.get("macro_top_k", 5),
            micro_top_k=dg_cfg.get("micro_top_k", 10),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str,
                  project_root: str = "") -> "DualGrainedRetriever":
        """
        从 YAML 配置文件创建双粒度检索器（便捷方法）

        参数:
            yaml_path: YAML 配置文件路径
            project_root: 项目根目录

        返回:
            DualGrainedRetriever: 完整配置的检索器
        """
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not project_root:
            project_root = config.get("global", {}).get("project_root", "")
        return cls.from_config(config, project_root)

    def load(self) -> bool:
        """
        加载所有检索组件

        返回:
            bool: 是否全部加载成功
        """
        dense_ok = self.dense.load()
        symbolic_ok = self.symbolic.load()
        if not dense_ok:
            logger.warning("稠密检索器加载失败，仅使用符号检索")
        if not symbolic_ok:
            logger.warning("符号检索器加载失败，仅使用稠密检索")
        return dense_ok or symbolic_ok

    # ================================================================
    # 宏观层检索 (Macro-Level Retrieval)
    # ================================================================

    def macro_retrieve(self, natural_language_step: str,
                       top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        宏观层检索：在将非形式化步骤映射为形式化中间状态时执行

        在 Mathlib 中进行全局语义检索，找到与当前推理步骤最相关的
        定理和引理，用于锚定中间状态的生成。

        参数:
            natural_language_step: 自然语言描述的推理步骤
                例: "利用加法交换律将 a + b 改写为 b + a"
            top_k: 返回结果数量（默认使用 macro_top_k）

        返回:
            List[RetrievalResult]: 相关引理列表（用于注入到 LLM 提示中）
        """
        top_k = top_k or self.macro_top_k
        cache_key = f"macro:{hashlib.md5(natural_language_step.encode()).hexdigest()}"

        if cache_key in self._cache:
            return self._cache[cache_key][:top_k]

        # 稠密检索：基于自然语言语义
        dense_results = self.dense.search(natural_language_step, top_k=top_k * 2)

        # 符号检索：从自然语言中提取可能的类型关键词
        symbolic_results = self.symbolic.search(natural_language_step, top_k=top_k * 2)

        # RRF 融合
        fused = self._reciprocal_rank_fusion(dense_results, symbolic_results, top_k)

        # 缓存
        self._update_cache(cache_key, fused)
        return fused

    # ================================================================
    # 微观层检索 (Micro-Level Retrieval)
    # ================================================================

    def micro_retrieve(self, proof_state: str,
                       top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        微观层检索：在生成连接两个相邻状态的具体 Tactics 时执行

        基于当前局部证明状态进行细粒度前提检索，为战术生成提供
        精确的上下文知识。

        参数:
            proof_state: 当前 Lean4 证明状态
                例: "⊢ ∀ (n : ℕ), n + 0 = n"
            top_k: 返回结果数量（默认使用 micro_top_k）

        返回:
            List[RetrievalResult]: 局部相关前提列表
        """
        top_k = top_k or self.micro_top_k
        cache_key = f"micro:{hashlib.md5(proof_state.encode()).hexdigest()}"

        if cache_key in self._cache:
            return self._cache[cache_key][:top_k]

        # 稠密检索：基于形式化状态语义
        dense_results = self.dense.search(proof_state, top_k=top_k * 2)

        # 符号检索：基于类型签名匹配
        symbolic_results = self.symbolic.search(proof_state, top_k=top_k * 2)

        # RRF 融合
        fused = self._reciprocal_rank_fusion(dense_results, symbolic_results, top_k)

        # 缓存
        self._update_cache(cache_key, fused)
        return fused

    # ================================================================
    # 检索上下文格式化（供 LLM Prompt 注入）
    # ================================================================

    def format_macro_context(self, results: List[RetrievalResult],
                              max_premises: int = 5) -> str:
        """
        将宏观层检索结果格式化为 LLM Prompt 上下文

        用于在生成中间状态时，强制 LLM 引用检索到的引理。

        参数:
            results: 检索结果列表
            max_premises: 最大引用数量

        返回:
            str: 格式化的上下文文本
        """
        if not results:
            return ""

        lines = ["[Retrieved Premises for State Planning]"]
        for i, r in enumerate(results[:max_premises]):
            lines.append(f"  {i+1}. {r.name} : {r.type_expr}")
        lines.append("[End Premises]")
        lines.append("You MUST reference at least one of the above premises "
                     "when generating the intermediate state.")
        return "\n".join(lines)

    def format_micro_context(self, results: List[RetrievalResult],
                              max_premises: int = 10) -> str:
        """
        将微观层检索结果格式化为 LLM Prompt 上下文

        用于在生成具体 Tactic 时，提供相关前提作为参考。

        参数:
            results: 检索结果列表
            max_premises: 最大引用数量

        返回:
            str: 格式化的上下文文本
        """
        if not results:
            return ""

        lines = ["[Retrieved Premises for Tactic Generation]"]
        for i, r in enumerate(results[:max_premises]):
            lines.append(f"  {i+1}. {r.name} : {r.type_expr}")
        lines.append("[End Premises]")
        return "\n".join(lines)

    # ================================================================
    # RRF 融合算法
    # ================================================================

    def _reciprocal_rank_fusion(self,
                                 list_a: List[RetrievalResult],
                                 list_b: List[RetrievalResult],
                                 top_k: int) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) 算法

        将多个检索结果列表融合为统一的排序列表。
        RRF 得分 = Σ 1 / (k + rank_i)，k = self.rrf_k

        参数:
            list_a: 第一个检索结果列表（如稠密检索）
            list_b: 第二个检索结果列表（如符号检索）
            top_k: 返回结果数量

        返回:
            List[RetrievalResult]: 融合后的结果列表
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        for rank, r in enumerate(list_a):
            key = r.name
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = r

        for rank, r in enumerate(list_b):
            key = r.name
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = r

        # 按融合得分排序
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            result.score = scores[key]
            result.source = "fused"
            results.append(result)

        return results

    def _update_cache(self, key: str, results: List[RetrievalResult]):
        """更新缓存（带大小限制）"""
        if len(self._cache) >= self._cache_max_size:
            # FIFO 淘汰
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = results

    def clear_cache(self):
        """清空检索缓存"""
        self._cache.clear()
