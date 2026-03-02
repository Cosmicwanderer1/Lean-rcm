"""
FAISS 索引构建脚本 —— 从语料库构建向量索引
@author ygw
创建日期: 2026-02-28

将 corpus.jsonl 中的定理文本编码为稠密向量，构建 FAISS IVF-Flat 索引。
支持增量构建、断点续做、GPU 加速等特性。

运行方式:
    cd /root/autodl-tmp/RTAP
    python scripts/build_faiss_index.py
    python scripts/build_faiss_index.py --config configs/retrieval.yaml
    python scripts/build_faiss_index.py --corpus data/vector_db/corpus.jsonl --gpu

输出:
    data/vector_db/faiss_index.bin    — FAISS 索引文件
    data/vector_db/corpus_meta.json   — 语料库元数据（id → doc 映射）
    data/vector_db/index_stats.json   — 索引构建统计信息

依赖:
    pip install sentence-transformers faiss-gpu  (或 faiss-cpu)
"""

import sys
import os
import json
import time
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    参数:
        config_path: str, 配置文件路径

    返回:
        Dict: 配置字典
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """
    加载 JSONL 格式语料库

    参数:
        corpus_path: str, 语料库文件路径

    返回:
        List[Dict]: 文档列表，每个文档包含 name, type_expr, doc_text 等字段
    """
    logger.info(f"加载语料库: {corpus_path}")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    corpus.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"语料库加载完成: {len(corpus)} 条文档")
    return corpus


def load_encoder(model_name: str, device: str = "cuda",
                 cache_dir: Optional[str] = None):
    """
    加载 Sentence-Transformers 编码器

    参数:
        model_name: str, 模型名称或本地路径（如 intfloat/e5-large-v2）
        device: str, 运行设备 ('cuda' / 'cpu')
        cache_dir: str, 模型缓存目录

    返回:
        SentenceTransformer: 编码器实例
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"加载编码器: {model_name} (device={device})")
    start = time.time()

    kwargs = {"device": device}
    if cache_dir:
        kwargs["cache_folder"] = cache_dir

    encoder = SentenceTransformer(model_name, **kwargs)
    elapsed = time.time() - start
    logger.info(f"编码器加载完成, 耗时 {elapsed:.1f}s, "
                f"维度={encoder.get_sentence_embedding_dimension()}")

    return encoder


def encode_corpus(encoder, corpus: List[Dict[str, str]],
                  batch_size: int = 64,
                  text_field: str = "doc_text",
                  prefix: str = "passage: ",
                  max_length: int = 512,
                  save_embeddings_path: Optional[str] = None) -> np.ndarray:
    """
    将语料库编码为稠密向量矩阵

    参数:
        encoder: SentenceTransformer 编码器
        corpus: List[Dict], 语料库文档列表
        batch_size: int, 编码批次大小
        text_field: str, 用于编码的文本字段名
        prefix: str, 编码前缀（E5 模型需要 "passage: "）
        max_length: int, 最大文本长度
        save_embeddings_path: str, 若非空则保存原始向量到此路径

    返回:
        np.ndarray: 编码向量矩阵 (N, D), float32, L2 归一化
    """
    logger.info(f"开始编码 {len(corpus)} 条文档 (batch_size={batch_size})...")

    # 准备编码文本
    texts = []
    for doc in corpus:
        text = doc.get(text_field, "")
        if not text:
            text = f"{doc.get('name', '')} : {doc.get('type_expr', '')}"
        # 截断过长文本
        if len(text) > max_length * 4:  # 粗略字符截断
            text = text[:max_length * 4]
        texts.append(f"{prefix}{text}")

    # 分批编码（带进度显示）
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(texts))
        batch = texts[batch_start:batch_end]

        emb = encoder.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 归一化（使内积 = 余弦相似度）
        )
        all_embeddings.append(emb)

        # 进度日志
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / total_batches * 100
            docs_done = batch_end
            rate = docs_done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - docs_done) / rate if rate > 0 else 0
            logger.info(f"  编码进度: {progress:.1f}% ({docs_done}/{len(texts)}) "
                        f"速率: {rate:.0f} docs/s, ETA: {eta:.0f}s")

    # 合并
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    total_time = time.time() - start_time
    logger.info(f"编码完成: shape={embeddings.shape}, 总耗时 {total_time:.1f}s, "
                f"平均 {len(corpus)/total_time:.0f} docs/s")

    # 可选保存原始向量
    if save_embeddings_path:
        Path(save_embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_embeddings_path, embeddings)
        logger.info(f"原始向量已保存: {save_embeddings_path} "
                    f"({embeddings.nbytes / 1024 / 1024:.1f} MB)")

    return embeddings


def build_faiss_index(embeddings: np.ndarray,
                      index_type: str = "ivf_flat",
                      nlist: int = 256,
                      nprobe: int = 32,
                      use_gpu: bool = False) -> Any:
    """
    构建 FAISS 向量索引

    参数:
        embeddings: np.ndarray, 向量矩阵 (N, D)
        index_type: str, 索引类型 ('flat', 'ivf_flat', 'ivf_pq')
        nlist: int, IVF 聚类中心数
        nprobe: int, 搜索时探测的聚类数
        use_gpu: bool, 是否使用 GPU 加速

    返回:
        faiss.Index: 构建完成的索引
    """
    import faiss

    n_vectors, dim = embeddings.shape
    logger.info(f"构建 FAISS 索引: type={index_type}, "
                f"n={n_vectors}, dim={dim}, nlist={nlist}")

    start_time = time.time()

    if index_type == "flat":
        # 暴力搜索（小规模精确检索）
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    elif index_type == "ivf_flat":
        # IVF-Flat（中大规模近似检索，精度高）
        # 自适应 nlist：至少保证每个聚类有 39 个样本用于训练
        effective_nlist = min(nlist, max(1, n_vectors // 39))
        if effective_nlist != nlist:
            logger.info(f"  nlist 自适应调整: {nlist} → {effective_nlist} "
                        f"(确保 n_vectors/nlist >= 39)")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist,
                                   faiss.METRIC_INNER_PRODUCT)

        # GPU 训练（如果可用）
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info("  使用 GPU 加速训练...")
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            gpu_index.train(embeddings)
            gpu_index.add(embeddings)
            # 转回 CPU 以便保存
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            index.train(embeddings)
            index.add(embeddings)

        # 设置搜索时的 nprobe
        index.nprobe = min(nprobe, effective_nlist)

    elif index_type == "ivf_pq":
        # IVF-PQ（大规模量化检索，内存高效）
        effective_nlist = min(nlist, max(1, n_vectors // 39))
        # PQ 子空间数必须能整除维度
        m = 8  # 默认 8 个子空间
        while dim % m != 0 and m > 1:
            m -= 1
        nbits = 8

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, effective_nlist, m, nbits,
                                 faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(nprobe, effective_nlist)

    else:
        raise ValueError(f"不支持的索引类型: {index_type}")

    elapsed = time.time() - start_time
    logger.info(f"索引构建完成: ntotal={index.ntotal}, 耗时 {elapsed:.1f}s")

    return index


def verify_index(index, embeddings: np.ndarray,
                 corpus: List[Dict[str, str]],
                 n_queries: int = 10) -> Dict[str, Any]:
    """
    验证索引质量（自检索 + 随机查询）

    参数:
        index: faiss.Index, 构建完成的索引
        embeddings: np.ndarray, 编码向量
        corpus: List[Dict], 语料库
        n_queries: int, 测试查询数量

    返回:
        Dict: 验证结果（recall@1, recall@10, 平均延迟等）
    """
    import faiss

    logger.info(f"验证索引质量 (n_queries={n_queries})...")

    n = len(corpus)
    # 随机选取 n_queries 个样本进行自检索
    rng = np.random.RandomState(42)
    query_ids = rng.choice(n, size=min(n_queries, n), replace=False)

    query_vectors = embeddings[query_ids]

    # 搜索
    start = time.time()
    scores, indices = index.search(query_vectors, 10)
    latency = (time.time() - start) / len(query_ids) * 1000  # ms/query

    # 计算 recall@1 和 recall@10
    recall_1 = sum(1 for i, qid in enumerate(query_ids) if qid in indices[i, :1]) / len(query_ids)
    recall_10 = sum(1 for i, qid in enumerate(query_ids) if qid in indices[i, :10]) / len(query_ids)

    result = {
        "recall_at_1": recall_1,
        "recall_at_10": recall_10,
        "avg_latency_ms": latency,
        "n_queries": len(query_ids),
    }

    logger.info(f"验证结果: recall@1={recall_1:.2%}, recall@10={recall_10:.2%}, "
                f"延迟={latency:.1f}ms/query")

    # 打印几个样例
    for i in range(min(3, len(query_ids))):
        qid = query_ids[i]
        q_name = corpus[qid].get("name", "???")
        top1_id = indices[i, 0]
        top1_name = corpus[top1_id].get("name", "???") if top1_id < n else "???"
        top1_score = scores[i, 0]
        match = "✓" if top1_id == qid else "✗"
        logger.info(f"  样例 {i+1}: query={q_name} → top1={top1_name} "
                    f"(score={top1_score:.4f}) {match}")

    return result


def save_index_and_meta(index, corpus: List[Dict[str, str]],
                        index_path: str, meta_path: str,
                        stats_path: str, stats: Dict[str, Any]):
    """
    保存索引、元数据和统计信息

    参数:
        index: faiss.Index, FAISS 索引
        corpus: List[Dict], 语料库
        index_path: str, 索引输出路径
        meta_path: str, 元数据输出路径
        stats_path: str, 统计信息输出路径
        stats: Dict, 包含构建统计和验证结果
    """
    import faiss

    # 确保目录存在
    for p in [index_path, meta_path, stats_path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    # 保存 FAISS 索引
    faiss.write_index(index, index_path)
    index_size_mb = os.path.getsize(index_path) / 1024 / 1024
    logger.info(f"FAISS 索引已保存: {index_path} ({index_size_mb:.1f} MB)")

    # 保存语料库元数据（JSON 格式，供 retriever.py 加载）
    meta = []
    for doc in corpus:
        meta.append({
            "name": doc.get("name", ""),
            "type_expr": doc.get("type_expr", ""),
            "module_path": doc.get("module_path", ""),
            "docstring": doc.get("docstring", doc.get("doc_text", "")),
        })

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    meta_size_mb = os.path.getsize(meta_path) / 1024 / 1024
    logger.info(f"元数据已保存: {meta_path} ({meta_size_mb:.1f} MB)")

    # 保存统计信息
    stats["index_file_size_mb"] = index_size_mb
    stats["meta_file_size_mb"] = meta_size_mb
    stats["build_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_path}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="构建 FAISS 向量索引")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "retrieval.yaml"),
                        help="配置文件路径")
    parser.add_argument("--corpus", type=str, default=None,
                        help="语料库文件路径（覆盖配置）")
    parser.add_argument("--model", type=str, default=None,
                        help="编码器模型名称（覆盖配置）")
    parser.add_argument("--index-type", type=str, default=None,
                        choices=["flat", "ivf_flat", "ivf_pq"],
                        help="索引类型（覆盖配置）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="编码批次大小（覆盖配置）")
    parser.add_argument("--gpu", action="store_true",
                        help="使用 GPU 加速索引构建")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="保存原始向量到 .npy 文件")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过索引质量验证")
    args = parser.parse_args()

    # 加载配置
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"已加载配置: {args.config}")

    global_config = config.get("global", {})
    encoder_config = config.get("encoder", {})
    faiss_config = config.get("faiss_index", {})
    corpus_config = config.get("corpus", {})
    project_root = global_config.get("project_root", PROJECT_ROOT)

    def resolve_path(p):
        """将相对路径解析为绝对路径"""
        if p and not os.path.isabs(p):
            return os.path.join(project_root, p)
        return p

    # 解析参数（命令行 > 配置文件 > 默认值）
    corpus_path = args.corpus or resolve_path(
        corpus_config.get("output_path", "data/vector_db/corpus.jsonl")
    )
    model_name = args.model or encoder_config.get("model_name", "intfloat/e5-large-v2")
    index_type = args.index_type or faiss_config.get("type", "ivf_flat")
    batch_size = args.batch_size or encoder_config.get("batch_size", 64)
    nlist = faiss_config.get("nlist", 256)
    nprobe = faiss_config.get("nprobe", 32)
    device = encoder_config.get("device", "cuda")
    use_gpu = args.gpu or faiss_config.get("gpu_build", False)
    model_cache_dir = resolve_path(encoder_config.get("cache_dir")) if encoder_config.get("cache_dir") else None

    # 输出路径
    index_path = resolve_path(faiss_config.get("index_path", "data/vector_db/faiss_index.bin"))
    meta_path = resolve_path(faiss_config.get("meta_path", "data/vector_db/corpus_meta.json"))
    stats_path = resolve_path(faiss_config.get("stats_path", "data/vector_db/index_stats.json"))
    embeddings_path = resolve_path("data/vector_db/embeddings.npy") if args.save_embeddings else None

    # 打印配置摘要
    logger.info(f"\n{'='*60}")
    logger.info(f"FAISS 索引构建配置:")
    logger.info(f"  编码器: {model_name}")
    logger.info(f"  设备: {device}")
    logger.info(f"  索引类型: {index_type}")
    logger.info(f"  nlist: {nlist}, nprobe: {nprobe}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  GPU 加速: {use_gpu}")
    logger.info(f"  语料库: {corpus_path}")
    logger.info(f"  索引输出: {index_path}")
    logger.info(f"{'='*60}\n")

    # Step 1: 加载语料库
    if not os.path.exists(corpus_path):
        logger.error(f"语料库文件不存在: {corpus_path}")
        logger.info("请先运行: python scripts/build_retrieval_corpus.py")
        return False

    corpus = load_corpus(corpus_path)
    if not corpus:
        logger.error("语料库为空，终止构建")
        return False

    stats = {"corpus_size": len(corpus)}

    # Step 2: 加载编码器
    encoder = load_encoder(model_name, device=device, cache_dir=model_cache_dir)

    # Step 3: 编码语料库
    embeddings = encode_corpus(
        encoder, corpus,
        batch_size=batch_size,
        text_field="doc_text",
        prefix="passage: ",
        save_embeddings_path=embeddings_path,
    )
    stats["embedding_dim"] = int(embeddings.shape[1])
    stats["encoding_device"] = device

    # Step 4: 构建索引
    index = build_faiss_index(
        embeddings,
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        use_gpu=use_gpu,
    )
    stats["index_type"] = index_type
    stats["nlist"] = nlist
    stats["nprobe"] = nprobe
    stats["ntotal"] = index.ntotal

    # Step 5: 验证索引
    if not args.skip_verify:
        verify_result = verify_index(index, embeddings, corpus, n_queries=50)
        stats["verification"] = verify_result
    else:
        logger.info("跳过索引验证")

    # Step 6: 保存
    save_index_and_meta(index, corpus, index_path, meta_path, stats_path, stats)

    # 最终摘要
    logger.info(f"\n{'='*60}")
    logger.info(f"索引构建完成!")
    logger.info(f"  语料库: {len(corpus)} 条文档")
    logger.info(f"  向量维度: {embeddings.shape[1]}")
    logger.info(f"  索引向量数: {index.ntotal}")
    logger.info(f"  索引文件: {index_path}")
    logger.info(f"  元数据文件: {meta_path}")
    if not args.skip_verify:
        logger.info(f"  Recall@1: {verify_result['recall_at_1']:.2%}")
        logger.info(f"  Recall@10: {verify_result['recall_at_10']:.2%}")
        logger.info(f"  查询延迟: {verify_result['avg_latency_ms']:.1f}ms")
    logger.info(f"{'='*60}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
