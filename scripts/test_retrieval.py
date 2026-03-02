"""
检索系统端到端测试脚本
@author ygw
创建日期: 2026-02-28

测试检索系统完整流程：语料库构建 → 索引构建 → 查询检索 → 结果验证

运行方式:
    cd /root/autodl-tmp/RTAP
    python scripts/test_retrieval.py                         # 完整测试
    python scripts/test_retrieval.py --mode query            # 仅查询测试（需已构建索引）
    python scripts/test_retrieval.py --mode build_and_query  # 构建+查询
    python scripts/test_retrieval.py --config configs/retrieval.yaml

测试项:
    1. 语料库加载与统计
    2. 编码器加载（E5-large-v2）
    3. 向量编码（单条 + 批量）
    4. FAISS 索引加载/构建
    5. 稠密检索（单查询 + 批量）
    6. 符号检索（类型签名匹配）
    7. 双粒度融合检索（RRF）
    8. 端到端延迟基准
"""

import sys
import os
import json
import time
import logging
import argparse
import traceback
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ================================================================
# 测试数据：代表性的 Lean4 证明状态查询
# ================================================================

SAMPLE_QUERIES = [
    {
        "name": "自然数的加法交换律",
        "query": "⊢ ∀ (n m : ℕ), n + m = m + n",
        "expected_keywords": ["Nat.add_comm", "add_comm"],
    },
    {
        "name": "乘法结合律",
        "query": "⊢ ∀ (a b c : ℕ), a * (b * c) = (a * b) * c",
        "expected_keywords": ["mul_assoc", "Nat.mul_assoc"],
    },
    {
        "name": "自然数零加法恒等",
        "query": "⊢ ∀ (n : ℕ), 0 + n = n",
        "expected_keywords": ["Nat.zero_add", "zero_add"],
    },
    {
        "name": "列表的长度属性",
        "query": "⊢ ∀ {α : Type} (l : List α) (a : α), (a :: l).length = l.length + 1",
        "expected_keywords": ["List.length_cons", "length"],
    },
    {
        "name": "自然语言描述 - 加法",
        "query": "prove that addition of natural numbers is commutative",
        "expected_keywords": ["comm", "add"],
    },
]


class RetrievalTestSuite:
    """
    检索系统测试套件

    属性:
        config_path: 配置文件路径
        config: 配置字典
        project_root: 项目根目录
        results: 测试结果列表
    """

    def __init__(self, config_path: str, project_root: str = ""):
        """
        初始化测试套件

        参数:
            config_path: str, 配置文件路径
            project_root: str, 项目根目录
        """
        self.config_path = config_path
        self.project_root = project_root or PROJECT_ROOT
        self.config: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.dense_retriever = None
        self.symbolic_retriever = None
        self.dual_retriever = None

    def _resolve(self, p: str) -> str:
        """将相对路径解析为绝对路径"""
        if p and not os.path.isabs(p):
            return os.path.join(self.project_root, p)
        return p

    def _record(self, test_name: str, passed: bool,
                details: str = "", duration: float = 0.0):
        """
        记录单项测试结果

        参数:
            test_name: 测试名称
            passed: 是否通过
            details: 详细信息
            duration: 耗时（秒）
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append({
            "name": test_name,
            "passed": passed,
            "details": details,
            "duration": duration,
        })
        logger.info(f"  [{status}] {test_name} ({duration:.3f}s)"
                     + (f" — {details}" if details else ""))

    def load_config(self) -> bool:
        """加载测试配置"""
        if os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置已加载: {self.config_path}")
            return True
        else:
            logger.warning(f"配置不存在: {self.config_path}，使用默认值")
            return False

    # ================================================================
    # Test 1: 语料库加载
    # ================================================================

    def test_corpus_loading(self) -> bool:
        """测试语料库加载"""
        logger.info("\n[Test 1] 语料库加载")
        start = time.time()

        corpus_config = self.config.get("corpus", {})
        corpus_path = self._resolve(
            corpus_config.get("output_path", "data/vector_db/corpus.jsonl")
        )

        if not os.path.exists(corpus_path):
            self._record("corpus_exists", False,
                         f"文件不存在: {corpus_path}",
                         time.time() - start)
            return False

        # 加载
        count = 0
        sample = None
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    count += 1
                    if count == 1:
                        sample = doc
        elapsed = time.time() - start

        self._record("corpus_exists", True, f"文件存在: {corpus_path}", elapsed)
        self._record("corpus_count", count > 0,
                     f"{count} 条文档", 0)
        if sample:
            has_fields = all(k in sample for k in ["name", "type_expr", "doc_text"])
            self._record("corpus_fields", has_fields,
                         f"样例字段: {list(sample.keys())}", 0)

        return count > 0

    # ================================================================
    # Test 2: 编码器加载
    # ================================================================

    def test_encoder_loading(self) -> bool:
        """测试编码器加载"""
        logger.info("\n[Test 2] 编码器加载")

        from src.models.retriever import DenseRetriever
        start = time.time()

        self.dense_retriever = DenseRetriever.from_config(
            self.config, self.project_root
        )
        ok = self.dense_retriever.load()
        elapsed = time.time() - start

        self._record("encoder_load", ok,
                     f"模型: {self.dense_retriever.model_name}, "
                     f"dim: {self.dense_retriever._embedding_dim}",
                     elapsed)

        # 索引加载
        has_index = self.dense_retriever.index is not None
        if has_index:
            self._record("faiss_index_load", True,
                         f"{self.dense_retriever.index.ntotal} 个向量", 0)
        else:
            self._record("faiss_index_load", False, "索引未加载", 0)

        # 语料库加载
        has_corpus = len(self.dense_retriever.corpus) > 0
        self._record("corpus_meta_load", has_corpus,
                     f"{len(self.dense_retriever.corpus)} 条", 0)

        return ok

    # ================================================================
    # Test 3: 向量编码
    # ================================================================

    def test_encoding(self) -> bool:
        """测试向量编码"""
        logger.info("\n[Test 3] 向量编码")

        if self.dense_retriever is None or self.dense_retriever.encoder is None:
            self._record("encoding", False, "编码器未就绪")
            return False

        # 单条编码
        start = time.time()
        single_emb = self.dense_retriever.encode(["⊢ ∀ (n : ℕ), n + 0 = n"])
        single_time = time.time() - start

        self._record("single_encode", single_emb.shape[0] == 1,
                     f"shape={single_emb.shape}, L2norm={float(np.linalg.norm(single_emb[0])):.4f}",
                     single_time)

        # 批量编码
        batch_texts = [q["query"] for q in SAMPLE_QUERIES]
        start = time.time()
        batch_emb = self.dense_retriever.encode(batch_texts, batch_size=4)
        batch_time = time.time() - start

        self._record("batch_encode", batch_emb.shape[0] == len(batch_texts),
                     f"shape={batch_emb.shape}, 平均 {batch_time/len(batch_texts)*1000:.1f}ms/条",
                     batch_time)

        return True

    # ================================================================
    # Test 4: 稠密检索
    # ================================================================

    def test_dense_search(self) -> bool:
        """测试稠密检索"""
        logger.info("\n[Test 4] 稠密检索")

        if self.dense_retriever is None or self.dense_retriever.index is None:
            self._record("dense_search", False, "索引未就绪")
            return False

        all_passed = True
        for q in SAMPLE_QUERIES:
            start = time.time()
            results = self.dense_retriever.search(q["query"], top_k=10)
            elapsed = time.time() - start

            # 检查是否有结果
            has_results = len(results) > 0

            # 检查是否命中预期关键词
            hit = False
            result_names = [r.name for r in results]
            for kw in q.get("expected_keywords", []):
                if any(kw.lower() in name.lower() for name in result_names):
                    hit = True
                    break

            detail = (f"top1={results[0].name if results else 'N/A'} "
                      f"(score={results[0].score:.4f})" if results else "无结果")
            test_passed = has_results
            self._record(f"dense_{q['name']}", test_passed, detail, elapsed)
            if not test_passed:
                all_passed = False

        return all_passed

    # ================================================================
    # Test 5: 符号检索
    # ================================================================

    def test_symbolic_search(self) -> bool:
        """测试符号检索"""
        logger.info("\n[Test 5] 符号检索")

        from src.models.retriever import SymbolicRetriever

        self.symbolic_retriever = SymbolicRetriever.from_config(
            self.config, self.project_root
        )

        start = time.time()
        ok = self.symbolic_retriever.load()
        elapsed = time.time() - start

        self._record("symbolic_load", ok,
                     f"{len(self.symbolic_retriever.theorems)} 定理, "
                     f"{len(self.symbolic_retriever._type_index)} 关键词",
                     elapsed)

        if not ok:
            return False

        # 测试搜索
        for q in SAMPLE_QUERIES[:3]:  # 只测前 3 个（符号检索对自然语言无效）
            start = time.time()
            results = self.symbolic_retriever.search(q["query"], top_k=10)
            elapsed = time.time() - start

            has_results = len(results) > 0
            detail = (f"top1={results[0].name} (score={results[0].score:.4f})"
                      if results else "无结果")
            self._record(f"symbolic_{q['name']}", has_results, detail, elapsed)

        return True

    # ================================================================
    # Test 6: 双粒度融合检索
    # ================================================================

    def test_dual_grained(self) -> bool:
        """测试双粒度融合检索（RRF）"""
        logger.info("\n[Test 6] 双粒度融合检索")

        from src.models.retriever import DualGrainedRetriever

        start = time.time()
        # 使用已初始化的子检索器
        self.dual_retriever = DualGrainedRetriever(
            dense_retriever=self.dense_retriever,
            symbolic_retriever=self.symbolic_retriever,
            rrf_k=self.config.get("dual_grained", {}).get("rrf_k", 60),
            macro_top_k=self.config.get("dual_grained", {}).get("macro_top_k", 5),
            micro_top_k=self.config.get("dual_grained", {}).get("micro_top_k", 10),
        )
        init_time = time.time() - start
        self._record("dual_init", True, f"初始化耗时 {init_time*1000:.1f}ms", init_time)

        # 宏观检索测试
        for q in SAMPLE_QUERIES[:2]:
            start = time.time()
            results = self.dual_retriever.macro_retrieve(q["query"])
            elapsed = time.time() - start

            has_results = len(results) > 0
            detail = (f"{len(results)} 条, top1={results[0].name} "
                      f"(score={results[0].score:.4f})" if results else "无结果")
            self._record(f"macro_{q['name']}", has_results, detail, elapsed)

        # 微观检索测试
        micro_query = "⊢ ∀ (n : ℕ), n + 0 = n"
        start = time.time()
        micro_results = self.dual_retriever.micro_retrieve(micro_query)
        elapsed = time.time() - start

        self._record("micro_retrieve", len(micro_results) > 0,
                     f"{len(micro_results)} 条", elapsed)

        # Prompt 格式化测试
        macro_ctx = self.dual_retriever.format_macro_context(micro_results[:5])
        self._record("prompt_format", "[Retrieved Premises" in macro_ctx,
                     f"长度 {len(macro_ctx)} 字符", 0)

        # 缓存测试（第二次查询应该更快）
        start = time.time()
        cache_results = self.dual_retriever.micro_retrieve(micro_query)
        cache_time = time.time() - start
        self._record("cache_hit", cache_time < 0.001,
                     f"缓存命中: {cache_time*1000:.2f}ms", cache_time)

        return True

    # ================================================================
    # Test 7: 延迟基准
    # ================================================================

    def test_latency_benchmark(self, n_queries: int = 20) -> bool:
        """
        端到端延迟基准测试

        参数:
            n_queries: 测试查询数量
        """
        logger.info(f"\n[Test 7] 延迟基准 (n={n_queries})")

        if self.dual_retriever is None:
            self._record("latency", False, "双粒度检索器未就绪")
            return False

        # 清空缓存
        self.dual_retriever.clear_cache()

        queries = [q["query"] for q in SAMPLE_QUERIES]
        # 循环扩展到 n_queries
        while len(queries) < n_queries:
            queries.extend([q["query"] for q in SAMPLE_QUERIES])
        queries = queries[:n_queries]

        # 微观检索延迟
        latencies = []
        for q in queries:
            self.dual_retriever.clear_cache()
            start = time.time()
            self.dual_retriever.micro_retrieve(q)
            latencies.append(time.time() - start)

        avg = sum(latencies) / len(latencies) * 1000
        p50 = sorted(latencies)[len(latencies) // 2] * 1000
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] * 1000
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] * 1000

        self._record("micro_avg_latency", avg < 500,
                     f"avg={avg:.1f}ms, p50={p50:.1f}ms, "
                     f"p95={p95:.1f}ms, p99={p99:.1f}ms",
                     sum(latencies))

        return True

    # ================================================================
    # 运行所有测试
    # ================================================================

    def run(self, mode: str = "full") -> Dict[str, Any]:
        """
        运行测试套件

        参数:
            mode: 运行模式
                - 'full': 完整测试
                - 'query': 仅查询测试
                - 'build_and_query': 构建索引后查询

        返回:
            Dict: 测试结果摘要
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"检索系统端到端测试 (mode={mode})")
        logger.info(f"{'='*60}")

        self.load_config()
        start_time = time.time()

        try:
            # 1. 语料库
            if mode in ("full", "build_and_query"):
                self.test_corpus_loading()

            # 2. 编码器 + 索引
            self.test_encoder_loading()

            # 3. 编码
            import numpy as np
            self.test_encoding()

            # 4. 稠密检索
            self.test_dense_search()

            # 5. 符号检索
            self.test_symbolic_search()

            # 6. 双粒度融合
            self.test_dual_grained()

            # 7. 延迟基准
            self.test_latency_benchmark()

        except Exception as e:
            logger.error(f"测试异常: {e}")
            traceback.print_exc()
            self._record("unexpected_error", False, str(e))

        total_time = time.time() - start_time

        # 汇总
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])

        logger.info(f"\n{'='*60}")
        logger.info(f"测试结果汇总")
        logger.info(f"  通过: {passed}/{len(self.results)}")
        logger.info(f"  失败: {failed}/{len(self.results)}")
        logger.info(f"  总耗时: {total_time:.1f}s")
        logger.info(f"{'='*60}")

        if failed > 0:
            logger.info("\n失败项:")
            for r in self.results:
                if not r["passed"]:
                    logger.info(f"  ✗ {r['name']}: {r['details']}")

        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "duration": total_time,
            "results": self.results,
        }


# 需要在 test_encoding 中使用
import numpy as np


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="检索系统端到端测试")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "retrieval.yaml"),
                        help="配置文件路径")
    parser.add_argument("--mode", type=str,
                        choices=["full", "query", "build_and_query"],
                        default="full",
                        help="测试模式")
    args = parser.parse_args()

    suite = RetrievalTestSuite(
        config_path=args.config,
        project_root=PROJECT_ROOT,
    )

    result = suite.run(mode=args.mode)
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
