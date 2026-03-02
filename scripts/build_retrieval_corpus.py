"""
语料库构建脚本 —— 从 traces 数据提取检索语料库
@author ygw
创建日期: 2026-02-28

从 traces_with_states.jsonl 中提取定理信息，构建用于稠密检索的语料库。
同时支持从 Pantograph env_catalog 直接提取（用于补充或独立使用）。

运行方式:
    cd /root/autodl-tmp/RTAP
    python scripts/build_retrieval_corpus.py
    python scripts/build_retrieval_corpus.py --source pantograph  # 从 Lean 环境提取
    python scripts/build_retrieval_corpus.py --config configs/retrieval.yaml

输出:
    data/vector_db/corpus.jsonl     — 语料库（每行一个定理的 JSON）
    data/vector_db/corpus_stats.json — 语料库统计信息
"""

import sys
import os
import json
import time
import logging
import argparse
import re
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import Counter

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
        config_path: 配置文件路径

    返回:
        Dict: 配置字典
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_module_name(file_path: str) -> str:
    """
    从文件路径提取模块名

    参数:
        file_path: Lean 源码文件路径（如 .../Mathlib/Algebra/AddConstMap/Basic.lean）

    返回:
        str: 模块名（如 Mathlib.Algebra.AddConstMap.Basic）
    """
    # 匹配 Mathlib/ 或 Init/ 开头的路径
    patterns = [
        r'(Mathlib/[\w/]+)\.lean',
        r'(Init/[\w/]+)\.lean',
        r'(Lean/[\w/]+)\.lean',
    ]
    for pattern in patterns:
        match = re.search(pattern, file_path)
        if match:
            return match.group(1).replace('/', '.')
    # 回退：取文件名部分
    basename = os.path.basename(file_path)
    return basename.replace('.lean', '')


def extract_from_traces(input_path: str,
                        name_field: str = "theorem_full_name",
                        type_field: str = "theorem_type",
                        module_field: str = "file_path",
                        deduplicate: bool = True) -> List[Dict[str, str]]:
    """
    从 traces JSONL 文件提取定理语料库

    参数:
        input_path: traces JSONL 文件路径
        name_field: 定理名称字段名
        type_field: 定理类型字段名
        module_field: 模块路径字段名
        deduplicate: 是否按名称去重

    返回:
        List[Dict]: 语料库条目列表
    """
    logger.info(f"从 traces 提取语料库: {input_path}")

    corpus = []
    seen_names: Set[str] = set()
    total_lines = 0
    skipped_dup = 0
    skipped_empty = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if total_lines % 10000 == 0:
                logger.info(f"  已处理 {total_lines} 行...")

            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            name = record.get(name_field, "").strip()
            type_expr = record.get(type_field, "").strip()
            file_path = record.get(module_field, "")

            # 跳过无名称或无类型的记录
            if not name or not type_expr:
                skipped_empty += 1
                continue

            # 去重
            if deduplicate and name in seen_names:
                skipped_dup += 1
                continue
            seen_names.add(name)

            # 提取模块名
            module_path = extract_module_name(file_path) if file_path else ""

            # 提取额外属性
            proof_steps = record.get("proof_steps", 0)
            proof_mode = record.get("proof_mode", "")
            namespace = record.get("namespace", "")

            # 构建语料库的文本表示（用于编码）
            # 格式: "name : type_expr" （E5 模型编码时会加 passage: 前缀）
            doc_text = f"{name} : {type_expr}"

            corpus.append({
                "name": name,
                "type_expr": type_expr,
                "module_path": module_path,
                "doc_text": doc_text,
                "proof_steps": proof_steps,
                "proof_mode": proof_mode,
                "namespace": namespace,
            })

    logger.info(f"traces 提取完成: {total_lines} 行 → {len(corpus)} 个定理"
                f"（去重跳过 {skipped_dup}，空值跳过 {skipped_empty}）")

    return corpus


def extract_from_pantograph(project_path: str,
                            imports: List[str] = None,
                            batch_size: int = 500,
                            timeout: int = 300) -> List[Dict[str, str]]:
    """
    从 Pantograph 环境直接提取定理语料库

    使用 Task 1.1 的 LeanServer 适配器层，通过 env_catalog + env_inspect
    提取环境中所有符号的类型信息。

    参数:
        project_path: Lean 项目路径
        imports: 导入的模块列表
        batch_size: 每批 inspect 的数量
        timeout: 总超时（秒）

    返回:
        List[Dict]: 语料库条目列表
    """
    from src.common.lean_server import LeanServer

    logger.info(f"从 Pantograph 提取语料库: project_path={project_path}, imports={imports}")

    server = LeanServer(
        imports=imports or ["Init"],
        project_path=project_path,
        timeout=120,
    )

    if not server.start():
        logger.error("Pantograph 服务器启动失败")
        return []

    try:
        # 获取所有符号名
        logger.info("获取环境符号列表...")
        all_names = server.env_catalog()
        logger.info(f"符号总数: {len(all_names)}")

        corpus = []
        failed = 0
        start_time = time.time()

        for i, name in enumerate(all_names):
            # 超时检查
            if time.time() - start_time > timeout:
                logger.warning(f"超时 ({timeout}s)，已提取 {len(corpus)} 个定理")
                break

            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  进度: {i+1}/{len(all_names)} "
                            f"({100*(i+1)/len(all_names):.1f}%) "
                            f"速率: {rate:.0f}/s, 已提取: {len(corpus)}")

            try:
                info = server.env_inspect(name)
                if info and "type" in info:
                    type_info = info["type"]
                    if isinstance(type_info, dict):
                        type_expr = type_info.get("pp", str(type_info))
                    else:
                        type_expr = str(type_info)

                    if type_expr:
                        # 提取模块名（从符号名推断）
                        parts = name.rsplit('.', 1)
                        module_path = parts[0] if len(parts) > 1 else ""

                        corpus.append({
                            "name": name,
                            "type_expr": type_expr,
                            "module_path": module_path,
                            "doc_text": f"{name} : {type_expr}",
                            "proof_steps": 0,
                            "proof_mode": "",
                            "namespace": module_path,
                        })
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.warning(f"  inspect 失败 ({name}): {e}")

        elapsed = time.time() - start_time
        logger.info(f"Pantograph 提取完成: {len(corpus)} 个定理, "
                    f"失败 {failed}, 耗时 {elapsed:.1f}s")

        return corpus

    finally:
        server.stop()


def compute_stats(corpus: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    计算语料库统计信息

    参数:
        corpus: 语料库条目列表

    返回:
        Dict: 统计信息字典
    """
    # 模块分布
    module_counter = Counter()
    namespace_counter = Counter()
    type_lengths = []
    name_lengths = []

    for doc in corpus:
        module = doc.get("module_path", "")
        if module:
            # 取顶级模块
            top_module = module.split('.')[0]
            module_counter[top_module] += 1
        namespace = doc.get("namespace", "")
        if namespace:
            namespace_counter[namespace] += 1

        type_lengths.append(len(doc.get("type_expr", "")))
        name_lengths.append(len(doc.get("name", "")))

    stats = {
        "total_documents": len(corpus),
        "unique_top_modules": dict(module_counter.most_common(20)),
        "top_namespaces": dict(namespace_counter.most_common(20)),
        "type_expr_length": {
            "mean": sum(type_lengths) / len(type_lengths) if type_lengths else 0,
            "max": max(type_lengths) if type_lengths else 0,
            "min": min(type_lengths) if type_lengths else 0,
        },
        "name_length": {
            "mean": sum(name_lengths) / len(name_lengths) if name_lengths else 0,
            "max": max(name_lengths) if name_lengths else 0,
            "min": min(name_lengths) if name_lengths else 0,
        },
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return stats


def save_corpus(corpus: List[Dict[str, str]],
                output_path: str,
                stats_path: str):
    """
    保存语料库到 JSONL 文件

    参数:
        corpus: 语料库条目列表
        output_path: JSONL 输出路径
        stats_path: 统计信息 JSON 输出路径
    """
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)

    # 保存 JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"语料库已保存: {output_path} ({len(corpus)} 条)")

    # 保存统计信息
    stats = compute_stats(corpus)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_path}")

    # 打印摘要
    logger.info(f"\n{'='*50}")
    logger.info(f"语料库摘要:")
    logger.info(f"  总文档数: {stats['total_documents']}")
    logger.info(f"  类型表达式平均长度: {stats['type_expr_length']['mean']:.1f}")
    logger.info(f"  顶级模块分布: {dict(list(stats['unique_top_modules'].items())[:5])}")
    logger.info(f"{'='*50}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="构建检索语料库")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "retrieval.yaml"),
                        help="配置文件路径")
    parser.add_argument("--source", type=str, choices=["traces", "pantograph", "both"],
                        default=None, help="数据源（覆盖配置文件）")
    parser.add_argument("--input", type=str, default=None,
                        help="traces 输入文件路径（覆盖配置文件）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（覆盖配置文件）")
    args = parser.parse_args()

    # 加载配置
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"已加载配置: {args.config}")
    else:
        logger.warning(f"配置文件不存在: {args.config}，使用默认值")

    corpus_config = config.get("corpus", {})
    global_config = config.get("global", {})
    project_root = global_config.get("project_root", PROJECT_ROOT)

    # 确定数据源
    source = args.source or corpus_config.get("source", "traces")

    # 确定路径
    def resolve_path(p):
        """将相对路径解析为绝对路径"""
        if p and not os.path.isabs(p):
            return os.path.join(project_root, p)
        return p

    output_path = args.output or resolve_path(corpus_config.get("output_path", "data/vector_db/corpus.jsonl"))
    stats_path = resolve_path(corpus_config.get("stats_path", "data/vector_db/corpus_stats.json"))

    logger.info(f"数据源: {source}")
    logger.info(f"输出路径: {output_path}")

    corpus = []

    # ---- traces 提取 ----
    if source in ("traces", "both"):
        traces_config = corpus_config.get("traces", {})
        input_path = args.input or resolve_path(
            traces_config.get("input_path", "data/processed/traces/traces_with_states.jsonl")
        )

        if not os.path.exists(input_path):
            logger.error(f"traces 文件不存在: {input_path}")
            return False

        traces_corpus = extract_from_traces(
            input_path=input_path,
            name_field=traces_config.get("name_field", "theorem_full_name"),
            type_field=traces_config.get("type_field", "theorem_type"),
            module_field=traces_config.get("module_field", "file_path"),
            deduplicate=traces_config.get("deduplicate", True),
        )
        corpus.extend(traces_corpus)

    # ---- Pantograph 提取 ----
    if source in ("pantograph", "both"):
        panto_config = corpus_config.get("pantograph", {})
        panto_project = resolve_path(
            panto_config.get("project_path", "workspace/PyPantograph")
        )
        panto_imports = panto_config.get("imports", ["Init"])

        panto_corpus = extract_from_pantograph(
            project_path=panto_project,
            imports=panto_imports,
            batch_size=panto_config.get("batch_size", 500),
            timeout=panto_config.get("timeout", 300),
        )

        # 合并并去重
        seen = {doc["name"] for doc in corpus}
        new_count = 0
        for doc in panto_corpus:
            if doc["name"] not in seen:
                corpus.append(doc)
                seen.add(doc["name"])
                new_count += 1
        logger.info(f"Pantograph 补充 {new_count} 个新定理")

    if not corpus:
        logger.error("语料库为空，请检查数据源配置")
        return False

    # 保存
    save_corpus(corpus, output_path, stats_path)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
