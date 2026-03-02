"""
RAG-CoS-MCTS 数据构建引擎 (Phase 2)
@author ygw

模块:
    ingestion: Mathlib4 全量追踪数据摄入
    cos_extractor: 状态链 (CoS) 提取
    thought_backtrans: Thought 回标（Teacher Model）
    augmentation: 错误注入 + 合成定理数据增强
    error_verifier: Pantograph 错误验证
    pipeline: 流水线编排器
"""
