#Lean-RCM
# RAG-CoS-MCTS 协作式定理证明框架

**Retrieval-Augmented and Metavariable-Aware Chain of States with Monte Carlo Tree Search**

---

## 项目简介

RCM 是一个面向 Lean4 的自动定理证明系统，基于三项核心创新：

1. **DG-RASP**（双粒度检索增强状态规划）— 宏观层锚定中间状态，微观层生成具体策略
2. **MAGC-MCTS**（元变量感知目标条件 MCTS）— 外层状态树 + 内层战术树的两级搜索
3. **RCRL**（反思认知修复循环）— 区分 ETR/ESR 路径的智能错误修复

基座模型：DeepSeek-Prover-V2-7B + LoRA
形式化内核：Pantograph REPL (Lean 4)
知识库：Mathlib4 (210K+ 定理)

## 项目结构

```
RTAP/
├── configs/                     # 配置文件
│   ├── data_pipeline.yaml       # Phase 2 数据流水线配置
│   ├── search.yaml              # MAGC-MCTS + DG-RASP + RCRL 搜索配置
│   └── training_sft.yaml        # Phase 3 SFT 训练配置
├── data/                        # 数据存放
│   ├── raw/mathlib4/            # Mathlib4 原始仓库
│   ├── processed/               # 处理后的数据集
│   └── vector_db/               # FAISS 向量索引
├── src/                         # 核心源代码
│   ├── common/                  # 通用工具层
│   │   ├── ast_parser.py        #   Lean4 AST 解析
│   │   ├── lean_server.py       #   Pantograph REPL 客户端
│   │   └── utils.py             #   文件 I/O、日志、检查点
│   ├── data_engine/             # Phase 2 数据构建
│   │   ├── ingestion.py         #   2.1 Mathlib 全量追踪
│   │   ├── cos_extractor.py     #   2.1 状态链 (CoS) 提取
│   │   ├── thought_backtrans.py #   2.3 Thought 回标 (Teacher Model)
│   │   ├── augmentation.py      #   2.2 错误注入 + 合成定理
│   │   ├── error_verifier.py    #   错误验证 (Pantograph)
│   │   └── pipeline.py          #   流水线编排
│   ├── models/                  # 模型层
│   │   ├── generator.py         #   Thought-CoS-Tactic 生成器
│   │   ├── retriever.py         #   DG-RASP 双粒度检索器
│   │   └── verifier.py          #   Pantograph 验证器
│   ├── search/                  # 搜索引擎
│   │   ├── magc_mcts.py         #   MAGC-MCTS 两级搜索
│   │   ├── rcrl.py              #   RCRL 反思修复循环
│   │   └── state_manager.py     #   元变量感知状态管理
│   └── trainer/                 # 训练器
│       ├── sft_trainer.py       #   Phase 3 SFT (trl + LoRA)
│       └── expert_iteration.py  #   Phase 4 Expert Iteration
├── benchmarks/                  # 评测
│   └── minif2f_v2/              #   miniF2F-v2 数据集
├── workspace/                   # 依赖工具
│   ├── LeanDojo-v2/             #   LeanDojo 追踪工具
│   └── PyPantograph/            #   Pantograph Python 绑定
├── assets/                      # 设计文档
├── scripts/                     # 脚本工具
└── requirements.txt             # Python 依赖
```

## 五阶段实施计划

| 阶段 | 名称 | 内容 |
|------|------|------|
| Phase 1 | 基础设施 | Pantograph 集成 + DG-RASP 检索管线 |
| Phase 2 | 数据构建 | CoS 提取 + 检索数据 + Thought 标注 + 数据增强 |
| Phase 3 | SFT 训练 | DeepSeek-Prover-V2-7B LoRA 微调 |
| Phase 4 | MCTS + EI | MAGC-MCTS 搜索 + Expert Iteration 自我博弈 |
| Phase 5 | 评估消融 | miniF2F / ProofNet / FATE-M 评测 |

## 三项创新

### 1. DG-RASP — 双粒度检索增强状态规划

- **宏观层**：将当前证明状态编码后在 Mathlib 全量向量库中检索相似定理，锚定中间状态
- **微观层**：提取当前 Goal 的类型签名，在定理前提库中检索可用引理，辅助策略生成
- **融合**：Reciprocal Rank Fusion (RRF) 融合稠密检索与符号检索结果

### 2. MAGC-MCTS — 元变量感知两级搜索

- **外层状态树**：节点 = DG-RASP 宏观锚定的中间状态，UCB1 策略引导方向选择
- **内层战术树**：节点 = 连接相邻中间状态的底层 Tactic 序列，Pantograph 实时验证
- **元变量管理**：通过 Pantograph 追踪元变量绑定与传播，支持独立子目标并行搜索

### 3. RCRL — 反思认知修复循环

- **错误分类**：基于规则模式将 Lean4 内核错误分为 8 大类
- **反思诊断**：生成 `<reflection>` 标签的语义化分析，帮助 LLM 理解错误本质
- **动态分流**：
  - **ETR (Error-Tactic Repair)**：错误局限于当前策略 → 同状态修复
  - **ESR (Error-State Repair)**：错误源于上游规划 → 回退到外层树

## 快速开始

### 环境配置

```bash
conda create -n lean-egnp python=3.11
conda activate lean-egnp
pip install -r requirements.txt

cd workspace/PyPantograph
python build-pantograph.py
```

### 数据流水线 (Phase 2)

```bash
python -m src.data_engine.pipeline --config configs/data_pipeline.yaml
```

### SFT 训练 (Phase 3)

```bash
python -m src.trainer.sft_trainer --config configs/training_sft.yaml
```

### MCTS 搜索 + Expert Iteration (Phase 4)

```bash
python -m src.trainer.expert_iteration \
    --config configs/search.yaml \
    --theorems data/processed/theorem_pool.jsonl
```

## 评测指标

| 基准 | 指标 |
|------|------|
| miniF2F-test | Pass@1, Pass@32, Pass@64 |
| ProofNet | Pass@1 |
| FATE-M (抽象代数) | Pass@32 |

## 技术栈

- **基座模型**: DeepSeek-Prover-V2-7B
- **微调**: trl + peft (LoRA)
- **检索**: sentence-transformers + FAISS
- **形式化验证**: Pantograph REPL (Lean 4)
- **数据追踪**: LeanDojo-v2
- **知识库**: Mathlib4

## 相关链接

- [Lean4 官方文档](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 仓库](https://github.com/leanprover-community/mathlib4)
- [LeanDojo 文档](https://leandojo.org/)

---

@author hwj
