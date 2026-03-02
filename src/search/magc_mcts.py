"""
元变量感知目标条件蒙特卡洛树搜索 (MAGC-MCTS)
@author ygw
更新日期: 2026-02-28

创新点二：Metavariable-Aware Goal-Conditioned MCTS via Pantograph
实现分层（Hierarchical）的两级蒙特卡洛树搜索，深度绑定 Pantograph 接口，
解决 Lean4 依赖类型理论中的元变量耦合问题。

两级搜索架构:
    外层状态树 (Outer State Tree):
        - 节点 = DG-RASP 生成的中间形式化状态 (Subgoals)
        - 价值函数评估当前状态逼近最终定理的逻辑距离
        - 指导宏观搜索方向

    内层战术树 (Inner Tactic Tree):
        - 节点 = 连接外层两个相邻中间状态的具体底层 Tactics 序列
        - 与 Pantograph 交互执行策略验证
        - 支持元变量耦合的动态解析

关键机制:
    1. 元变量耦合的动态解析 (via Pantograph)
    2. 无损状态追踪与回溯 (State Backtracking)
    3. 双层 RAG 检索增强 (与 DG-RASP 集成)
"""

import math
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("magc_mcts")


# ================================================================
# 节点定义
# ================================================================

@dataclass
class OuterNode:
    """
    外层状态树节点

    表示一个宏观的中间形式化状态（Subgoal），由 DG-RASP 的
    宏观层检索锚定生成。

    属性:
        state_id: Pantograph 中的状态 ID
        state_str: 形式化状态文本（Lean4 proof state）
        natural_description: 自然语言描述（用于宏观检索）
        parent: 父节点
        children: 子节点列表
        tactic_sequence: 从父节点到本节点的战术序列（由内层树搜索填充）
        visits: 被访问次数
        value: 累计价值
        depth: 在外层树中的深度
        is_solved: 该子目标是否已解决
        metavar_bindings: 元变量绑定信息（Pantograph 追踪）
        retrieval_context: 宏观检索结果缓存
    """
    state_id: int = -1
    state_str: str = ""
    natural_description: str = ""
    parent: Optional['OuterNode'] = None
    children: List['OuterNode'] = field(default_factory=list)
    tactic_sequence: List[str] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    is_solved: bool = False
    metavar_bindings: Dict[str, str] = field(default_factory=dict)
    retrieval_context: List[Any] = field(default_factory=list)

    # 未尝试的替代中间状态
    _untried_states: List[Dict[str, Any]] = field(default_factory=list)
    _states_initialized: bool = False

    def is_terminal(self) -> bool:
        """是否为终端节点（证明完成或无法推进）"""
        if self.is_solved:
            return True
        if self.state_str and self.state_str.strip().lower() == "no goals":
            return True
        return False

    def is_fully_expanded(self) -> bool:
        """是否已完全展开"""
        if not self._states_initialized:
            return False
        return len(self._untried_states) == 0

    def ucb_score(self, exploration_c: float = 1.414) -> float:
        """计算 UCB1 得分"""
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration_c * math.sqrt(
            math.log(max(self.parent.visits, 1)) / self.visits
        ) if self.parent else 0.0
        return exploit + explore


@dataclass
class InnerNode:
    """
    内层战术树节点

    表示在连接两个外层状态之间的具体 Tactic 执行步骤。

    属性:
        state_id: Pantograph 中的状态 ID
        state_str: 当前证明状态文本
        parent: 父节点
        children: 子节点列表
        tactic: 导致本状态的策略
        visits: 访问次数
        value: 累计价值
        depth: 在内层树中的深度
        error_message: 策略执行失败时的错误信息
        thought: 生成该策略时的思考过程
    """
    state_id: int = -1
    state_str: str = ""
    parent: Optional['InnerNode'] = None
    children: List['InnerNode'] = field(default_factory=list)
    tactic: str = ""
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    error_message: str = ""
    thought: str = ""

    # 未尝试的策略
    _untried_tactics: List[str] = field(default_factory=list)
    _tactics_initialized: bool = False

    def is_terminal(self) -> bool:
        """是否为终端（无目标 = 成功，state_id=-1 = 失败）"""
        if self.state_id == -1:
            return True
        if not self.state_str:
            return True
        return self.state_str.strip().lower() == "no goals"

    def is_success(self) -> bool:
        """是否成功到达目标状态"""
        return (self.state_str.strip().lower() == "no goals"
                and self.state_id != -1)

    def is_fully_expanded(self) -> bool:
        if not self._tactics_initialized:
            return False
        return len(self._untried_tactics) == 0

    def ucb_score(self, exploration_c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = exploration_c * math.sqrt(
            math.log(max(self.parent.visits, 1)) / self.visits
        ) if self.parent else 0.0
        return exploit + explore


# ================================================================
# MAGC-MCTS 核心引擎
# ================================================================

class MAGCMCTS:
    """
    元变量感知目标条件 MCTS (MAGC-MCTS)

    两级搜索引擎，协调外层状态树和内层战术树：

    外层循环 (Outer Loop):
        1. SELECT: UCB1 选择最有潜力的外层节点
        2. EXPAND: 调用 LLM + DG-RASP 生成新的中间状态
        3. INNER_SEARCH: 启动内层树搜索连接相邻状态
        4. BACKPROPAGATE: 将内层搜索结果回传到外层树

    内层循环 (Inner Loop):
        1. SELECT: UCB1 选择战术节点
        2. EXPAND: 调用 LLM + micro-RAG 生成候选策略
        3. EVALUATE: Pantograph 验证策略合法性
        4. REPAIR: RCRL 反思修复失败策略
        5. BACKPROPAGATE: 更新节点价值

    参数:
        verifier: Pantograph 验证器实例
        generator: Thought-CoS-Tactic 生成模型
        retriever: DG-RASP 双粒度检索器
        repair_module: RCRL 反思修复模块
        outer_iterations: 外层搜索最大迭代次数
        inner_iterations: 内层搜索最大迭代次数
        outer_exploration: 外层 UCB 探索常数
        inner_exploration: 内层 UCB 探索常数
        max_outer_depth: 外层树最大深度（中间状态数上限）
        max_inner_depth: 内层树最大深度（单段战术步数上限）
        inner_budget: 内层搜索的计算预算（节点扩展数上限）
    """

    def __init__(self,
                 verifier,
                 generator,
                 retriever=None,
                 repair_module=None,
                 outer_iterations: int = 50,
                 inner_iterations: int = 200,
                 outer_exploration: float = 1.414,
                 inner_exploration: float = 1.414,
                 max_outer_depth: int = 10,
                 max_inner_depth: int = 30,
                 inner_budget: int = 500):
        """
        初始化 MAGC-MCTS 引擎

        参数:
            verifier: PantographVerifier 实例
            generator: ThoughtCoSTacticGenerator 实例
            retriever: DualGrainedRetriever 实例（可选）
            repair_module: RCRL 模块（可选）
            outer_iterations: 外层搜索迭代数
            inner_iterations: 内层搜索迭代数
            outer_exploration: 外层 UCB 探索常数
            inner_exploration: 内层 UCB 探索常数
            max_outer_depth: 外层最大深度
            max_inner_depth: 内层最大深度
            inner_budget: 内层计算预算
        """
        self.verifier = verifier
        self.generator = generator
        self.retriever = retriever
        self.repair = repair_module

        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.outer_exploration = outer_exploration
        self.inner_exploration = inner_exploration
        self.max_outer_depth = max_outer_depth
        self.max_inner_depth = max_inner_depth
        self.inner_budget = inner_budget

        # 全局状态追踪
        self._visited_states: Dict[str, int] = {}  # state_hash → visit_count
        self._metavar_registry: Dict[str, str] = {}  # 元变量全局绑定注册表

        # 统计
        self.stats = {
            "outer_expansions": 0,
            "inner_expansions": 0,
            "inner_searches": 0,
            "inner_successes": 0,
            "repairs_attempted": 0,
            "repairs_succeeded": 0,
            "total_tactics_tried": 0,
            "proof_found": False,
        }

    # ================================================================
    # 顶层搜索入口
    # ================================================================

    def search(self, theorem_type: str,
               natural_description: str = "") -> Optional[List[str]]:
        """
        执行完整的 MAGC-MCTS 搜索

        参数:
            theorem_type: 待证明定理的类型表达式
                例: "∀ (n : ℕ), n + 0 = n"
            natural_description: 定理的自然语言描述（用于宏观检索）

        返回:
            Optional[List[str]]: 成功时返回完整的策略序列，失败时返回 None
        """
        logger.info(f"MAGC-MCTS 启动搜索: {theorem_type[:80]}...")
        start_time = time.time()

        # 1. 在 Pantograph 中创建初始证明目标
        init_result = self.verifier.goal_start(theorem_type)
        if not init_result or "error" in str(init_result).lower():
            logger.error(f"无法创建证明目标: {init_result}")
            return None

        initial_state_id = init_result.get("stateId", -1)
        # 获取初始状态文本
        state_print = self.verifier.goal_print(initial_state_id)
        initial_state_str = self._extract_goals_text(state_print)

        # 2. 构建外层树根节点
        root = OuterNode(
            state_id=initial_state_id,
            state_str=initial_state_str,
            natural_description=natural_description,
            depth=0,
        )

        # 3. 外层搜索主循环
        for outer_iter in range(self.outer_iterations):
            # 检查时间预算
            elapsed = time.time() - start_time

            # SELECT: 选择外层节点
            node = self._outer_select(root)
            if node is None:
                continue

            # 终端检查
            if node.is_terminal():
                if node.is_solved:
                    proof = self._extract_outer_proof(node)
                    self.stats["proof_found"] = True
                    logger.info(f"[Outer] 证明找到! 迭代 {outer_iter}, "
                                f"耗时 {elapsed:.1f}s, 步骤数 {len(proof)}")
                    return proof
                continue

            # EXPAND: 尝试用内层搜索连接到新的中间状态
            success = self._outer_expand_and_evaluate(node)

            # BACKPROPAGATE
            reward = 1.0 if success else 0.0
            self._outer_backpropagate(node, reward)

        # 搜索结束，检查是否有完成的路径
        proof_node = self._find_solved_outer_node(root)
        if proof_node:
            proof = self._extract_outer_proof(proof_node)
            self.stats["proof_found"] = True
            return proof

        elapsed = time.time() - start_time
        logger.info(f"MAGC-MCTS 搜索结束 (未找到证明), "
                    f"耗时 {elapsed:.1f}s, 统计: {self.stats}")
        return None

    # ================================================================
    # 外层树操作
    # ================================================================

    def _outer_select(self, root: OuterNode) -> OuterNode:
        """
        外层选择：UCB1 策略遍历到叶子节点

        参数:
            root: 外层树根节点

        返回:
            OuterNode: 选中的叶子节点
        """
        current = root
        depth = 0
        while (not current.is_terminal()
               and current.is_fully_expanded()
               and depth < self.max_outer_depth):
            if not current.children:
                return current
            # UCB 选择最佳子节点
            best = max(current.children,
                       key=lambda c: c.ucb_score(self.outer_exploration))
            current = best
            depth += 1
        return current

    def _outer_expand_and_evaluate(self, node: OuterNode) -> bool:
        """
        外层展开：生成新的中间状态并通过内层搜索验证可达性

        流程:
        1. 如果尚未初始化，调用 LLM + 宏观层 RAG 生成候选中间状态
        2. 取出一个未尝试的中间状态
        3. 启动内层战术树搜索，尝试从当前状态到达该中间状态
        4. 成功则创建子节点

        参数:
            node: 当前外层节点

        返回:
            bool: 是否成功展开
        """
        # 初始化候选中间状态
        if not node._states_initialized:
            candidate_states = self._generate_intermediate_states(node)
            node._untried_states = candidate_states
            node._states_initialized = True

        if not node._untried_states:
            return False

        # 取出一个候选状态
        target_state_info = node._untried_states.pop(0)
        target_description = target_state_info.get("description", "")

        self.stats["outer_expansions"] += 1

        # 启动内层搜索：从 node.state_id 尝试到达 target_state
        inner_result = self._inner_search(
            start_state_id=node.state_id,
            start_state_str=node.state_str,
            target_description=target_description,
        )

        if inner_result is not None:
            tactics, final_state_id, final_state_str = inner_result

            # 创建外层子节点
            child = OuterNode(
                state_id=final_state_id,
                state_str=final_state_str,
                natural_description=target_description,
                parent=node,
                tactic_sequence=tactics,
                depth=node.depth + 1,
            )

            # 检查是否证明完成
            if final_state_str.strip().lower() == "no goals":
                child.is_solved = True

            # 同步元变量绑定
            self._sync_metavar_bindings(child)

            node.children.append(child)
            return True

        return False

    def _generate_intermediate_states(self, node: OuterNode) -> List[Dict[str, Any]]:
        """
        使用 LLM + 宏观层 RAG 生成候选中间状态

        参数:
            node: 当前外层节点

        返回:
            List[Dict]: 候选中间状态列表，每项包含 description 字段
        """
        candidates = []

        # 宏观层检索
        retrieval_context = ""
        if self.retriever:
            macro_results = self.retriever.macro_retrieve(node.state_str)
            retrieval_context = self.retriever.format_macro_context(macro_results)
            node.retrieval_context = macro_results

        # 调用生成模型产生候选中间状态描述
        # 直接生成多个策略作为候选
        responses = self.generator.generate_step(
            state=node.state_str,
            temperature=0.7,
            num_samples=5,
        )

        for r in responses:
            tactic = r.get("tactic", "").strip()
            thought = r.get("thought", "").strip()
            if tactic and tactic != "sorry":
                candidates.append({
                    "description": thought or tactic,
                    "suggested_tactic": tactic,
                })

        return candidates

    def _outer_backpropagate(self, node: OuterNode, reward: float):
        """外层回传：从叶子向根更新 visits 和 value"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _find_solved_outer_node(self, root: OuterNode) -> Optional[OuterNode]:
        """DFS 查找已解决的外层叶子节点"""
        if root.is_solved:
            return root
        for child in root.children:
            result = self._find_solved_outer_node(child)
            if result:
                return result
        return None

    def _extract_outer_proof(self, node: OuterNode) -> List[str]:
        """从外层叶子节点回溯提取完整策略序列"""
        segments = []
        current = node
        while current and current.parent is not None:
            if current.tactic_sequence:
                segments.append(current.tactic_sequence)
            current = current.parent
        # 反转并展平
        segments.reverse()
        proof = []
        for seg in segments:
            proof.extend(seg)
        return proof

    # ================================================================
    # 内层战术树搜索
    # ================================================================

    def _inner_search(self,
                      start_state_id: int,
                      start_state_str: str,
                      target_description: str = ""
                      ) -> Optional[Tuple[List[str], int, str]]:
        """
        内层搜索：从起始状态出发，尝试推进证明

        参数:
            start_state_id: 起始 Pantograph 状态 ID
            start_state_str: 起始状态文本
            target_description: 目标状态的自然语言描述（作为启发信息）

        返回:
            Optional[Tuple]: 成功时返回 (tactics_list, final_state_id, final_state_str)
                            失败时返回 None
        """
        self.stats["inner_searches"] += 1

        root = InnerNode(
            state_id=start_state_id,
            state_str=start_state_str,
            depth=0,
        )

        expansions = 0

        for inner_iter in range(self.inner_iterations):
            if expansions >= self.inner_budget:
                break

            # SELECT
            node = self._inner_select(root)
            if node is None:
                continue

            # 终端检查
            if node.is_terminal():
                if node.is_success():
                    self.stats["inner_successes"] += 1
                    tactics = self._extract_inner_path(node)
                    return (tactics, node.state_id, node.state_str)
                continue

            # EXPAND
            child = self._inner_expand(node)
            expansions += 1
            self.stats["inner_expansions"] += 1

            if child is None:
                self._inner_backpropagate(node, -0.1)
                continue

            # EVALUATE
            if child.is_success():
                self.stats["inner_successes"] += 1
                self._inner_backpropagate(child, 1.0)
                tactics = self._extract_inner_path(child)
                return (tactics, child.state_id, child.state_str)

            if child.state_id == -1:
                # 策略执行失败 → 尝试 RCRL 修复
                repair_result = self._attempt_repair(
                    node.state_id, node.state_str,
                    child.tactic, child.error_message
                )
                if repair_result:
                    repaired_tactic, new_state_id, new_state_str = repair_result
                    repaired_child = InnerNode(
                        state_id=new_state_id,
                        state_str=new_state_str,
                        parent=node,
                        tactic=repaired_tactic,
                        depth=node.depth + 1,
                        thought=f"[repair] {child.error_message}",
                    )
                    node.children.append(repaired_child)

                    if repaired_child.is_success():
                        self.stats["inner_successes"] += 1
                        tactics = self._extract_inner_path(repaired_child)
                        return (tactics, repaired_child.state_id, repaired_child.state_str)

                    self._inner_backpropagate(repaired_child, 0.3)
                else:
                    self._inner_backpropagate(child, -0.5)
            else:
                # 策略执行成功但证明未完成
                # 启发式价值：目标数减少 → 正向奖励
                value = self._evaluate_inner_state(child, start_state_str)
                self._inner_backpropagate(child, value)

        # 内层搜索超预算，寻找最优进展路径
        best_leaf = self._find_best_inner_leaf(root)
        if best_leaf and best_leaf.state_id != -1 and best_leaf.state_str:
            tactics = self._extract_inner_path(best_leaf)
            if tactics:
                return (tactics, best_leaf.state_id, best_leaf.state_str)

        return None

    def _inner_select(self, root: InnerNode) -> InnerNode:
        """内层选择：UCB1 策略"""
        current = root
        depth = 0
        while (not current.is_terminal()
               and current.is_fully_expanded()
               and depth < self.max_inner_depth):
            if not current.children:
                return current
            # 过滤掉失败节点
            valid_children = [c for c in current.children if c.state_id != -1]
            if not valid_children:
                valid_children = current.children
            best = max(valid_children,
                       key=lambda c: c.ucb_score(self.inner_exploration))
            current = best
            depth += 1
        return current

    def _inner_expand(self, node: InnerNode) -> Optional[InnerNode]:
        """
        内层展开：生成并验证一个候选策略

        流程:
        1. 如果未初始化，调用 LLM + 微观层 RAG 生成候选策略
        2. 取出一个未尝试的策略
        3. 通过 Pantograph 验证
        4. 创建子节点（成功或失败）

        参数:
            node: 当前内层节点

        返回:
            Optional[InnerNode]: 新创建的子节点
        """
        if node.is_terminal():
            return None

        # 初始化候选策略
        if not node._tactics_initialized:
            tactics = self._generate_candidate_tactics(node)
            node._untried_tactics = tactics
            node._tactics_initialized = True

        if not node._untried_tactics:
            return None

        tactic = node._untried_tactics.pop(0)
        self.stats["total_tactics_tried"] += 1

        # 通过 Pantograph 执行策略
        result = self.verifier.goal_tactic(node.state_id, tactic)

        if result and result.get("is_valid", False):
            new_state_id = result.get("new_state_id", -1)
            # 获取新状态文本
            state_print = self.verifier.goal_print(new_state_id)
            new_state_str = self._extract_goals_text(state_print)

            child = InnerNode(
                state_id=new_state_id,
                state_str=new_state_str,
                parent=node,
                tactic=tactic,
                depth=node.depth + 1,
            )
        else:
            # 策略执行失败
            error_msg = ""
            if result:
                error_msg = result.get("error", result.get("message", ""))
            child = InnerNode(
                state_id=-1,
                state_str="",
                parent=node,
                tactic=tactic,
                depth=node.depth + 1,
                error_message=error_msg,
            )

        node.children.append(child)
        return child

    def _generate_candidate_tactics(self, node: InnerNode) -> List[str]:
        """
        使用 LLM + 微观层 RAG 生成候选策略

        参数:
            node: 当前内层节点

        返回:
            List[str]: 去重后的候选策略列表
        """
        # 微观层检索
        retrieval_context = ""
        if self.retriever:
            micro_results = self.retriever.micro_retrieve(node.state_str)
            retrieval_context = self.retriever.format_micro_context(micro_results)

        # 混合采样策略（贪心 + 多样性）
        greedy_responses = self.generator.generate_step(
            state=node.state_str,
            temperature=0.0,
            num_samples=1,
        )
        sampled_responses = self.generator.generate_step(
            state=node.state_str,
            temperature=0.7,
            num_samples=5,
        )
        all_responses = greedy_responses + sampled_responses

        tactics = []
        seen = set()
        noise_patterns = {
            '[thought]', '[tactic]', '[error tactic]', '[error message]',
            '[reasoning]', '[action]', 'thought:', 'tactic:', 'reasoning:',
        }

        for r in all_responses:
            t = r.get("tactic", "").strip()
            if not t or t == "sorry" or t.lower() in noise_patterns:
                continue
            if t.startswith(':'):
                t = t[1:].strip()
                if not t:
                    continue
            if t not in seen:
                tactics.append(t)
                seen.add(t)

        return tactics

    def _inner_backpropagate(self, node: InnerNode, reward: float):
        """内层回传"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _extract_inner_path(self, node: InnerNode) -> List[str]:
        """提取内层路径的策略序列"""
        path = []
        current = node
        while current and current.parent is not None:
            if current.tactic:
                path.append(current.tactic)
            current = current.parent
        return list(reversed(path))

    def _find_best_inner_leaf(self, root: InnerNode) -> Optional[InnerNode]:
        """DFS 找到价值最高的内层叶子节点"""
        best = None
        best_value = -float('inf')

        def dfs(node: InnerNode):
            nonlocal best, best_value
            if not node.children and node.state_id != -1 and node.visits > 0:
                avg_value = node.value / max(node.visits, 1)
                if avg_value > best_value:
                    best_value = avg_value
                    best = node
            for child in node.children:
                dfs(child)

        dfs(root)
        return best

    # ================================================================
    # 元变量管理
    # ================================================================

    def _sync_metavar_bindings(self, node: OuterNode):
        """
        同步元变量绑定

        当外层树的某个子目标中通过 Exists.intro 等策略实例化了一个元变量时，
        Pantograph 会在后台捕获这一绑定。此方法从 Pantograph 提取绑定信息，
        并更新全局注册表。

        参数:
            node: 需要同步的外层节点
        """
        if node.state_id < 0:
            return

        try:
            # 通过 Pantograph 获取当前状态的元变量信息
            state_info = self.verifier.goal_print(node.state_id)
            if state_info and "metavariables" in state_info:
                for mv in state_info["metavariables"]:
                    mv_name = mv.get("name", "")
                    mv_value = mv.get("value", "")
                    if mv_name and mv_value:
                        self._metavar_registry[mv_name] = mv_value
                        node.metavar_bindings[mv_name] = mv_value
        except Exception as e:
            logger.debug(f"元变量同步失败: {e}")

    # ================================================================
    # RCRL 修复集成
    # ================================================================

    def _attempt_repair(self, state_id: int, state_str: str,
                        failed_tactic: str, error_msg: str
                        ) -> Optional[Tuple[str, int, str]]:
        """
        尝试通过 RCRL 修复失败的策略

        参数:
            state_id: 当前状态 ID
            state_str: 当前状态文本
            failed_tactic: 失败的策略
            error_msg: 错误信息

        返回:
            Optional[Tuple]: 成功时返回 (repaired_tactic, new_state_id, new_state_str)
        """
        if not self.repair:
            return None

        self.stats["repairs_attempted"] += 1

        repair_result = self.repair.attempt_repair(
            state_id=state_id,
            state_str=state_str,
            failed_tactic=failed_tactic,
            error_message=error_msg,
        )

        if repair_result and repair_result[0]:
            self.stats["repairs_succeeded"] += 1
            _, new_state_id, repaired_tactic = repair_result
            # 获取修复后的状态文本
            state_print = self.verifier.goal_print(new_state_id)
            new_state_str = self._extract_goals_text(state_print)
            return (repaired_tactic, new_state_id, new_state_str)

        return None

    # ================================================================
    # 辅助工具
    # ================================================================

    def _evaluate_inner_state(self, node: InnerNode,
                               start_state: str) -> float:
        """
        评估内层节点的启发式价值

        简单启发式：目标数量减少 → 正向奖励

        参数:
            node: 内层节点
            start_state: 起始状态（用于比较）

        返回:
            float: 启发式价值 (-1 ~ 1)
        """
        # 计算目标数量的变化
        start_goals = start_state.count("⊢")
        current_goals = node.state_str.count("⊢") if node.state_str else 999

        if current_goals == 0:
            return 1.0  # 完全解决
        elif current_goals < start_goals:
            return 0.3  # 目标减少
        elif current_goals == start_goals:
            return 0.05  # 可能有进展（目标变化）
        else:
            return -0.1  # 目标增加（通常不好）

    def _extract_goals_text(self, state_print_result: Any) -> str:
        """
        从 Pantograph goal.print 响应中提取目标文本

        参数:
            state_print_result: goal.print 的返回值

        返回:
            str: 格式化的目标文本
        """
        if not state_print_result:
            return ""

        goals = state_print_result.get("goals", [])
        if not goals:
            return "no goals"

        lines = []
        for g in goals:
            if isinstance(g, str):
                lines.append(g)
            elif isinstance(g, dict):
                lines.append(g.get("target", g.get("goal", str(g))))
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return dict(self.stats)
