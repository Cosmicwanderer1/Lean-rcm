"""
元变量感知状态管理器 (Metavariable-Aware State Manager)
@author ygw
更新日期: 2026-02-28

为 MAGC-MCTS 提供证明状态的生命周期管理，核心职责：
1. 状态注册与追踪 — 维护所有访问过的证明状态
2. 元变量耦合管理 — 追踪 Lean4 元变量的绑定与传播
3. 状态独立性判断 — 当子目标不共享元变量时可并行搜索
4. 状态回溯 — 利用 Pantograph 的无损快照机制
5. 状态去重与缓存 — 基于语义哈希避免重复搜索
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("state_manager")


# ================================================================
# 数据结构
# ================================================================

@dataclass
class ManagedState:
    """
    受管理的证明状态

    属性:
        state_id: Pantograph 内部状态 ID
        state_str: 形式化状态文本
        state_hash: 状态内容哈希（用于去重）
        parent_id: 父状态 ID（-1 表示根）
        tactic: 从父状态到本状态的策略
        depth: 状态深度
        goals: 分解后的子目标列表
        metavar_set: 本状态涉及的元变量集合
        metavar_bindings: 本状态中已绑定的元变量 {名称: 值}
        is_solved: 是否已解决
        is_dead: 是否已标记为死端
        visit_count: 被访问次数
        created_at: 创建时间戳
    """
    state_id: int = -1
    state_str: str = ""
    state_hash: str = ""
    parent_id: int = -1
    tactic: str = ""
    depth: int = 0
    goals: List[str] = field(default_factory=list)
    metavar_set: Set[str] = field(default_factory=set)
    metavar_bindings: Dict[str, str] = field(default_factory=dict)
    is_solved: bool = False
    is_dead: bool = False
    visit_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class MetavarInfo:
    """
    元变量信息

    属性:
        name: 元变量名称 (如 ?m.123)
        var_type: 元变量的类型
        bound_value: 已绑定的值（None 表示未绑定）
        created_by_state: 创建该元变量的状态 ID
        bound_by_state: 绑定该元变量的状态 ID
        dependent_states: 依赖该元变量的状态 ID 集合
    """
    name: str = ""
    var_type: str = ""
    bound_value: Optional[str] = None
    created_by_state: int = -1
    bound_by_state: int = -1
    dependent_states: Set[int] = field(default_factory=set)


# ================================================================
# 状态管理器
# ================================================================

class MetavarAwareStateManager:
    """
    元变量感知状态管理器

    核心功能:
    1. 状态注册 — 每个新的 Pantograph 状态都被注册和跟踪
    2. 元变量追踪 — 维护元变量的生命周期（创建→绑定→传播）
    3. 独立性检测 — 判断两个子目标是否共享元变量
    4. 回溯支持 — 提供安全的状态回退机制
    5. 去重与缓存 — 基于语义哈希的状态去重

    与 Pantograph 的协作:
        Pantograph 内部维护一棵证明树，每个节点有唯一的 state_id。
        本管理器在此之上建立元变量关联图，追踪哪些目标共享元变量。
        当某个子目标通过 Exists.intro 绑定了一个元变量，
        所有依赖该元变量的其他子目标都会收到通知。

    参数:
        verifier: PantographVerifier 实例
        max_states: 最大管理状态数（内存保护）
    """

    def __init__(self, verifier=None, max_states: int = 10000):
        """
        初始化状态管理器

        参数:
            verifier: PantographVerifier 实例（用于状态查询）
            max_states: 最大管理状态数
        """
        self.verifier = verifier
        self.max_states = max_states

        # 状态注册表
        self._states: Dict[int, ManagedState] = {}
        # 元变量注册表
        self._metavars: Dict[str, MetavarInfo] = {}
        # 状态哈希 → state_id（去重）
        self._hash_index: Dict[str, int] = {}
        # 元变量 → 依赖的状态ID列表
        self._metavar_deps: Dict[str, Set[int]] = defaultdict(set)
        # 子目标的独立性缓存
        self._independence_cache: Dict[Tuple[int, int], bool] = {}

    # ================================================================
    # 状态注册
    # ================================================================

    def register_state(self,
                       state_id: int,
                       state_str: str,
                       parent_id: int = -1,
                       tactic: str = "",
                       depth: int = 0) -> ManagedState:
        """
        注册一个新的证明状态

        参数:
            state_id: Pantograph 状态 ID
            state_str: 状态文本
            parent_id: 父状态 ID
            tactic: 产生此状态的策略
            depth: 状态深度

        返回:
            ManagedState: 注册后的受管理状态
        """
        # 计算哈希
        state_hash = self._compute_hash(state_str)

        # 检查去重
        if state_hash in self._hash_index:
            existing_id = self._hash_index[state_hash]
            if existing_id in self._states:
                existing = self._states[existing_id]
                existing.visit_count += 1
                logger.debug(
                    f"状态去重: state_id={state_id} → "
                    f"已存在 state_id={existing_id}"
                )
                return existing

        # 解析子目标
        goals = self._parse_goals(state_str)

        # 提取元变量
        metavar_set = self._extract_metavars(state_str)

        # 创建受管理状态
        managed = ManagedState(
            state_id=state_id,
            state_str=state_str,
            state_hash=state_hash,
            parent_id=parent_id,
            tactic=tactic,
            depth=depth,
            goals=goals,
            metavar_set=metavar_set,
            visit_count=1,
        )

        # 注册
        self._states[state_id] = managed
        self._hash_index[state_hash] = state_id

        # 注册元变量依赖
        for mv in metavar_set:
            self._metavar_deps[mv].add(state_id)
            if mv not in self._metavars:
                self._metavars[mv] = MetavarInfo(
                    name=mv,
                    created_by_state=state_id,
                )
            self._metavars[mv].dependent_states.add(state_id)

        # 内存保护
        if len(self._states) > self.max_states:
            self._evict_old_states()

        return managed

    def get_state(self, state_id: int) -> Optional[ManagedState]:
        """
        获取受管理状态

        参数:
            state_id: 状态 ID

        返回:
            Optional[ManagedState]: 状态对象，不存在则返回 None
        """
        return self._states.get(state_id)

    def mark_solved(self, state_id: int):
        """标记状态为已解决"""
        if state_id in self._states:
            self._states[state_id].is_solved = True

    def mark_dead(self, state_id: int):
        """标记状态为死端（不可推进）"""
        if state_id in self._states:
            self._states[state_id].is_dead = True

    # ================================================================
    # 元变量管理
    # ================================================================

    def bind_metavar(self, name: str, value: str, bound_by_state: int):
        """
        绑定一个元变量的值

        当 Pantograph 通过某个策略确定了一个元变量的具体值时调用。
        该绑定会传播到所有依赖该元变量的状态。

        参数:
            name: 元变量名称
            value: 绑定值
            bound_by_state: 绑定操作发生的状态 ID
        """
        if name not in self._metavars:
            self._metavars[name] = MetavarInfo(name=name)

        info = self._metavars[name]
        info.bound_value = value
        info.bound_by_state = bound_by_state

        # 传播绑定到所有依赖状态
        for dep_state_id in info.dependent_states:
            if dep_state_id in self._states:
                self._states[dep_state_id].metavar_bindings[name] = value

        # 清除相关的独立性缓存
        self._invalidate_independence_cache(name)

        logger.debug(
            f"元变量绑定: {name}={value[:50]}... "
            f"(来自 state={bound_by_state}, "
            f"传播到 {len(info.dependent_states)} 个状态)"
        )

    def get_unbound_metavars(self, state_id: int) -> Set[str]:
        """
        获取指定状态中未绑定的元变量

        参数:
            state_id: 状态 ID

        返回:
            Set[str]: 未绑定的元变量名称集合
        """
        state = self._states.get(state_id)
        if not state:
            return set()

        unbound = set()
        for mv in state.metavar_set:
            info = self._metavars.get(mv)
            if info and info.bound_value is None:
                unbound.add(mv)
        return unbound

    # ================================================================
    # 独立性判断
    # ================================================================

    def are_goals_independent(self, state_id_a: int, state_id_b: int) -> bool:
        """
        判断两个子目标是否独立（不共享元变量）

        独立的子目标可以并行搜索，因为它们的解不会相互影响。

        参数:
            state_id_a: 子目标 A 的状态 ID
            state_id_b: 子目标 B 的状态 ID

        返回:
            bool: True 表示独立，False 表示存在耦合
        """
        cache_key = (min(state_id_a, state_id_b), max(state_id_a, state_id_b))
        if cache_key in self._independence_cache:
            return self._independence_cache[cache_key]

        state_a = self._states.get(state_id_a)
        state_b = self._states.get(state_id_b)

        if not state_a or not state_b:
            return True

        # 检查是否共享未绑定的元变量
        shared_metavars = state_a.metavar_set & state_b.metavar_set
        if not shared_metavars:
            self._independence_cache[cache_key] = True
            return True

        # 如果共享的元变量都已绑定，则仍然独立
        all_bound = all(
            self._metavars.get(mv) and self._metavars[mv].bound_value is not None
            for mv in shared_metavars
        )

        self._independence_cache[cache_key] = all_bound
        return all_bound

    def get_coupled_groups(self, state_ids: List[int]) -> List[List[int]]:
        """
        将状态 ID 列表按元变量耦合关系分组

        耦合的状态（共享未绑定元变量）会被分在同一组，
        不同组之间的搜索可以并行进行。

        参数:
            state_ids: 状态 ID 列表

        返回:
            List[List[int]]: 分组后的状态 ID 列表
        """
        if not state_ids:
            return []

        # 使用 Union-Find 进行分组
        parent = {sid: sid for sid in state_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(state_ids)):
            for j in range(i + 1, len(state_ids)):
                if not self.are_goals_independent(state_ids[i], state_ids[j]):
                    union(state_ids[i], state_ids[j])

        # 收集分组
        groups: Dict[int, List[int]] = defaultdict(list)
        for sid in state_ids:
            groups[find(sid)].append(sid)

        return list(groups.values())

    # ================================================================
    # 状态回溯
    # ================================================================

    def get_backtrack_path(self, state_id: int) -> List[Tuple[int, str]]:
        """
        获取从根到指定状态的回溯路径

        参数:
            state_id: 目标状态 ID

        返回:
            List[Tuple[int, str]]: [(state_id, tactic), ...] 从根到目标
        """
        path = []
        current_id = state_id

        while current_id >= 0 and current_id in self._states:
            state = self._states[current_id]
            path.append((state.state_id, state.tactic))
            if state.parent_id == current_id:
                break  # 防止环
            current_id = state.parent_id

        return list(reversed(path))

    def get_ancestor_states(self, state_id: int, n: int = 3) -> List[ManagedState]:
        """
        获取最近 n 个祖先状态

        参数:
            state_id: 当前状态 ID
            n: 祖先数量

        返回:
            List[ManagedState]: 祖先状态列表（距离从近到远）
        """
        ancestors = []
        current_id = state_id

        while len(ancestors) < n and current_id >= 0:
            state = self._states.get(current_id)
            if not state:
                break
            if current_id != state_id:
                ancestors.append(state)
            current_id = state.parent_id

        return ancestors

    # ================================================================
    # 统计与查询
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取状态管理器统计信息"""
        total_states = len(self._states)
        solved_count = sum(1 for s in self._states.values() if s.is_solved)
        dead_count = sum(1 for s in self._states.values() if s.is_dead)
        total_metavars = len(self._metavars)
        bound_metavars = sum(
            1 for m in self._metavars.values() if m.bound_value is not None
        )

        return {
            "total_states": total_states,
            "solved_states": solved_count,
            "dead_states": dead_count,
            "active_states": total_states - solved_count - dead_count,
            "total_metavars": total_metavars,
            "bound_metavars": bound_metavars,
            "unbound_metavars": total_metavars - bound_metavars,
            "unique_hashes": len(self._hash_index),
        }

    def get_all_active_states(self) -> List[ManagedState]:
        """获取所有活跃（未解决且非死端）的状态"""
        return [
            s for s in self._states.values()
            if not s.is_solved and not s.is_dead
        ]

    # ================================================================
    # 内部工具
    # ================================================================

    @staticmethod
    def _compute_hash(state_str: str) -> str:
        """
        计算状态内容哈希

        参数:
            state_str: 状态文本

        返回:
            str: SHA256 哈希值（前16位）
        """
        # 规范化：去除多余空白
        normalized = " ".join(state_str.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _parse_goals(state_str: str) -> List[str]:
        """
        解析状态文本中的子目标

        参数:
            state_str: 状态文本

        返回:
            List[str]: 子目标列表
        """
        if not state_str or state_str.strip().lower() == "no goals":
            return []

        goals = []
        # 按 ⊢ 分割，每个 ⊢ 后面到下一行起始是一个目标
        parts = state_str.split("⊢")
        for i, part in enumerate(parts):
            if i == 0:
                continue  # ⊢ 之前是假设上下文
            goal = part.strip()
            # 截断到下一个假设块（以变量声明开始的行）
            lines = goal.split("\n")
            goal_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and ":" in stripped and not stripped.startswith("⊢"):
                    # 可能是下一个目标的假设部分
                    if goal_lines:  # 只有在已经收集了目标内容后才截断
                        break
                goal_lines.append(line)
            if goal_lines:
                goals.append("\n".join(goal_lines).strip())

        # 如果上面的解析没有结果，将整个文本作为单个目标
        if not goals and state_str.strip():
            goals.append(state_str.strip())

        return goals

    @staticmethod
    def _extract_metavars(state_str: str) -> Set[str]:
        """
        从状态文本中提取元变量名称

        Lean4 中元变量的典型格式: ?m.123, ?_uniq.456

        参数:
            state_str: 状态文本

        返回:
            Set[str]: 元变量名称集合
        """
        import re
        # 匹配 ?标识符.数字 格式的元变量
        pattern = r'\?[\w]+\.\d+'
        matches = re.findall(pattern, state_str)
        return set(matches)

    def _invalidate_independence_cache(self, metavar_name: str):
        """元变量绑定变化后清除相关的独立性缓存"""
        deps = self._metavar_deps.get(metavar_name, set())
        to_remove = []
        for key in self._independence_cache:
            a, b = key
            if a in deps or b in deps:
                to_remove.append(key)
        for key in to_remove:
            del self._independence_cache[key]

    def _evict_old_states(self):
        """淘汰旧状态以释放内存"""
        if len(self._states) <= self.max_states:
            return

        # 按 visit_count 排序，淘汰访问最少的
        sorted_states = sorted(
            self._states.items(),
            key=lambda x: (x[1].is_solved, x[1].visit_count)
        )

        evict_count = len(self._states) - self.max_states + 100  # 多淘汰一些
        for i in range(min(evict_count, len(sorted_states))):
            state_id, state = sorted_states[i]
            if state.is_solved:
                break  # 不淘汰已解决的状态
            # 清理哈希索引
            if state.state_hash in self._hash_index:
                del self._hash_index[state.state_hash]
            del self._states[state_id]

        logger.debug(f"状态淘汰: 淘汰 {evict_count} 个状态, "
                     f"剩余 {len(self._states)} 个")
