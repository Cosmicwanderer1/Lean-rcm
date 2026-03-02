"""
反思认知修复循环 (RCRL - Reflective Cognitive Repair Loop)
@author ygw
更新日期: 2026-02-28

创新点三：核心错误修复机制

当 MAGC-MCTS 内层树中某个策略被 Lean4 内核拒绝时，RCRL 启动修复流程：
1. 捕获 Lean4 内核的精确错误信息
2. 生成 <reflection> 标签包裹的反思内容（语义诊断）
3. 根据错误类型动态分流到 ETR 或 ESR 路径

修复路径:
    ETR (Error-Tactic Repair):
        错误局限于当前策略（类型不匹配、参数错误等）
        → 在同一 proof state 下生成修正策略
        → Pantograph 重新验证

    ESR (Error-State Repair):
        错误源于上游规划问题（错误的中间状态选择）
        → 回退到外层状态树的上一层
        → 通知 MAGC-MCTS 外层树进行重新展开

关键机制:
    - <thought> 增强生成：在策略生成前增加内心独白，提高一次通过率
    - <reflection> 反思诊断：将 Lean4 错误转化为语义化理解，
      区分类型检查失败、未知标识符、策略不适用等不同错误
    - 修复预算控制：避免无限重试
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("rcrl")


# ================================================================
# 错误分类
# ================================================================

class ErrorCategory(Enum):
    """Lean4 策略错误分类"""
    TYPE_MISMATCH = "类型不匹配"
    UNKNOWN_IDENTIFIER = "未知标识符"
    TACTIC_FAILED = "策略执行失败"
    UNSOLVED_GOALS = "存在未解决的子目标"
    TIMEOUT = "执行超时"
    PARSE_ERROR = "语法解析错误"
    ELABORATION_ERROR = "细化错误"
    METAVAR_UNRESOLVED = "元变量未解析"
    OTHER = "其他错误"


class RepairRoute(Enum):
    """修复路由"""
    ETR = "ETR"  # Error-Tactic Repair（局部策略修复）
    ESR = "ESR"  # Error-State Repair（状态级回退修复）


# ================================================================
# 数据结构
# ================================================================

@dataclass
class DiagnosisResult:
    """
    反思诊断结果

    属性:
        category: 错误类别
        route: 修复路由 (ETR / ESR)
        reflection: 反思文本（<reflection> 标签内容）
        semantic_description: 错误的语义化描述
        suggestion: 修复建议
        confidence: 诊断置信度 (0 ~ 1)
    """
    category: ErrorCategory = ErrorCategory.OTHER
    route: RepairRoute = RepairRoute.ETR
    reflection: str = ""
    semantic_description: str = ""
    suggestion: str = ""
    confidence: float = 0.5


@dataclass
class RepairResult:
    """
    修复结果

    属性:
        success: 修复是否成功
        route: 使用的修复路由
        repaired_tactic: 修复后的策略
        new_state_id: 修复成功后的新状态 ID
        new_state_str: 修复成功后的新状态文本
        attempts: 尝试次数
        reflections: 所有反思记录
    """
    success: bool = False
    route: RepairRoute = RepairRoute.ETR
    repaired_tactic: str = ""
    new_state_id: int = -1
    new_state_str: str = ""
    attempts: int = 0
    reflections: List[str] = field(default_factory=list)


# ================================================================
# 错误分类器
# ================================================================

class ErrorClassifier:
    """
    Lean4 错误分类器

    将 Lean4 内核返回的错误信息分类为预定义的 ErrorCategory，
    并判断应使用 ETR 还是 ESR 路径修复。

    分类策略基于规则模式匹配 + 上下文语义分析。

    ESR 触发条件（需要回退到外层树修复）:
        - 当前状态本身不合理（从错误上游继承的问题）
        - 连续多次 ETR 修复失败
        - 错误涉及无法在当前层面解决的类型约束
    """

    # 错误模式→类别映射
    ERROR_PATTERNS = [
        (r"type mismatch", ErrorCategory.TYPE_MISMATCH),
        (r"has type.*but is expected to have type", ErrorCategory.TYPE_MISMATCH),
        (r"application type mismatch", ErrorCategory.TYPE_MISMATCH),
        (r"unknown identifier", ErrorCategory.UNKNOWN_IDENTIFIER),
        (r"unknown constant", ErrorCategory.UNKNOWN_IDENTIFIER),
        (r"tactic .* failed", ErrorCategory.TACTIC_FAILED),
        (r"unsolved goals", ErrorCategory.UNSOLVED_GOALS),
        (r"timeout", ErrorCategory.TIMEOUT),
        (r"expected token", ErrorCategory.PARSE_ERROR),
        (r"unexpected token", ErrorCategory.PARSE_ERROR),
        (r"elaboration error", ErrorCategory.ELABORATION_ERROR),
        (r"don't know how to synthesize", ErrorCategory.ELABORATION_ERROR),
        (r"failed to synthesize", ErrorCategory.ELABORATION_ERROR),
        (r"\?m\.\d+", ErrorCategory.METAVAR_UNRESOLVED),
        (r"unresolved.*metavar", ErrorCategory.METAVAR_UNRESOLVED),
    ]

    # 倾向于 ESR 的错误类别
    ESR_PRONE_CATEGORIES = {
        ErrorCategory.ELABORATION_ERROR,
        ErrorCategory.METAVAR_UNRESOLVED,
    }

    def classify(self, error_message: str,
                 state_str: str = "",
                 consecutive_failures: int = 0) -> Tuple[ErrorCategory, RepairRoute]:
        """
        对错误信息进行分类并决定修复路由

        参数:
            error_message: Lean4 内核错误信息
            state_str: 当前证明状态（用于上下文分析）
            consecutive_failures: 连续失败次数

        返回:
            Tuple[ErrorCategory, RepairRoute]: (错误类别, 修复路由)
        """
        category = self._match_category(error_message)

        # 决定修复路由
        route = RepairRoute.ETR  # 默认使用 ETR

        # 条件1: 错误类别本身倾向于 ESR
        if category in self.ESR_PRONE_CATEGORIES:
            route = RepairRoute.ESR

        # 条件2: 连续失败多次，说明当前状态可能有上游问题
        if consecutive_failures >= 3:
            route = RepairRoute.ESR

        # 条件3: 超时通常需要更高层面的策略调整
        if category == ErrorCategory.TIMEOUT:
            route = RepairRoute.ESR

        return category, route

    def _match_category(self, error_message: str) -> ErrorCategory:
        """基于正则模式匹配错误类别"""
        lower_msg = error_message.lower()
        for pattern, category in self.ERROR_PATTERNS:
            if re.search(pattern, lower_msg):
                return category
        return ErrorCategory.OTHER


# ================================================================
# RCRL 主模块
# ================================================================

class ReflectiveCognitiveRepairLoop:
    """
    反思认知修复循环 (RCRL)

    核心模块，整合错误分类、反思诊断和修复执行。

    生命周期 (单次修复):
        1. 接收失败信息 (state, tactic, error_msg)
        2. 错误分类 → ErrorCategory + RepairRoute
        3. 生成 <reflection> 反思诊断
        4. 按路由执行修复:
           - ETR: 在同一状态下生成修正策略 → Pantograph 验证
           - ESR: 标记需要外层回退 → 返回信号给 MAGC-MCTS
        5. 结果反馈

    参数:
        verifier: PantographVerifier 实例
        generator: ThoughtCoSTacticGenerator 实例
        max_etr_attempts: ETR 路径最大重试次数
        repair_temperature: 修复生成时的采样温度
        enable_reflection: 是否启用反思诊断（调试时可关闭）
    """

    def __init__(self,
                 verifier,
                 generator,
                 max_etr_attempts: int = 3,
                 repair_temperature: float = 0.3,
                 enable_reflection: bool = True):
        """
        初始化 RCRL

        参数:
            verifier: PantographVerifier 实例
            generator: ThoughtCoSTacticGenerator 实例
            max_etr_attempts: ETR 路径最大重试次数
            repair_temperature: 修复生成时的采样温度
            enable_reflection: 是否启用反思诊断
        """
        self.verifier = verifier
        self.generator = generator
        self.max_etr_attempts = max_etr_attempts
        self.repair_temperature = repair_temperature
        self.enable_reflection = enable_reflection

        self.classifier = ErrorClassifier()

        # 统计
        self.stats = {
            "total_attempts": 0,
            "etr_attempts": 0,
            "etr_successes": 0,
            "esr_triggers": 0,
            "reflections_generated": 0,
        }

    # ================================================================
    # 顶层修复入口
    # ================================================================

    def attempt_repair(self,
                       state_id: int,
                       state_str: str,
                       failed_tactic: str,
                       error_message: str
                       ) -> Tuple[bool, int, str]:
        """
        尝试修复失败的策略

        参数:
            state_id: 当前 Pantograph 状态 ID
            state_str: 当前证明状态文本
            failed_tactic: 失败的策略
            error_message: Lean4 内核错误信息

        返回:
            Tuple[bool, int, str]:
                (修复是否成功, 新状态ID(-1表示失败或ESR), 修复后的策略)
                - ETR成功: (True, new_state_id, repaired_tactic)
                - ETR失败: (False, -1, "")
                - ESR信号: (False, -2, "ESR")  -2 表示需要外层回退
        """
        self.stats["total_attempts"] += 1

        # 1. 错误分类
        category, route = self.classifier.classify(
            error_message, state_str, consecutive_failures=0
        )

        logger.info(
            f"RCRL 诊断: 类别={category.value}, 路由={route.value}, "
            f"策略='{failed_tactic[:50]}...'"
        )

        # 2. 反思诊断
        diagnosis = self._generate_reflection(
            state_str, failed_tactic, error_message, category, route
        )

        # 3. 按路由执行修复
        if route == RepairRoute.ESR:
            self.stats["esr_triggers"] += 1
            logger.info(f"RCRL → ESR: 需要外层状态回退")
            return (False, -2, "ESR")

        # ETR 路径: 在同一状态下尝试修复
        return self._execute_etr(
            state_id, state_str, failed_tactic, error_message, diagnosis
        )

    # ================================================================
    # 反思诊断
    # ================================================================

    def _generate_reflection(self,
                             state_str: str,
                             failed_tactic: str,
                             error_message: str,
                             category: ErrorCategory,
                             route: RepairRoute) -> DiagnosisResult:
        """
        生成 <reflection> 反思诊断

        将 Lean4 内核的机器格式错误转化为语义化理解，帮助 LLM
        更好地生成修复策略。

        参数:
            state_str: 当前证明状态
            failed_tactic: 失败的策略
            error_message: 错误信息
            category: 已分类的错误类别
            route: 决定的修复路由

        返回:
            DiagnosisResult: 诊断结果
        """
        self.stats["reflections_generated"] += 1

        if not self.enable_reflection:
            return DiagnosisResult(
                category=category,
                route=route,
                reflection="[reflection disabled]",
            )

        # 构建反思内容
        reflection_lines = [
            f"<reflection>",
            f"错误类别: {category.value}",
            f"失败策略: {failed_tactic}",
            f"内核反馈: {error_message[:200]}",
        ]

        # 根据错误类别生成特定反思
        semantic_desc, suggestion = self._category_specific_reflection(
            category, state_str, failed_tactic, error_message
        )

        reflection_lines.extend([
            f"语义诊断: {semantic_desc}",
            f"修复建议: {suggestion}",
            f"修复路由: {route.value}",
            f"</reflection>",
        ])

        reflection_text = "\n".join(reflection_lines)
        logger.debug(f"反思内容:\n{reflection_text}")

        return DiagnosisResult(
            category=category,
            route=route,
            reflection=reflection_text,
            semantic_description=semantic_desc,
            suggestion=suggestion,
            confidence=self._estimate_confidence(category),
        )

    def _category_specific_reflection(self,
                                      category: ErrorCategory,
                                      state_str: str,
                                      failed_tactic: str,
                                      error_message: str
                                      ) -> Tuple[str, str]:
        """
        按错误类别生成特定的语义诊断和修复建议

        参数:
            category: 错误类别
            state_str: 当前状态
            failed_tactic: 失败策略
            error_message: 错误信息

        返回:
            Tuple[str, str]: (语义描述, 修复建议)
        """
        if category == ErrorCategory.TYPE_MISMATCH:
            # 提取期望类型和实际类型
            expected = self._extract_pattern(
                error_message, r"expected to have type\s+(.+?)(?:\n|$)"
            )
            actual = self._extract_pattern(
                error_message, r"has type\s+(.+?)(?:\n|but)"
            )
            desc = f"策略产生的项的类型与目标类型不匹配"
            if expected and actual:
                desc += f"。期望: {expected[:80]}，实际: {actual[:80]}"
            suggestion = (
                "尝试使用类型转换函数(如 Nat.cast, Int.ofNat)，"
                "或检查假设中是否有类型一致的引理"
            )
            return desc, suggestion

        elif category == ErrorCategory.UNKNOWN_IDENTIFIER:
            identifier = self._extract_pattern(
                error_message, r"unknown (?:identifier|constant) ['\"]?(\w+)"
            )
            desc = f"引用了不存在的标识符"
            if identifier:
                desc += f" '{identifier}'"
            suggestion = (
                "检查是否需要导入额外的 Mathlib 模块，"
                "或使用相似的已知引理名称替换"
            )
            return desc, suggestion

        elif category == ErrorCategory.TACTIC_FAILED:
            # 提取具体策略名称
            tactic_name = self._extract_pattern(
                error_message, r"tactic ['\"]?(\w+)['\"]? failed"
            )
            desc = f"策略 '{tactic_name or failed_tactic}' 在当前状态下不适用"
            suggestion = (
                "当前目标的形状可能不匹配该策略的前提条件。"
                "尝试先用 simp/norm_num 简化，或使用不同的策略分解目标"
            )
            return desc, suggestion

        elif category == ErrorCategory.UNSOLVED_GOALS:
            desc = "策略执行后仍留有未解决的子目标"
            suggestion = (
                "策略可能部分有效但不够完整。"
                "尝试添加 <;> 连接后续策略，或使用 all_goals 包裹"
            )
            return desc, suggestion

        elif category == ErrorCategory.PARSE_ERROR:
            desc = "策略文本存在语法错误"
            suggestion = (
                "检查括号配对、关键字拼写、参数格式。"
                "确保使用 Lean4 语法而非 Lean3 语法"
            )
            return desc, suggestion

        elif category == ErrorCategory.ELABORATION_ERROR:
            desc = "Lean4 无法推断出策略中的隐式参数"
            suggestion = (
                "显式提供类型标注或实例参数。"
                "使用 @显式前缀 或 (· : Type) 标注"
            )
            return desc, suggestion

        elif category == ErrorCategory.METAVAR_UNRESOLVED:
            desc = "存在未解析的元变量，可能是证明不完整"
            suggestion = (
                "需要在上游步骤中提供更多信息来绑定元变量。"
                "考虑使用 refine/exact 显式提供参数值"
            )
            return desc, suggestion

        elif category == ErrorCategory.TIMEOUT:
            desc = "策略执行超时，可能是决策过程过于复杂"
            suggestion = "简化策略，避免使用 decide/omega 处理大规模表达式"
            return desc, suggestion

        else:
            desc = f"未分类的错误: {error_message[:100]}"
            suggestion = "尝试使用更基础的策略（intro/apply/exact）替代"
            return desc, suggestion

    def _estimate_confidence(self, category: ErrorCategory) -> float:
        """估计诊断置信度"""
        confidence_map = {
            ErrorCategory.TYPE_MISMATCH: 0.8,
            ErrorCategory.UNKNOWN_IDENTIFIER: 0.9,
            ErrorCategory.TACTIC_FAILED: 0.7,
            ErrorCategory.UNSOLVED_GOALS: 0.6,
            ErrorCategory.PARSE_ERROR: 0.95,
            ErrorCategory.ELABORATION_ERROR: 0.6,
            ErrorCategory.METAVAR_UNRESOLVED: 0.5,
            ErrorCategory.TIMEOUT: 0.85,
            ErrorCategory.OTHER: 0.3,
        }
        return confidence_map.get(category, 0.3)

    # ================================================================
    # ETR 执行
    # ================================================================

    def _execute_etr(self,
                     state_id: int,
                     state_str: str,
                     failed_tactic: str,
                     error_message: str,
                     diagnosis: DiagnosisResult
                     ) -> Tuple[bool, int, str]:
        """
        执行 ETR (Error-Tactic Repair) 修复

        在同一 proof state 下，利用错误信息 + 反思诊断，
        生成修正策略并通过 Pantograph 验证。

        参数:
            state_id: 当前状态 ID
            state_str: 当前状态文本
            failed_tactic: 失败的策略
            error_message: 错误信息
            diagnosis: 反思诊断结果

        返回:
            Tuple[bool, int, str]: (成功, 新状态ID, 修复后策略)
        """
        for attempt in range(self.max_etr_attempts):
            self.stats["etr_attempts"] += 1

            # 调用生成模型的错误修正接口
            corrections = self.generator.generate_correction(
                state=state_str,
                error_tactic=failed_tactic,
                error_message=error_message,
                temperature=self.repair_temperature,
                num_samples=1,
            )

            if not corrections:
                continue

            candidate = corrections[0]
            corrected_tactic = candidate.get("tactic", "").strip()

            if not corrected_tactic or corrected_tactic == "sorry":
                continue

            # 跳过与原策略完全相同的修正
            if corrected_tactic == failed_tactic:
                continue

            # 通过 Pantograph 验证修正策略
            result = self.verifier.goal_tactic(state_id, corrected_tactic)
            if result and result.get("is_valid", False):
                new_state_id = result.get("new_state_id", -1)
                self.stats["etr_successes"] += 1
                logger.info(
                    f"ETR 修复成功 (尝试 {attempt + 1}/{self.max_etr_attempts}): "
                    f"'{failed_tactic[:30]}...' → '{corrected_tactic[:30]}...'"
                )
                return (True, new_state_id, corrected_tactic)
            else:
                # 更新错误信息以便下一轮修复
                if result:
                    error_message = result.get("error", error_message)
                failed_tactic = corrected_tactic

        # 所有 ETR 尝试失败
        # 如果连续失败次数达到阈值，升级为 ESR
        logger.info(
            f"ETR 修复失败 (已尝试 {self.max_etr_attempts} 次), "
            f"升级为 ESR 建议"
        )
        return (False, -1, "")

    # ================================================================
    # 完整修复流程（带多轮重试）
    # ================================================================

    def full_repair_cycle(self,
                          state_id: int,
                          state_str: str,
                          failed_tactic: str,
                          error_message: str,
                          max_rounds: int = 2
                          ) -> RepairResult:
        """
        完整的修复循环（多轮 ETR → ESR 升级）

        参数:
            state_id: 当前状态 ID
            state_str: 当前状态文本
            failed_tactic: 失败的策略
            error_message: 错误信息
            max_rounds: 最大修复轮次

        返回:
            RepairResult: 完整的修复结果
        """
        result = RepairResult()
        current_tactic = failed_tactic
        current_error = error_message
        consecutive_failures = 0

        for round_idx in range(max_rounds):
            # 分类
            category, route = self.classifier.classify(
                current_error, state_str, consecutive_failures
            )

            # 生成反思
            diagnosis = self._generate_reflection(
                state_str, current_tactic, current_error, category, route
            )
            result.reflections.append(diagnosis.reflection)

            if route == RepairRoute.ESR:
                result.route = RepairRoute.ESR
                result.success = False
                result.attempts = round_idx + 1
                logger.info(f"RCRL 完整循环: 第 {round_idx + 1} 轮触发 ESR")
                return result

            # ETR 修复
            success, new_id, repaired = self._execute_etr(
                state_id, state_str, current_tactic, current_error, diagnosis
            )

            result.attempts = round_idx + 1

            if success:
                result.success = True
                result.route = RepairRoute.ETR
                result.repaired_tactic = repaired
                result.new_state_id = new_id
                # 获取新状态文本
                state_print = self.verifier.goal_print(new_id)
                if state_print:
                    goals = state_print.get("goals", [])
                    result.new_state_str = "\n".join(
                        g if isinstance(g, str) else g.get("target", str(g))
                        for g in goals
                    ) if goals else "no goals"
                return result

            consecutive_failures += 1

        # 所有轮次失败
        result.success = False
        result.route = RepairRoute.ETR
        return result

    # ================================================================
    # 辅助工具
    # ================================================================

    @staticmethod
    def _extract_pattern(text: str, pattern: str) -> str:
        """
        从文本中提取正则匹配的第一个组

        参数:
            text: 源文本
            pattern: 正则表达式（含一个捕获组）

        返回:
            str: 匹配内容，无匹配时返回空字符串
        """
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def get_stats(self) -> Dict[str, Any]:
        """获取 RCRL 统计信息"""
        return dict(self.stats)

    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
