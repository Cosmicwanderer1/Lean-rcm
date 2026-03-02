"""
Lean4 语法树解析器
@author ygw
更新日期: 2026-02-06

解析 Lean4 代码的抽象语法树（AST），提取定理声明、策略、依赖等结构化信息。
用于 W5 全量追踪阶段，从 Mathlib4 源码中提取定理和证明结构。
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TheoremInfo:
    """
    定理信息数据类

    属性:
        name: 定理名称
        full_name: 完整限定名（含命名空间）
        type_expr: 定理类型表达式
        proof_body: 证明体（原始文本）
        tactics: 提取的策略列表
        proof_mode: 证明模式（tactic / term / mixed）
        file_path: 所在文件路径
        line_start: 起始行号
        line_end: 结束行号
        namespace: 命名空间
        imports: 依赖的 import 列表
        attributes: 定理属性标签（如 @[simp]）
    """
    name: str = ""
    full_name: str = ""
    type_expr: str = ""
    proof_body: str = ""
    tactics: List[str] = field(default_factory=list)
    proof_mode: str = "unknown"
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    namespace: str = ""
    imports: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "type_expr": self.type_expr,
            "proof_body": self.proof_body,
            "tactics": self.tactics,
            "proof_mode": self.proof_mode,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "namespace": self.namespace,
            "imports": self.imports,
            "attributes": self.attributes,
        }


@dataclass
class TacticInfo:
    """
    策略信息数据类

    属性:
        name: 策略名称（如 simp, apply, rw）
        full_text: 策略完整文本
        arguments: 策略参数列表
        line_number: 所在行号
        indent_level: 缩进层级
        is_combinator: 是否为组合策略（如 <;>, ;）
    """
    name: str = ""
    full_text: str = ""
    arguments: List[str] = field(default_factory=list)
    line_number: int = 0
    indent_level: int = 0
    is_combinator: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "full_text": self.full_text,
            "arguments": self.arguments,
            "line_number": self.line_number,
            "indent_level": self.indent_level,
            "is_combinator": self.is_combinator,
        }


class ASTParser:
    """
    Lean4 抽象语法树解析器

    基于正则表达式的轻量级解析器，用于从 Lean4 源码中提取：
    1. 定理声明（theorem / lemma / example）
    2. 策略序列（tactic block）
    3. 导入依赖（import）
    4. 命名空间（namespace / section）

    注意：这不是完整的 Lean4 parser，而是面向数据提取的实用工具。
    对于需要精确 AST 的场景，应使用 LeanDojo 的 trace 功能。
    """

    # 定理声明关键字
    THEOREM_KEYWORDS = ["theorem", "lemma", "example", "def"]

    # 已知的 Lean4 策略名称（用于识别 tactic 模式）
    KNOWN_TACTICS = [
        "intro", "intros", "apply", "exact", "rw", "rewrite", "simp",
        "ring", "norm_num", "omega", "linarith", "nlinarith", "field_simp",
        "push_neg", "contrapose", "by_contra", "by_cases", "cases",
        "induction", "rcases", "obtain", "have", "let", "show", "suffices",
        "calc", "constructor", "ext", "funext", "congr", "refine",
        "use", "exists", "trivial", "assumption", "contradiction",
        "exfalso", "absurd", "decide", "norm_cast", "push_cast",
        "rfl", "symm", "trans", "gcongr", "positivity", "polyrith",
        "aesop", "tauto", "finish", "tidy", "hint", "suggest",
        "library_search", "exact?", "apply?", "rw?", "simp?",
        "specialize", "generalize", "clear", "rename_i", "subst",
        "split", "left", "right", "injections", "injection",
    ]

    # 策略组合子
    TACTIC_COMBINATORS = ["<;>", ";", "·", ">>", "|||"]

    def __init__(self):
        """初始化 AST 解析器"""
        # 编译常用正则表达式
        self._theorem_pattern = re.compile(
            r'^(\s*)((?:@\[.*?\]\s*)*)'           # 属性标签
            r'((?:private|protected|noncomputable)\s+)*'  # 修饰符
            r'(theorem|lemma)\s+'                  # 关键字
            r'(\w[\w.\']*)\s*'                     # 定理名
            r'((?:\{[^}]*\}|\([^)]*\)|\[[^\]]*\]|\s)*)'  # 隐式/显式参数
            r'\s*:\s*'                             # 冒号
            r'(.*?)\s*:=\s*'                       # 类型表达式
            r'(.*)',                                # 证明体开始
            re.DOTALL
        )

        self._import_pattern = re.compile(r'^import\s+(.+)$', re.MULTILINE)
        self._namespace_pattern = re.compile(r'^namespace\s+(\S+)', re.MULTILINE)
        self._end_namespace_pattern = re.compile(r'^end\s+(\S+)', re.MULTILINE)
        self._tactic_pattern = re.compile(
            r'^\s*(' + '|'.join(re.escape(t) for t in self.KNOWN_TACTICS) + r')\b',
        )

    def parse_file(self, file_path: str) -> List[TheoremInfo]:
        """
        解析整个 Lean4 文件，提取所有定理信息

        参数:
            file_path: Lean4 文件路径

        返回:
            List[TheoremInfo]: 定理信息列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return []

        # 提取文件级信息
        imports = self._extract_imports(content)
        namespaces = self._extract_namespaces(content)

        # 提取定理
        theorems = self._extract_theorems(content, file_path)

        # 补充文件级信息
        for thm in theorems:
            thm.imports = imports
            thm.file_path = file_path
            # 确定命名空间
            thm.namespace = self._find_namespace_at_line(namespaces, thm.line_start)
            if thm.namespace and thm.name:
                thm.full_name = f"{thm.namespace}.{thm.name}"
            else:
                thm.full_name = thm.name

        logger.info(f"从 {file_path} 提取到 {len(theorems)} 个定理")
        return theorems

    def _extract_imports(self, content: str) -> List[str]:
        """
        提取文件的 import 声明

        参数:
            content: 文件内容

        返回:
            List[str]: import 模块列表
        """
        return self._import_pattern.findall(content)

    def _extract_namespaces(self, content: str) -> List[Tuple[str, int, int]]:
        """
        提取命名空间范围

        参数:
            content: 文件内容

        返回:
            List[Tuple]: (命名空间名, 起始行, 结束行) 列表
        """
        lines = content.split('\n')
        namespace_stack = []
        namespaces = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # 检测 namespace 开始
            ns_match = re.match(r'^namespace\s+(\S+)', stripped)
            if ns_match:
                namespace_stack.append((ns_match.group(1), i + 1))

            # 检测 namespace 结束
            end_match = re.match(r'^end\s+(\S+)', stripped)
            if end_match and namespace_stack:
                ns_name, start_line = namespace_stack.pop()
                namespaces.append((ns_name, start_line, i + 1))

        return namespaces

    def _find_namespace_at_line(self, namespaces: List[Tuple[str, int, int]],
                                 line: int) -> str:
        """
        查找指定行所在的命名空间

        参数:
            namespaces: 命名空间列表
            line: 行号

        返回:
            str: 命名空间名称
        """
        for ns_name, start, end in namespaces:
            if start <= line <= end:
                return ns_name
        return ""

    def _extract_theorems(self, content: str, file_path: str) -> List[TheoremInfo]:
        """
        从文件内容中提取所有定理

        参数:
            content: 文件内容
            file_path: 文件路径

        返回:
            List[TheoremInfo]: 定理列表
        """
        theorems = []
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 跳过注释行
            if stripped.startswith('--') or stripped.startswith('/-'):
                if stripped.startswith('/-'):
                    # 跳过多行注释
                    while i < len(lines) and '-/' not in lines[i]:
                        i += 1
                i += 1
                continue

            # 检测定理/引理声明
            for keyword in ["theorem", "lemma"]:
                if re.match(rf'^\s*((?:@\[.*?\]\s*)*(?:(?:private|protected|noncomputable)\s+)*){keyword}\s+', line):
                    thm = self._parse_theorem_block(lines, i, keyword)
                    if thm is not None:
                        theorems.append(thm)
                    break

            i += 1

        return theorems

    def _parse_theorem_block(self, lines: List[str], start_idx: int,
                              keyword: str) -> Optional[TheoremInfo]:
        """
        解析单个定理块

        参数:
            lines: 文件所有行
            start_idx: 定理起始行索引
            keyword: 关键字（theorem / lemma）

        返回:
            TheoremInfo: 定理信息，解析失败返回 None
        """
        try:
            # 收集定理声明的完整文本（可能跨多行）
            block_lines = []
            brace_depth = 0
            paren_depth = 0
            found_assign = False
            end_idx = start_idx

            for j in range(start_idx, min(start_idx + 300, len(lines))):
                line = lines[j]
                block_lines.append(line)

                # 跟踪括号深度
                for ch in line:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                    elif ch == '(':
                        paren_depth += 1
                    elif ch == ')':
                        paren_depth -= 1

                if ':=' in line:
                    found_assign = True

                # 判断定理块结束条件
                if found_assign:
                    # 如果使用 by 开始的 tactic 证明
                    if brace_depth <= 0 and paren_depth <= 0:
                        # 检查下一行是否还是证明的一部分
                        if j + 1 < len(lines):
                            next_line = lines[j + 1].strip()
                            # 如果下一行是空行或新的声明，则当前定理结束
                            if (not next_line or
                                any(next_line.startswith(kw) for kw in
                                    ["theorem", "lemma", "def", "instance",
                                     "class", "structure", "namespace", "end",
                                     "section", "import", "open", "#",
                                     "variable", "noncomputable", "private",
                                     "protected", "attribute", "@["])):
                                end_idx = j
                                break
                        else:
                            end_idx = j
                            break

                end_idx = j

            block_text = '\n'.join(block_lines)

            # 提取定理名称
            name_match = re.search(
                rf'{keyword}\s+(\w[\w.\']*)', block_text
            )
            name = name_match.group(1) if name_match else ""

            # 提取属性标签
            attributes = re.findall(r'@\[([^\]]+)\]', block_text)

            # 提取类型表达式（:= 之前的部分）
            type_expr = ""
            assign_pos = block_text.find(':=')
            if assign_pos > 0:
                # 从名称后到 := 之间的内容
                colon_section = block_text[:assign_pos]
                # 找到最后一个 : 作为类型开始
                type_start = colon_section.rfind(':')
                if type_start > 0:
                    type_expr = colon_section[type_start + 1:].strip()

            # 提取证明体
            proof_body = ""
            if assign_pos >= 0:
                proof_body = block_text[assign_pos + 2:].strip()

            # 判断证明模式并提取策略
            proof_mode, tactics = self._analyze_proof(proof_body)

            return TheoremInfo(
                name=name,
                type_expr=type_expr,
                proof_body=proof_body,
                tactics=tactics,
                proof_mode=proof_mode,
                line_start=start_idx + 1,
                line_end=end_idx + 1,
                attributes=attributes,
            )

        except Exception as e:
            logger.warning(f"解析定理块失败 (行 {start_idx + 1}): {e}")
            return None

    def _analyze_proof(self, proof_body: str) -> Tuple[str, List[str]]:
        """
        分析证明体，判断证明模式并提取策略

        参数:
            proof_body: 证明体文本

        返回:
            Tuple[str, List[str]]: (证明模式, 策略列表)
        """
        stripped = proof_body.strip()

        if not stripped:
            return ("empty", [])

        # 检查是否以 by 开头（tactic 模式）
        if stripped.startswith("by"):
            tactics = self._extract_tactics(stripped[2:].strip())
            return ("tactic", tactics)

        # 检查是否包含 tactic 块（如 by { ... }）
        if "by" in stripped:
            by_pos = stripped.find("by")
            after_by = stripped[by_pos + 2:].strip()
            tactics = self._extract_tactics(after_by)
            if tactics:
                return ("mixed", tactics)

        # term 模式
        return ("term", [])

    def _extract_tactics(self, tactic_block: str) -> List[str]:
        """
        从 tactic 块中提取策略列表

        参数:
            tactic_block: tactic 块文本

        返回:
            List[str]: 策略列表
        """
        tactics = []
        lines = tactic_block.split('\n')

        for line in lines:
            stripped = line.strip()

            # 跳过空行和注释
            if not stripped or stripped.startswith('--'):
                continue

            # 移除行内注释
            comment_pos = stripped.find('--')
            if comment_pos > 0:
                stripped = stripped[:comment_pos].strip()

            # 处理分号分隔的多个策略
            if ';' in stripped and '<;>' not in stripped:
                parts = stripped.split(';')
                for part in parts:
                    part = part.strip()
                    if part:
                        tactics.append(part)
            elif stripped:
                # 移除前导的 · (bullet point)
                if stripped.startswith('·'):
                    stripped = stripped[1:].strip()
                if stripped:
                    tactics.append(stripped)

        return tactics

    def parse_tactic(self, tactic_str: str) -> TacticInfo:
        """
        解析单个策略字符串，提取结构化信息

        参数:
            tactic_str: 策略字符串

        返回:
            TacticInfo: 策略的结构化表示
        """
        stripped = tactic_str.strip()

        # 提取策略名称（第一个单词）
        name_match = re.match(r'(\w+)', stripped)
        name = name_match.group(1) if name_match else stripped

        # 提取参数（策略名之后的部分）
        arguments = []
        if name_match:
            args_text = stripped[name_match.end():].strip()
            if args_text:
                # 简单的参数分割（按空格，但保留括号内的内容）
                arguments = self._split_arguments(args_text)

        # 判断是否为组合策略
        is_combinator = any(c in stripped for c in self.TACTIC_COMBINATORS)

        return TacticInfo(
            name=name,
            full_text=stripped,
            arguments=arguments,
            is_combinator=is_combinator,
        )

    def _split_arguments(self, args_text: str) -> List[str]:
        """
        分割策略参数，保留括号内的完整性

        参数:
            args_text: 参数文本

        返回:
            List[str]: 参数列表
        """
        args = []
        current = ""
        depth = 0

        for ch in args_text:
            if ch in '([{':
                depth += 1
                current += ch
            elif ch in ')]}':
                depth -= 1
                current += ch
            elif ch == ' ' and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += ch

        if current.strip():
            args.append(current.strip())

        return args

    def extract_dependencies(self, lean_code: str) -> List[str]:
        """
        提取代码中的 import 依赖

        参数:
            lean_code: Lean 代码字符串

        返回:
            List[str]: 依赖的模块列表
        """
        return self._extract_imports(lean_code)

    def is_tactic_proof(self, proof_body: str) -> bool:
        """
        判断证明体是否为 tactic 模式

        参数:
            proof_body: 证明体文本

        返回:
            bool: 是否为 tactic 模式
        """
        mode, _ = self._analyze_proof(proof_body)
        return mode in ("tactic", "mixed")

    def count_proof_steps(self, proof_body: str) -> int:
        """
        计算证明步数

        参数:
            proof_body: 证明体文本

        返回:
            int: 证明步数
        """
        _, tactics = self._analyze_proof(proof_body)
        return len(tactics)
