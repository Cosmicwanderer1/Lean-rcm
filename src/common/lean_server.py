"""
Lean Server 交互接口 —— PyPantograph 适配器层
@author ygw
更新日期: 2026-02-28

基于 PyPantograph Server 的适配器层封装，提供与 Lean4 证明环境的全功能交互。
覆盖 Task 1.1 的三大核心能力：
  1. 独立子目标求解（goal_continue / goal_resume / Site 参数）
  2. 无损状态回溯（goal_save / goal_load）
  3. 元变量耦合追踪（sibling_dep 解析 / goal_subsume）

同时保留向后兼容的便利方法（run_proof、extract_state_pair、LeanServerPool）。

底层: workspace/PyPantograph/pantograph/server.py (Server 类)
上层: 本文件 (LeanServer 适配器)
"""

import sys
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

# 将 PyPantograph 路径加入 sys.path，以便直接 import pantograph
_PYPANTOGRAPH_DIR = str(Path(__file__).resolve().parent.parent.parent / "workspace" / "PyPantograph")
if _PYPANTOGRAPH_DIR not in sys.path:
    sys.path.insert(0, _PYPANTOGRAPH_DIR)

# 导入 PyPantograph 核心模块
from pantograph.server import Server as PantographServer
from pantograph.expr import (
    Expr,
    Goal,
    GoalState as PantoGoalState,
    Variable,
    Site,
    Subsumption,
    Tactic,
    TacticHave,
    TacticLet,
    TacticExpr,
    TacticDraft,
    TacticMode,
)
from pantograph.message import (
    Message,
    Severity,
    TacticFailure,
    ParseError,
    ServerError,
)
from pantograph.data import (
    TacticInvocation,
    CompilationUnit,
    SearchTarget,
    CheckTrackResult,
)
from pantograph.utils import get_lean_path as _panto_get_lean_path

logger = logging.getLogger(__name__)


def _find_lean_project_dir(base_path: str) -> str:
    """
    自动定位 Lean 项目根目录（包含 lakefile.lean 或 lakefile.toml 的目录）
    @author ygw

    搜索策略（按优先级）：
    1. base_path 本身
    2. base_path/src/（PyPantograph 的实际 Lean 项目在 src/ 下）
    3. base_path 下所有直接子目录

    参数:
        base_path: 候选基础路径

    返回:
        str: 找到的 Lean 项目目录（含 lakefile），未找到则返回原路径
    """
    lakefile_names = ["lakefile.lean", "lakefile.toml", "Lakefile.lean"]
    bp = Path(base_path)

    # 策略 1：base_path 本身就是 Lean 项目
    for lf in lakefile_names:
        if (bp / lf).is_file():
            logger.info(f"Lean 项目目录（直接匹配）: {base_path}")
            return str(bp)

    # 策略 2：检查 src/ 子目录（PyPantograph 的结构）
    src_dir = bp / "src"
    if src_dir.is_dir():
        for lf in lakefile_names:
            if (src_dir / lf).is_file():
                result = str(src_dir)
                logger.info(f"Lean 项目目录（src/ 子目录）: {result}")
                return result

    # 策略 3：搜索直接子目录
    if bp.is_dir():
        for child in bp.iterdir():
            if child.is_dir():
                for lf in lakefile_names:
                    if (child / lf).is_file():
                        result = str(child)
                        logger.info(f"Lean 项目目录（子目录匹配）: {result}")
                        return result

    logger.warning(f"未找到 lakefile，保持原路径: {base_path}")
    return str(bp)


def _get_lean_path_from_toolchain(project_path: str) -> Optional[str]:
    """
    从 lean-toolchain 文件和 elan 工具链推断 LEAN_PATH
    @author ygw

    读取 lean-toolchain 获取版本号，然后构建：
    - elan 工具链 lib 路径（含 Init 模块）
    - 项目自身的 .lake/build/lib
    - 包的 build/lib

    参数:
        project_path: Lean 项目路径（含 lean-toolchain）

    返回:
        str: 构建的 LEAN_PATH，或 None
    """
    paths = []
    pp = Path(project_path)

    # 1. 读取 lean-toolchain 文件
    toolchain_file = pp / "lean-toolchain"
    if not toolchain_file.is_file():
        # 尝试父目录
        toolchain_file = pp.parent / "lean-toolchain"
    if not toolchain_file.is_file():
        logger.warning(f"未找到 lean-toolchain 文件: {pp}")
        return None

    try:
        tc_content = toolchain_file.read_text().strip()
        # 格式: "leanprover/lean4:v4.26.0" 或 "leanprover--lean4---v4.26.0"
        logger.info(f"lean-toolchain 内容: {tc_content}")

        # 提取版本号部分
        if ":" in tc_content:
            # "leanprover/lean4:v4.26.0" -> "v4.26.0"
            tc_version = tc_content.split(":")[-1].strip()
        else:
            tc_version = tc_content.strip()

        # 构建 elan 工具链目录名（格式: leanprover--lean4---v4.26.0）
        tc_dir_name = tc_content.replace("/", "--").replace(":", "---")

        # 搜索 elan 工具链 lib 目录
        elan_home = Path.home() / ".elan"
        tc_lib_candidates = [
            elan_home / "toolchains" / tc_dir_name / "lib" / "lean",
            elan_home / "toolchains" / tc_dir_name / "lib" / "lean4" / "library",
            elan_home / "toolchains" / tc_dir_name / "lib",
        ]

        for tc_lib in tc_lib_candidates:
            if tc_lib.is_dir():
                # 验证 Init.olean 存在
                init_olean = tc_lib / "Init.olean"
                if init_olean.is_file():
                    paths.append(str(tc_lib))
                    logger.info(f"elan 工具链 lib: {tc_lib} (Init.olean ✓)")
                    break
                else:
                    # 即使没有 Init.olean，也加入（某些版本结构不同）
                    paths.append(str(tc_lib))
                    logger.info(f"elan 工具链 lib: {tc_lib} (Init.olean 未找到，仍加入)")
                    break
        else:
            logger.warning(f"未找到 elan 工具链 lib 目录，候选: {tc_lib_candidates}")

    except Exception as e:
        logger.warning(f"解析 lean-toolchain 失败: {e}")

    # 2. 项目自身的 .lake/build/lib
    own_lib = pp / ".lake" / "build" / "lib"
    if own_lib.is_dir():
        paths.append(str(own_lib))
        logger.info(f"项目 build/lib: {own_lib}")

    # 3. 包的 build/lib（遍历 .lake/packages/*/）
    lake_packages = pp / ".lake" / "packages"
    if lake_packages.is_dir():
        for pkg_dir in sorted(lake_packages.iterdir()):
            if not pkg_dir.is_dir():
                continue
            # 包可能有 .lake/build/lib 或直接 lib
            pkg_build_lib = pkg_dir / ".lake" / "build" / "lib"
            pkg_lib = pkg_dir / "lib"
            if pkg_build_lib.is_dir():
                paths.append(str(pkg_build_lib))
                logger.info(f"包 build/lib: {pkg_build_lib}")
            elif pkg_lib.is_dir():
                paths.append(str(pkg_lib))
                logger.info(f"包 lib: {pkg_lib}")

    if paths:
        lean_path = os.pathsep.join(paths)
        logger.info(f"从 toolchain 构建 LEAN_PATH={lean_path}")
        return lean_path

    return None


def _find_repl_binary(project_path: Optional[str] = None) -> Optional[str]:
    """
    自动搜索 pantograph-repl 二进制文件
    @author ygw

    按优先级搜索以下位置：
    1. 环境变量 PANTOGRAPH_REPL_PATH
    2. PyPantograph 目录下的 pantograph/pantograph-repl（官方路径）
    3. PyPantograph 目录下的 src/.lake/build/bin/repl（lake build 产物）
    4. project_path 下的 .lake/build/bin/repl
    5. project_path 下的 pantograph-repl

    参数:
        project_path: Lean 项目路径

    返回:
        str: 找到的二进制路径，或 None
    """
    candidates = []

    # 1. 环境变量
    env_path = os.environ.get("PANTOGRAPH_REPL_PATH")
    if env_path:
        candidates.append(env_path)

    # 2-3. PyPantograph 目录下的常见位置
    panto_dir = Path(_PYPANTOGRAPH_DIR)
    candidates.extend([
        str(panto_dir / "pantograph" / "pantograph-repl"),
        str(panto_dir / "src" / ".lake" / "build" / "bin" / "repl"),
    ])

    # 4-5. project_path 下的位置
    if project_path:
        pp = Path(project_path)
        candidates.extend([
            str(pp / ".lake" / "build" / "bin" / "repl"),
            str(pp / "pantograph-repl"),
            str(pp / "src" / ".lake" / "build" / "bin" / "repl"),
        ])

    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            logger.info(f"找到 pantograph-repl: {c}")
            return c

    logger.warning(f"未找到 pantograph-repl 二进制，搜索路径: {candidates}")
    return None


def _get_lean_path_safe(project_path: str, timeout: int = 30) -> Optional[str]:
    """
    安全获取 LEAN_PATH，带超时保护
    @author ygw

    先检查环境变量，再尝试通过 lake 命令获取。

    参数:
        project_path: Lean 项目路径
        timeout: lake 命令超时（秒）

    返回:
        str: LEAN_PATH 值，或 None
    """
    # 1. 优先使用环境变量
    env_lp = os.environ.get("LEAN_PATH")
    if env_lp:
        logger.info(f"使用环境变量 LEAN_PATH={env_lp}")
        return env_lp

    # 2. 尝试 lake 命令（带超时）
    import subprocess
    try:
        logger.info(f"执行 lake env printenv LEAN_PATH (timeout={timeout}s) ...")
        result = subprocess.run(
            ["lake", "env", "printenv", "LEAN_PATH"],
            capture_output=True, text=True,
            cwd=project_path, timeout=timeout
        )
        if result.returncode == 0 and result.stdout.strip():
            lean_path = result.stdout.strip()
            logger.info(f"lake 返回 LEAN_PATH={lean_path}")
            return lean_path
        else:
            logger.warning(f"lake 命令返回码={result.returncode}, "
                           f"stdout={result.stdout[:200]}, stderr={result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"lake env printenv LEAN_PATH 超时 ({timeout}s)")
    except FileNotFoundError:
        logger.warning("lake 命令不可用")
    except Exception as e:
        logger.warning(f"获取 LEAN_PATH 失败: {e}")

    # 3. 从 lean-toolchain + elan 工具链 + .lake 目录推断（最可靠的回退策略）
    logger.info("尝试从 lean-toolchain + elan 工具链推断 LEAN_PATH ...")
    tc_lean_path = _get_lean_path_from_toolchain(project_path)
    if tc_lean_path:
        return tc_lean_path

    # 4. 最后的回退：仅从 .lake 目录推断（不含 elan lib）
    lake_packages = Path(project_path) / ".lake" / "packages"
    if lake_packages.is_dir():
        paths = []
        for pkg_dir in lake_packages.iterdir():
            lib_dir = pkg_dir / "lib"
            if lib_dir.is_dir():
                paths.append(str(lib_dir))
            build_lib = pkg_dir / ".lake" / "build" / "lib"
            if build_lib.is_dir():
                paths.append(str(build_lib))
        # 加上项目自身的 build/lib
        own_lib = Path(project_path) / ".lake" / "build" / "lib"
        if own_lib.is_dir():
            paths.append(str(own_lib))
        if paths:
            lean_path = os.pathsep.join(paths)
            logger.info(f"从 .lake/packages 推断 LEAN_PATH={lean_path}")
            return lean_path

    logger.warning("无法获取 LEAN_PATH")
    return None


# ================================================================
# 兼容数据类：保持旧接口的返回类型不变
# ================================================================

@dataclass
class GoalState:
    """
    证明目标状态数据类（兼容旧接口 + 新增元变量信息）

    属性:
        state_id: 状态唯一标识
        goals: 当前待证明的目标列表（字符串格式，兼容旧代码）
        raw_response: 原始响应数据
        panto_state: 底层 PyPantograph GoalState 对象（新增，用于高级操作）
    """
    state_id: int = -1
    goals: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    panto_state: Optional[Any] = field(default=None, repr=False)

    def is_solved(self) -> bool:
        """判断是否所有目标已解决"""
        return len(self.goals) == 0 and self.state_id >= 0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（不含 panto_state）"""
        return {
            "state_id": self.state_id,
            "goals": self.goals,
            "raw_response": self.raw_response,
        }

    @staticmethod
    def from_panto(panto_state: PantoGoalState) -> 'GoalState':
        """
        从 PyPantograph GoalState 转换为兼容的 GoalState

        参数:
            panto_state: PyPantograph 的 GoalState 对象

        返回:
            GoalState: 兼容旧接口的 GoalState
        """
        goals_str = [str(g) for g in panto_state.goals]
        return GoalState(
            state_id=panto_state.state_id,
            goals=goals_str,
            raw_response={},
            panto_state=panto_state,
        )

    def get_goal_objects(self) -> List[Goal]:
        """
        获取底层 Goal 对象列表（含元变量依赖信息 sibling_dep）

        返回:
            List[Goal]: PyPantograph Goal 对象列表，无底层状态时返回空列表
        """
        if self.panto_state and hasattr(self.panto_state, 'goals'):
            return self.panto_state.goals
        return []

    def get_metavar_deps(self) -> Dict[int, Optional[set]]:
        """
        获取每个目标的元变量依赖关系

        返回:
            Dict[int, Optional[set]]: {goal_index: sibling_dep_set}
                sibling_dep 为 None 表示无依赖，set 为依赖的兄弟目标索引集合
        """
        deps = {}
        for i, goal in enumerate(self.get_goal_objects()):
            deps[i] = goal.sibling_dep
        return deps

    def has_metavar_coupling(self) -> bool:
        """
        检查当前目标间是否存在元变量耦合

        返回:
            bool: 存在耦合返回 True
        """
        for dep in self.get_metavar_deps().values():
            if dep is not None and len(dep) > 0:
                return True
        return False


@dataclass
class TacticResult:
    """
    策略执行结果数据类

    属性:
        success: 策略是否执行成功
        state_before: 执行前的状态
        state_after: 执行后的状态
        tactic: 执行的策略
        error_message: 错误信息（失败时）
        duration_ms: 执行耗时（毫秒）
    """
    success: bool = False
    state_before: Optional[GoalState] = None
    state_after: Optional[GoalState] = None
    tactic: str = ""
    error_message: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "success": self.success,
            "state_before": self.state_before.to_dict() if self.state_before else None,
            "state_after": self.state_after.to_dict() if self.state_after else None,
            "tactic": self.tactic,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


class LeanServer:
    """
    Lean4 服务器客户端 —— PyPantograph Server 适配器。

    底层调用 PyPantograph 的 Server 类，上层提供：
    1. 兼容旧接口的便利方法（goal_start, goal_tactic, run_proof 等）
    2. Task 1.1 三大核心能力：
       - 独立子目标求解：goal_continue, goal_resume, goal_tactic_on_goal
       - 无损状态回溯：goal_save, goal_load
       - 元变量耦合追踪：goal_subsume, 自动 sibling_dep 解析
    3. 环境检索能力：env_catalog, env_inspect, env_module_read
    4. 前端处理能力：tactic_invocations, load_sorry, check_compile

    使用示例:
        server = LeanServer(imports=["Mathlib"])
        server.start()
        state = server.goal_start("∀ (n : Nat), n + 0 = n")
        result = server.goal_tactic(state.state_id, "intro n")
        server.stop()
    """

    def __init__(self,
                 executable_path: str = None,
                 timeout: int = 60,
                 project_path: Optional[str] = None,
                 imports: Optional[List[str]] = None,
                 options: Optional[Dict[str, Any]] = None,
                 core_options: Optional[List[str]] = None,
                 repl_path: Optional[str] = None,
                 lean_path: Optional[str] = None):
        """
        初始化 Lean 服务器
        @author ygw

        参数:
            executable_path: 兼容旧接口，等同于 repl_path
            timeout: 命令执行超时时间（秒）
            project_path: Lean 项目路径（用于加载 Mathlib 等依赖）
            imports: 启动时导入的模块列表（如 ["Mathlib"]）
            options: 传给 Pantograph 的选项字典
            core_options: 传给 Lean core 的选项列表
            repl_path: pantograph-repl 二进制路径（None 则自动搜索）
            lean_path: LEAN_PATH 环境变量值（None 则自动获取）
        """
        # repl_path 优先级: repl_path > executable_path > 自动搜索
        self.repl_path = repl_path or executable_path
        self.timeout = timeout
        self.project_path = project_path
        self.imports = imports or ["Init"]
        self.options = options or {}
        self.core_options = core_options or []
        self.lean_path = lean_path  # 可传入预计算的 LEAN_PATH

        # 兼容旧属性名
        self.executable_path = self.repl_path

        self._server: Optional[PantographServer] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._command_counter = 0

    def start(self) -> bool:
        """
        启动 Pantograph repl 进程
        @author ygw

        启动流程（手动控制，避免 PyPantograph 内部阻塞）：
        1. 自动搜索 pantograph-repl 二进制
        2. 安全获取 LEAN_PATH（带超时保护）
        3. 用 _sync_init=False 创建 Server 实例
        4. 注入 proc_path 和 lean_path
        5. 调用 restart() 启动进程

        返回:
            bool: 启动是否成功
        """
        if self._is_running and self._server:
            logger.warning("Lean 服务器已在运行中，跳过重复启动")
            return True

        try:
            # 第 0 步：自动定位 Lean 项目目录（lakefile.lean 所在位置）
            effective_project_path = self.project_path
            if effective_project_path:
                effective_project_path = _find_lean_project_dir(effective_project_path)
                if effective_project_path != self.project_path:
                    logger.info(f"project_path 已自动修正: {self.project_path} -> {effective_project_path}")
            
            logger.info(f"启动 Pantograph Server: imports={self.imports}, "
                        f"project_path={effective_project_path}, timeout={self.timeout}")

            # 第 1 步：定位 repl 二进制
            repl = self.repl_path or _find_repl_binary(effective_project_path)
            if not repl:
                logger.error("找不到 pantograph-repl 二进制！请设置 repl_path 参数或 "
                             "PANTOGRAPH_REPL_PATH 环境变量")
                return False
            logger.info(f"使用 repl 二进制: {repl}")

            # 第 2 步：获取 LEAN_PATH（使用修正后的项目路径）
            lean_path = self.lean_path
            if not lean_path and effective_project_path:
                lean_path = _get_lean_path_safe(effective_project_path, timeout=30)
            if lean_path:
                logger.info(f"LEAN_PATH: {lean_path[:200]}{'...' if len(str(lean_path)) > 200 else ''}")
            else:
                logger.warning("未获取到 LEAN_PATH，将仅使用 Init 模块")

            # 第 3 步：用 _sync_init=False 创建 Server（跳过内部的 get_lean_path 和 restart）
            self._server = PantographServer(
                imports=self.imports,
                project_path=effective_project_path,
                options=self.options,
                core_options=self.core_options,
                timeout=self.timeout,
                _sync_init=False,
            )

            # 第 4 步：手动注入 proc_path 和 lean_path
            self._server.proc_path = repl
            self._server.lean_path = lean_path

            # 第 5 步：启动进程（同步）
            logger.info("正在启动 repl 进程并等待 ready 信号...")
            self._server.restart()

            self._is_running = True
            self._command_counter = 0
            logger.info("Pantograph Server 启动成功")
            return True

        except Exception as e:
            logger.error(f"启动 Pantograph Server 失败: {e}")
            import traceback
            traceback.print_exc()
            self._server = None
            self._is_running = False
            return False

    def stop(self):
        """停止 Pantograph repl 进程，释放资源"""
        if self._server:
            try:
                self._server._close()
            except Exception as e:
                logger.error(f"停止服务器时出错: {e}")
            finally:
                self._server = None
                self._is_running = False
                logger.info("Pantograph Server 已停止")

    def restart(self) -> bool:
        """
        重启服务器

        返回:
            bool: 重启是否成功
        """
        self.stop()
        return self.start()

    def is_running(self) -> bool:
        """
        检查服务器是否在运行

        返回:
            bool: 是否在运行
        """
        if not self._is_running or not self._server:
            return False
        # 检查底层进程是否存活
        if self._server.proc is None:
            self._is_running = False
            return False
        if self._server.proc.returncode is not None:
            self._is_running = False
            return False
        return True

    def get_server(self) -> PantographServer:
        """
        获取底层 PyPantograph Server 实例（供高级用户直接调用）

        返回:
            PantographServer: 底层服务器实例

        异常:
            RuntimeError: 服务器未启动时抛出
        """
        if not self._server or not self._is_running:
            raise RuntimeError("Pantograph Server 未启动，请先调用 start()")
        return self._server

    # ================================================================
    # 核心 API：证明目标管理
    # ================================================================

    def goal_start(self, expr: str) -> Optional[GoalState]:
        """
        创建新的证明目标

        参数:
            expr: Lean4 表达式字符串（如 "∀ (n : Nat), n + 0 = n"）

        返回:
            GoalState: 初始目标状态，失败返回 None
        """
        try:
            panto_state = self._server.goal_start(expr)
            state = GoalState.from_panto(panto_state)
            logger.info(f"创建目标成功，state_id={state.state_id}, 目标数={len(state.goals)}")
            return state
        except (ServerError, Exception) as e:
            logger.error(f"创建目标失败: {e}")
            return None

    def goal_tactic(self, state_id: int, tactic: str,
                    goal_id: Optional[int] = None) -> TacticResult:
        """
        在指定状态上执行策略

        参数:
            state_id: 目标状态 ID
            tactic: 要执行的策略字符串
            goal_id: 指定在哪个子目标上执行（None 表示默认第一个目标）

        返回:
            TacticResult: 策略执行结果
        """
        start_time = time.time()

        try:
            # 需要获取对应的 PantoGoalState
            panto_state = self._find_or_create_panto_state(state_id)
            if panto_state is None:
                duration_ms = (time.time() - start_time) * 1000
                return TacticResult(
                    success=False,
                    tactic=tactic,
                    error_message=f"找不到 state_id={state_id} 对应的状态",
                    duration_ms=duration_ms,
                )

            # 构建 Site 参数（指定子目标）
            site = Site(goal_id=goal_id) if goal_id is not None else Site()

            # 执行策略
            new_panto_state = self._server.goal_tactic(panto_state, tactic, site)
            new_state = GoalState.from_panto(new_panto_state)

            duration_ms = (time.time() - start_time) * 1000
            self._command_counter += 1

            return TacticResult(
                success=True,
                state_after=new_state,
                tactic=tactic,
                duration_ms=duration_ms,
            )

        except TacticFailure as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            return TacticResult(
                success=False,
                tactic=tactic,
                error_message=error_msg,
                duration_ms=duration_ms,
            )
        except (ServerError, ParseError, Exception) as e:
            duration_ms = (time.time() - start_time) * 1000
            return TacticResult(
                success=False,
                tactic=tactic,
                error_message=str(e),
                duration_ms=duration_ms,
            )

    def goal_tactic_on_goal(self, state: GoalState, tactic: Union[str, Tactic],
                            goal_id: int = 0,
                            auto_resume: Optional[bool] = None) -> TacticResult:
        """
        在指定子目标上执行策略（使用 GoalState 对象而非 state_id）

        此方法直接操作 GoalState 对象，支持完整的 Tactic 类型（包括
        TacticHave、TacticLet、TacticExpr、TacticDraft 等高级策略），
        并支持 Site 的 auto_resume 参数。

        参数:
            state: 当前 GoalState 对象（必须含 panto_state）
            tactic: 策略（str 或 Tactic 联合类型）
            goal_id: 子目标索引（默认 0）
            auto_resume: 是否自动恢复休眠目标（None 使用默认行为）

        返回:
            TacticResult: 策略执行结果
        """
        start_time = time.time()

        if state.panto_state is None:
            duration_ms = (time.time() - start_time) * 1000
            return TacticResult(
                success=False,
                tactic=str(tactic),
                error_message="GoalState 缺少 panto_state，无法执行高级策略",
                duration_ms=duration_ms,
            )

        try:
            site = Site(goal_id=goal_id, auto_resume=auto_resume)
            new_panto_state = self._server.goal_tactic(state.panto_state, tactic, site)
            new_state = GoalState.from_panto(new_panto_state)

            duration_ms = (time.time() - start_time) * 1000
            self._command_counter += 1

            return TacticResult(
                success=True,
                state_after=new_state,
                tactic=str(tactic),
                duration_ms=duration_ms,
            )

        except TacticFailure as e:
            duration_ms = (time.time() - start_time) * 1000
            return TacticResult(
                success=False,
                tactic=str(tactic),
                error_message=str(e),
                duration_ms=duration_ms,
            )
        except (ServerError, ParseError, Exception) as e:
            duration_ms = (time.time() - start_time) * 1000
            return TacticResult(
                success=False,
                tactic=str(tactic),
                error_message=str(e),
                duration_ms=duration_ms,
            )

    # ================================================================
    # 核心能力 1：独立子目标求解
    # ================================================================

    def goal_continue(self, target: GoalState, branch: GoalState) -> Optional[GoalState]:
        """
        子目标切换：在 target 上完成 branch 的搜索后，恢复 target 的搜索。

        对应 Pantograph 的 goal.continue 命令。
        当 MCTS 在一个子树上完成搜索后，需要切换到另一个子树时使用。

        参数:
            target: 要恢复搜索的目标状态
            branch: 已完成搜索的分支状态

        返回:
            GoalState: 合并后的新状态，失败返回 None
        """
        if target.panto_state is None or branch.panto_state is None:
            logger.error("goal_continue 需要两个有效的 panto_state")
            return None

        try:
            result = self._server.goal_continue(target.panto_state, branch.panto_state)
            return GoalState.from_panto(result)
        except (ServerError, Exception) as e:
            logger.error(f"goal_continue 失败: {e}")
            return None

    def goal_resume(self, state: GoalState, goals: List[Goal]) -> Optional[GoalState]:
        """
        恢复休眠目标：将指定的 goals 重新带入搜索范围。

        当 Pantograph 运行在非 automaticMode 时，求解一个目标后，
        其他目标可能进入休眠状态。此方法将它们唤醒。

        参数:
            state: 当前状态
            goals: 要恢复的 Goal 对象列表

        返回:
            GoalState: 包含已恢复目标的新状态，失败返回 None
        """
        if state.panto_state is None:
            logger.error("goal_resume 需要有效的 panto_state")
            return None

        try:
            result = self._server.goal_resume(state.panto_state, goals)
            return GoalState.from_panto(result)
        except (ServerError, Exception) as e:
            logger.error(f"goal_resume 失败: {e}")
            return None

    # ================================================================
    # 核心能力 2：无损状态回溯
    # ================================================================

    def goal_save(self, state: GoalState, path: str) -> bool:
        """
        保存目标状态快照到文件（无损回溯的基础）

        用于 MCTS 搜索中保存关键节点的内核快照，以便后续无损回溯。
        当内层战术树搜索失败时，可通过 goal_load 恢复到此快照。

        参数:
            state: 要保存的目标状态
            path: 保存路径

        返回:
            bool: 保存是否成功
        """
        if state.panto_state is None:
            logger.error("goal_save 需要有效的 panto_state")
            return False

        try:
            self._server.goal_save(state.panto_state, path)
            logger.info(f"目标状态已保存: state_id={state.state_id}, path={path}")
            return True
        except (ServerError, Exception) as e:
            logger.error(f"goal_save 失败: {e}")
            return False

    def goal_load(self, path: str) -> Optional[GoalState]:
        """
        从文件加载目标状态快照（无损回溯的恢复操作）

        恢复之前保存的内核快照，用于 MCTS 回溯到之前的搜索节点。
        注意：用户需自行确保环境与保存时一致。

        参数:
            path: 快照文件路径

        返回:
            GoalState: 恢复的目标状态，失败返回 None
        """
        try:
            panto_state = self._server.goal_load(path)
            state = GoalState.from_panto(panto_state)
            logger.info(f"目标状态已加载: state_id={state.state_id}, path={path}")
            return state
        except (ServerError, Exception) as e:
            logger.error(f"goal_load 失败: {e}")
            return None

    # ================================================================
    # 核心能力 3：元变量耦合追踪
    # ================================================================

    def goal_subsume(self, state: GoalState, goal: Goal,
                     candidates: List[Goal],
                     src_state: Optional[GoalState] = None
                     ) -> Optional[Tuple[Subsumption, Optional[GoalState], Optional[Goal]]]:
        """
        元变量归并检测：检查一个目标是否被候选目标归并。

        这是 MAGC-MCTS 中处理元变量耦合的核心方法。当一个分支中的
        变量被实例化后，Pantograph 需要检查其他分支的目标是否因此
        被归并（Subsumed），从而避免重复搜索。

        参数:
            state: 当前目标状态
            goal: 待检查的目标
            candidates: 候选归并目标列表
            src_state: 候选目标所在的源状态（可选）

        返回:
            Tuple[Subsumption, Optional[GoalState], Optional[Goal]]:
                - Subsumption: 归并类型（NONE / SUBSUMED / CYCLE）
                - GoalState: 归并后的新状态（仅当非循环时）
                - Goal: 归并者目标（仅当发生归并时）
            失败返回 None
        """
        if state.panto_state is None:
            logger.error("goal_subsume 需要有效的 panto_state")
            return None

        try:
            src_panto = src_state.panto_state if src_state and src_state.panto_state else None
            sub, new_panto_state, subsumptor = self._server.goal_subsume(
                state.panto_state, goal, candidates, src_panto
            )

            new_state = None
            if new_panto_state is not None:
                new_state = GoalState.from_panto(new_panto_state)

            return (sub, new_state, subsumptor)

        except (ServerError, Exception) as e:
            logger.error(f"goal_subsume 失败: {e}")
            return None

    def analyze_metavar_coupling(self, state: GoalState) -> Dict[str, Any]:
        """
        分析当前状态中目标间的元变量耦合图

        用于 MCTS 搜索策略决策：判断哪些子目标可以独立并行搜索，
        哪些需要按序处理。

        参数:
            state: 当前目标状态

        返回:
            Dict: {
                "total_goals": 目标总数,
                "has_coupling": 是否存在耦合,
                "coupling_graph": {goal_idx: set(dependent_goal_indices)},
                "independent_goals": [可独立搜索的目标索引列表],
                "coupled_groups": [[耦合目标组1], [耦合目标组2], ...]
            }
        """
        goals = state.get_goal_objects()
        n = len(goals)

        coupling_graph = {}
        for i, goal in enumerate(goals):
            deps = goal.sibling_dep
            coupling_graph[i] = set(deps) if deps else set()

        # 使用 Union-Find 发现连通分量
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, deps in coupling_graph.items():
            for j in deps:
                if 0 <= j < n:
                    union(i, j)

        # 构建连通分量
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        coupled_groups = [g for g in groups.values() if len(g) > 1]
        independent_goals = [g[0] for g in groups.values() if len(g) == 1]

        return {
            "total_goals": n,
            "has_coupling": len(coupled_groups) > 0,
            "coupling_graph": {k: list(v) for k, v in coupling_graph.items()},
            "independent_goals": independent_goals,
            "coupled_groups": coupled_groups,
        }

    # ================================================================
    # 环境查询 API
    # ================================================================

    def env_inspect(self, name: str, print_value: bool = False,
                    print_dependency: bool = False) -> Optional[Dict[str, Any]]:
        """
        查询 Lean 环境中常量的类型信息

        参数:
            name: 常量的完整名称（如 "Nat.add_comm"）
            print_value: 是否打印值
            print_dependency: 是否打印依赖

        返回:
            Dict: 包含 type、value、dependency 等字段的响应字典，失败返回 None
        """
        try:
            result = self._server.env_inspect(name, print_value, print_dependency)
            return result
        except (ServerError, Exception) as e:
            logger.error(f"env_inspect 失败 ({name}): {e}")
            return None

    def env_catalog(self, module_prefix: Optional[str] = None,
                    invert_filter: bool = False) -> List[str]:
        """
        列出环境中所有符号名

        参数:
            module_prefix: 模块前缀过滤（如 "Mathlib.Topology"）
            invert_filter: 反转过滤条件

        返回:
            List[str]: 符号名列表
        """
        try:
            return self._server.env_catalog(module_prefix, invert_filter)
        except (ServerError, Exception) as e:
            logger.error(f"env_catalog 失败: {e}")
            return []

    def env_module_read(self, module: str) -> Optional[Dict]:
        """
        读取指定 Lean 模块的内容

        参数:
            module: 模块名（如 "Mathlib.Topology.Basic"）

        返回:
            Dict: 模块内容信息，失败返回 None
        """
        try:
            return self._server.env_module_read(module)
        except (ServerError, Exception) as e:
            logger.error(f"env_module_read 失败 ({module}): {e}")
            return None

    def env_add(self, name: str, levels: List[str], t: str, v: str,
                is_theorem: bool = True) -> bool:
        """
        向环境添加定义

        参数:
            name: 定义名称
            levels: universe levels
            t: 类型表达式
            v: 值表达式
            is_theorem: 是否为定理

        返回:
            bool: 是否添加成功
        """
        try:
            self._server.env_add(name, levels, t, v, is_theorem)
            return True
        except (ServerError, Exception) as e:
            logger.error(f"env_add 失败 ({name}): {e}")
            return False

    # ================================================================
    # 前端处理 API
    # ================================================================

    def tactic_invocations(self, file_name: str) -> List[CompilationUnit]:
        """
        提取文件中的策略调用点

        可用于 CoS 数据提取：从 Lean 源码中提取所有策略调用的
        前后目标状态。

        参数:
            file_name: Lean 源码文件路径

        返回:
            List[CompilationUnit]: 编译单元列表
        """
        try:
            return self._server.tactic_invocations(file_name)
        except (ServerError, Exception) as e:
            logger.error(f"tactic_invocations 失败 ({file_name}): {e}")
            return []

    def load_sorry(self, src: str, binder_name: Optional[str] = None,
                   ignore_values: bool = False) -> List[SearchTarget]:
        """
        从含 sorry 的代码中提取搜索目标

        参数:
            src: Lean 源码字符串
            binder_name: 绑定名称（可选）
            ignore_values: 是否忽略值

        返回:
            List[SearchTarget]: 搜索目标列表
        """
        try:
            return self._server.load_sorry(src, binder_name, ignore_values)
        except (ServerError, Exception) as e:
            logger.error(f"load_sorry 失败: {e}")
            return []

    def check_compile(self, code: str,
                      new_constants: bool = False,
                      read_header: bool = False) -> Optional[List[CompilationUnit]]:
        """
        检查 Lean 代码是否能编译通过

        参数:
            code: Lean 源码字符串
            new_constants: 是否收集新常量
            read_header: 是否读取头部

        返回:
            List[CompilationUnit]: 编译单元列表，失败返回 None
        """
        try:
            return self._server.check_compile(code, new_constants, read_header)
        except (ServerError, Exception) as e:
            logger.error(f"check_compile 失败: {e}")
            return None

    def load_header(self, header: str) -> bool:
        """
        从头部加载环境

        参数:
            header: 头部代码（如 import 语句）

        返回:
            bool: 是否成功
        """
        try:
            self._server.load_header(header)
            return True
        except (ServerError, Exception) as e:
            logger.error(f"load_header 失败: {e}")
            return False

    def load_definitions(self, snippet: str) -> bool:
        """
        加载定义到环境中

        参数:
            snippet: Lean 代码片段

        返回:
            bool: 是否成功
        """
        try:
            self._server.load_definitions(snippet)
            return True
        except (ServerError, Exception) as e:
            logger.error(f"load_definitions 失败: {e}")
            return False

    # ================================================================
    # 状态管理辅助
    # ================================================================

    def goal_print(self, state_id: int) -> Optional[str]:
        """
        打印指定状态的详细信息（兼容旧接口）

        参数:
            state_id: 目标状态 ID

        返回:
            str: 状态的文本表示，失败返回 None
        """
        try:
            # 通过底层 run 命令获取
            result = self._server.run('goal.print', {
                'stateId': state_id,
                'goals': True,
            })
            if "error" in result:
                logger.error(f"打印状态失败: {result}")
                return None
            goals = result.get('goals', [])
            if not goals:
                return "goals accomplished"
            # 格式化输出
            parts = []
            for g in goals:
                target = g.get('target', {}).get('pp', str(g.get('target', '')))
                variables = g.get('vars', [])
                var_strs = []
                for v in variables:
                    vname = v.get('userName', '_')
                    vtype = v.get('type', {}).get('pp', str(v.get('type', '')))
                    var_strs.append(f"{vname} : {vtype}")
                part = "\n".join(var_strs) + f"\n⊢ {target}" if var_strs else f"⊢ {target}"
                parts.append(part)
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"goal_print 失败: {e}")
            return None

    def goal_delete(self, state_id: int) -> bool:
        """
        删除指定状态，释放服务器端资源

        参数:
            state_id: 目标状态 ID

        返回:
            bool: 是否删除成功
        """
        try:
            result = self._server.run('goal.delete', {'stateIds': [state_id]})
            return "error" not in result
        except Exception:
            return False

    def gc(self):
        """手动触发垃圾回收，释放已删除的 GoalState 内存"""
        try:
            self._server.gc()
        except Exception as e:
            logger.warning(f"gc 失败: {e}")

    # ================================================================
    # 便利方法：兼容旧接口
    # ================================================================

    def goal_start_copy(self, copy_from: str) -> Optional[GoalState]:
        """
        从已有定理/定义创建证明目标

        参数:
            copy_from: 常量全名（如 "Nat.add_comm"）

        返回:
            GoalState: 初始目标状态，失败返回 None
        """
        try:
            result = self._server.run('goal.start', {"copyFrom": copy_from})
            if "error" in result:
                logger.error(f"copyFrom 创建目标失败 ({copy_from}): {result}")
                return None
            # 手动构建 PantoGoalState
            panto_state = PantoGoalState(
                state_id=result["stateId"],
                goals=[Goal.sentence(result["root"], copy_from)],
                messages=[],
                _sentinel=self._server.to_remove_goal_states,
            )
            return GoalState.from_panto(panto_state)
        except (ServerError, Exception) as e:
            logger.error(f"copyFrom 创建目标失败 ({copy_from}): {e}")
            return None

    def goal_start_with_fallback(self, theorem_full_name: str
                                  ) -> Union[Tuple[Optional['GoalState'], bool], Optional['GoalState']]:
        """
        创建证明目标，带多级回退策略

        回退顺序:
        1. copyFrom（直接引用常量名，变量已自动引入）
        2. env.inspect 获取类型 → goal_start(expr=type)（变量未引入，需 intros）

        参数:
            theorem_full_name: 定理全名（如 "Nat.add_comm"）

        返回:
            Tuple[Optional[GoalState], bool]:
                - GoalState: 初始目标状态，所有策略均失败返回 None
                - bool: 是否使用了 goal_start(expr) 路径（True=需要 intros 引入变量）
        @author ygw 2026-03-01 v2: 返回 needs_intro 标记
        """
        # 策略 1: copyFrom（变量已在上下文中，不需要 intros）
        state = self.goal_start_copy(theorem_full_name)
        if state is not None:
            return state, False

        # 策略 2: env.inspect 获取类型（变量在 ∀ 里，需要 intros）
        logger.info(f"copyFrom 失败，尝试 env.inspect 回退: {theorem_full_name}")
        inspect_result = self.env_inspect(theorem_full_name)
        if inspect_result and "type" in inspect_result:
            type_info = inspect_result["type"]
            if isinstance(type_info, dict):
                type_expr = type_info.get("pp", type_info.get("default", ""))
            else:
                type_expr = str(type_info)

            if type_expr:
                state = self.goal_start(type_expr)
                if state is not None:
                    logger.info(f"env.inspect 回退成功: {theorem_full_name}")
                    return state, True

        logger.warning(f"所有 goal_start 策略均失败: {theorem_full_name}")
        return None, False

    def run_proof(self, theorem_expr: str, tactics: List[str]) -> List[TacticResult]:
        """
        执行完整的证明过程，记录每一步的状态变化

        参数:
            theorem_expr: 定理表达式
            tactics: 策略列表

        返回:
            List[TacticResult]: 每步策略的执行结果
        """
        results = []

        initial_state = self.goal_start(theorem_expr)
        if initial_state is None:
            logger.error(f"无法创建证明目标: {theorem_expr}")
            return results

        current_state = initial_state

        for i, tactic in enumerate(tactics):
            # 获取执行前状态文本
            state_before_text = self.goal_print(current_state.state_id)

            result = self.goal_tactic(current_state.state_id, tactic)

            # 补充执行前状态信息
            if state_before_text:
                result.state_before = GoalState(
                    state_id=current_state.state_id,
                    goals=[state_before_text]
                )

            results.append(result)

            if not result.success:
                logger.warning(f"策略执行失败 (步骤 {i+1}/{len(tactics)}): "
                               f"tactic='{tactic}', error='{result.error_message}'")
                break

            if result.state_after:
                current_state = result.state_after
                if result.state_after.is_solved():
                    logger.info(f"证明完成！共 {i+1} 步")
                    break

        return results

    def extract_state_pair(self, theorem_expr: str, tactic: str,
                           preceding_tactics: Optional[List[str]] = None
                           ) -> Optional[Tuple[str, str]]:
        """
        提取单个策略执行前后的状态对 (S_pre, S_post)

        参数:
            theorem_expr: 定理表达式
            tactic: 目标策略
            preceding_tactics: 在目标策略之前需要执行的策略列表

        返回:
            Tuple[str, str]: (执行前状态, 执行后状态)，失败返回 None
        """
        initial_state = self.goal_start(theorem_expr)
        if initial_state is None:
            return None

        current_state = initial_state

        # 先执行前置策略
        if preceding_tactics:
            for pre_tactic in preceding_tactics:
                result = self.goal_tactic(current_state.state_id, pre_tactic)
                if not result.success:
                    logger.warning(f"前置策略执行失败: {pre_tactic}")
                    return None
                if result.state_after:
                    current_state = result.state_after

        # 获取执行前状态
        state_pre = self.goal_print(current_state.state_id)
        if state_pre is None:
            return None

        # 执行目标策略
        result = self.goal_tactic(current_state.state_id, tactic)
        if not result.success:
            return None

        # 获取执行后状态
        state_post = ""
        if result.state_after:
            if result.state_after.is_solved():
                state_post = "goals accomplished"
            else:
                post_text = self.goal_print(result.state_after.state_id)
                state_post = post_text if post_text else ""

        return (state_pre, state_post)

    # ================================================================
    # 内部辅助方法
    # ================================================================

    def _find_or_create_panto_state(self, state_id: int) -> Optional[PantoGoalState]:
        """
        根据 state_id 获取或构建 PantoGoalState

        由于旧接口仅传递 state_id，需要构建一个最小化的 PantoGoalState
        以便调用 PyPantograph 的方法。

        参数:
            state_id: 状态 ID

        返回:
            PantoGoalState: 底层状态对象，失败返回 None
        """
        try:
            # 先通过 goal.print 获取目标信息
            result = self._server.run('goal.print', {
                'stateId': state_id,
                'goals': True,
            })
            if "error" in result:
                return None

            goals = result.get('goals', [])
            return PantoGoalState.parse_inner(
                state_id, goals, [],
                self._server.to_remove_goal_states
            )
        except Exception as e:
            logger.error(f"_find_or_create_panto_state 失败 (state_id={state_id}): {e}")
            return None

    # ================================================================
    # 上下文管理器支持
    # ================================================================

    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动停止服务器"""
        self.stop()
        return False


@contextmanager
def create_lean_server(executable_path: str = None,
                       timeout: int = 60,
                       project_path: str = None,
                       imports: Optional[List[str]] = None,
                       repl_path: Optional[str] = None,
                       lean_path: Optional[str] = None):
    """
    创建 LeanServer 的上下文管理器工厂函数
    @author ygw

    参数:
        executable_path: 兼容旧接口，等同 repl_path
        timeout: 命令超时时间（秒）
        project_path: Lean 项目路径
        imports: 启动时导入的模块列表
        repl_path: pantograph-repl 二进制路径
        lean_path: LEAN_PATH 环境变量值

    使用示例:
        with create_lean_server(imports=["Mathlib"]) as server:
            state = server.goal_start("∀ (n : Nat), n + 0 = n")
            result = server.goal_tactic(state.state_id, "intro n")
    """
    server = LeanServer(
        executable_path=executable_path,
        timeout=timeout,
        project_path=project_path,
        imports=imports,
        repl_path=repl_path,
        lean_path=lean_path,
    )

    try:
        if not server.start():
            raise RuntimeError("无法启动 Pantograph Server")
        yield server
    finally:
        server.stop()


class LeanServerPool:
    """
    Lean 服务器连接池，支持并发场景下的多实例管理。

    用于 CoS 提取等需要并发访问 Pantograph 的场景。

    使用示例:
        pool = LeanServerPool(max_size=4)
        pool.start_all()
        server = pool.acquire()
        # ... 使用 server ...
        pool.release(server)
        pool.stop_all()
    """

    def __init__(self,
                 max_size: int = 4,
                 executable_path: str = None,
                 timeout: int = 60,
                 project_path: str = None,
                 imports: Optional[List[str]] = None,
                 repl_path: Optional[str] = None,
                 lean_path: Optional[str] = None):
        """
        初始化服务器连接池
        @author ygw

        参数:
            max_size: 最大连接数
            executable_path: 兼容旧接口，等同 repl_path
            timeout: 命令超时时间（秒）
            project_path: Lean 项目路径
            imports: 启动时导入的模块列表
            repl_path: pantograph-repl 二进制路径
            lean_path: LEAN_PATH 环境变量值
        """
        self.max_size = max_size
        self.executable_path = executable_path
        self.repl_path = repl_path
        self.lean_path = lean_path
        self.timeout = timeout
        self.project_path = project_path
        self.imports = imports or ["Init"]
        self._pool: List[LeanServer] = []
        self._available: List[LeanServer] = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_size)

    def start_all(self) -> int:
        """
        启动所有服务器实例

        返回:
            int: 成功启动的实例数
        """
        success_count = 0
        for i in range(self.max_size):
            server = LeanServer(
                executable_path=self.executable_path,
                timeout=self.timeout,
                project_path=self.project_path,
                imports=self.imports,
                repl_path=self.repl_path,
                lean_path=self.lean_path,
            )
            if server.start():
                self._pool.append(server)
                self._available.append(server)
                success_count += 1
                logger.info(f"服务器实例 {i+1}/{self.max_size} 启动成功")
            else:
                logger.error(f"服务器实例 {i+1}/{self.max_size} 启动失败")

        logger.info(f"服务器池启动完成: {success_count}/{self.max_size} 个实例就绪")
        return success_count

    def stop_all(self):
        """停止所有服务器实例"""
        for server in self._pool:
            server.stop()
        self._pool.clear()
        self._available.clear()
        logger.info("服务器池已全部停止")

    def acquire(self) -> Optional[LeanServer]:
        """
        从池中获取一个可用的服务器实例（阻塞等待）

        返回:
            LeanServer: 可用的服务器实例，池耗尽返回 None
        """
        self._semaphore.acquire()
        with self._lock:
            if self._available:
                server = self._available.pop()
                if server.is_running():
                    return server
                else:
                    logger.warning("获取到的服务器实例已停止，尝试重启")
                    if server.start():
                        return server
                    if server in self._pool:
                        self._pool.remove(server)
                    logger.error("服务器实例重启失败，已从池中移除")
                    self._semaphore.release()
                    return None
            logger.error("服务器池可用列表为空")
            self._semaphore.release()
            return None

    def release(self, server: LeanServer):
        """
        归还服务器实例到池中

        参数:
            server: 要归还的服务器实例
        """
        with self._lock:
            if server.is_running():
                self._available.append(server)
            else:
                logger.warning("归还的服务器实例已停止，尝试重启")
                if server.start():
                    self._available.append(server)
                else:
                    if server in self._pool:
                        self._pool.remove(server)
                    logger.error("服务器实例重启失败，已从池中移除")
        self._semaphore.release()

    @contextmanager
    def get_server(self):
        """
        上下文管理器方式获取和归还服务器

        使用示例:
            with pool.get_server() as server:
                state = server.goal_start(expr)
        """
        server = self.acquire()
        if server is None:
            raise RuntimeError("无法从池中获取服务器实例")
        try:
            yield server
        finally:
            self.release(server)

    def __enter__(self):
        """支持 with 语句"""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动停止所有实例"""
        self.stop_all()
        return False
