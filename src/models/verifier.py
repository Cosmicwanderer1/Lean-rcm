"""
Pantograph 形式化验证器
@author ygw
更新日期: 2026-02-27

基于 Pantograph 的 Lean 4 形式化环境验证器。
提供与 Lean 4 编译器的交互接口，用于:
    - 在评测流水线中验证模型生成的 tactic 是否合法
    - 在 MCTS 搜索中作为环境 step 执行器
    - 支持多步证明验证 (verify_proof)

协议参考: scripts/test_pantograph.py 中已验证的 JSON RPC 接口
"""

import json
import logging
import os
import subprocess
from typing import Dict, Any, Optional, List

logger = logging.getLogger("verifier")


class PantographVerifier:
    """
    基于 Pantograph 的 Lean 4 形式化环境验证器

    核心职责:
        - 启动并维护 Pantograph 进程与 Lean 4 编译器的交互
        - 通过 goal.start 启动证明目标
        - 通过 goal.tactic 逐步执行策略
        - 通过 goal.print 获取当前证明状态
        - 提供 verify_proof 高层接口供评测流水线调用

    协议对齐:
        命令格式与 scripts/test_pantograph.py 严格一致:
        - goal.start: {"cmd": "goal.start", "expr": "..."} -> {"stateId": N}
        - goal.tactic: {"cmd": "goal.tactic", "stateId": N, "goalId": G, "tactic": "..."}
                       -> {"nextStateId": M, "goals": [...]}
        - goal.print:  {"cmd": "goal.print", "stateId": N} -> {"goals": [...]}
        - reset:       {"cmd": "reset"} -> {}

    参数:
        project_path (str): Lean 项目的根目录路径
        pantograph_path (str): Pantograph 可执行文件路径
        timeout (float): 单次命令超时秒数
    """

    # 默认可执行文件路径 — 对齐服务器实际部署位置
    DEFAULT_PANTOGRAPH_PATH = (
        "/root/autodl-tmp/RTAP/workspace/PyPantograph/src/.lake/build/bin/repl"
    )

    # repl 必须在此目录下运行 (包含 lakefile.lean)
    DEFAULT_PANTOGRAPH_CWD = (
        "/root/autodl-tmp/RTAP/workspace/PyPantograph/src"
    )

    # Mathlib 项目目录 — 用于获取 Mathlib 的 LEAN_PATH
    DEFAULT_MATHLIB_PATH = (
        "/root/autodl-tmp/RTAP/data/raw/mathlib4"
    )

    def __init__(self,
                 project_path: str = "/root/autodl-tmp/RTAP",
                 pantograph_path: Optional[str] = None,
                 pantograph_cwd: Optional[str] = None,
                 mathlib_path: Optional[str] = None,
                 imports: Optional[List[str]] = None,
                 timeout: float = 30.0):
        """
        初始化验证器

        参数:
            project_path (str): RTAP 项目的根目录路径
            pantograph_path (str): Pantograph 可执行文件路径，None 则使用默认路径
            pantograph_cwd (str): repl 进程的工作目录 (必须包含 lakefile.lean)，
                                  None 则使用默认路径 PyPantograph/src
            mathlib_path (str): Mathlib4 项目目录路径，用于获取 Mathlib 的 LEAN_PATH，
                                None 则使用默认路径
            imports (List[str]): repl 启动时加载的 Lean 模块列表，
                                 如 ["Mathlib"]，None 则默认加载 Mathlib
            timeout (float): 单次命令的超时秒数
        """
        self.project_path = project_path
        self.pantograph_path = pantograph_path or self.DEFAULT_PANTOGRAPH_PATH
        self.pantograph_cwd = pantograph_cwd or self.DEFAULT_PANTOGRAPH_CWD
        self.mathlib_path = mathlib_path or self.DEFAULT_MATHLIB_PATH
        self.imports = imports if imports is not None else ["Mathlib"]
        self.timeout = timeout
        self.process = None
        self._is_ready = False

    # ================================================================
    # 生命周期管理
    # ================================================================

    def start_server(self) -> bool:
        """
        启动 Pantograph 交互进程并确认就绪

        返回:
            bool: 启动是否成功
        """
        if self.process and self.process.poll() is None:
            logger.warning("Pantograph 进程已在运行，跳过重复启动")
            return True

        try:
            # 1. 构建 LEAN_PATH 环境变量
            #    需要合并两个来源:
            #    - PyPantograph/src 的 LEAN_PATH (包含 LSpec, Pantograph lib)
            #    - Mathlib4 项目的 LEAN_PATH (包含 Mathlib 及其依赖: batteries, Qq, aesop 等)
            env = os.environ.copy()
            lean_paths = []

            # 1a. 获取 Mathlib 的 LEAN_PATH (包含 Mathlib 本体及所有依赖)
            try:
                mathlib_result = subprocess.run(
                    ["lake", "env", "printenv", "LEAN_PATH"],
                    cwd=self.mathlib_path,
                    capture_output=True, text=True, timeout=60
                )
                if mathlib_result.returncode == 0 and mathlib_result.stdout.strip():
                    lean_paths.append(mathlib_result.stdout.strip())
                    logger.info(f"Mathlib LEAN_PATH 已获取 ({len(mathlib_result.stdout.strip().split(':'))} 个路径)")
            except Exception as e:
                logger.warning(f"获取 Mathlib LEAN_PATH 失败 ({e})")

            # 1b. 获取 PyPantograph 的 LEAN_PATH (包含 LSpec, Pantograph)
            try:
                panto_result = subprocess.run(
                    ["lake", "env", "printenv", "LEAN_PATH"],
                    cwd=self.pantograph_cwd,
                    capture_output=True, text=True, timeout=30
                )
                if panto_result.returncode == 0 and panto_result.stdout.strip():
                    lean_paths.append(panto_result.stdout.strip())
                    logger.info(f"PyPantograph LEAN_PATH 已获取")
            except Exception as e:
                logger.warning(f"获取 PyPantograph LEAN_PATH 失败 ({e})")

            # 1c. 合并去重 (保持顺序，Mathlib 优先)
            if lean_paths:
                seen = set()
                merged = []
                for path_str in lean_paths:
                    for p in path_str.split(":"):
                        if p and p not in seen:
                            seen.add(p)
                            merged.append(p)
                env["LEAN_PATH"] = ":".join(merged)
                logger.info(f"合并后 LEAN_PATH 共 {len(merged)} 个路径")

            # 2. 构建启动命令: repl [Module1] [Module2] ...
            cmd = [self.pantograph_path] + self.imports
            # cwd 必须是包含 lakefile.lean 的 PyPantograph/src 目录
            # 否则 repl 无法找到 Lean 环境，会挂起或报错
            self.process = subprocess.Popen(
                cmd,
                cwd=self.pantograph_cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # repl 启动后第一行输出是 "ready."，必须读取并丢弃
            # 否则后续 JSON 命令的响应读取会错位
            ready_line = self._readline_with_timeout()
            if ready_line is None:
                stderr_output = self.process.stderr.read() if self.process.poll() is not None else ""
                logger.error(f"Pantograph 启动超时，未收到 ready 信号: {stderr_output}")
                self.close()
                return False

            ready_line = ready_line.strip()
            if ready_line != "ready.":
                logger.warning(f"Pantograph 首行输出非预期: '{ready_line}'，继续尝试...")

            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                logger.error(f"Pantograph 启动后立即退出: {stderr_output}")
                return False

            # 发送 reset 命令确认通信正常
            response = self._send_raw({"cmd": "reset"})
            if response is not None:
                self._is_ready = True
                logger.info("Pantograph 验证引擎已启动并就绪")
                return True
            else:
                logger.error("Pantograph 启动成功但 reset 命令无响应")
                self.close()
                return False

        except FileNotFoundError:
            logger.error(f"Pantograph 可执行文件不存在: {self.pantograph_path}")
            return False
        except Exception as e:
            logger.error(f"Pantograph 启动失败: {e}")
            return False

    def close(self):
        """关闭验证器进程，释放资源"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self._is_ready = False
            logger.info("Pantograph 验证引擎已关闭")

    def restart(self) -> bool:
        """
        重启 Pantograph 进程 (崩溃恢复)

        返回:
            bool: 重启是否成功
        """
        logger.warning("正在重启 Pantograph...")
        self.close()
        return self.start_server()

    def __enter__(self):
        """支持 context manager: with PantographVerifier() as v: ..."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动关闭"""
        self.close()
        return False

    @property
    def is_alive(self) -> bool:
        """检查 Pantograph 进程是否存活"""
        return self.process is not None and self.process.poll() is None

    # ================================================================
    # 底层通信
    # ================================================================

    def _send_raw(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        向 Pantograph 发送 JSON 命令并等待响应 (带超时保护)

        参数:
            command (Dict): 命令字典

        返回:
            Dict: 响应字典，通信失败返回 None
        """
        if not self.process or self.process.poll() is not None:
            logger.error("Pantograph 进程未运行")
            return None

        try:
            # Pantograph repl 协议格式: "命令名 {参数JSON}\n"
            # 例: goal.start {"expr": "..."}\n
            # 注意: 不是 {"cmd": "goal.start", "expr": "..."}\n
            cmd_name = command.pop("cmd")
            payload_str = json.dumps(command) if command else "{}"
            cmd_str = f"{cmd_name} {payload_str}\n"
            # 恢复 command 字典，避免影响调用方
            command["cmd"] = cmd_name

            self.process.stdin.write(cmd_str)
            self.process.stdin.flush()

            # 带超时的读取
            response_str = self._readline_with_timeout()
            if response_str:
                return json.loads(response_str)
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Pantograph 响应 JSON 解析失败: {e}")
            return None
        except BrokenPipeError:
            logger.error("Pantograph 进程已崩溃 (BrokenPipe)")
            self._is_ready = False
            return None
        except Exception as e:
            logger.error(f"发送命令时出错: {e}")
            return None

    def _readline_with_timeout(self) -> Optional[str]:
        """
        带超时保护的 readline，防止 Pantograph 挂起导致死锁

        返回:
            str: 读取到的一行文本，超时返回 None
        """
        import threading

        result = [None]

        def _read():
            try:
                result[0] = self.process.stdout.readline()
            except Exception:
                result[0] = None

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            logger.warning(f"Pantograph 响应超时 ({self.timeout}s)")
            return None

        return result[0]

    # ================================================================
    # Pantograph 协议命令
    # ================================================================

    def reset(self) -> bool:
        """
        重置 Pantograph 状态 (题目间切换时调用)

        返回:
            bool: 重置是否成功
        """
        response = self._send_raw({"cmd": "reset"})
        if response is not None and "error" not in response:
            return True
        logger.warning(f"reset 失败: {response}")
        return False

    def load_header(self, header: str) -> bool:
        """
        加载 Lean 4 头部代码到 Pantograph 环境

        通过 frontend.process + inheritEnv=True 将 import / open / set_option 等
        声明注入到当前 Pantograph 环境中，使后续 goal.start 可以使用对应命名空间
        中的符号和记号。

        典型 header 内容:
            import Mathlib
            import Aesop
            set_option maxHeartbeats 0
            open BigOperators Real Nat Rat Finset

        参数:
            header (str): Lean 4 头部代码

        返回:
            bool: 加载是否成功
        """
        if not header or not header.strip():
            return True

        response = self._send_raw({
            "cmd": "frontend.process",
            "file": header,
            "readHeader": True,
            "inheritEnv": True,
            "newConstants": False
        })
        if response is not None and "error" not in response:
            logger.info("Header 环境已加载")
            return True

        error = response.get("error", "未知错误") if response else "无响应"
        logger.warning(f"load_header 失败: {error}")
        return False

    def goal_start(self, expr: str) -> Optional[int]:
        """
        从定理表达式启动证明目标

        参数:
            expr (str): Lean 4 定理表达式 (如 "forall (n : Nat), n + 0 = n")

        返回:
            int: 初始状态 ID，失败返回 None
        """
        response = self._send_raw({
            "cmd": "goal.start",
            "expr": expr
        })
        if response and "stateId" in response:
            return response["stateId"]

        error = response.get("error", "未知错误") if response else "无响应"
        logger.error(f"goal.start 失败: {error}")
        return None

    def goal_tactic(self, state_id: int, tactic: str,
                    goal_id: int = 0) -> Dict[str, Any]:
        """
        在指定状态节点上执行 Tactic

        参数:
            state_id (int): Pantograph 维护的当前状态 ID
            tactic (str): 模型生成的 Lean 4 策略
            goal_id (int): 目标 ID (多目标时指定，默认 0 即第一个目标)

        返回:
            Dict: 执行结果:
                - is_success (bool): 是否彻底完成证明 (goals 为空)
                - is_valid (bool): 策略是否合法且无报错
                - new_state_id (int|None): 产生的新状态 ID
                - goals (List): 剩余目标列表
                - error_msg (str|None): 报错信息
        """
        cmd = {
            "cmd": "goal.tactic",
            "stateId": state_id,
            "goalId": goal_id,
            "tactic": tactic
        }

        response = self._send_raw(cmd)

        result = {
            "is_success": False,
            "is_valid": False,
            "new_state_id": None,
            "goals": [],
            "error_msg": None,
        }

        if not response:
            result["error_msg"] = "Pantograph 无响应"
            return result

        if "error" in response:
            result["error_msg"] = response["error"]
            return result

        # parseError 表示策略语法解析失败
        if "parseError" in response:
            result["error_msg"] = response.get("desc", response["parseError"])
            return result

        # 策略执行成功 — 解析 nextStateId 和 goals
        if "nextStateId" in response:
            result["new_state_id"] = response["nextStateId"]
            result["goals"] = response.get("goals", [])

            # hasSorry / hasUnsafe 检查:
            # 策略产生了 sorry 或 unsafe 代码，视为无效
            if response.get("hasSorry", False):
                result["error_msg"] = "Tactic generated sorry"
                return result
            if response.get("hasUnsafe", False):
                result["error_msg"] = "Tactic generated unsafe"
                return result

            result["is_valid"] = True
            # goals 为空表示证明完成
            if not result["goals"]:
                result["is_success"] = True

        return result

    def goal_print(self, state_id: int) -> Optional[List[str]]:
        """
        获取指定状态的目标列表

        Pantograph 的 goal.print 必须显式传 goals=True 才会返回目标详情。
        返回的 goals 是结构化对象列表，每个目标包含:
            - vars: 变量列表 (每个含 userName, type.pp)
            - target: 目标表达式 (含 pp 字段)
        本方法将其格式化为模型可读的字符串形式:
            "x : ℤ\\nh₀ : 0 < x\\n⊢ x = 8"

        参数:
            state_id (int): 状态 ID

        返回:
            List[str]: 目标字符串列表，失败返回 None
        """
        response = self._send_raw({
            "cmd": "goal.print",
            "stateId": state_id,
            "goals": True
        })
        if response and "goals" in response:
            return self._format_goals(response["goals"])

        error = response.get("error", "未知错误") if response else "无响应"
        logger.warning(f"goal.print 失败: {error}")
        return None

    @staticmethod
    def _format_goals(raw_goals: list) -> List[str]:
        """
        将 Pantograph 返回的结构化目标对象转为可读字符串

        参数:
            raw_goals (list): Pantograph goal.print 返回的原始目标列表，
                每项为 dict 包含 vars, target, name, userName 等字段

        返回:
            List[str]: 格式化后的目标字符串列表
        """
        goal_strs = []
        for g in raw_goals:
            # 如果 goal 已经是字符串 (兼容旧版协议)，直接使用
            if isinstance(g, str):
                goal_strs.append(g)
                continue

            parts = []
            # 格式化假设变量: "name : type" 或 "name : type := value"
            for v in g.get("vars", []):
                name = v.get("userName", "_")
                t_obj = v.get("type", {})
                t = t_obj.get("pp", str(t_obj)) if isinstance(t_obj, dict) else str(t_obj)
                line = f"{name} : {t}"
                val = v.get("value")
                if val:
                    val_pp = val.get("pp", str(val)) if isinstance(val, dict) else str(val)
                    line += f" := {val_pp}"
                parts.append(line)

            # 格式化目标结论: "⊢ target"
            target_obj = g.get("target", {})
            target = target_obj.get("pp", str(target_obj)) if isinstance(target_obj, dict) else str(target_obj)
            parts.append(f"⊢ {target}")
            goal_strs.append("\n".join(parts))

        return goal_strs

    # ================================================================
    # 高层验证接口 (评测流水线使用)
    # ================================================================

    def verify_proof(self, theorem_expr: str,
                     tactics: List[str]) -> Dict[str, Any]:
        """
        验证完整证明: 给定定理表达式和策略序列，判断是否能完成证明。

        这是评测流水线的核心入口。流程:
            1. goal.start 启动定理
            2. 依次 goal.tactic 执行每个策略
            3. 检查最终是否 goals 为空

        参数:
            theorem_expr (str): Lean 4 定理表达式
            tactics (List[str]): 策略序列

        返回:
            Dict: 验证结果:
                - success (bool): 证明是否完成
                - steps_executed (int): 成功执行的策略步数
                - total_steps (int): 策略总数
                - error_step (int|None): 出错步骤的索引 (0-based)
                - error_msg (str|None): 出错信息
                - remaining_goals (List): 剩余未证目标
        """
        result = {
            "success": False,
            "steps_executed": 0,
            "total_steps": len(tactics),
            "error_step": None,
            "error_msg": None,
            "remaining_goals": [],
        }

        if not self.is_alive:
            result["error_msg"] = "Pantograph 进程未运行"
            return result

        # 1. 启动证明目标
        state_id = self.goal_start(theorem_expr)
        if state_id is None:
            result["error_msg"] = f"无法启动证明目标: {theorem_expr[:100]}"
            return result

        # 2. 逐步执行策略
        current_state_id = state_id
        for step_idx, tactic in enumerate(tactics):
            if not tactic or not tactic.strip():
                continue

            step_result = self.goal_tactic(current_state_id, tactic.strip())

            if not step_result["is_valid"]:
                result["error_step"] = step_idx
                result["error_msg"] = step_result["error_msg"]
                return result

            result["steps_executed"] = step_idx + 1
            current_state_id = step_result["new_state_id"]

            # 证明完成
            if step_result["is_success"]:
                result["success"] = True
                return result

        # 所有策略执行完毕但证明未完成 — 获取剩余目标
        remaining = self.goal_print(current_state_id)
        result["remaining_goals"] = remaining or []

        return result

    def verify_single_tactic(self, theorem_expr: str,
                             tactic: str) -> Dict[str, Any]:
        """
        单步验证: 对一个定理只执行一步 tactic，返回结果。
        适用于 Pass@1 评测中模型只生成单步策略的场景。

        参数:
            theorem_expr (str): Lean 4 定理表达式
            tactic (str): 单个 Lean 4 策略

        返回:
            Dict: 同 verify_proof 返回格式
        """
        return self.verify_proof(theorem_expr, [tactic])
