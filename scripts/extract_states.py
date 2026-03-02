"""
状态提取脚本
@author ygw

基于 Pantograph 提取 Lean 证明的 Tactic State
"""

import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 导入路径配置
try:
    from config_paths import PathConfig
    DEFAULT_PANTOGRAPH_PATH = PathConfig.PANTOGRAPH_EXECUTABLE
except ImportError:
    DEFAULT_PANTOGRAPH_PATH = "/root/autodl-tmp/RTAP/workspace/PyPantograph/build/bin/pantograph"


@dataclass
class TacticState:
    """策略状态数据类"""
    state_id: int
    goals: List[str]
    hypotheses: List[Dict[str, str]]
    goal_type: str


class StateExtractor:
    """状态提取器"""

    def __init__(self, pantograph_path: str = DEFAULT_PANTOGRAPH_PATH):
        """
        初始化状态提取器

        参数:
            pantograph_path: Pantograph 可执行文件路径
        """
        self.pantograph_path = pantograph_path
        self.process = None

    def start(self) -> bool:
        """
        启动 Pantograph 服务

        返回:
            bool: 启动是否成功
        """
        try:
            self.process = subprocess.Popen(
                [self.pantograph_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            return True
        except Exception as e:
            print(f"启动失败: {e}")
            return False

    def send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送命令到 Pantograph

        参数:
            command: 命令字典

        返回:
            Dict: 服务器响应
        """
        if not self.process:
            return None

        try:
            command_str = json.dumps(command) + "\n"
            self.process.stdin.write(command_str)
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            return json.loads(response_line) if response_line else None
        except Exception as e:
            print(f"命令发送失败: {e}")
            return None

    def extract_initial_state(self, theorem_expr: str) -> Optional[TacticState]:
        """
        提取定理的初始状态

        参数:
            theorem_expr: 定理表达式

        返回:
            TacticState: 初始状态
        """
        # 启动目标
        response = self.send_command({
            "cmd": "goal.start",
            "expr": theorem_expr
        })

        if not response or "stateId" not in response:
            return None

        state_id = response["stateId"]

        # 获取状态详情
        state_info = self.send_command({
            "cmd": "goal.print",
            "stateId": state_id
        })

        if not state_info or "goals" not in state_info:
            return None

        # 解析状态
        goals = state_info.get("goals", [])
        if not goals:
            return None

        first_goal = goals[0]
        return TacticState(
            state_id=state_id,
            goals=[g.get("target", "") for g in goals],
            hypotheses=first_goal.get("hypotheses", []),
            goal_type=first_goal.get("target", "")
        )

    def extract_state_after_tactic(self,
                                   state_id: int,
                                   goal_id: int,
                                   tactic: str) -> Optional[Tuple[TacticState, bool]]:
        """
        提取应用策略后的状态

        参数:
            state_id: 当前状态 ID
            goal_id: 目标 ID
            tactic: 策略字符串

        返回:
            Tuple: (新状态, 是否成功)
        """
        # 应用策略
        response = self.send_command({
            "cmd": "goal.tactic",
            "stateId": state_id,
            "goalId": goal_id,
            "tactic": tactic
        })

        if not response:
            return None, False

        # 检查是否有错误
        if "error" in response:
            return None, False

        # 检查是否完成证明
        if "goals" in response and len(response["goals"]) == 0:
            return TacticState(
                state_id=response.get("nextStateId", -1),
                goals=[],
                hypotheses=[],
                goal_type="Proof completed"
            ), True

        # 获取新状态
        next_state_id = response.get("nextStateId")
        if not next_state_id:
            return None, False

        state_info = self.send_command({
            "cmd": "goal.print",
            "stateId": next_state_id
        })

        if not state_info or "goals" not in state_info:
            return None, False

        goals = state_info.get("goals", [])
        if not goals:
            return TacticState(
                state_id=next_state_id,
                goals=[],
                hypotheses=[],
                goal_type="Proof completed"
            ), True

        first_goal = goals[0]
        return TacticState(
            state_id=next_state_id,
            goals=[g.get("target", "") for g in goals],
            hypotheses=first_goal.get("hypotheses", []),
            goal_type=first_goal.get("target", "")
        ), True

    def extract_proof_trace(self,
                           theorem_expr: str,
                           tactics: List[str]) -> List[Dict[str, Any]]:
        """
        提取完整的证明轨迹

        参数:
            theorem_expr: 定理表达式
            tactics: 策略列表

        返回:
            List[Dict]: 证明轨迹（状态-策略对）
        """
        trace = []

        # 获取初始状态
        initial_state = self.extract_initial_state(theorem_expr)
        if not initial_state:
            return trace

        trace.append({
            "step": 0,
            "state": {
                "state_id": initial_state.state_id,
                "goals": initial_state.goals,
                "hypotheses": initial_state.hypotheses,
                "goal_type": initial_state.goal_type
            },
            "tactic": None,
            "next_state": None
        })

        current_state_id = initial_state.state_id
        current_goal_id = 0

        # 逐步应用策略
        for i, tactic in enumerate(tactics):
            next_state, success = self.extract_state_after_tactic(
                current_state_id,
                current_goal_id,
                tactic
            )

            if not success or not next_state:
                trace.append({
                    "step": i + 1,
                    "state": None,
                    "tactic": tactic,
                    "next_state": None,
                    "error": "策略应用失败"
                })
                break

            trace.append({
                "step": i + 1,
                "state": trace[-1]["state"],
                "tactic": tactic,
                "next_state": {
                    "state_id": next_state.state_id,
                    "goals": next_state.goals,
                    "hypotheses": next_state.hypotheses,
                    "goal_type": next_state.goal_type
                }
            })

            current_state_id = next_state.state_id

            # 如果证明完成，退出
            if len(next_state.goals) == 0:
                break

        return trace

    def stop(self):
        """停止 Pantograph 服务"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)


def main():
    """主函数 - 示例用法"""
    extractor = StateExtractor()

    if not extractor.start():
        print("无法启动状态提取器")
        return

    print("状态提取器已启动\n")

    # 示例：提取简单定理的证明轨迹
    theorem = "∀ (p q : Prop), p → q → p ∧ q"
    tactics = [
        "intro p q",
        "intro hp hq",
        "constructor",
        "exact hp",
        "exact hq"
    ]

    print(f"定理: {theorem}")
    print(f"策略: {tactics}\n")

    trace = extractor.extract_proof_trace(theorem, tactics)

    print("=" * 60)
    print("证明轨迹")
    print("=" * 60)

    for entry in trace:
        print(f"\n步骤 {entry['step']}:")
        if entry['tactic']:
            print(f"  策略: {entry['tactic']}")
        if entry.get('next_state'):
            print(f"  目标数: {len(entry['next_state']['goals'])}")
            if entry['next_state']['goals']:
                print(f"  当前目标: {entry['next_state']['goal_type']}")
        if entry.get('error'):
            print(f"  错误: {entry['error']}")

    extractor.stop()
    print("\n状态提取器已停止")


if __name__ == "__main__":
    main()
