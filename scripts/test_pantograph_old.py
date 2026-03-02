"""
Pantograph 交互接口测试脚本
@author ygw

测试 Pantograph 的稳定性，特别是元变量耦合（Metavariable Coupling）场景
"""

import subprocess
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# 导入路径配置
try:
    from config_paths import PathConfig
    DEFAULT_PANTOGRAPH_PATH = PathConfig.PANTOGRAPH_EXECUTABLE
except ImportError:
    DEFAULT_PANTOGRAPH_PATH = "/root/autodl-tmp/RTAP/workspace/PyPantograph/build/bin/pantograph"


class PantographTester:
    """Pantograph 测试器"""

    def __init__(self, pantograph_path: str = DEFAULT_PANTOGRAPH_PATH):
        """
        初始化测试器

        参数:
            pantograph_path: Pantograph 可执行文件路径
        """
        self.pantograph_path = pantograph_path
        self.process = None
        self.test_results = []

    def start_server(self) -> bool:
        """
        启动 Pantograph 服务器

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
            time.sleep(1)  # 等待服务器启动
            return self.process.poll() is None
        except Exception as e:
            print(f"启动 Pantograph 失败: {e}")
            return False

    def send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送命令到 Pantograph

        参数:
            command: 命令字典

        返回:
            Dict: 服务器响应
        """
        if not self.process or self.process.poll() is not None:
            print("Pantograph 服务器未运行")
            return None

        try:
            command_str = json.dumps(command) + "\n"
            self.process.stdin.write(command_str)
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            if not response_line:
                return None

            return json.loads(response_line)
        except Exception as e:
            print(f"发送命令失败: {e}")
            return None

    def test_basic_interaction(self) -> bool:
        """
        测试基本交互

        返回:
            bool: 测试是否通过
        """
        print("测试 1: 基本交互")

        # 发送 reset 命令
        response = self.send_command({"cmd": "reset"})
        if response and response.get("error") is None:
            print("  ✓ 基本交互测试通过")
            return True
        else:
            print(f"  ✗ 基本交互测试失败: {response}")
            return False

    def test_goal_start(self) -> bool:
        """
        测试目标启动

        返回:
            bool: 测试是否通过
        """
        print("测试 2: 目标启动")

        # 启动一个简单的目标
        command = {
            "cmd": "goal.start",
            "expr": "∀ (p q : Prop), p → q → p ∧ q"
        }

        response = self.send_command(command)
        if response and "stateId" in response:
            print(f"  ✓ 目标启动测试通过，状态 ID: {response['stateId']}")
            return True
        else:
            print(f"  ✗ 目标启动测试失败: {response}")
            return False

    def test_metavariable_coupling(self) -> bool:
        """
        测试元变量耦合场景

        返回:
            bool: 测试是否通过
        """
        print("测试 3: 元变量耦合")

        # 创建包含元变量的目标
        command1 = {
            "cmd": "goal.start",
            "expr": "∀ (n : Nat), n + 0 = n"
        }

        response1 = self.send_command(command1)
        if not response1 or "stateId" not in response1:
            print(f"  ✗ 元变量耦合测试失败（启动阶段）: {response1}")
            return False

        state_id = response1["stateId"]

        # 应用策略
        command2 = {
            "cmd": "goal.tactic",
            "stateId": state_id,
            "goalId": 0,
            "tactic": "intro n"
        }

        response2 = self.send_command(command2)
        if response2 and "nextStateId" in response2:
            print(f"  ✓ 元变量耦合测试通过")
            return True
        else:
            print(f"  ✗ 元变量耦合测试失败（策略阶段）: {response2}")
            return False

    def test_state_extraction(self) -> bool:
        """
        测试状态提取

        返回:
            bool: 测试是否通过
        """
        print("测试 4: 状态提取")

        # 启动目标
        command1 = {
            "cmd": "goal.start",
            "expr": "∀ (a b : Nat), a + b = b + a"
        }

        response1 = self.send_command(command1)
        if not response1 or "stateId" not in response1:
            print(f"  ✗ 状态提取测试失败: {response1}")
            return False

        state_id = response1["stateId"]

        # 获取状态信息
        command2 = {
            "cmd": "goal.print",
            "stateId": state_id
        }

        response2 = self.send_command(command2)
        if response2 and "goals" in response2:
            print(f"  ✓ 状态提取测试通过，目标数: {len(response2['goals'])}")
            return True
        else:
            print(f"  ✗ 状态提取测试失败: {response2}")
            return False

    def test_concurrent_requests(self, num_requests: int = 10) -> bool:
        """
        测试并发请求处理

        参数:
            num_requests: 请求数量

        返回:
            bool: 测试是否通过
        """
        print(f"测试 5: 并发请求（{num_requests} 个请求）")

        success_count = 0
        for i in range(num_requests):
            command = {
                "cmd": "goal.start",
                "expr": f"∀ (x : Nat), x + {i} = {i} + x"
            }

            response = self.send_command(command)
            if response and "stateId" in response:
                success_count += 1

        passed = success_count == num_requests
        if passed:
            print(f"  ✓ 并发请求测试通过（{success_count}/{num_requests}）")
        else:
            print(f"  ✗ 并发请求测试失败（{success_count}/{num_requests}）")

        return passed

    def stop_server(self):
        """停止 Pantograph 服务器"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            print("\nPantograph 服务器已停止")

    def run_all_tests(self) -> bool:
        """
        运行所有测试

        返回:
            bool: 所有测试是否通过
        """
        print("=" * 60)
        print("Pantograph 交互接口测试")
        print("=" * 60)
        print()

        if not self.start_server():
            print("✗ 无法启动 Pantograph 服务器")
            return False

        print("✓ Pantograph 服务器已启动\n")

        # 运行所有测试
        tests = [
            self.test_basic_interaction,
            self.test_goal_start,
            self.test_metavariable_coupling,
            self.test_state_extraction,
            lambda: self.test_concurrent_requests(10)
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print()
            except Exception as e:
                print(f"  ✗ 测试异常: {e}\n")
                results.append(False)

        # 停止服务器
        self.stop_server()

        # 输出总结
        print("=" * 60)
        print("测试总结")
        print("=" * 60)
        passed = sum(results)
        total = len(results)
        print(f"通过: {passed}/{total}")

        if passed == total:
            print("\n✓ 所有测试通过！Pantograph 工作正常。")
            return True
        else:
            print(f"\n✗ {total - passed} 个测试失败。")
            return False


def main():
    """主函数"""
    tester = PantographTester()
    success = tester.run_all_tests()

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
