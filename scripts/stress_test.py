"""
压力测试脚本
@author ygw

对 Lean 交互环境进行压力测试，验证稳定性和性能
"""

import subprocess
import json
import time
import threading
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入路径配置
try:
    from config_paths import PathConfig
    DEFAULT_PANTOGRAPH_PATH = PathConfig.PANTOGRAPH_EXECUTABLE
except ImportError:
    DEFAULT_PANTOGRAPH_PATH = "/root/autodl-tmp/RTAP/workspace/PyPantograph/build/bin/pantograph"


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    success: bool
    duration: float
    error_message: str = ""


class StressTest:
    """压力测试类"""

    def __init__(self, pantograph_path: str = DEFAULT_PANTOGRAPH_PATH):
        """
        初始化压力测试

        参数:
            pantograph_path: Pantograph 可执行文件路径
        """
        self.pantograph_path = pantograph_path
        self.results: List[TestResult] = []

    def create_process(self) -> subprocess.Popen:
        """
        创建 Pantograph 进程

        返回:
            Popen: 进程对象
        """
        return subprocess.Popen(
            [self.pantograph_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_command(self, process: subprocess.Popen, command: Dict[str, Any]) -> Tuple[bool, float]:
        """
        发送命令并测量响应时间

        参数:
            process: Pantograph 进程
            command: 命令字典

        返回:
            Tuple: (是否成功, 响应时间)
        """
        try:
            start_time = time.time()

            command_str = json.dumps(command) + "\n"
            process.stdin.write(command_str)
            process.stdin.flush()

            response_line = process.stdout.readline()
            duration = time.time() - start_time

            if not response_line:
                return False, duration

            response = json.loads(response_line)
            success = "error" not in response

            return success, duration
        except Exception as e:
            return False, time.time() - start_time

    def test_sequential_requests(self, num_requests: int = 100) -> TestResult:
        """
        测试顺序请求处理

        参数:
            num_requests: 请求数量

        返回:
            TestResult: 测试结果
        """
        print(f"测试 1: 顺序请求 ({num_requests} 个)")

        process = self.create_process()
        time.sleep(1)

        durations = []
        success_count = 0
        start_time = time.time()

        for i in range(num_requests):
            command = {
                "cmd": "goal.start",
                "expr": f"∀ (x : Nat), x + {i} = {i} + x"
            }

            success, duration = self.send_command(process, command)
            durations.append(duration)

            if success:
                success_count += 1

            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{num_requests}")

        total_duration = time.time() - start_time
        process.terminate()
        process.wait(timeout=5)

        avg_duration = statistics.mean(durations) if durations else 0
        success_rate = success_count / num_requests * 100

        print(f"  成功率: {success_rate:.1f}%")
        print(f"  平均响应时间: {avg_duration*1000:.2f}ms")
        print(f"  总耗时: {total_duration:.2f}s")

        return TestResult(
            test_name="顺序请求测试",
            success=success_rate >= 95,
            duration=total_duration,
            error_message=f"成功率: {success_rate:.1f}%" if success_rate < 95 else ""
        )

    def test_concurrent_requests(self, num_threads: int = 10, requests_per_thread: int = 10) -> TestResult:
        """
        测试并发请求处理

        参数:
            num_threads: 线程数
            requests_per_thread: 每个线程的请求数

        返回:
            TestResult: 测试结果
        """
        print(f"测试 2: 并发请求 ({num_threads} 线程 × {requests_per_thread} 请求)")

        def worker(thread_id: int) -> Tuple[int, int]:
            """工作线程"""
            process = self.create_process()
            time.sleep(0.5)

            success_count = 0
            for i in range(requests_per_thread):
                command = {
                    "cmd": "goal.start",
                    "expr": f"∀ (x : Nat), x + {thread_id * 100 + i} = {thread_id * 100 + i} + x"
                }

                success, _ = self.send_command(process, command)
                if success:
                    success_count += 1

            process.terminate()
            process.wait(timeout=5)

            return success_count, requests_per_thread

        start_time = time.time()
        total_success = 0
        total_requests = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]

            for future in as_completed(futures):
                success, total = future.result()
                total_success += success
                total_requests += total

        total_duration = time.time() - start_time
        success_rate = total_success / total_requests * 100

        print(f"  成功率: {success_rate:.1f}%")
        print(f"  总耗时: {total_duration:.2f}s")
        print(f"  吞吐量: {total_requests / total_duration:.2f} 请求/秒")

        return TestResult(
            test_name="并发请求测试",
            success=success_rate >= 90,
            duration=total_duration,
            error_message=f"成功率: {success_rate:.1f}%" if success_rate < 90 else ""
        )

    def test_long_running_session(self, duration_seconds: int = 60) -> TestResult:
        """
        测试长时间运行会话

        参数:
            duration_seconds: 运行时长（秒）

        返回:
            TestResult: 测试结果
        """
        print(f"测试 3: 长时间运行会话 ({duration_seconds} 秒)")

        process = self.create_process()
        time.sleep(1)

        start_time = time.time()
        request_count = 0
        success_count = 0

        while time.time() - start_time < duration_seconds:
            command = {
                "cmd": "goal.start",
                "expr": f"∀ (x : Nat), x + {request_count} = {request_count} + x"
            }

            success, _ = self.send_command(process, command)
            request_count += 1

            if success:
                success_count += 1

            if request_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  进度: {elapsed:.0f}s / {duration_seconds}s, 请求数: {request_count}")

            time.sleep(0.1)  # 避免过载

        total_duration = time.time() - start_time
        process.terminate()
        process.wait(timeout=5)

        success_rate = success_count / request_count * 100 if request_count > 0 else 0

        print(f"  总请求数: {request_count}")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  平均速率: {request_count / total_duration:.2f} 请求/秒")

        return TestResult(
            test_name="长时间运行测试",
            success=success_rate >= 95,
            duration=total_duration,
            error_message=f"成功率: {success_rate:.1f}%" if success_rate < 95 else ""
        )

    def test_complex_proofs(self, num_proofs: int = 20) -> TestResult:
        """
        测试复杂证明处理

        参数:
            num_proofs: 证明数量

        返回:
            TestResult: 测试结果
        """
        print(f"测试 4: 复杂证明处理 ({num_proofs} 个)")

        process = self.create_process()
        time.sleep(1)

        complex_theorems = [
            "∀ (n : Nat), n + 0 = n",
            "∀ (a b c : Nat), a + (b + c) = (a + b) + c",
            "∀ (p q : Prop), p ∧ q → q ∧ p",
            "∀ (p q r : Prop), (p → q) → (q → r) → (p → r)",
            "∀ (n m : Nat), n + m = m + n"
        ]

        success_count = 0
        start_time = time.time()

        for i in range(num_proofs):
            theorem = complex_theorems[i % len(complex_theorems)]
            command = {
                "cmd": "goal.start",
                "expr": theorem
            }

            success, _ = self.send_command(process, command)
            if success:
                success_count += 1

            if (i + 1) % 5 == 0:
                print(f"  进度: {i + 1}/{num_proofs}")

        total_duration = time.time() - start_time
        process.terminate()
        process.wait(timeout=5)

        success_rate = success_count / num_proofs * 100

        print(f"  成功率: {success_rate:.1f}%")
        print(f"  总耗时: {total_duration:.2f}s")

        return TestResult(
            test_name="复杂证明测试",
            success=success_rate >= 95,
            duration=total_duration,
            error_message=f"成功率: {success_rate:.1f}%" if success_rate < 95 else ""
        )

    def run_all_tests(self) -> bool:
        """
        运行所有压力测试

        返回:
            bool: 所有测试是否通过
        """
        print("=" * 60)
        print("Lean 环境压力测试")
        print("=" * 60)
        print()

        # 运行所有测试
        self.results.append(self.test_sequential_requests(100))
        print()

        self.results.append(self.test_concurrent_requests(10, 10))
        print()

        self.results.append(self.test_long_running_session(30))
        print()

        self.results.append(self.test_complex_proofs(20))
        print()

        # 输出总结
        print("=" * 60)
        print("测试总结")
        print("=" * 60)

        for result in self.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(f"{status} | {result.test_name} ({result.duration:.2f}s)")
            if result.error_message:
                print(f"       {result.error_message}")

        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)

        print()
        print(f"通过: {passed}/{total}")

        if passed == total:
            print("\n✓ 所有压力测试通过！环境稳定可靠。")
            return True
        else:
            print(f"\n✗ {total - passed} 个测试失败。")
            return False


def main():
    """主函数"""
    tester = StressTest()
    success = tester.run_all_tests()

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
