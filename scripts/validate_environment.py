"""
环境验证脚本
@author ygw

验证 LeanDojo-v2、Pantograph 和 Mathlib4 的安装和配置
"""

import subprocess
import sys
import os
from pathlib import Path

# 导入路径配置
try:
    from config_paths import PathConfig
except ImportError:
    # 如果无法导入，使用默认路径
    class PathConfig:
        MATHLIB4_PATH = "/root/autodl-tmp/RTAP/data/raw/mathlib4"
        PANTOGRAPH_PATH = "/root/autodl-tmp/RTAP/workspace/PyPantograph"


class EnvironmentValidator:
    """环境验证器"""

    def __init__(self):
        """初始化验证器"""
        self.results = []
        self.passed = 0
        self.failed = 0

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """
        记录测试结果

        参数:
            test_name: 测试名称
            passed: 是否通过
            message: 附加消息
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append(f"{status} | {test_name}: {message}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def check_lean_version(self) -> bool:
        """
        检查 Lean 版本

        返回:
            bool: 是否通过检查
        """
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            version = result.stdout.strip()
            # 接受 Lean 4.x 的任何版本
            passed = "Lean (version 4." in version
            self.log_result("Lean 版本检查", passed, version)
            return passed
        except Exception as e:
            self.log_result("Lean 版本检查", False, str(e))
            return False

    def check_mathlib4(self) -> bool:
        """
        检查 Mathlib4 安装

        返回:
            bool: 是否通过检查
        """
        try:
            mathlib_path = Path(PathConfig.MATHLIB4_PATH)
            if not mathlib_path.exists():
                self.log_result("Mathlib4 安装检查", False, "目录不存在")
                return False

            # 检查关键文件
            lakefile = mathlib_path / "lakefile.lean"
            if not lakefile.exists():
                self.log_result("Mathlib4 安装检查", False, "lakefile.lean 不存在")
                return False

            self.log_result("Mathlib4 安装检查", True, f"路径: {mathlib_path}")
            return True
        except Exception as e:
            self.log_result("Mathlib4 安装检查", False, str(e))
            return False

    def check_pantograph(self) -> bool:
        """
        检查 Pantograph 安装

        返回:
            bool: 是否通过检查
        """
        try:
            pantograph_path = Path(PathConfig.PANTOGRAPH_PATH)
            if not pantograph_path.exists():
                self.log_result("Pantograph 安装检查", False, "目录不存在")
                return False

            # 检查可执行文件（repl）
            executable = Path(PathConfig.PANTOGRAPH_EXECUTABLE)
            if not executable.exists():
                self.log_result("Pantograph 安装检查", False, f"可执行文件不存在: {executable}")
                return False

            self.log_result("Pantograph 安装检查", True, f"路径: {pantograph_path}")
            return True
        except Exception as e:
            self.log_result("Pantograph 安装检查", False, str(e))
            return False

    def check_leandojo(self) -> bool:
        """
        检查 LeanDojo 安装

        返回:
            bool: 是否通过检查
        """
        try:
            import lean_dojo_v2
            version = getattr(lean_dojo_v2, "__version__", "unknown")
            self.log_result("LeanDojo 安装检查", True, f"版本: {version}")
            return True
        except ImportError as e:
            self.log_result("LeanDojo 安装检查", False, "无法导入 lean_dojo_v2")
            return False
        except Exception as e:
            self.log_result("LeanDojo 安装检查", False, str(e))
            return False

    def check_python_dependencies(self) -> bool:
        """
        检查 Python 依赖

        返回:
            bool: 是否通过检查
        """
        required_packages = [
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "yaml"
        ]

        all_passed = True
        for package in required_packages:
            try:
                __import__(package)
                self.log_result(f"Python 包: {package}", True, "已安装")
            except ImportError:
                self.log_result(f"Python 包: {package}", False, "未安装")
                all_passed = False

        return all_passed

    def run_all_checks(self):
        """运行所有检查"""
        print("=" * 60)
        print("RTAP-v3 环境验证")
        print("=" * 60)
        print()

        # 运行所有检查
        self.check_lean_version()
        self.check_mathlib4()
        self.check_pantograph()
        self.check_leandojo()
        self.check_python_dependencies()

        # 输出结果
        print()
        print("=" * 60)
        print("验证结果")
        print("=" * 60)
        for result in self.results:
            print(result)

        print()
        print(f"总计: {self.passed} 通过, {self.failed} 失败")
        print("=" * 60)

        return self.failed == 0


def main():
    """主函数"""
    validator = EnvironmentValidator()
    success = validator.run_all_checks()

    if success:
        print("\n✓ 所有检查通过！环境配置正确。")
        sys.exit(0)
    else:
        print("\n✗ 部分检查失败，请检查环境配置。")
        sys.exit(1)


if __name__ == "__main__":
    main()
