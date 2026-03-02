"""
路径配置文件
@author ygw

集中管理所有工具和数据的路径配置
"""

import os
from pathlib import Path


class PathConfig:
    """路径配置类"""

    # 项目根目录
    PROJECT_ROOT = "/root/autodl-tmp/RTAP"

    # Workspace 目录
    WORKSPACE_ROOT = os.path.join(PROJECT_ROOT, "workspace")

    # 工具路径
    PANTOGRAPH_PATH = os.path.join(WORKSPACE_ROOT, "PyPantograph", "src")
    PANTOGRAPH_EXECUTABLE = os.path.join(PANTOGRAPH_PATH, ".lake", "build", "bin", "repl")

    LEANDOJO_PATH = os.path.join(WORKSPACE_ROOT, "LeanDojo-v2")

    # 数据路径
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    MATHLIB4_PATH = os.path.join(DATA_ROOT, "raw", "mathlib4")

    # 处理后的数据路径
    PROCESSED_DATA_PATH = os.path.join(DATA_ROOT, "processed")
    TRACES_PATH = os.path.join(PROCESSED_DATA_PATH, "traces")
    COS_DATASET_PATH = os.path.join(PROCESSED_DATA_PATH, "cos_dataset")
    ERROR_CORRECTION_PATH = os.path.join(PROCESSED_DATA_PATH, "error_correction")

    # 向量数据库路径
    VECTOR_DB_PATH = os.path.join(DATA_ROOT, "vector_db")

    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        directories = [
            cls.WORKSPACE_ROOT,
            cls.DATA_ROOT,
            cls.PROCESSED_DATA_PATH,
            cls.TRACES_PATH,
            cls.COS_DATASET_PATH,
            cls.ERROR_CORRECTION_PATH,
            cls.VECTOR_DB_PATH,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ 目录已就绪: {directory}")

    @classmethod
    def check_installations(cls):
        """检查必要的工具是否已安装"""
        checks = {
            "PyPantograph": Path(cls.PANTOGRAPH_PATH).exists(),
            "LeanDojo-v2": Path(cls.LEANDOJO_PATH).exists(),
            "Mathlib4": Path(cls.MATHLIB4_PATH).exists(),
        }

        print("\n工具安装检查:")
        for tool, exists in checks.items():
            status = "✓" if exists else "✗"
            print(f"{status} {tool}: {'已安装' if exists else '未安装'}")

        return all(checks.values())


def main():
    """主函数 - 显示配置信息"""
    print("=" * 60)
    print("RTAP-v3 路径配置")
    print("=" * 60)
    print()

    print("项目根目录:", PathConfig.PROJECT_ROOT)
    print()

    print("工具路径:")
    print(f"  PyPantograph: {PathConfig.PANTOGRAPH_PATH}")
    print(f"  LeanDojo-v2: {PathConfig.LEANDOJO_PATH}")
    print()

    print("数据路径:")
    print(f"  Mathlib4: {PathConfig.MATHLIB4_PATH}")
    print(f"  处理后数据: {PathConfig.PROCESSED_DATA_PATH}")
    print(f"  向量数据库: {PathConfig.VECTOR_DB_PATH}")
    print()

    # 确保目录存在
    PathConfig.ensure_directories()
    print()

    # 检查安装
    all_installed = PathConfig.check_installations()

    if not all_installed:
        print("\n⚠️  部分工具未安装，请按照以下步骤安装：")
        print()

        if not Path(PathConfig.MATHLIB4_PATH).exists():
            print("安装 Mathlib4:")
            print(f"  cd {PathConfig.DATA_ROOT}/raw")
            print("  git clone https://github.com/leanprover-community/mathlib4.git")
            print(f"  cd {PathConfig.MATHLIB4_PATH}")
            print("  lake build")
            print()


if __name__ == "__main__":
    main()
