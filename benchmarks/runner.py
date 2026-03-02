"""
RTAP-v3 SFT Phase 1 基线评测流水线
@author ygw
更新日期: 2026-02-27

核心功能:
    1. 加载并解析 MiniF2F-v2 测试集
    2. 调度 Generator (LoRA) 进行单路贪婪解码 (Pass@1 Baseline)
    3. 调度 PantographVerifier 进行实时 Lean 4 环境验证
    4. 统计 Pass@1 准确率并输出详细的错误追踪日志
"""

import os
import sys
import re
import json
import time
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# 将项目根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.generator import ThoughtCoSTacticGenerator
from src.models.verifier import PantographVerifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmarks/eval_minif2f.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("runner")

class MiniF2FLoader:
    """MiniF2F-v2 数据集解析器 (支持 JSONL 格式)"""

    @staticmethod
    def load_jsonl(filepath: str, split: str = "test") -> List[Dict[str, str]]:
        """
        从 MiniF2F-v2 JSONL 文件加载定理，提取 Pantograph goal.start 所需的类型表达式。

        参数:
            filepath (str): JSONL 文件路径
            split (str): 数据集分割，"test" 或 "valid"，空字符串则加载全部

        返回:
            List[Dict]: 每项包含 name, expr (Lean 4 类型表达式)
        """
        if not os.path.exists(filepath):
            logger.error(f"找不到数据集文件: {filepath}")
            return []

        theorems = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # 按 split 过滤 (如果指定)
                if split and record.get("split", "") != split:
                    continue

                name = record.get("name", "")
                formal = record.get("formal_statement", "")
                if not formal:
                    continue

                # 从 formal_statement 提取类型表达式
                expr = MiniF2FLoader._extract_type_expr(formal)
                if expr:
                    theorems.append({
                        "name": name,
                        "expr": expr,
                        "header": record.get("header", "")
                    })
                else:
                    logger.warning(f"无法从 {name} 提取类型表达式")

        logger.info(f"成功从 {os.path.basename(filepath)} 提取了 {len(theorems)} 个测试定理")
        return theorems

    @staticmethod
    def _extract_type_expr(formal_statement: str) -> str:
        """
        从 formal_statement 提取 Pantograph goal.start 所需的类型表达式。

        formal_statement 格式: "theorem NAME BINDERS : TYPE := by"
        提取规则:
            1. 去掉 "theorem NAME" 前缀和 ":= by" / ":=" 后缀
            2. 跳过所有括号包裹的 binder 组 ((...), {...}, [...])
            3. 下一个 ':' 即为 binder 与结论的分隔符
            4. 组合为 "∀ BINDERS, CONCLUSION"

        参数:
            formal_statement (str): 完整的 Lean 4 定理声明

        返回:
            str: 可传给 goal.start 的类型表达式
        """
        body = formal_statement.strip()

        # 1. 去掉 "theorem NAME" 前缀
        if body.startswith("theorem "):
            body = body[len("theorem "):]
            # 跳过定理名 (字母/数字/下划线/撇号)
            i = 0
            while i < len(body) and (body[i].isalnum() or body[i] in "_'"):
                i += 1
            body = body[i:]

        # 2. 去掉 ":= by" 或 ":=" 后缀 (在 depth 0 找到 :=)
        depth = 0
        assign_pos = -1
        for i in range(len(body) - 1):
            ch = body[i]
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            elif ch == ':' and i + 1 < len(body) and body[i + 1] == '=' and depth == 0:
                assign_pos = i
                break
        if assign_pos >= 0:
            body = body[:assign_pos]

        # 3. 标准化空白
        body = ' '.join(body.split())

        # 4. 跳过所有 binder 组，找到分隔 ':'
        #    binder 组以 ( [ { 开头，在 depth 0
        i = 0
        n = len(body)

        # 跳过前导空格
        while i < n and body[i] == ' ':
            i += 1

        # 如果直接以 ':' 开头 — 无 binder
        if i < n and body[i] == ':':
            return body[i + 1:].strip()

        # 逐个跳过 binder 组
        while i < n:
            if body[i] in '([{':
                # 跳过整个 binder 组 (匹配括号)
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if body[i] in '([{':
                        depth += 1
                    elif body[i] in ')]}':
                        depth -= 1
                    i += 1
                # 跳过 binder 组后的空格
                while i < n and body[i] == ' ':
                    i += 1
                # 如果下一个是 ':' 且不是 ':=' — 找到分隔符
                if i < n and body[i] == ':' and (i + 1 >= n or body[i + 1] != '='):
                    args = body[:i].strip()
                    conclusion = body[i + 1:].strip()
                    if args:
                        return f"∀ {args}, {conclusion}"
                    return conclusion
                # 否则继续检查下一个 binder 组
            else:
                # 不以括号开头 — 不可能再有 binder，找 ':'
                while i < n:
                    if body[i] == ':' and (i + 1 >= n or body[i + 1] != '='):
                        args = body[:i].strip()
                        conclusion = body[i + 1:].strip()
                        if args:
                            return f"∀ {args}, {conclusion}"
                        return conclusion
                    i += 1
                break

        # 如果找不到分隔符，返回整个 body
        logger.warning(f"未能找到类型分隔符: {body[:80]}...")
        return body

    @staticmethod
    def load_lean_file(filepath: str) -> List[Dict[str, str]]:
        """
        从 Lean 4 源码文件中解析提取 Theorem 声明 (旧方法，建议用 load_jsonl)
        """
        if not os.path.exists(filepath):
            logger.error(f"找不到数据集文件: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'theorem\s+([a-zA-Z0-9_]+)\s*(.*?)\s*:\s*(.*?)\s*:='
        matches = re.findall(pattern, content, re.DOTALL)

        theorems = []
        for match in matches:
            name, args, statement = match
            args = args.strip()
            statement = statement.strip()

            if args:
                expr = f"∀ {args}, {statement}"
            else:
                expr = statement

            expr = " ".join(expr.split())

            theorems.append({
                "name": name,
                "expr": expr
            })

        logger.info(f"成功从 {os.path.basename(filepath)} 提取了 {len(theorems)} 个定理")
        return theorems


class BenchmarkRunner:
    """基线测试运行器"""

    def __init__(self,
                 generator: ThoughtCoSTacticGenerator,
                 verifier: PantographVerifier,
                 max_steps: int = 20):
        self.generator = generator
        self.verifier = verifier
        self.max_steps = max_steps
        self.results = []
        self._loaded_header = None  # 缓存已加载的 header，避免重复加载

    def _ensure_header(self, header: str):
        """
        确保 Pantograph 环境已加载指定 header (open/set_option 等)

        仅在 header 发生变化时才重新加载，减少不必要的 frontend.process 调用。

        参数:
            header (str): Lean 4 头部代码
        """
        if not header or header == self._loaded_header:
            return
        if self.verifier.load_header(header):
            self._loaded_header = header
        else:
            logger.warning("Header 加载失败，某些符号/记号可能无法解析")

    def evaluate_problem(self, theorem: Dict[str, str]) -> Dict[str, Any]:
        """
        评测单个定理 (无回溯贪婪搜索)
        """
        name = theorem["name"]
        expr = theorem["expr"]
        header = theorem.get("header", "")

        record = {
            "name": name,
            "expr": expr,
            "success": False,
            "error_msg": None,
            "steps": [],
            "reason": "Unknown"
        }

        # 0. 加载 header 环境 (open 命名空间、set_option 等)
        self._ensure_header(header)

        # 1. 在 Pantograph 中启动目标
        state_id = self.verifier.goal_start(expr)
        if state_id is None:
            record["error_msg"] = "Pantograph 无法解析或启动该定理表达式"
            record["reason"] = "Init Error"
            return record
            
        current_state_id = state_id
        
        # 2. 单路步进搜索
        for step in range(self.max_steps):
            # 获取当前目标的文本表示
            goals = self.verifier.goal_print(current_state_id)
            if goals is None:
                record["error_msg"] = "无法获取目标状态"
                record["reason"] = "Pantograph Error"
                break
                
            if len(goals) == 0:
                record["success"] = True
                record["reason"] = "No Goals"
                break
                
            # 将多目标拼接作为输入状态送给模型
            state_str = "\n\n".join(goals)
            
            # 模型生成下一步 (temperature=0.0 保证贪婪解码，最稳定的一步)
            responses = self.generator.generate_step(state_str, temperature=0.0, num_samples=1)
            tactic = responses[0]['tactic']
            thought = responses[0]['thought']
            
            step_record = {
                "step": step + 1,
                "state": state_str,
                "thought": thought,
                "tactic": tactic
            }
            record["steps"].append(step_record)
            
            # 如果模型吐出空白或 sorry，直接判定失败
            if not tactic or tactic.strip() == "sorry":
                record["error_msg"] = "模型输出了空白策略或 sorry"
                record["reason"] = "Empty Tactic"
                break
                
            # 提交给 Lean 4 环境验证
            res = self.verifier.goal_tactic(current_state_id, tactic)
            
            if res["is_success"]:
                record["success"] = True
                record["reason"] = "No Goals"
                break
            elif res["is_valid"]:
                # 策略合法，更新到新状态节点
                current_state_id = res["new_state_id"]
            else:
                # 策略非法 (如类型不匹配、找不到标识符) -> 基线测试直接失败，不启用 ETR 纠错
                record["error_msg"] = res["error_msg"]
                record["reason"] = "Tactic Error"
                break
        else:
            # 循环正常结束且未 break，说明达到了最大步数
            record["reason"] = "Max Steps Reached"
            
        # 重置环境，准备下一题
        self.verifier.reset()
        return record

    def run_evaluation(self, test_data: List[Dict[str, str]], output_file: str):
        """运行完整评测集"""
        logger.info(f"🚀 开始评测，共 {len(test_data)} 题")
        
        success_count = 0
        total = len(test_data)
        
        pbar = tqdm(test_data, desc="评测进度")
        for idx, theorem in enumerate(pbar):
            # 如果有遇到导致环境崩溃的严重错误，重启 Pantograph
            if not self.verifier.is_alive:
                self.verifier.restart()
                self._loaded_header = None  # 重启后需要重新加载 header
                
            result = self.evaluate_problem(theorem)
            self.results.append(result)
            
            if result["success"]:
                success_count += 1
                
            # 实时更新 tqdm 面板上的 Pass@1 分数
            current_pass_rate = (success_count / (idx + 1)) * 100
            pbar.set_postfix({"Pass@1": f"{current_pass_rate:.1f}%", "Success": success_count})
            
            # 每评测一题就将结果落盘，防止中途断电白跑
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
                
        final_pass_rate = (success_count / total) * 100
        logger.info(f"✅ 评测完成! 最终 Pass@1 准确率: {final_pass_rate:.2f}% ({success_count}/{total})")
        logger.info(f"结果已保存至: {output_file}")


class SearchBenchmarkRunner(BenchmarkRunner):
    """带 MCTS/RMaxTS 搜索的评测运行器"""

    def __init__(self,
                 generator: ThoughtCoSTacticGenerator,
                 verifier: PantographVerifier,
                 max_steps: int = 20,
                 max_iterations: int = 50):
        super().__init__(generator, verifier, max_steps)
        self.max_iterations = max_iterations
        from src.search.rmax_ts import RMaxTS
        # 初始化搜索算法
        self.search_algo = RMaxTS(
            verifier=verifier,
            generator=generator,
            max_iterations=max_iterations,
            max_depth=max_steps
        )

    def evaluate_problem(self, theorem: Dict[str, str]) -> Dict[str, Any]:
        """对于带有搜索的评测单题逻辑"""
        name = theorem["name"]
        expr = theorem["expr"]
        header = theorem.get("header", "")

        record = {
            "name": name,
            "expr": expr,
            "success": False,
            "error_msg": None,
            "steps": [],
            "reason": "Unknown"
        }

        self._ensure_header(header)
        state_id = self.verifier.goal_start(expr)
        if state_id is None:
            record["error_msg"] = "Pantograph 无法解析或启动该定理表达式"
            record["reason"] = "Init Error"
            return record
            
        goals = self.verifier.goal_print(state_id)
        if goals is None:
            record["error_msg"] = "无法获取目标状态"
            record["reason"] = "Pantograph Error"
            self.verifier.reset()
            return record
            
        if len(goals) == 0:
            record["success"] = True
            record["reason"] = "No Goals"
            self.verifier.reset()
            return record
            
        state_str = "\n\n".join(goals)
        
        # 执行搜索
        path = self.search_algo.search(initial_state_id=state_id, initial_state_str=state_str)
        if path:
            record["success"] = True
            record["reason"] = "Proof Found"
            for i, tac in enumerate(path):
                record["steps"].append({
                    "step": i + 1,
                    "tactic": tac
                })
        else:
            record["success"] = False
            record["reason"] = "Search Failed"

        self.verifier.reset()
        return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTAP Benchmark Runner")
    parser.add_argument("--search", action="store_true", help="启用 MCTS/RMaxTS 搜索模式")
    parser.add_argument("--max_iters", type=int, default=50, help="搜索模式下的最大迭代次数")
    parser.add_argument("--max_steps", type=int, default=15, help="证明最大步数/搜索深度")
    args = parser.parse_args()

    # 路径配置
    base_model_path = "/root/autodl-tmp/models/DeepSeek-Prover-V2-7B"
    lora_path = "checkpoints/sft_phase1/final/"
    project_path = "/root/autodl-tmp/RTAP"

    # MiniF2F-v2 JSONL 数据集 (v2c = 竞赛干净版)
    test_file_path = "benchmarks/minif2f_v2/datasets/miniF2F_v2c.jsonl"
    
    if args.search:
        output_file = "benchmarks/search_results.json"
    else:
        output_file = "benchmarks/pass_at_1_results.json"

    # 1. 解析数据集
    logger.info("正在加载 MiniF2F-v2 测试集...")
    test_data = MiniF2FLoader.load_jsonl(test_file_path, split="test")
    if not test_data:
        # 如果没有 test split，尝试加载 valid split
        logger.warning("未找到 test split，尝试加载 valid split...")
        test_data = MiniF2FLoader.load_jsonl(test_file_path, split="valid")
    if not test_data:
        # 加载全部
        logger.warning("未找到指定 split，加载全部数据...")
        test_data = MiniF2FLoader.load_jsonl(test_file_path, split="")
    if not test_data:
        logger.error(f"无法加载测试集 {test_file_path}，请检查路径！")
        return

    # 为了快速验证流程，先切片测试前 10 题
    test_data = test_data[:10]

    # 2. 初始化环境与模型
    verifier = PantographVerifier(project_path=project_path, timeout=60.0)
    generator = ThoughtCoSTacticGenerator(model_path=base_model_path, lora_path=lora_path)

    logger.info("正在将模型载入显存...")
    generator.load_model()

    # 3. 执行测试
    with verifier:  # 使用 with 语句自动管理 Pantograph 的启动和关闭
        if args.search:
            logger.info(f"启用 MCTS/RMaxTS 搜索评测模式 (max_iters={args.max_iters}, max_depth={args.max_steps})")
            runner = SearchBenchmarkRunner(generator=generator, verifier=verifier, max_steps=args.max_steps, max_iterations=args.max_iters)
        else:
            logger.info("使用单路贪婪解码 (Pass@1) 基线评测")
            runner = BenchmarkRunner(generator=generator, verifier=verifier, max_steps=args.max_steps)
            
        runner.run_evaluation(test_data, output_file)

if __name__ == "__main__":
    main()