"""
Thought 反向翻译模块
@author ygw
更新日期: 2026-02-11

W7 核心模块：解决 Mathlib 缺乏思维注释的问题。
使用 Teacher Model (DeepSeek-V3 / GPT-4o) 对每个状态转换生成自然语言 Thought。
Prompt 策略: 输入 (S_pre, Tactic, S_post)，要求模型解释推理逻辑。
支持异步并发调用，大幅提升吞吐量。

技术产出:
- 自然语言 Thought（如"利用归纳法拆解问题..."）
- (Theorem, Thought, CoS, Tactic) 四元组数据集
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utils import (
    load_yaml, save_jsonl, load_jsonl, ensure_dir,
    setup_logging, ProgressTracker, get_timestamp,
    batch_iter, retry_with_backoff
)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """You are a Lean4 proof reasoning annotator. Given one proof step with its before/after states and the tactic used, explain the reasoning behind this step.

[Proof Step]
S_pre: {state_pre}
Tactic: {tactic}
S_post: {state_post}

[Rules]
Write 1-2 sentences explaining the reasoning. You MUST:
- State which specific goal, hypothesis, or expression the tactic targets
- Describe how the proof state changes (what is eliminated, introduced, or rewritten)
- NEVER use vague phrases: "obviously", "clearly", "trivially", "it is easy to see", "straightforward"
- Do NOT restate the problem; only describe the reasoning
- Output the reasoning text directly with no numbering, headings, or prefixes"""


# 基于策略类型的模板 Thought（API 不可用时的回退方案）
TACTIC_THOUGHT_TEMPLATES = {
    "intro": "通过引入假设变量，将全称量词消去，简化证明目标。使用 {tactic} 将前提引入上下文。",
    "intros": "批量引入多个假设变量到上下文中，消去目标中的全称量词和蕴含前提。",
    "apply": "识别到当前目标可以通过已知引理或假设直接推导。使用 {tactic} 将目标归约为子目标。",
    "rw": "观察到目标中存在可以改写的表达式。使用 {tactic} 进行等式改写，简化目标形式。",
    "simp": "当前目标可以通过自动化简策略解决。使用 {tactic} 调用简化引理库进行化简。",
    "exact": "当前目标与上下文中的某个项完全匹配。使用 {tactic} 直接提供证明项。",
    "cases": "需要对某个归纳类型进行分情况讨论。使用 {tactic} 将目标拆分为多个子情况。",
    "induction": "识别到需要使用数学归纳法。使用 {tactic} 建立基础情况和归纳步骤。",
    "have": "引入一个中间引理来辅助证明。使用 {tactic} 建立局部假设，分步推进证明。",
    "constructor": "目标是一个存在性命题或合取命题。使用 {tactic} 构造证据或分别证明各部分。",
    "calc": "使用计算链进行逐步推导。通过 {tactic} 建立等式或不等式的传递链。",
    "obtain": "从存在性假设中提取见证值。使用 {tactic} 解构存在量词，获取具体的值和性质。",
    "refine": "提供目标的部分证明项，留下未完成的子目标。使用 {tactic} 逐步细化证明结构。",
    "ring": "目标是一个环上的等式。使用 {tactic} 自动验证多项式等式。",
    "omega": "目标涉及自然数或整数的线性算术。使用 {tactic} 自动求解线性不等式系统。",
    "norm_num": "目标涉及具体数值计算。使用 {tactic} 自动化数值规范化和验证。",
    "linarith": "目标可以通过线性算术推导得出。使用 {tactic} 从上下文中的线性不等式推导结论。",
    "contradiction": "上下文中存在矛盾的假设。使用 {tactic} 从矛盾中推导出任意结论。",
    "split": "目标是一个合取命题。使用 {tactic} 将其拆分为两个独立的子目标分别证明。",
    "left": "目标是一个析取命题，选择证明左侧分支。",
    "right": "目标是一个析取命题，选择证明右侧分支。",
}


class TeacherModelClient:
    """
    Teacher Model API 客户端
    支持 DeepSeek-V3、GPT-4o 等 OpenAI 兼容 API。
    包含速率限制、重试、备用模型切换等功能。
    支持同步和异步两种调用方式。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 Teacher Model 客户端"""
        self.provider = config.get("provider", "deepseek-v3")
        self.model_name = config.get("model_name", "deepseek-chat")
        self.api_base = config.get("api_base", "https://api.deepseek.com/v1")
        self.api_key_env = config.get("api_key_env", "DEEPSEEK_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "")
        self.fallback_config = config.get("fallback", None)
        self._fallback_client: Optional['TeacherModelClient'] = None
        if not self.api_key:
            logger.warning(f"API Key 环境变量 {self.api_key_env} 未设置，将使用模板回退方案")

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 512, top_p: float = 0.95) -> Optional[str]:
        """同步调用 Teacher Model 生成文本（保留兼容）"""
        if not self.api_key:
            return self._template_fallback(prompt)
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            def do_request():
                resp = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers, json=payload, timeout=60
                )
                resp.raise_for_status()
                return resp.json()

            result = retry_with_backoff(do_request, max_retries=3, base_delay=2.0)
            choices = result.get("choices", [])
            if choices:
                return choices[0]["message"]["content"].strip()
            return None
        except ImportError:
            logger.warning("requests 库未安装，使用模板回退方案")
            return self._template_fallback(prompt)
        except Exception as e:
            logger.warning(f"API 调用失败: {e}")
            if self.fallback_config and not self._fallback_client:
                self._fallback_client = TeacherModelClient(self.fallback_config)
            if self._fallback_client:
                return self._fallback_client.generate(prompt, temperature, max_tokens, top_p)
            return self._template_fallback(prompt)

    async def async_generate(self, session: 'aiohttp.ClientSession',
                             prompt: str, temperature: float = 0.7,
                             max_tokens: int = 512, top_p: float = 0.95) -> Optional[str]:
        """异步调用 Teacher Model 生成文本"""
        if not self.api_key:
            return self._template_fallback(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        url = f"{self.api_base}/chat/completions"
        for attempt in range(4):
            try:
                async with session.post(url, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt * 2, 30)
                        logger.warning(f"API 429 限速，等待 {wait}s 后重试")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    result = await resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"].strip()
                    return None
            except asyncio.TimeoutError:
                logger.warning(f"API 超时 (attempt {attempt + 1}/4)")
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                logger.warning(f"API 异步调用失败 (attempt {attempt + 1}/4): {e}")
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        return None

    def _template_fallback(self, prompt: str) -> str:
        """模板回退方案：当 API 不可用时，生成基于规则的 Thought"""
        tactic = ""
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Tactic:"):
                tactic = stripped.split(":", 1)[-1].strip()
                break
        tactic_name = tactic.split()[0] if tactic else "unknown"
        template = TACTIC_THOUGHT_TEMPLATES.get(
            tactic_name,
            "在当前证明状态下，选择策略 {tactic} 来推进证明。该策略能够有效地转换目标状态。"
        )
        return template.format(tactic=tactic)


class ThoughtBacktranslator:
    """
    Thought 生成器，使用教师模型进行反向翻译。
    支持异步并发调用（aiohttp），大幅提升吞吐量。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 Thought 生成器"""
        self.config = config
        thought_config = config["thought_generation"]
        self.input_path = thought_config.get("input_path", "")
        self.output_path = thought_config.get("output_path", "")
        # 生成参数
        gen_config = thought_config.get("generation", {})
        self.temperature = gen_config.get("temperature", 0.7)
        self.max_tokens = gen_config.get("max_tokens", 512)
        self.top_p = gen_config.get("top_p", 0.95)
        # Prompt 模板
        self.prompt_template = thought_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        # 批处理参数
        batch_config = thought_config.get("batch", {})
        self.batch_size = batch_config.get("batch_size", 20)
        self.rate_limit_rpm = batch_config.get("rate_limit_rpm", 60)
        self.max_concurrent = batch_config.get("max_concurrent_requests", 20)
        # 质量过滤
        quality_config = thought_config.get("quality_filter", {})
        self.min_length = quality_config.get("min_length", 20)
        self.max_length = quality_config.get("max_length", 500)
        self.require_keywords = quality_config.get("require_keywords", [])
        self.forbidden_patterns = quality_config.get("forbidden_patterns", [])
        # 初始化 Teacher Model 客户端
        teacher_config = thought_config.get("teacher_model", {})
        self.client = TeacherModelClient(teacher_config)
        # 速率限制（同步模式用）
        self._request_interval = 60.0 / max(self.rate_limit_rpm, 1)
        self._last_request_time = 0.0
        # 统计
        self.stats = {
            "total_samples": 0, "processed": 0,
            "generated": 0, "filtered": 0, "errors": 0,
        }

    def run(self) -> str:
        """执行 Thought 生成（自动选择异步或同步模式）"""
        if HAS_AIOHTTP and self.client.api_key:
            print(f"[W7] 使用异步并发模式，并发数={self.max_concurrent}", flush=True)
            return asyncio.run(self._async_run())
        else:
            if not HAS_AIOHTTP:
                print("[W7] aiohttp 未安装，使用同步模式", flush=True)
            return self._sync_run()

    async def _async_run(self) -> str:
        """异步并发执行 Thought 生成"""
        logger.info("=" * 60)
        logger.info("W7 思维回溯 (Thought Back-translation) 开始 [异步并发模式]")
        logger.info("=" * 60)
        start_time = time.time()
        ensure_dir(self.output_path)

        input_file = os.path.join(self.input_path, "cos_flat.jsonl")
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return ""

        print(f"[W7] 加载数据: {input_file}", flush=True)
        samples = load_jsonl(input_file)
        self.stats["total_samples"] = len(samples)
        print(f"[W7] 加载完成: {len(samples)} 条样本", flush=True)

        output_file = os.path.join(self.output_path, "thought_dataset.jsonl")
        if os.path.exists(output_file):
            os.remove(output_file)

        semaphore = asyncio.Semaphore(self.max_concurrent)
        rps = self.rate_limit_rpm / 60.0
        interval = 1.0 / rps if rps > 0 else 0.5

        results = []
        processed = 0
        last_print_time = time.time()

        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 5)
        timeout = aiohttp.ClientTimeout(total=90)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def process_one(sample: Dict) -> Optional[Dict]:
                nonlocal processed
                async with semaphore:
                    prompt = self.prompt_template.format(
                        state_pre=sample.get("state_before", ""),
                        tactic=sample.get("tactic", ""),
                        state_post=sample.get("state_after", ""),
                    )
                    thought = await self.client.async_generate(
                        session, prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                    )
                    processed += 1
                    if thought and self._validate_thought(thought):
                        sample_copy = dict(sample)
                        sample_copy["thought"] = thought
                        self.stats["generated"] += 1
                        return sample_copy
                    elif thought:
                        self.stats["filtered"] += 1
                    else:
                        self.stats["errors"] += 1
                    return None

            # 分块处理，每块 max_concurrent * 3
            chunk_size = self.max_concurrent * 3
            for chunk_start in range(0, len(samples), chunk_size):
                chunk = samples[chunk_start:chunk_start + chunk_size]
                tasks = []
                for sample in chunk:
                    tasks.append(asyncio.create_task(process_one(sample)))
                    if interval > 0:
                        await asyncio.sleep(interval)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, Exception):
                        logger.warning(f"任务异常: {r}")
                        self.stats["errors"] += 1
                    elif r is not None:
                        results.append(r)

                self.stats["processed"] = processed

                if len(results) >= 500:
                    save_jsonl(results, output_file, mode='a')
                    results.clear()

                now = time.time()
                if now - last_print_time >= 10:
                    elapsed = now - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    remaining_h = ((len(samples) - processed) / speed / 3600) if speed > 0 else 0
                    print(
                        f"[W7] 进度: {processed}/{len(samples)} "
                        f"({processed/len(samples)*100:.1f}%) | "
                        f"生成: {self.stats['generated']} | "
                        f"过滤: {self.stats['filtered']} | "
                        f"错误: {self.stats['errors']} | "
                        f"速度: {speed:.1f}/s | "
                        f"剩余: {remaining_h:.1f}h",
                        flush=True,
                    )
                    last_print_time = now

        if results:
            save_jsonl(results, output_file, mode='a')

        elapsed = time.time() - start_time
        self.stats["processed"] = processed
        print(
            f"\n[W7] 完成! 总计: {processed}, "
            f"生成: {self.stats['generated']}, "
            f"过滤: {self.stats['filtered']}, "
            f"错误: {self.stats['errors']}, "
            f"耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)",
            flush=True,
        )
        self._save_stats(elapsed)
        return output_file

    def _sync_run(self) -> str:
        """同步执行 Thought 生成（回退方案）"""
        logger.info("=" * 60)
        logger.info("W7 思维回溯 (Thought Back-translation) 开始 [同步模式]")
        logger.info("=" * 60)
        start_time = time.time()
        ensure_dir(self.output_path)

        input_file = os.path.join(self.input_path, "cos_flat.jsonl")
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return ""

        print(f"[W7] 加载数据: {input_file}", flush=True)
        samples = load_jsonl(input_file)
        self.stats["total_samples"] = len(samples)
        print(f"[W7] 加载完成: {len(samples)} 条样本", flush=True)

        output_file = os.path.join(self.output_path, "thought_dataset.jsonl")
        tracker = ProgressTracker(total=len(samples), desc="Thought 生成", log_interval=50)

        results = []
        for batch in batch_iter(samples, self.batch_size):
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            for _ in batch:
                tracker.update(success=True)
            if len(results) >= 500:
                save_jsonl(results, output_file, mode='a')
                results.clear()

        if results:
            save_jsonl(results, output_file, mode='a')

        tracker.finish()
        elapsed = time.time() - start_time
        self._save_stats(elapsed)
        return output_file

    def _save_stats(self, elapsed: float):
        """保存统计信息"""
        logger.info("=" * 60)
        logger.info(f"W7 思维回溯完成 | 生成: {self.stats['generated']} | 过滤: {self.stats['filtered']} | 耗时: {elapsed:.1f}s")
        logger.info("=" * 60)
        stats_file = os.path.join(self.output_path, "thought_stats.json")
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        self.stats["timestamp"] = get_timestamp()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """处理一批样本（同步模式用）"""
        results = []
        for sample in batch:
            try:
                thought = self._generate_single_thought(sample)
                self.stats["processed"] += 1
                if thought and self._validate_thought(thought):
                    sample["thought"] = thought
                    results.append(sample)
                    self.stats["generated"] += 1
                else:
                    self.stats["filtered"] += 1
            except Exception as e:
                logger.debug(f"生成 Thought 失败: {e}")
                self.stats["errors"] += 1
        return results

    def _generate_single_thought(self, sample: Dict) -> Optional[str]:
        """为单个样本生成 Thought（同步模式用）"""
        now = time.time()
        elapsed_since_last = now - self._last_request_time
        if elapsed_since_last < self._request_interval:
            time.sleep(self._request_interval - elapsed_since_last)
        self._last_request_time = time.time()
        prompt = self.prompt_template.format(
            state_pre=sample.get("state_before", ""),
            tactic=sample.get("tactic", ""),
            state_post=sample.get("state_after", ""),
        )
        return self.client.generate(
            prompt=prompt, temperature=self.temperature,
            max_tokens=self.max_tokens, top_p=self.top_p,
        )

    def _validate_thought(self, thought: str) -> bool:
        """验证 Thought 质量"""
        if not thought or len(thought) < self.min_length or len(thought) > self.max_length:
            return False
        for pattern in self.forbidden_patterns:
            if pattern in thought:
                return False
        if self.require_keywords and not any(kw in thought for kw in self.require_keywords):
            return False
        return True

    # 兼容旧接口
    def generate_thought(self, state: str, tactic: str) -> Optional[str]:
        """为给定的状态和策略生成 Thought"""
        sample = {"state_before": state, "tactic": tactic, "state_after": ""}
        return self._generate_single_thought(sample)

    def batch_generate(self, cos_dataset: List[Dict]) -> List[Dict]:
        """批量生成 Thought"""
        return self._process_batch(cos_dataset)

    def validate_thought(self, thought: str) -> bool:
        """验证 Thought 质量"""
        return self._validate_thought(thought)


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="W7 思维回溯")
    parser.add_argument("--config", type=str, default="configs/data_pipeline.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/thought_backtrans.log", encoding="utf-8"),
        ],
    )
    print("[W7] 加载配置...", flush=True)
    config = load_yaml(args.config)
    print("[W7] 初始化 ThoughtBacktranslator...", flush=True)
    translator = ThoughtBacktranslator(config)
    print("[W7] 开始运行...", flush=True)
    output_file = translator.run()
    print(f"\nThought 生成完成！输出文件: {output_file}")


if __name__ == "__main__":
    main()
