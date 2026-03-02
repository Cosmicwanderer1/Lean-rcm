"""
Thought-CoS-Tactic 生成模型
@author ygw
更新日期: 2026-02-27

基于 DeepSeek-Prover-V2-7B 和 LoRA 权重，用于在 MCTS 搜索树中节点展开时
生成非形式化思考 (Thought)、中间状态预估 (CoS) 以及具体的动作指令 (Tactic)。

核心能力:
    - 加载基座模型 + LoRA adapter，merge_and_unload 后高效推理
    - 根据 Lean 4 证明状态生成 Thought + Tactic (对齐 SFT Phase 1 训练格式)
    - 支持多样本采样 (num_samples > 1，用于 Pass@k 评测)
    - 支持错误修正推理 (ETR/ESR)
    - 支持批量推理 (评测场景)
"""

import os
import re
import gc
import json
import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger("generator")


class ThoughtCoSTacticGenerator:
    """
    Thought-CoS-Tactic 生成模型

    核心职责:
        - 加载 DeepSeek-Prover-V2-7B 基座 + LoRA adapter
        - 根据 Lean 4 证明状态生成 Thought + Tactic
        - 支持 batch 推理和多样本采样 (Pass@k 评测)
        - 支持错误修正推理 (ETR/ESR)

    参数:
        model_path (str): 基础模型路径 (本地路径)
        lora_path (str): 微调后 LoRA 权重的路径
        device (str): 运行设备
        system_prompt (str): 自定义系统提示词
    """

    # 默认系统提示词 — 与 training_sft.yaml 中 phase1_system 完全对齐
    DEFAULT_SYSTEM_PROMPT = (
        "You are a Lean 4 theorem proving assistant. Given a proof state, "
        "first provide a brief reasoning (Thought) explaining your strategy, "
        "then output the exact Lean 4 tactic to apply.\n"
        "Rules:\n"
        "- Thought: 1-3 sentences explaining the reasoning.\n"
        "- Tactic: exactly one Lean 4 tactic, no code fences.\n"
        "- Never use sorry or admit."
    )

    # 错误修正系统提示词后缀 — 与 sft_trainer.py ErrorCorrectionDataset 对齐
    EC_SYSTEM_SUFFIX = (
        "\nAdditionally, you can fix incorrect tactics. "
        "When given a proof state with an erroneous tactic and its error message, "
        "provide the corrected tactic."
    )

    def __init__(self,
                 model_path: str = "/root/autodl-tmp/models/DeepSeek-Prover-V2-7B",
                 lora_path: str = "checkpoints/sft_phase1/final",
                 device: str = "cuda",
                 system_prompt: Optional[str] = None):
        """
        初始化生成模型

        参数:
            model_path (str): 基础模型路径 (本地路径或 HuggingFace ID)
            lora_path (str): LoRA 权重路径。为 None 则只加载基础模型。
            device (str): 运行设备，默认 "cuda"
            system_prompt (str): 自定义系统提示词，None 时使用默认
        """
        self.base_model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        加载模型与 LoRA 权重，并为推理进行优化

        流程:
            1. 加载 Tokenizer，设置 pad_token
            2. 以 bfloat16 加载基座模型 (~14GB 显存)
            3. 如有 LoRA 权重，加载并合并到基座模型 (merge_and_unload)
            4. 设置为 eval 模式
        """
        logger.info(f"正在加载 Tokenizer: {self.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )
        # DeepSeek 模型需要设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"正在加载 Base Model (bfloat16): {self.base_model_path}")
        # bfloat16: 7B 模型约占 14GB 显存，适合单张 3090/4090
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )

        if self.lora_path and os.path.exists(self.lora_path):
            logger.info(f"正在加载并合并 LoRA 权重: {self.lora_path}")
            peft_model = PeftModel.from_pretrained(base_model, self.lora_path)
            # merge_and_unload: 将 LoRA 分支权重融合回基座，消除推理时的额外计算开销
            self.model = peft_model.merge_and_unload()
            logger.info("LoRA 权重合并完成")
        else:
            logger.warning("未找到 LoRA 权重或未指定，仅使用基础模型进行推理。")
            self.model = base_model

        self.model.eval()
        logger.info("模型加载完毕，已进入 eval 模式")

    def is_loaded(self) -> bool:
        """
        检查模型是否已加载

        返回:
            bool: 模型是否就绪
        """
        return self.model is not None and self.tokenizer is not None

    # ================================================================
    # Prompt 构建 (严格对齐 SFT 训练格式)
    # ================================================================

    def _build_prompt(self, state: str,
                      error_tactic: Optional[str] = None,
                      error_message: Optional[str] = None) -> str:
        """
        构建供大模型推理的 Prompt，使用 Chat 模板对齐 SFT 训练时的格式。

        关键对齐点:
            - system: 与 training_sft.yaml 中 phase1_system 一致
            - user: 训练时为纯 state_before 文本 (非包裹格式)
            - 错误修正模式: user 包含 [Error Tactic] 和 [Error Message]

        参数:
            state (str): 当前 Lean 4 证明状态
            error_tactic (str): 错误策略 (仅 ETR/ESR 模式)
            error_message (str): 错误信息 (仅 ETR/ESR 模式)

        返回:
            str: 格式化后的 prompt 字符串
        """
        # 确定系统提示词
        if error_tactic is not None:
            system_content = self.system_prompt + self.EC_SYSTEM_SUFFIX
        else:
            system_content = self.system_prompt

        # 构建用户消息 — 与 sft_trainer.py 中的格式严格一致
        if error_tactic is not None:
            # 错误修正模式: 与 ErrorCorrectionDataset._load_and_convert() 对齐
            user_content = (
                f"{state}\n\n"
                f"[Error Tactic] {error_tactic}\n"
                f"[Error Message] {error_message or 'unknown error'}"
            )
        else:
            # 标准模式: 与 Phase1Dataset._load_and_convert() 对齐
            # 训练时 user content 就是纯 state_before
            user_content = state

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # 使用模型自带的 chat template，确保与训练时格式一致
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    # ================================================================
    # 核心推理: 单步生成
    # ================================================================

    def generate_step(self, state: str,
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      num_samples: int = 1,
                      top_p: float = 0.95,
                      repetition_penalty: float = 1.2) -> List[Dict[str, str]]:
        """
        给定证明状态，生成下一步的 Thought 和 Tactic

        参数:
            state (str): 当前 Lean 4 证明状态
            max_new_tokens (int): 最大生成 token 数
            temperature (float): 采样温度 (>0 多样性采样, 0 贪心解码)
            num_samples (int): 候选策略数 (Pass@k 评测时设为 k)
            top_p (float): nucleus sampling 概率阈值
            repetition_penalty (float): 重复惩罚系数 (>1.0 抑制重复，1.0 不惩罚)

        返回:
            List[Dict[str, str]]: 候选策略列表，每个元素包含:
                - raw_output: 模型原始输出
                - thought: 解析出的推理过程
                - tactic: 解析出的 Lean 4 策略
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model()。")

        prompt = self._build_prompt(state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # 构建生成参数 — 始终 num_return_sequences=1，通过外层循环实现多样本
        # 原因: num_return_sequences=N 会同时分配 N 份 KV cache，
        #        在 24GB 4090 上 N>=4 时极易 OOM (7B bf16 模型本身占 ~14GB)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        results = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            thought, tactic = self._parse_output(generated_text)
            results.append({
                "raw_output": generated_text,
                "thought": thought,
                "tactic": tactic
            })

            # 及时释放本轮输出张量，防止显存累积
            del outputs
            torch.cuda.empty_cache()

        return results

    # ================================================================
    # 错误修正推理 (ETR/ESR)
    # ================================================================

    def generate_correction(self, state: str,
                            error_tactic: str,
                            error_message: str,
                            max_new_tokens: int = 512,
                            temperature: float = 0.3,
                            num_samples: int = 1,
                            repetition_penalty: float = 1.2) -> List[Dict[str, str]]:
        """
        错误修正推理: 给定出错的状态、错误策略和错误信息，生成修正后的策略

        参数:
            state (str): 当前 Lean 4 证明状态
            error_tactic (str): 导致错误的策略
            error_message (str): Lean 4 编译器返回的错误信息
            max_new_tokens (int): 最大生成 token 数
            temperature (float): 采样温度 (修正场景偏低以减少发散)
            num_samples (int): 候选修正策略数
            repetition_penalty (float): 重复惩罚系数

        返回:
            List[Dict[str, str]]: 修正策略列表，格式同 generate_step
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载。请先调用 load_model()。")

        prompt = self._build_prompt(
            state,
            error_tactic=error_tactic,
            error_message=error_message
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False

        results = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            thought, tactic = self._parse_output(generated_text)
            results.append({
                "raw_output": generated_text,
                "thought": thought,
                "tactic": tactic
            })

            del outputs
            torch.cuda.empty_cache()

        return results

    # ================================================================
    # 批量推理 (评测场景)
    # ================================================================

    def batch_generate(self, states: List[str],
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       num_samples: int = 1) -> List[List[Dict[str, str]]]:
        """
        批量生成策略，适用于评测场景。

        逐条生成以避免 padding 导致的输出质量下降。
        当 num_samples > 1 时，每个 state 生成 num_samples 个候选。

        参数:
            states (List[str]): 证明状态列表
            max_new_tokens (int): 最大生成 token 数
            temperature (float): 采样温度
            num_samples (int): 每个状态的候选策略数

        返回:
            List[List[Dict]]: 外层对应每个 state，内层对应 num_samples 个候选
        """
        all_results = []
        for idx, state in enumerate(states):
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"批量推理进度: {idx + 1}/{len(states)}")
            results = self.generate_step(
                state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_samples=num_samples
            )
            all_results.append(results)
        logger.info(f"批量推理完成: 共 {len(states)} 个状态, "
                     f"每个 {num_samples} 个候选")
        return all_results

    # ================================================================
    # 输出解析
    # ================================================================

    def _parse_output(self, text: str) -> Tuple[str, str]:
        """
        解析模型输出，提取 Thought 和 Tactic。
        高鲁棒性正则解析器，兼容 SFT 训练格式及模型可能产生的多种变体。

        支持的格式 (按优先级):
            1. [Thought] xxx \n [Tactic] yyy   — 标准 SFT 训练格式
            2. [Tactic] yyy                    — 无 Thought 的训练格式
            3. Thought: xxx \n Tactic: yyy     — 冒号分隔变体
            4. ### Thought \n xxx \n ### Tactic — Markdown 格式变体
            5. <thought>xxx</thought>          — XML 标签变体
            6. 纯文本                           — 兜底：第一个有效行作为 tactic

        参数:
            text (str): 模型原始输出文本

        返回:
            Tuple[str, str]: (thought, tactic) 二元组
        """
        if not text or not text.strip():
            return "", ""

        thought = ""
        tactic = text.strip()

        # === 策略 1: 标准 SFT 训练格式 [Thought] ... [Tactic] ... ===
        bracket_match = re.search(
            r'\[Thought\]\s*(.*?)\s*\[Tactic\]\s*(.*)',
            text, re.DOTALL
        )
        if bracket_match:
            thought = bracket_match.group(1).strip()
            tactic = bracket_match.group(2).strip()
            return thought, self._clean_tactic(tactic)

        # === 策略 1b: 仅 [Tactic] 标记 (模型跳过了 Thought) ===
        tactic_only_match = re.search(r'\[Tactic\]\s*(.*)', text, re.DOTALL)
        if tactic_only_match:
            tactic = tactic_only_match.group(1).strip()
            return thought, self._clean_tactic(tactic)

        # === 策略 1c: 模型混淆输出 [Error Tactic] xxx — 提取 xxx 作为 tactic ===
        # 模型在采样模式下可能将 ETR 格式标签当前缀输出
        error_tactic_match = re.search(r'\[Error Tactic\]\s*(.*)', text, re.DOTALL)
        if error_tactic_match:
            raw_tactic = error_tactic_match.group(1).strip()
            if raw_tactic:
                # 取第一个看起来像 tactic 的片段 (到分号或换行)
                first_part = re.split(r'[;\n]', raw_tactic)[0].strip()
                if first_part and first_part.lower() not in ('sorry', 'admit'):
                    return "", self._clean_tactic(first_part)

        # === 策略 2: 冒号分隔格式 Thought: ... Tactic: ... ===
        colon_match = re.search(
            r'(?:Thought|Reasoning)\s*:\s*(.*?)\s*(?:Tactic|Action)\s*:\s*(.*)',
            text, re.DOTALL | re.IGNORECASE
        )
        if colon_match:
            thought = colon_match.group(1).strip()
            tactic = colon_match.group(2).strip()
            return thought, self._clean_tactic(tactic)

        # === 策略 3: Markdown 标题格式 ### Thought ... ### Tactic ... ===
        md_match = re.search(
            r'#{1,3}\s*(?:Thought|Reasoning)\s*\n(.*?)'
            r'#{1,3}\s*(?:Tactic|Action)\s*\n(.*)',
            text, re.DOTALL | re.IGNORECASE
        )
        if md_match:
            thought = md_match.group(1).strip()
            tactic = md_match.group(2).strip()
            return thought, self._clean_tactic(tactic)

        # === 策略 4: XML 标签格式 <thought>...</thought> <tactic>...</tactic> ===
        xml_thought = re.search(
            r'<thought>(.*?)</thought>', text, re.DOTALL | re.IGNORECASE
        )
        xml_tactic = re.search(
            r'<tactic>(.*?)</tactic>', text, re.DOTALL | re.IGNORECASE
        )
        if xml_tactic:
            thought = xml_thought.group(1).strip() if xml_thought else ""
            tactic = xml_tactic.group(1).strip()
            return thought, self._clean_tactic(tactic)

        # === 策略 5: 兜底 — 取第一个非空、非格式标签行作为 tactic ===
        # 跳过模型输出中的格式标签行 (如 [Thought], [Tactic], Thought:, etc.)
        format_label_pattern = re.compile(
            r'^\[?(Thought|Tactic|Reasoning|Action|Error Tactic|Error Message)\]?\s*:?\s*$',
            re.IGNORECASE
        )
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        for line in lines:
            if format_label_pattern.match(line):
                continue
            # 跳过以冒号开头的残留 (如 ": lia")
            if line.startswith(':'):
                line = line[1:].strip()
            if line:
                tactic = line
                break

        return thought, self._clean_tactic(tactic)

    @staticmethod
    def _clean_tactic(tactic: str) -> str:
        """
        清理 tactic 字符串: 去除代码围栏、多余换行、trailing 注释、重复项等

        Lean 4 tactic 可以跨多行 (如 rw [a, b,\\n  c, d])，因此不能
        简单地取第一行。采用括号感知的合并策略:
            1. 将所有行合并为单行
            2. 找到第一个括号完全闭合的位置作为 tactic 结束点
            3. 去除 trailing 文本 (模型可能在 tactic 后追加解释)
            4. 检测并去除方括号列表中的重复项 (模型退化生成)

        参数:
            tactic (str): 原始 tactic 字符串 (可能多行)

        返回:
            str: 清理后的 tactic (单行、无围栏、无注释、无重复)
        """
        if not tactic:
            return ""

        # 去除 Markdown 代码围栏 ```lean4 ... ```
        tactic = re.sub(r'^```(?:lean4?|lean)?\s*', '', tactic)
        tactic = re.sub(r'\s*```\s*$', '', tactic)

        # 将所有行合并为单行 (保留空格分隔)
        lines = [line.strip() for line in tactic.split('\n') if line.strip()]
        if not lines:
            return ""
        full_text = ' '.join(lines)

        # 括号感知截断: 找到第一个括号完全闭合的点 (含最大长度保护)
        result = ThoughtCoSTacticGenerator._extract_balanced_tactic(full_text)

        # 去除方括号列表中的连续重复项 (模型退化生成防护)
        result = ThoughtCoSTacticGenerator._dedup_bracket_items(result)

        # 去除行尾 Lean 注释 (-- comment)
        result = re.sub(r'\s+--\s+.*$', '', result)

        # 拒绝 sorry / admit
        if result.lower().strip() in ('sorry', 'admit'):
            return ""

        return result.strip()

    @staticmethod
    def _extract_balanced_tactic(text: str, max_chars: int = 600) -> str:
        """
        从文本中提取第一个完整的 tactic (括号平衡)，含最大长度保护

        规则:
            - 如果 tactic 包含 ([{ 等开括号，追踪到对应闭括号为止
            - 如果没有括号，取到第一个句号或换行处
            - 处理 \"...\" 和 Lean 字符串字面量中的括号 (不计入深度)
            - 如果在 max_chars 内括号未闭合，在最后一个逗号处截断并强制闭合

        参数:
            text (str): 合并后的单行文本
            max_chars (int): 最大允许字符数，超过后强制截断闭合

        返回:
            str: 第一个完整 tactic
        """
        depth = 0
        max_depth = 0
        in_string = False
        i = 0
        n = len(text)
        # 记录括号栈，用于强制闭合
        bracket_stack = []
        # 记录最外层括号内最后一个逗号的位置
        last_comma_at_depth1 = -1

        while i < n:
            ch = text[i]

            # 跳过字符串字面量中的内容
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
                i += 1
                continue
            if in_string:
                i += 1
                continue

            if ch in '([{':
                depth += 1
                max_depth = max(max_depth, depth)
                bracket_stack.append(ch)
            elif ch in ')]}':
                depth -= 1
                if bracket_stack:
                    bracket_stack.pop()
                # 最外层括号闭合: tactic 结束
                if depth == 0 and max_depth > 0:
                    return text[:i + 1].strip()

            # 记录深度 1 处的逗号位置 (用于截断)
            if ch == ',' and depth == 1:
                last_comma_at_depth1 = i

            # 超过 max_chars 且仍在括号内: 强制截断
            if i >= max_chars and depth > 0:
                if last_comma_at_depth1 > 0:
                    # 在最后一个逗号处截断，补上对应的闭括号
                    truncated = text[:last_comma_at_depth1].rstrip()
                    # 补全所有未闭合的括号
                    close_map = {'(': ')', '[': ']', '{': '}'}
                    for open_br in reversed(bracket_stack):
                        truncated += close_map.get(open_br, '')
                    return truncated.strip()
                # 没有逗号，直接截断并闭合
                break

            i += 1

        # 如果没有括号，或括号未闭合，返回全部文本
        # (未闭合时让 Lean 报错，保留完整信息以便调试)
        return text.strip()

    @staticmethod
    def _dedup_bracket_items(tactic: str) -> str:
        """
        检测并去除 tactic 中方括号列表内的连续重复项

        典型退化输出: rw [a, b, ← c, ← c, ← c, ← c, ..., ← c]
        修复后:       rw [a, b, ← c]

        规则:
            - 找到 tactic 中所有 [...] 区域
            - 按逗号分割列表项
            - 连续出现相同项超过 2 次的，只保留前 2 次
            - 重新组装

        参数:
            tactic (str): 已提取的 tactic 字符串

        返回:
            str: 去重后的 tactic
        """
        # 匹配 rw/simp/erw/simp_rw 等后面的方括号列表
        bracket_pattern = re.compile(
            r'(\b(?:rw|erw|simp|simp_rw|norm_cast|push_cast|pull_cast)\s*)\[([^\]]*)\]'
        )

        def dedup_items(match):
            prefix = match.group(1)
            items_str = match.group(2)
            items = [item.strip() for item in items_str.split(',')]

            # 去除连续重复: 同一 item 连续出现 > 2 次的只保留 2 次
            deduped = []
            prev_item = None
            repeat_count = 0
            for item in items:
                if not item:
                    continue
                if item == prev_item:
                    repeat_count += 1
                    if repeat_count <= 2:
                        deduped.append(item)
                    # repeat_count > 2: 跳过
                else:
                    prev_item = item
                    repeat_count = 1
                    deduped.append(item)

            return prefix + '[' + ', '.join(deduped) + ']'

        return bracket_pattern.sub(dedup_items, tactic)

    # ================================================================
    # 配置加载辅助
    # ================================================================

    @classmethod
    def from_config(cls, config_path: str,
                    checkpoint_subdir: str = "final") -> "ThoughtCoSTacticGenerator":
        """
        从 YAML 配置文件创建 Generator 实例

        参数:
            config_path (str): YAML 配置文件路径 (如 configs/training_sft.yaml)
            checkpoint_subdir (str): checkpoint 子目录名

        返回:
            ThoughtCoSTacticGenerator: 已配置但未加载模型的实例
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        ckpt_cfg = config.get("checkpoint", {})
        prompt_cfg = config.get("prompts", {})

        # 模型路径: 优先本地路径
        local_path = model_cfg.get("local_model_path", "")
        model_path = local_path if local_path and os.path.isdir(local_path) \
            else model_cfg.get("base_model", "deepseek-ai/DeepSeek-Prover-V2-7B")

        # LoRA 路径
        output_dir = ckpt_cfg.get("output_dir", "checkpoints/sft_phase1")
        lora_path = os.path.join(output_dir, checkpoint_subdir)

        # 系统提示词
        system_prompt = prompt_cfg.get("phase1_system", None)

        return cls(
            model_path=model_path,
            lora_path=lora_path,
            system_prompt=system_prompt
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息摘要

        返回:
            Dict[str, Any]: 模型配置与状态信息
        """
        info = {
            "base_model_path": self.base_model_path,
            "lora_path": self.lora_path,
            "device": self.device,
            "is_loaded": self.is_loaded(),
        }
        if self.is_loaded():
            info["model_dtype"] = str(next(self.model.parameters()).dtype)
            info["vocab_size"] = self.tokenizer.vocab_size
            total_params = sum(p.numel() for p in self.model.parameters())
            info["total_params"] = f"{total_params / 1e9:.2f}B"
        return info
