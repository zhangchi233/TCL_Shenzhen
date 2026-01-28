#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoT 蒸馏流水线

两步流程:
1. Step 1: 给定问题生成 CoT 思考过程，然后根据问题和标准答案改写 CoT，
   用 </think> 分隔思考和答案，形成符合自然语言规范的思考步骤
2. Step 2: 对 CoT 和答案进行修订，去除幻觉，优化训练数据质量
"""

import asyncio
import json
import os
import re
import pandas as pd
from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import ServerError

# ==========================================
# 配置
# ==========================================
CONCURRENCY_LIMIT = 64  # 并发数
MAX_RETRIES = 3
RETRY_DELAY = 10


def parse_json(text):
    """解析 JSON 格式"""
    try:
        json_str = re.search(r'\{.*\}', text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        return {}


class CoTDistillationPipeline:
    """CoT 蒸馏流水线"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key).aio
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def _generate_with_retry(self, contents: list, max_retries: int = MAX_RETRIES):
        """带重试的生成请求"""
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=contents
                )
                return response.text
            except ServerError as e:
                last_exc = e
                msg = str(e).lower()
                if ("503" in msg or "overloaded" in msg) and attempt < max_retries:
                    print(f"⚠️ Server busy, retry {attempt}/{max_retries}...")
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                raise
            except Exception as e:
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown error in _generate_with_retry")
    
    # ==========================================
    # Step 1: 生成 CoT 并与答案融合
    # ==========================================
    
    def _build_cot_generation_prompt(self):
        """构建 CoT 生成和融合的 Prompt"""
        return """
        # 角色
        你是一位显示技术领域的专家教授，擅长将复杂问题分解为清晰的思考步骤。

        # 任务
        给定一个【问题】、【图片】和【标准答案】，你需要：
        1. **生成思考过程 (Chain of Thought)**: 模拟一个专业人员仅根据问题和图片，逐步分析推理得出答案的过程
        2. **融合答案**: 将思考过程和标准答案自然地融合在一起

        # 重要约束 ⚠️
        - 思考过程必须**只基于问题和图片**进行推理
        - **禁止**出现以下表述：
          - "根据文中..."、"文中提到..."、"上下文中..."
          - "根据提供的信息..."、"材料中显示..."
          - 任何暗示有额外文本材料的表述
        - 思考过程应该像是在**直接观察图片并思考问题**，而非阅读文档
        - 可以使用："从图中可以看到..."、"观察图片..."、"根据图示..."、"图中显示..."

        # 输出格式要求
        - 思考部分和答案部分用 `</think>` 分隔
        - 思考部分应该：
          - 从问题出发，明确要解决什么
          - 观察图片中的关键信息
          - 结合专业知识进行推理
          - 使用自然流畅的语言，避免机械化的"步骤1、步骤2"
        - 答案部分应该：
          - 基于标准答案进行适当润色
          - 确保与思考过程逻辑一致
          - 专业且详尽
          - 同样不能出现"文中"、"上下文"等表述

        # 输出 JSON 格式
        {
            "thinking_process": "<思考过程，只基于问题和图片的自然语言推理>",
            "final_answer": "<基于标准答案润色后的最终答案>",
            "combined_output": "<思考过程></think><最终答案>"
        }
        """
    
    async def generate_cot_and_merge(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: 生成 CoT 并与答案融合
        
        输入: item 包含 question, answer, image_path (可选), context (可选)
        输出: 添加 cot_thinking, cot_answer, cot_combined 字段
        """
        question = item.get("question", item.get("refined_question", ""))
        answer = item.get("answer", item.get("refined_answer", ""))
        context = item.get("caption_expanded", item.get("context", ""))
        
        if not question or not answer:
            return {**item, "cot_error": "Missing question or answer"}
        
        async with self.semaphore:
            try:
                system_prompt = self._build_cot_generation_prompt()
                
                user_content = f"""
                ### 问题
                {question}

                ### 标准答案 (仅供参考，用于确保答案方向正确)
                {answer}

                请仔细观察图片，然后生成思考过程并与答案融合。
                注意：思考过程必须只基于问题和图片，不要引用任何"文中"、"上下文"等表述。
                """
                # 准备内容
                contents = [system_prompt, user_content]
                
                # 如果有图片，添加图片
                image_path = item.get("image_path")
                if image_path and os.path.exists(image_path):
                    try:
                        contents.append(
                            types.Part.from_bytes(
                                data=open(image_path, "rb").read(),
                                mime_type="image/jpeg",
                            )
                        )
                    except Exception as e:
                        print(f"⚠️ [ID: {item.get('id')}] 图片加载失败: {e}")
                
                # 调用 API
                result_text = await self._generate_with_retry(contents)
                result_json = parse_json(result_text)
                
                # 构建 combined output
                thinking = result_json.get("thinking_process", "")
                final_ans = result_json.get("final_answer", answer)
                combined = result_json.get("combined_output", f"{thinking}</think>{final_ans}")
                
                # 确保格式正确
                if "</think>" not in combined:
                    combined = f"{thinking}</think>{final_ans}"
                
                print(f"✅ [ID: {item.get('id')}] Step 1 完成: CoT 生成")
                
                return {
                    **item,
                    "cot_thinking": thinking,
                    "cot_answer": final_ans,
                    "cot_combined": combined,
                    "step1_status": "success"
                }
                
            except Exception as e:
                print(f"❌ [ID: {item.get('id')}] Step 1 错误: {str(e)}")
                return {
                    **item,
                    "cot_error": str(e),
                    "step1_status": "error"
                }
    
    # ==========================================
    # Step 2: 修订 CoT 和答案，去除幻觉
    # ==========================================
    
    def _build_refinement_prompt(self):
        """构建修订和去幻觉的 Prompt"""
        return """
        # 角色
        你是一位严格的数据质量审核专家，专门负责检查和修订 AI 生成的训练数据。

        # 任务
        审核并修订给定的【思考过程 + 答案】组合，确保：
        1. **无幻觉**: 所有信息必须基于提供的图片和问题，不能凭空捏造
        2. **逻辑一致**: 思考过程必须逻辑自洽，与最终答案一致
        3. **专业准确**: 技术术语使用准确，符合显示技术领域规范
        4. **自然流畅**: 语言表达自然，适合作为高质量训练数据

        # 检查重点 ⚠️
        1. **幻觉检测**: 是否包含图片中不可见的具体数值、型号、结构等
        2. **禁止表述检测**: 是否出现以下违规表述（必须删除或改写）：
           - "根据文中..."、"文中提到..."、"上下文中..."
           - "根据提供的信息..."、"材料中显示..."、"文本中..."
           - 任何暗示有额外文本材料的表述
        3. **推理正确性**: 思考步骤是否能从图片和问题逻辑推导出答案
        4. **信息泄露**: 思考过程是否过早暴露答案
        5. **格式规范**: </think> 分隔符使用是否正确

        # 修订原则
        - 将"文中提到"改为"从图中可以观察到"或直接陈述
        - 将"上下文显示"改为"根据图示"或"观察图片可知"
        - 确保思考过程读起来像是在**直接分析图片和问题**

        # 输出 JSON 格式
        {
            "needs_revision": <bool>,
            "revision_reason": "<如果需要修订，说明原因>",
            "hallucination_detected": "<检测到的幻觉内容，如果有>",
            "forbidden_expressions_found": "<检测到的违规表述，如果有>",
            "refined_thinking": "<修订后的思考过程，不含违规表述>",
            "refined_answer": "<修订后的答案>",
            "refined_combined": "<修订后的完整输出，格式: 思考过程</think>答案>",
            "quality_score": <0.0-1.0 的质量评分>
        }
        """
    
    async def refine_cot_and_answer(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: 修订 CoT 和答案，去除幻觉
        
        输入: item 包含 cot_combined (来自 Step 1)
        输出: 添加修订后的字段
        """
        # 检查 Step 1 是否成功
        if item.get("step1_status") != "success":
            return {**item, "step2_status": "skipped", "skip_reason": "Step 1 failed"}
        
        cot_combined = item.get("cot_combined", "")
        question = item.get("question", item.get("refined_question", ""))
        original_answer = item.get("answer", item.get("refined_answer", ""))
        context = item.get("caption_expanded", item.get("context", ""))
        
        if not cot_combined:
            return {**item, "step2_status": "error", "step2_error": "No CoT combined output"}
        
        async with self.semaphore:
            try:
                system_prompt = self._build_refinement_prompt()
                
                user_content = f"""
                ### 原始问题
                {question}

                ### 原始标准答案
                {original_answer}

                ### 待审核的 CoT + 答案组合
                {cot_combined}

                请严格审核上述内容，特别注意：
                1. 检查是否有"文中"、"上下文"、"材料中"等违规表述
                2. 检查是否有图片中看不到的幻觉信息
                3. 确保思考过程只基于问题和图片进行推理
                """
                                # 准备内容
                contents = [system_prompt, user_content]
                
                # 如果有图片，添加图片
                image_path = item.get("image_path")
                if image_path and os.path.exists(image_path):
                    try:
                        contents.append(
                            types.Part.from_bytes(
                                data=open(image_path, "rb").read(),
                                mime_type="image/jpeg",
                            )
                        )
                    except Exception as e:
                        print(f"⚠️ [ID: {item.get('id')}] 图片加载失败: {e}")
                
                # 调用 API
                result_text = await self._generate_with_retry(contents)
                result_json = parse_json(result_text)
                
                # 提取结果
                needs_revision = result_json.get("needs_revision", False)
                refined_combined = result_json.get("refined_combined", cot_combined)
                
                # 确保格式正确
                if "</think>" not in refined_combined:
                    refined_thinking = result_json.get("refined_thinking", "")
                    refined_ans = result_json.get("refined_answer", original_answer)
                    refined_combined = f"{refined_thinking}</think>{refined_ans}"
                
                quality_score = result_json.get("quality_score", 0.5)
                
                print(f"🔧 [ID: {item.get('id')}] Step 2 完成: 修订={needs_revision}, 质量={quality_score}")
                
                return {
                    **item,
                    "final_cot_combined": refined_combined,
                    "final_thinking": result_json.get("refined_thinking", item.get("cot_thinking", "")),
                    "final_answer": result_json.get("refined_answer", item.get("cot_answer", "")),
                    "was_revised": needs_revision,
                    "revision_reason": result_json.get("revision_reason", ""),
                    "hallucination_detected": result_json.get("hallucination_detected", ""),
                    "quality_score": quality_score,
                    "step2_status": "success"
                }
                
            except Exception as e:
                print(f"❌ [ID: {item.get('id')}] Step 2 错误: {str(e)}")
                return {
                    **item,
                    "final_cot_combined": cot_combined,  # 使用 Step 1 的结果
                    "step2_status": "error",
                    "step2_error": str(e)
                }
    
    # ==========================================
    # 运行完整流水线
    # ==========================================
    
    async def run_pipeline(self, dataset: List[Dict[str, Any]], quality_threshold: float = 0.6) -> pd.DataFrame:
        """
        运行完整的两步 CoT 蒸馏流水线
        
        Args:
            dataset: 输入数据列表，每条包含 question, answer, image_path (可选)
            quality_threshold: 质量分数阈值，低于此值的数据将被标记
            
        Returns:
            处理后的 DataFrame
        """
        
        
        print("=" * 60)
        print("🚀 开始 CoT 蒸馏流水线")
        print(f"   数据量: {len(dataset)}")
        print(f"   并发数: {CONCURRENCY_LIMIT}")
        print(f"   质量阈值: {quality_threshold}")
        print("=" * 60)
        
        # Step 1: 生成 CoT 并融合
        print("\n📝 Step 1: 生成 CoT 思考过程并与答案融合...")
        step1_tasks = [self.generate_cot_and_merge(item) for item in dataset]
        step1_results = await asyncio.gather(*step1_tasks)
        
        step1_success = sum(1 for r in step1_results if r.get("step1_status") == "success")
        print(f"   ✅ Step 1 完成: {step1_success}/{len(step1_results)} 成功")
        
        # Step 2: 修订和去幻觉
        print("\n🔧 Step 2: 修订 CoT 和答案，去除幻觉...")
        step2_tasks = [self.refine_cot_and_answer(item) for item in step1_results]
        step2_results = await asyncio.gather(*step2_tasks)
        
        step2_success = sum(1 for r in step2_results if r.get("step2_status") == "success")
        print(f"   ✅ Step 2 完成: {step2_success}/{len(step2_results)} 成功")
        
        # 统计
        revised_count = sum(1 for r in step2_results if r.get("was_revised", False))
        high_quality_count = sum(1 for r in step2_results if r.get("quality_score", 0) >= quality_threshold)
        
        print("\n📊 统计:")
        print(f"   - 被修订的数据: {revised_count}")
        print(f"   - 高质量数据 (>={quality_threshold}): {high_quality_count}")
        
        # 添加是否通过质量检查的标记
        for item in step2_results:
            item["quality_passed"] = item.get("quality_score", 0) >= quality_threshold
        
        return pd.DataFrame(step2_results)


# ==========================================
# 主函数
# ==========================================
from argparse import ArgumentParser
arg_parser = ArgumentParser()
arg_parser.add_argument("--api_key", type=str, default="AIzaSyCjhCgDEZ05AGFkRWSGRRPCOWULbvvjOlw", help="Google API Key")
arg_parser.add_argument("--input_file", type=str, default="/home/maxzhang/datapipeline/temp_images/vqa.json", help="输入 JSON 文件路径")
arg_parser.add_argument("--output_file", type=str, default="/home/maxzhang/datapipeline/temp_images/cot_distilled_output.json", help="输出 JSON 文件路径")
args = arg_parser.parse_args()

async def main():
    # 配置
    GOOGLE_API_KEY = args.api_key
    INPUT_FILE = args.input_file  # 输入文件
    OUTPUT_FILE = args.output_file  # 输出文件
    
    # 加载数据
    print(f"📂 加载数据: {INPUT_FILE}")
    data = pd.read_json(INPUT_FILE)
    data["id"] = range(1, len(data) + 1)
    
    # 处理图片路径
    if "image_path" in data.columns:
        data["image_path"] = data["image_path"].apply(
            lambda x: "./temp_images/" + x.split("/")[-1] if x else None
        )
    
    print(f"   数据量: {len(data)}")
    print(data.head())
    
    # 转换为字典列表
    dataset = data.to_dict('records')
    
    # 可选: 采样测试
    # dataset = dataset[:10]
    
    # 初始化流水线
    pipeline = CoTDistillationPipeline(api_key=GOOGLE_API_KEY)
    
    # 运行
    results = await pipeline.run_pipeline(dataset, quality_threshold=0.6)
    
    # 保存结果
    print(f"\n💾 保存结果到: {OUTPUT_FILE}")
    results.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=2)
    
    # 同时保存 CSV 方便查看
    csv_path = OUTPUT_FILE.replace(".json", ".csv")
    results.to_csv(csv_path, index=False)
    print(f"   CSV 格式: {csv_path}")
    
    # 打印示例
    print("\n📋 结果示例:")
    if len(results) > 0:
        sample = results.iloc[0]
        print(f"   问题: {sample.get('question', sample.get('+', ''))[:50]}...")
        print(f"   最终输出: {sample.get('final_cot_combined', '')[:100]}...")
        print(f"   质量分数: {sample.get('quality_score', 'N/A')}")
    
    print("\n🎉 CoT 蒸馏完成!")


if __name__ == "__main__":
    asyncio.run(main())
