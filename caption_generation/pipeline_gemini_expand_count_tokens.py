import json
import asyncio
import base64
import json
import os
import pandas as pd
import io
from openai import AsyncOpenAI
from typing import List, Dict, Any
from hintfactory import HintFactory
from google import genai
from google.genai import types
_uploaded_files_cache: Dict[str, Any] = {}
def parse_json_list(json_data: str) -> list:
    """
    解析 JSON 列表内容的函数。
    
    参数:
        json_data (str): JSON 格式的字符串，表示一个列表。
    
    返回:
        list: 解析后的 Python 列表。
    """
    import re 
    # 解析  ```json xxxx``` 中xxx的内容
    
    try:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, json_data, re.DOTALL)
        json_data = match.group(1).strip() if match else json_data.strip()
        # 将 JSON 字符串解析为 Python 列表
        parsed_list = json.loads(json_data)
        
        # 确保解析结果是一个列表
        if not isinstance(parsed_list, list):
            raise ValueError("JSON 数据不是一个列表")
        
        # 对列表中的每个元素进行处理（这里简单打印）
        for item in parsed_list:
            print(f"解析到的内容: {item}")
        
        return parsed_list
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return json_data
    except ValueError as e:
        print(f"数据错误: {e}")
        return json_data

# 假设你封装了两个基础调用函数
# mock_llm_call(prompt): 输入文本prompt，返回文本结果 (用于 Step 1, 2)
# mock_vlm_call(image, prompt): 输入图片和文本prompt，返回文本结果 (用于 Step 3, 4)

def step1_extract_entities(original_caption):
    """
    第一步：实体提取
    目的：从原始caption中识别需要扩充的钩子。
    模型：纯文本 LLM
    """
    prompt = f"""
    # Role
    你是一个信息抽取专家。

    # Input
    原始描述: "{original_caption}"

    # Task
    请从描述中提取出具有“百科全书式背景挖掘价值”的关键词。
    重点关注：专有名词（人名、地名、产品名）、特定时间/事件、专有术语。
    忽略：通用形容词（如“红色的”、“大的”）。

    # Output Format
    仅输出一个JSON列表，例如: ["埃菲尔铁塔", "1889年世博会", "钢铁结构"]
    """
    
    # 实际调用 (这里用伪代码代替)
    # response = mock_llm_call(prompt)
    # return json.loads(response) 
    return prompt # 返回 prompt 供你测试
import json

def step1_5_extract_contextual_knowledge(original_caption, context):
    """
    步骤 1.5：基于原文语境的知识提取
    目的：从原文(context)中提取与图片Caption直接相关的背景知识要点（如实验条件、特定数据含义、背后的逻辑等）。
    模型：纯文本 LLM
    """
    prompt = f"""
    # Role
    你是一个专业的信息抽取与总结专家。

    # Input
    1. [图片原始描述 (Caption)]: "{original_caption}"
    2. [原文语境 (Context)]: "{context}"

    # Task
    你的任务是阅读“原文语境”，从中筛选出有助于解释或补充“图片原始描述”的关键背景知识。
    请提取以下几类信息（如果原文中有提到）：
    
    1. **技术/方法 (Methodology)**: 图片是如何生成的？（例如：拍摄设备、成像技术、绘图算法、显微镜型号等）。
    2. **对象/材料 (Subject/Material)**: 图片中具体展示了什么特定物质、人物或数据集？（例如：实验用到的金属、特定年份的人口数据、特定的代码模块）。
    3. **现象/结论 (Observation/Conclusion)**: 原文如何解释图片中的现象？（例如：曲线上升代表温度升高、红色区域代表高注意力权重）。
    4. **特定背景 (Specific Context)**: 时间、地点、特定事件或项目名称。

    # Constraints
    - 提取的内容必须与[图片原始描述]有强相关性。
    - 输出应为**完整的陈述句或短语**，清楚地说明属性和值（参考Output Example）。
    - 如果原文中没有相关信息，不要编造。

    # Output Example
    
    1. "该图片的拍摄方式为电子透镜",
    2. "实验材料包括铜和金",
    3. "红色曲线表示未经过优化的基线模型",
    4. "观测到放热现象，该曲线图显示了温度变化"
    

    # Output Format
    请以Bullet Point形式输出提取到的背景知识，每条独立成行。
    例如：
    1. "该图片的拍摄方式为电子透镜",
    2. "实验材料包括铜和金",
    3. "红色曲线表示未经过优化的基线模型",
    4. "观测到放热现象，该曲线图显示了温度变化"
    
    """
    
    # 模拟调用
    # response = mock_llm_call(prompt)
    # return json.loads(response)
    return prompt
def step2_generate_background(original_caption, entities):
    """
    第二步：背景知识生成
    目的：利用模型内部知识库生成背景信息（暂不考虑是否与图片冲突，先发散）。
    模型：纯文本 LLM
    """
    entities_str = ", ".join(entities)
    
    prompt = f"""
    # Role
    你是一个博学的知识库助手。

    # Context
    原始语境: "{original_caption}"
    需要扩充的实体: [{entities_str}]

    # Task
    请为上述实体生成背景知识简报。
    
    # Requirements
    1. 每个实体的介绍要详细具体，返回所有可能相关的背景知识。
    2. 保持客观中立，不要加入主观评论。
    
    # Output Format
    请直接输出整理好的背景知识文本段落，不需要JSON。
    """
    
    # response = mock_llm_call(prompt)
    return prompt

def step3_visual_fact_check(image, original_caption, raw_background_info):
    """
    第三步：基于视觉的冲突检测与过滤 (核心步骤)
    目的：利用 VLM 执行 Tip 1 (视觉介入) 和 Tip 2 (真理层级)。
    模型：VLM (Vision Language Model)
    """
    
    prompt = f"""
    # Role
    你是一名严苛的“视觉法医”和事实核查员。你的任务是清洗背景知识，确保其准确性。

    # Inputs
    1. [Image]: 参考输入的图片。
    2. [Original Caption] (用户提供的原始描述): "{original_caption}"
    3. [Draft Background] (待核查的背景知识): "{raw_background_info}"

    # Protocol: The Hierarchy of Truth (真理层级)
    在进行核查时，必须严格遵守以下优先权顺序：
    1. **Tier 1 (最高级): 用户原始描述 (Original Caption)**。这是绝对真理。如果背景知识与此冲突，必须修改或删除背景知识。
    2. **Tier 2: 图片视觉事实 (Visual Evidence)**。如果背景知识描述了图片中不存在的物体、错误的颜色、或错误的环境（例如背景说“这是雨天”，但图片明显是晴天），以图片为准。
    3. **Tier 3: 外部背景知识**。只有在不违反 Tier 1 和 Tier 2 的情况下，才能保留。

    # Task
    1. 逐条检查 [Draft Background] 中的信息。
    2. **删除**任何与图片内容直接冲突的视觉描述（如错误的颜色、位置、动作）。
    3. **保留**图片看不出来但合理的深层背景（如历史年份、设计理念、人物生平）。
    4. **修正**任何与 [Original Caption] 逻辑相悖的内容。

    # Output
    输出一段清洗后的、绝对安全的背景知识文本。如果某条信息被删除，不需要解释，直接输出剩下的部分即可。
    """
    
    # response = mock_vlm_call(image, prompt)
    return prompt

def step4_integrate_final_caption(original_caption, verified_background, original_context):
    """
    第四步：最终融合 (多源信息合成)
    目的：将 1.视觉事实、2.原文具体语境、3.通用背景知识 融合成一段高信息密度的精准描述。
    模型：纯文本 LLM (GPT-4o / Claude 3.5 Sonnet 等效果最佳)
    """
    
    prompt = f"""
    # Role
    你是一位专业的科学/技术编辑或资深特稿撰写人。你的专长是将图像的视觉信息与复杂的文本背景完美结合，产出清晰、准确且信息丰富的说明文字。

    # Input Data Sources
    1. [Visual Anchor - 视觉锚点]: "{original_caption}"
       (这是图片中实际看到的内容，作为描述的基础框架)
       
    2. [Specific Context - 原文语境]: "{original_context}"
       (这是图片所属文章的片段，提供了具体的实验对象、数据来源或特定事件细节)
       
    3. [General Knowledge - 验证背景]: "{verified_background}"
       (这是关于术语、原理或历史的通用解释，已经过核实)

    # Task
    请撰写一段最终的图片说明（Caption）。你需要以[Visual Anchor]为核心，利用[Specific Context]将模糊的视觉元素具体化，并适时引用[General Knowledge]来解释晦涩的概念。

    # Writing Guidelines (严格执行)
    1. **视觉主导，细节填充 (Visual-First, Context-Filled)**:
       - 必须以描述画面开始。
       - **具体化**: 不要说“图显示了一个金属的反应”，要结合[Specific Context]说“图显示了**铜(Copper)**在**电子透镜**下的放热反应”。
       - 只要原文语境中有提及，必须将图片中的通用物体替换为具体的专有名词。

    2. **知识融合与解释 (Explain with Background)**:
       - 当提到[Specific Context]中的专有名词时，如果该名词在[General Knowledge]中有解释，请自然地融入这些解释。
       - 例如：不要只说“使用了A算法”，要说“使用了A算法——一种常用于处理稀疏矩阵的高效计算方法...”。

    3. **消除歧义 (Zero Ambiguity)**:
       - **禁止滥用代词**: 尽量少用“它(it)”、“这个(this)”、“前者/后者”。
       - 如果必须指代，请重复名词或使用明确的限定词（如“该实验装置”、“图中的红色曲线”）。
       - 确保句子结构不会产生修饰歧义。

    4. **事实一致性 (Consistency Check)**:
       - 最终描述绝不能与[Visual Anchor]中的颜色、形状、数量等视觉事实冲突。
       - 如果[Specific Context]与[Visual Anchor]看似矛盾（例如文中说很多个，图里只有一个），以“图示为...”或“图中选取了...作为示例”来处理。

    # Output
    请直接输出最终生成的图片说明段落，无需解释你的思考过程。
    """
    
    # response = mock_llm_call(prompt)
    return prompt

async def google_call(async_client,image,prompt):
    cache = _uploaded_files_cache
    if image in cache:
        upload_file = cache[image]
        print(f"♻️ 复用已上传的图片: {image}")
    else:
        upload_file = await async_client.files.upload(file=image)
        cache[image] = upload_file
        print(f"📤 新上传图片: {image}")
    try:
        response 
        response = await async_client.models.generate_content(
                        model = "gemini-3-flash-preview",
                        contents= [
                        upload_file,
                        prompt]
                        )
     


        
        result_text = response.text
        if result_text is None:
            return await google_call(async_client,image,prompt)
        else:
            token_counts = response.usage_metadata.total_token_count
        return result_text,token_counts,response.usage_metadata.prompt_token_count
    except genai.errors.ServerError as e:
        # if it is 503 error, we can retry
        
        if e.code == 503:
            print(f"⚠️ Server busy, retrying...")
            # sleep for 10 s
            await asyncio.sleep(10)
            return await google_call(async_client,image,prompt)
        else:
            print(e)
            return None,0,0
async def openai_call(client,image,prompt,model):
    def encode_image(image):
        import base64
        image_type = image.split(".")[-1]
        return f"data:image/{image_type};base64," + base64.b64encode(open(image, "rb").read()).decode()
    response = await client.chat.completions.create(
        model = model,
        messages = [
            {"role": "system", "content": "You are a helpful assistant that helps people find information."},
            {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image(image)}}
            ]    
            }
        ]
    )
    
    return response.choices[0].message.content
# ==========================================
# 模拟执行流程 (Pipeline)
# ==========================================

async def process_single_caption(image,caption, context,use_openai=False,model ="gpt-4-vision-preview",api_key = "AIzaSyCjhCgDEZ05AGFkRWSGRRPCOWULbvvjOlw"):
    try:
        return await _process_single_caption(image,caption, context,use_openai=model,api_key=api_key)
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        return None
async def _process_single_caption(image,caption, context,use_openai=False,model ="gpt-4-vision-preview",api_key = "AIzaSyCjhCgDEZ05AGFkRWSGRRPCOWULbvvjOlw"):
    
    if use_openai:
        client = AsyncOpenAI(api_key=api_key)
    else:
        client = genai.Client(api_key=api_key).aio
    print("--- Start one Image ---")
    step1_prompt = step1_extract_entities(caption)
    #print(f"Step 1 Prompt: {step1_prompt}")
    response,count1,prompt_token_count1 = await (openai_call(client,image,step1_prompt,model) if use_openai else google_call(client,image,step1_prompt))
    entities = parse_json_list(response)
    
    step2_prompt = step2_generate_background(caption, entities)
    response,count2,prompt_token_count2 = await (openai_call(client,image,step2_prompt,model) if use_openai else google_call(client,image,step2_prompt))
    
    raw_background_info = response
   
    #context_response = step1_5_extract_contextual_knowledge(caption, context)
    
    
    
    
    step3_prompt = step3_visual_fact_check(image, caption, raw_background_info)
    response,count3,prompt_token_count3 = await (openai_call(client,image,step3_prompt,model) if use_openai else google_call(client,image,step3_prompt))
    verified_background = response
    
    step4_prompt = step4_integrate_final_caption(caption, verified_background,context)
    response,count4,prompt_token_count4 = await (openai_call(client,image,step4_prompt,model) if use_openai else google_call(client,image,step4_prompt))
    final_caption = response
    print(f"Step 1 Extracted: {entities}")
    # print(f"Step 1.5 Contextual Knowledge Prompt: {context}")
    print(f"Step 2 Generated Background Info: {raw_background_info}")
    print(f"Step 3 Verified Background Info: {verified_background}")
    print(f"Step 4 Final Caption: {final_caption}")
    return {
        "image_path":image,
        "caption_original": caption,
        "caption_expanded": final_caption,
        "entities_extracted": entities,
        "background_raw": raw_background_info,
        "background_verified": verified_background,
        "contextual_knowledge": context,
        "total_token_count": count1 + count2 + count3 + count4,
        "prompt_token_count": prompt_token_count1 + prompt_token_count2 + prompt_token_count3 + prompt_token_count4
    }

    
        
def main_pipeline(image_inputs, original_caption_inputs,contexts,cores = 16):
    from asyncio import Semaphore
    lock = Semaphore(cores)
    async def gather_tasks(lock):
        
        tasks = [
            process_single_caption(image,caption,context) for image,caption,context in zip(image_inputs,original_caption_inputs,contexts)
        ]
        async with lock:
            tasks = await asyncio.gather(*tasks)
        return [task for task in tasks if task !=None]
    return asyncio.run(gather_tasks(lock))
from argparse import ArgumentParser

parser = ArgumentParser(description="Process image captions")
parser.add_argument("--data", type=str, default="/home/maxzhang/datapipeline/sub_merged_caption_2.json", help="Path to the input data JSON file")
parser.add_argument("--img_path", type=str, default="/home/maxzhang/img_selected", help="Path to the directory containing images")
parser.add_argument("--save_path", type=str, default="expanded_captions.json", help="Path to save the output JSON file")
args = parser.parse_args()

if __name__=="__main__":
    import json 
    data = args.data
    IMG_PATH = args.img_path
    data = json.load(open(data))
    images,captions,contexts = [],[],[]
    for sample in data:
        img = sample["images"][0]
        #print(sample)
        img = os.path.join(IMG_PATH,img.split("/")[-3],img.split("/")[-1]) if "max_selected" not in img else os.path.join(IMG_PATH,img.split("/")[-2],img.split("/")[-1])
        caption = sample["caption"]
        context = sample["title"] +"\n".join(sample.get("sub_titles",[])) + "\n\n" + "\n".join(sample["related_text"])
        images.append(img)
        captions.append(caption)
        contexts.append(context)
    results = main_pipeline(images,captions,contexts,cores=16)
    
    save_path = args.save_path
    with open(save_path,"w") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    print("average token count:", sum([r["total_token_count"] for r in results])/len(results))
    print("average prompt token count:", sum([r["prompt_token_count"] for r in results])/len(results))

