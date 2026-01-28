import asyncio
import base64
import json
import os
import pandas as pd
import io
from openai import AsyncOpenAI
from typing import List, Dict, Any
from datapipeline.caption_generation.hintfactory import HintFactory
# ==========================================
# 1. 配置与定义
# ==========================================

# 请替换为你的 API Key
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
MODEL_NAME = "ep-20251028151816-lv47m"  # 必须使用支持 Vision 的模型
CONCURRENCY_LIMIT = 128  # 同时并发请求的数量 (根据你的 Tier 调整)

# 你的分类体系 (这里只截取部分作为示例，实际运行时请放入完整 CSV)，实际过程中只根据三级标签生产hint
TAXONOMY_CSV = """
名词细项
Micro-LED display
微型LED显示 (MicroLED)
Mini LED
OLED (有机发光二极管)
WOLED
Micro OLED
OLED-on-Silicon
硅基 OLED
TFT-LCD
GaN LED / InGaN LED / AlGaInP LED / GaAs LED
柔性显示 (Flexible Display)
柔性技术
透明显示
全息显示
光场显示
立体显示 (3D显示)
空间显示
平板显示
微显示器 (Micro-display)
薄膜晶体管 (TFT)
非晶硅TFT (a-Si TFT)
低温多晶硅 (LTPS TFT)
氧化物TFT (IGZO TFT / Metal Oxide TFT)
有源驱动显示 (AM-Driving)
像素电路 (Pixel Circuit)
像素驱动电路
GOA电路 (Gate Driver on Array)
像素电极
Level Shifter
短沟道效应 (Short Channel Effect)
热载流子效应 (Hot Carrier Effect)
Kink Effect (扭曲效应)
液晶材料
蓝相液晶 (Blue Phase LC)
正性液晶 / 负性液晶
液晶取向
OLED材料科学
OLED发光材料 (Emitters)
磷光 OLED (Phosphorescent)
TADF (热活化延迟荧光)
QLED (量子点显示)
QLED材料科学
量子点膜 (QDEF)
量子点背光源
偏光片 (Polarizer)
偏光片技术
彩色滤光片 (Color Filters)
柔性聚酰亚胺 (Flexible PI)
Chip On PI
化学气相沉积 (CVD)
物理气相沉积 (PVD)
蒸镀技术 (Evaporation)
喷墨打印技术 (Inkjet Printing)
光刻技术 (Photolithography)
干法刻蚀 (Dry Etching)
湿法刻蚀 (Wet Etching)
制程工艺
退火 (Annealing)
准分子激光退火 (ELA)
快速热退火 (RTA)
摩擦配向 (Rubbing)
光配向 (Photo-alignment)
配向膜 (Alignment Layer)
薄膜封装 (TFE)
封装技术
OLED 封装
芯片转移 (Mass Transfer)
芯片控制
集成技术
液晶显示 LCD
背光模组
直下式背光 / 侧入式背光
Local Dimming (局部调光)
动态调光
亮度均匀性
Source Driver (源极驱动)
TCON (时序控制器)
Chip On Film (COF)
Chip On Glass (COG)
触控技术
On-cell 触控
电容触控
LVDS / mini-LVDS
V-by-One
色域 (Color Gamut)
可视角 (Viewing Angle)
光学规格
亮点缺陷
画面串扰 (Crosstalk)
像素补偿 / 灰阶补偿
不确定其他类型

"""

# 图片类型定义
IMAGE_TYPES_LIST = [
    "photo (实物摄影)", 
    "diagram (原理/结构示意图)", 
    "chart (数据图表)", 
    "scan (显微镜/扫描图)", 
    "screenshot (屏幕截图)", 
    "table (表格)", 
    "measurement (热力图/仿真图)"
]

# ==========================================
# 2. 辅助工具函数
# ==========================================

def parse_taxonomy(csv_text="/mnt/storage/dataset/PPVL_reuslts_CN/RAG-Anything/taxonomy/test.csv"):
    """解析 CSV 为纯文本列表，供 Prompt 使用"""
    df = pd.read_csv(io.StringIO(csv_text.strip()))
    df.columns = [c.strip() for c in df.columns]
    # 格式化为: "L1 > L2 > L3" 供模型理解层级
    tags = []
    titles = "名词细项"
    for _, row in df.iterrows():
        tags.append(f"{row['名词细项']}")
    return tags

def encode_image(image_path):
    """将本地图片转换为 Base64 编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# ==========================================
# 3. 核心 Prompt 设计
# ==========================================
def build_system_extract_prompt(context):

    prompt = f"""
        ### 任务目标
        分析用户给定的【图片】和【文本上下文】，提取最具代表性的关键词。

        ### 文本上下文
        {context}

        ### 分析步骤 (请在提取前进行思考)
        1. **视觉锚定 (Visual Grounding)**：简要描述图片中实际可见的内容是什么（例如：特定的设备结构、流程图的某个阶段、特定的数据图表）。
        2. **图文对齐 (Alignment)**：检查文本中提到的哪些技术术语或实体，直接对应步骤1中观察到的视觉元素。
        3. **相关性过滤**：剔除那些虽然在文本中提到，但与当前图片内容无关的词汇。
        4. 

        ### 约束条件
        - 关键词必须出现在文本上下文中。
        - 关键词必须在图片中有视觉依据。
        - 避免抽象动词，侧重于名词、实体和具体的技术参数。
        - 请严格从给定的名词列表中选择一个最贴切的标签（只选一个）：

        ### 输出格式
        请返回一个包含两个字段的 JSON 对象：
        {{
            "reasoning": "简要解释图片内容与文本的对应关系...",
            "keywords": ["关键词"]
                }}
        """
    return prompt

def build_system_prompt(taxonomy_list):
    # 将列表拼接成字符串
    taxonomy_str = "\n".join([f"- {tag}" for tag in taxonomy_list])
    types_str = "\n".join([f"- {t}" for t in IMAGE_TYPES_LIST])
    print(taxonomy_str)
    return f"""
    # 你是一位显示技术（Display Technology）领域的资深研究员和数据标注专家。你具备极强的视觉识别能力和专业术语理解能力。

    # Task (任务)
    你的任务是分析提供的【图片】和【上下文文本】。
    请从提供的知识库中，精准识别出该数据对应的 **关键词** 和 **图片类型 (Image Type)**。

    # Knowledge Base (分类候选库)
    请严格从以下列表中选择一个最贴切的标签（只选一个）：
    {taxonomy_str}

    # Image Types (图片类型定义)
    请从以下列表中选择一个最符合视觉特征的类型：
    {types_str}

    # Rules (标注规则)
    1. **视觉优先原则 (Visual First)**: 
    - 首先看图判断 [Image Type]。
    - 如果是结构拆解，选 'diagram'；如果是真实拍摄的物体，选 'photo'；如果是数据坐标轴，选 'chart'。
    
    2. **语义推理原则 (Semantic Reasoning)**:
    - 结合【图片内容】和【文本描述】锁定具体**关键词**。
    - **同义词推断**: 如果文本提到 "Foldable Phone" (折叠手机)，你应该选择 "柔性显示 (Flexible Display)"。
    - **精确性**: 必须精确到具体的关键词。如果图片明显是 "Micro-LED"，不要只选宽泛的 "器件 (Devices)"。

    3. **格式要求**:
    - 必须输出标准的 JSON 格式，不要包含 Markdown 标记（如 ```json）。

    # Output JSON Schema (输出格式)
    {{
    "reasoning": "用简练的中文解释推理过程。例如：'图片展示了晶体管结构，且文中提到了LTPS，因此判定为 LTPS TFT。'",
    "image_type": "从定义的英文类型中选一个 (例如 'diagram')",
    "keywords": "精准匹配到的关键词名称 (例如 'Micro-LED display')",
    }}
    """

# ==========================================
# 4. 异步处理流水线
# ==========================================

class AsyncLabelingPipeline:
    def __init__(self, api_key="d075fa90-7412-4208-9776-188332b1f2f9", taxonomy_csv="/mnt/storage/dataset/PPVL_reuslts_CN/RAG-Anything/taxonomy/test.csv"):
        self.client = AsyncOpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3",
                                  api_key="d075fa90-7412-4208-9776-188332b1f2f9")

                                  
        self.taxonomy_list = parse_taxonomy(taxonomy_csv)
        self.system_prompt = build_system_prompt(self.taxonomy_list)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT) # 限制并发数

    async def process_single_item(self, item: Dict[str, Any]):
        """
        处理单条数据：Image + Context -> Classification
        item: {'id': 1, 'image_path': '...', 'context': '...'}

        # processing single data and return data tag in json format
        """
        async with self.semaphore: # 获取信号量
            try:
                # 1. 准备图片
                
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": []}
                ]
                
                # 2. 组装 User Message (Text + Image)
                content_payload = []
                
                # 添加文本上下文
                content_payload.append({
                    "type": "text", 
                    "text": f"Context/Caption: {item.get('related_text_strong', 'No text provided.')}"
                })
                
                # 添加图片 (如果存在)
                for image in item["images"]:
                    base64_image = encode_image(image)
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high" # 使用高分辨率模式以识别细节
                        }
                    })
                
                messages[1]["content"] = content_payload

                # 3. 异步调用 OpenAI
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1, # 低温度保证分类准确性
                    response_format={"type": "json_object"}, # 强制 JSON
                    max_tokens=300
                )

                # 4. 解析结果
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                
                # 合并原始 ID 以便追踪
                result_json['id'] = item['id']
                result_json["images"] = item["images"]
                result_json['original_context'] = item['related_text_strong']
                
                print(f"✅ [ID: {item['id']}] Processed: {result_json['l3_tag']} | {result_json['image_type']}")
                return result_json

            except Exception as e:
                print(f"❌ [ID: {item['id']}] Error: {str(e)}")
                return {
                    "id": item['id'], 
                    "error": str(e),
                    "l3_tag": "ERROR",
                    "image_type": "ERROR"
                }
    async def process_key_word_extraction(self,item):
        async with self.semaphore:
            try:
                # 1. 准备 Context 文本
                context_text = item.get('related_text_strong', '')
                
                # 2. 定义关键词提取专用的 System Prompt (CoT版)
                # 这里的 prompt 即使不动态插入 context 也可以，因为 context 会在 user message 里提供
                keyword_system_prompt = build_system_extract_prompt(context_text)
                messages = [
                    {"role": "system", "content": keyword_system_prompt},
                    {"role": "user", "content": []}
                ]

                # 3. 组装 User Message (Context + Image)
                content_payload = []
                
                # 3.1 注入文本上下文
                content_payload.append({
                    "type": "text", 
                    "text": f"【文本上下文】:\n{context_text}\n\n请开始分析图片并提取关键词："
                })
                
                # 3.2 注入图片
                images = item.get("image_path", [])
              
                if not images:
                    print(f"⚠️ [ID: {item['id']}] Warning: No images found.")
                else:
                    images = [images]
                for image in images:
                    # 假设 encode_image 是外部定义好的帮助函数
                    base64_image = encode_image(image) 
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                
                messages[1]["content"] = content_payload

                # 4. 异步调用 OpenAI
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME, # 确保使用支持 Vision 的模型 (如 gpt-4o)
                    messages=messages,
                    temperature=0.2, # 稍低的温度以保持提取的准确性
                    response_format={"type": "json_object"},
                    max_tokens=500   # 给 reasoning 留出足够的空间
                )

                # 5. 解析结果
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                
                # 6. 构造返回数据
                final_output = {
                    "id": item['id'],
                    "original_context": context_text,
                    "extracted_keywords": result_json.get("keywords", []),
                    "extraction_reasoning": result_json.get("reasoning", ""),
                    # 保留图片路径供后续检查，但不返回 base64 以节省内存
                    "image_count": len(images)
                }
                
                print(f"✅ [ID: {item['id']}] Extracted {len(final_output['extracted_keywords'])} keywords.")
                return final_output

            except Exception as e:
                print(f"❌ [ID: {item['id']}] Keyword Extraction Error: {str(e)}")
                return {
                    "id": item['id'], 
                    "error": str(e),
                    "extracted_keywords": [],
                    "extraction_reasoning": "ERROR"
                }

    
    def build_vqa_generation_prompt(self, l3_tag=None, image_type=None, hint=None,keyword=None):
        """
        构建生成 QA 对的 Prompt Template
        核心策略：Context + Image -> Specialized Hint -> QA Pair
        """
        if hint==None and keyword==None and image_type == None:
            return f"""
                # 角色
                你是一位专攻显示技术的教授。你正在为博士生制作一份考试数据集。

                # 输入数据
 
                - **上下文文本**: 见用户消息。
                - **视觉内容**: 见用户图片。

               

                # 任务
                **严格**基于提供的图片和上下文，创建一个高质量的问答（QA）对。

                # 要求
                1. **问题**: 避免像“图片里有什么？”这样的宽泛问题。相反，应针对可见的具体结构、数据数值或机制进行提问。
                2. **答案**: 必须能从提供的视觉或文本上下文中推导出来。不要凭空捏造输入中未包含的外部知识。
                3. **推理**: 逻辑清晰地解释答案是如何从视觉特征或文本中推导出来的。

                # 输出格式 (仅 JSON)
                {{
                    "question": "技术性问题",
                    "answer": "详细的答案",
                    "explanation": "基于视觉/文本证据的逐步推导过程"
                }}
                """
        else:
            return f"""
                # 角色
                你是一位专攻显示技术的教授。你正在为博士生制作一份考试数据集。

                # 输入数据
                - **图片类型**: {image_type}
                - **技术主题**: {l3_tag}
                - **上下文文本**: 见用户消息。
                - **视觉内容**: 见用户图片。

                # 指令 (提示词/Hint)
                {hint}
                # 相关的需要关注的的提示词
                {keyword}


                # 任务
                **严格**基于提供的图片和上下文，创建一个高质量的问答（QA）对。

                # 要求
                1. **问题**: 必须具体针对 {l3_tag} 技术。避免像“图片里有什么？”这样的宽泛问题。相反，应针对可见的具体结构、数据数值或机制进行提问。
                2. **答案**: 必须能从提供的视觉或文本上下文中推导出来。不要凭空捏造输入中未包含的外部知识。
                3. **推理**: 逻辑清晰地解释答案是如何从视觉特征或文本中推导出来的。

                # 输出格式 (仅 JSON)
                {{
                    "question": "技术性问题",
                    "answer": "详细的答案",
                    "explanation": "基于视觉/文本证据的逐步推导过程"
                }}
                """
        
    async def generate_qa_pair(self, item_with_class: Dict[str, Any],l3_tag=None, image_type=None,keyword =None):
        """
        第二阶段：根据分类结果生成 QA
        """
        async with self.semaphore:
            try:
                # 1. 获取动态 Hint
                l3_tag = item_with_class.get("l3_tag", None)
                image_type = item_with_class.get("image_type", None)
                if l3_tag ==None and image_type==None:
                    hint = None
                else:
                    hint = HintFactory.get_hint(l3_tag, image_type)

                # 2. 构建 Prompt
                system_prompt = self.build_vqa_generation_prompt(l3_tag, image_type, hint,keyword)
                
                # 3. 准备消息
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": []}
                ]
                
                content_payload = []
                content_payload.append({
                    "type": "text", 
                    "text": f"Context:{item_with_class.get('figure_title', '')} \n\n {item_with_class.get('related_text_strong', '')}"
                })
                
                # 重新读取图片 (或者在内存中传递 base64，取决于你的内存策略)
                # 这里假设 item 里还存着路径
                for image_path in item_with_class.get("images", []):
                    base64_image = encode_image(image_path)
                    if base64_image:
                        content_payload.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
                
                messages[1]["content"] = content_payload

                # 4. 调用 LLM 生成 QA
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7, # 稍微提高一点温度以增加问题的多样性
                    response_format={"type": "json_object"},
                    max_tokens=500
                )

                qa_result = json.loads(response.choices[0].message.content)
                
                # 5. 合并结果
                final_record = {**item_with_class,
                    "generated_results":{
                    "generated_question": qa_result.get("question"),
                    "generated_answer": qa_result.get("answer"),
                    "qa_reasoning": qa_result.get("reasoning"),
                    },
                    "used_hint": hint
                    }
                
                print(f"💡 [ID: {item_with_class['id']}] QA Generated: {qa_result.get('question')[:50]}...")
                return final_record

            except Exception as e:
                print(f"❌ [ID: {item_with_class['id']}] QA Gen Error: {str(e)}")
                return  {**item_with_class,
                    "generated_results":{
                    "generated_question": None,
                    "generated_answer": None,
                    "qa_reasoning":None,
                    },
                    "used_hint": "error"
                    }
    async def filterout_data(self, item: Dict[str, Any], question: str, answer: str, filter_out_threshold=0.6):
        """
        [评估阶段]
        输入：原始 item (含图片/Context), 生成的 question, 生成的 answer
        输出：是否保留 (Boolean), 评分 (Score), 理由 (Reason)
        """
        
        # 0. 基础检查：如果生成失败，直接过滤
        if not question or not answer:
            return {"pass": False, "score": 0.0, "reason": "Empty question or answer"}

        async with self.semaphore:
            try:
                # 1. 定义评估专用的 Prompt
                # 我们要求模型扮演一个“严格的审核员”，检查 QA 是否存在幻觉或逻辑错误
                eval_system_prompt = """
                你是一位专攻“显示技术”领域的数据集质量质检员（QA Auditor）。你的审核标准非常严格。
                
                # 任务
                根据提供的【图片】和【上下文文本】，评估生成的【问答对 (QA Pair)】的质量。
                
                # 评估维度 (Evaluation Criteria)
                1. **事实依据 (Grounding)**: 答案必须严格基于图片中可见的内容或提供的文本信息。**严禁幻觉 (No Hallucinations)**，即答案不能包含输入数据中不存在的外部事实。
                2. **相关性 (Relevance)**: 问题必须与“显示技术”紧密相关，且必须与当前提供的图片内容匹配。
                3. **可回答性 (Solvability)**: 仅凭提供的图片和上下文，逻辑上是否足以推导出这个答案？
                4. **具体性 (Specificity)**: 拒绝宽泛的问题（例如“这张图展示了什么？”）。问题必须针对具体的器件结构、数据参数、工艺流程或技术原理。

                # 输出格式 (JSON)
                请仅返回 JSON 对象，不要包含 Markdown 标记：
                {{
                    "score": <0.0 到 1.0 之间的浮点数>,  // 1.0 代表完美可用，< 0.6 代表质量差需过滤
                    "reason": "<用简练的中文解释评分理由，指出具体的优点或缺陷>",
                    "pass": <布尔值> // 如果 score >= 阈值 则为 true
                }}
                """

                # 2. 组装 User Message
                messages = [
                    {"role": "system", "content": eval_system_prompt},
                    {"role": "user", "content": []}
                ]

                content_payload = []

                # 2.1 注入评估对象 (QA Pair + Context)
                eval_text = (
                    f"### Context:\n{item.get('related_text_strong', 'No context provided')}\n\n"
                    f"### Generated Question:\n{question}\n\n"
                    f"### Generated Answer:\n{answer}\n\n"
                    f"### Instruction:\nEvaluate the QA pair against the image and context. Return JSON."
                )
                
                content_payload.append({
                    "type": "text", 
                    "text": eval_text
                })

                # 2.2 注入图片 (用于核实 Visual Grounding)
                # 注意：为了节省 Token，如果图片非常多，可以只取第一张，或者根据业务逻辑调整
                images = item.get("images", [])
                if images:
                    # 这里的 images 应该是路径列表，取第一张作为主要参考
                    # 如果你的逻辑需要多图，可以在这里循环添加
                    target_image_path = images[0] 
                    base64_image = encode_image(target_image_path)
                    if base64_image:
                        content_payload.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low" # 评估时用 low 模式通常足够且省钱，除非细节极其重要
                            }
                        })
                
                messages[1]["content"] = content_payload

                # 3. 调用 LLM
                response = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0, # 评估必须客观，使用 0 温度
                    response_format={"type": "json_object"},
                    max_tokens=200
                )

                # 4. 解析结果
                result_json = json.loads(response.choices[0].message.content)
                
                score = result_json.get("score", 0.0)
                is_pass = score >= filter_out_threshold

                print(f"⚖️ [ID: {item['id']}] Filter Check: Score {score} | Pass: {is_pass}")
                
                return {
                    "pass": is_pass,
                    "score": score,
                    "reason": result_json.get("reason", "No reason provided")
                }

            except Exception as e:
                print(f"⚠️ [ID: {item['id']}] Filter Error: {str(e)}")
                # 发生错误时，保守策略可以选 False (丢弃)，或者 True (保留人工清洗)
                # 这里选择保留但标记分数 -1
                return {"pass": False, "score": -1.0, "reason": f"Error: {str(e)}"}

    async def run_batch(self, dataset: List[Dict],use_hint = True):
        """
        两阶段流水线：
        1. Classify (分类)
        2. Generate QA (生成问答)
        """
        
        # 第一步任务列表
        if use_hint:
            print("Step 1: Classification & Tagging...")
            keyword_tasks = [self.process_key_word_extraction(item) for item in dataset]
            keyword_results = await asyncio.gather(*keyword_tasks)
            # 
            df_results = pd.DataFrame([sample for sample in keyword_results if "error" not in sample])
            if isinstance(dataset, pd.DataFrame):
                df_original = dataset.copy()
            else:
                # 如果传入的是 list of dicts
                df_original = pd.DataFrame(dataset)
            cols_to_update = ['extracted_keywords', 'extraction_reasoning']
            df_original_clean = df_original.drop(columns=[c for c in cols_to_update if c in df_original.columns])
            cols_to_update = ['extracted_keywords', 'extraction_reasoning']
            final_df = pd.merge(
                df_original_clean,
                df_results[['id'] + cols_to_update], # 只取需要的列
                on='id',
                how='left'
            )
            # print(final_df.head())
            # success_count = final_df['extracted_keywords'].notna().sum()
            # print(f"✅ Batch Complete. Processed: {len(final_df)} | Success: {success_count}")
            # print("\nStep 2: QA Generation based on Tags...")
            # # 过滤掉第一步失败的项
            # valid_results = [r for r in keyword_tasks if r.get("extracted_keywords") != "ERROR"]
            
            # # 第二步任务列表

            qa_tasks = []
            for item in final_df.to_dict("records"):
                l3_tag = item.get("category",None)
                image_type = item.get("label",None)
                keywords = item.get("extracted_keywords", None)
                qa_tasks.append(self.generate_qa_pair(item,l3_tag,image_type,keywords))
            qa_tasks = await asyncio.gather(*qa_tasks)
            qa_tasks = [task for task in qa_tasks if task["used_hint"]!="error"]
            qa_tasks = pd.DataFrame(qa_tasks)
            return qa_tasks
        if not use_hint:
            print("Step 1: Generate No Meta Data Response...")
            qa_tasks = [self.generate_qa_pair(item) for item in dataset]
            qa_tasks = await asyncio.gather(*qa_tasks)
            # print(qa_tasks)
            qa_tasks = [task for task in qa_tasks if task["used_hint"]!="error"]
            qa_tasks = pd.DataFrame(qa_tasks)
            return qa_tasks# ,final_results

# ==========================================
# 5. 主程序入口
# ==========================================
# from google import genai
# from google.genai import types

# with open('path/to/small-sample.jpg', 'rb') as f:
#     image_bytes = f.read()

# client = genai.Client()
# response = client.models.generate_content(
# model='gemini-2.5-flash',
# contents=[
#     types.Part.from_bytes(
#     data=image_bytes,
#     mime_type='image/jpeg',
#     ),
#     'Caption this image.'
# ]
# )

# print(response.text)
async def main():
    USE_HINT=True
    import duckdb
    db="/mnt/storage/database/db.duckdb"
    con = duckdb.connect(db)
    TABLE_NAME = "semiconductor_batch_4_1"
    try:
        con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS extracted_keywords JSON")
        con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS extraction_reasoning VARCHAR")
        con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS with_meta_data_results JSON")
        con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS without_meta_data_results JSON")
        
        
        print("✅ 表结构检查完毕 (列已存在或已添加)")
    except Exception as e:
        print(f"⚠️ 添加列时遇到提示 (可能已存在): {e}")
    
    
    data = con.execute(f"""select * from {TABLE_NAME} where with_meta_data_results is null""")
    
    
    
    data=data.fetchdf()
    data["id"] = data.index
    data.columns
 

    dataset = data
    # with open("/mnt/storage/dataset/PPVL_reuslts_CN/json_file/filter_qiuping_all_deduplicated_v0.3.1.json","r") as f:
    #     dataset = json.load(f)[:20]
    dataset = dataset.sample(n=min(50,len(dataset))).to_dict('records')
    # for i in range(len(dataset)):
    #     dataset[i]["id"] = i+1
    # # 为了演示，创建一个假的本地图片文件，防止代码报错
    # os.makedirs("test_images", exist_ok=True)


    # 2. 初始化 Pipeline
    pipeline = AsyncLabelingPipeline(api_key=API_KEY, taxonomy_csv=TAXONOMY_CSV)
    
    print(f"🚀 开始处理 {len(dataset)} 条数据 (并发数: {CONCURRENCY_LIMIT})...")
    
    # 3. 运行
    results = await pipeline.run_batch(dataset,use_hint=USE_HINT)
    
    # 4. 保存结果
    output_df = pd.DataFrame(results)
    print("\n🎉 处理完成！结果预览：")
    print(output_df.head())
    output_df.to_csv("keyword_results.csv", index=False)
    # 保存为 JSONL 或 CSV 供下一步使用
    output_df.to_json("keyword_results.jsonl", orient="records", lines=True, force_ascii=False)
    # update original table with the column keyword
    



    print("\n🔄 开始更新 DuckDB 原始表...")
    store_database=False
    con.register('batch_updates_view', output_df)   
    if USE_HINT and store_database:
        # 确保 keywords 列是 JSON 字符串格式，防止插入报错
        output_df['extracted_keywords_json'] = output_df['extracted_keywords'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        
        
        
        

        update_sql = f"""
        UPDATE {TABLE_NAME}
        SET 
            extracted_keywords = batch_updates_view.extracted_keywords,
            extraction_reasoning = batch_updates_view.extraction_reasoning,
            with_meta_data_results = batch_updates_view.generated_results,
            
        FROM batch_updates_view
        WHERE {TABLE_NAME}.rowid = batch_updates_view.id
        """
    elif store_database:      
        update_sql = f"""
        UPDATE {TABLE_NAME}
        SET 
            without_meta_data_results = batch_updates_view.generated_results,
            
        FROM batch_updates_view
        WHERE {TABLE_NAME}.rowid = batch_updates_view.id
        """
    if store_database:
        try:
            con.execute(update_sql)
            # 获取受影响的行数 (DuckDB 通常不直接返回行数，但如果不报错即成功)
            print(f"✅ 数据库更新成功！")
            
            # --- 验证更新 ---
            # 随机查一条看看
            check_df = con.execute(f"""
                SELECT *
                FROM {TABLE_NAME} 
                WHERE with_meta_data_results IS NOT NULL OR without_meta_data_results IS NOT NULL LIMIT 1 
            """).fetchdf()
            print("\n🔍 数据库更新验证 (Sample):")
            print(check_df)
            check_df.to_csv("final_results.csv",index=False)
            
        except Exception as e:
            print(f"❌ 数据库更新失败: {e}")
        
    # 关闭连接
    con.close()
if __name__ == "__main__":
    # Windows/Jupyter 环境可能需要 nest_asyncio
    # import nest_asyncio
    # nest_asyncio.apply()
    
    asyncio.run(main())