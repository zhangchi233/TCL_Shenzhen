basic_template="""
# Role
你是一个专业的AI数据生成专家。你的任务是基于提供的[图片Caption]、[分类标签]和[视觉提示Hint]，生成高质量的Visual Question Answering (VQA) 训练数据。

# Input Data
-图片
{image}
- **Image Caption (上下文):** {caption}
- **Image Type (图片类型):** {image_type}
- **Category Tags (标签):** {tags}
- **Focus Hint (生成指令):** {hint}

# Task Description
请根据上述信息，生成 {n} 对 Question-Answer (QA) 数据。

## 生成策略
1. **结合类型与标签:** 必须严格遵循 [Focus Hint] 指示的方向提问。
2. **利用 Caption:** 答案必须能从 Image Caption 或图片本身的视觉属性中推断出来，不能产生幻觉。
3. **多样的提问层次:**
   - **L1 (感知类):** 询问图中有什么，颜色，位置等（针对实物图）。
   - **L2 (关联类):** 询问组件之间的关系，或者图表数据的对比（针对架构图/图表）。
   - **L3 (推理类):** 结合 Tag 的专业知识，询问图片背后的原理或功能（这是重点）。

# Constraints
- **答案风格:** 答案应简洁明了，直接回答问题，不要废话。
- **避免指代不明:** 问题中尽量包含具体的 Tag 词汇，避免仅使用 "这个物体" 或 "这张图"。
- **格式要求:** 输出必须是纯 JSON 格式。

# Output Format (JSON)
[
  {
    "question": "基于{tags}，这张图展示了什么关键特征？",
    "answer": "它展示了...",
    "type": "reasoning"
  },
  {
    "question": "图中的{specific_component}起到了什么作用？",
    "answer": "它的作用是...",
    "type": "factual"
  }
]
"""