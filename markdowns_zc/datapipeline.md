# 数据处理流水线 (Data Pipeline) 指南

## 1. 数据概览与准备

下表记录了直接输入的原始数据路径及其处理逻辑。

| 数据路径 | 描述与处理逻辑 |
| :--- | :--- |
| `/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/filter_qiuping_keword_deduplicated_CN_v1.0.0_decay_lc.json` | **1. 加载 JSON:** 包含图像及相关的上下文文本信息。<br>**2. 唯一来源:** 这是流水线需要准备的唯一原始数据源。 |

> **提示**：流水线的所有工作文件已记录并上传至 GitLab。请首先从 GitLab 下载代码。

---

## 2. 处理流程详解

### 第一步：准备数据并合并多图
此步骤需在阿里云环境执行。

**脚本路径**：`./caption_generation/merge_images.py`

**运行命令**：
```bash
python merge_images.py --base_file /mnt/storage/MLLM/karol/merge/xxx.json --out_root ./temp_files
```
*`--base_file` 指向初始输入数据表格中的 JSON 路径。*

**输出结构**：
```plaintext
./temp_files/
    pdf_name1/
        img1.png
        ...
    pdf_name2/
        img1.png
        ...
```

### 第二步：更新输入文件并替换图像路径
将原始 JSON 中的单图路径替换为合并后的图像路径。

```python
import os 
import json 

# 合并后图像的存放路径
merged_images_path = "/mnt/workspace/MLLM/karol/merge_sub_images/merged_subimages/max_selected"
# 原始 JSON 文件
existed_json_files = "filter_qiuping_keword_deduplicated_CN_v1.0.0_decay_lc.json"

def storage_processed_data(images_path, existed_json_files):
    storage_dict = {}
    # 构建图像映射表
    for dirs, _, files in os.walk(images_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                # ... 内部逻辑 ...
                pass
    
    json_data = json.load(open(existed_json_files, "r"))
    selected_data = []
    
    for sample in json_data:
        # 逻辑：如果图片数量 > 1，计算 Bbox 并匹配合并后的新图
        # ... 路径替换算法 ...
        if isinstance(sample["images"], str):
            selected_data.append(sample)
    
    return selected_data

print("正在处理合并后的图像路径...")
processed_data = storage_processed_data(merged_images_path, existed_json_files)
with open("./sub_merged_caption.json", "w") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
print("处理完成，结果已保存至 sub_merged_caption.json")
```

### 第三步：迁移图像至新目录
如果我们在云端处理图像，需要将其同步到本地磁盘供后续步骤使用。

```python
import json 
import os 
import shutil 
import tqdm

# 合并后的 JSON 路径（支持多个文件合并）
JSON_FILES = [
    "/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/temp/sub_merged_caption.json",
    "/mnt/storage/MLLM/karol/merge_sub_images/merged_subimages/temp/sub_merged_caption_2.json"
]

selected_img_path = "./img_selected"
os.makedirs(selected_img_path, exist_ok=True)

all_data = []
for p in JSON_FILES:
    all_data.extend(json.load(open(p)))

for sample in tqdm.tqdm(all_data):
    img_path = sample["images"][0]
    # 根据路径结构自动创建子目录
    img_copy_dir = img_path.split("/")[-2] if "max_selected" in img_path else img_path.split("/")[-3]
    dest_dir = os.path.join(selected_img_path, img_copy_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(dest_dir, os.path.basename(img_path))):
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

print("图像复制完成。")
```

---

## 3. 执行自动化流水线

### 阶段 A: 生成描述 (Caption)
1. **Doubao 基础描述生成**:
   ```bash
   python caption_generation/doubao_initial_caption.py --read_path ./sub_merged_caption.json --save_path ./caption_generation/temp/doubao_initial_caption.json 
   ```
2. **Gemini 优化与 Token 计数**:
   ```bash
   python caption_generation/pipeline_gemini_expand_count_tokens.py --data ./caption_generation/temp/doubao_initial_caption.json --img_path ./img_selected --save_path ./caption_generation/temp/gemini_refined_caption.json
   ```

### 阶段 B: 构建训练数据 (VQA)
1. **生成最终 VQA JSON**:
   ```bash
   python pipeline_gemini_build_vqa.py --input_path ./caption_generation/temp/gemini_refined_caption.json --output_path ./temp_images/vqa.json
   ```
2. **CoT (思维链) 蒸馏**:
   ```bash
   python distill_cot.py --api_key "YOUR_GOOGLE_API_KEY" --input_file ./temp_images/vqa.json --output_file ./temp_images/cot_distilled_output.json
   ```

**最终产物**: `./temp_images/cot_distilled_output.json` (包含 CoT 蒸馏的 VQA 数据集)
