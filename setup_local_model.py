import json
from pathlib import Path

# ================= 核心配置 =================
# 注意：这里去掉了末尾的 /models
real_model_path = "/mnt/storage/models/mineru/models--opendatalab--PDF-Extract-Kit-1.0/snapshots/cce3a518cac69a48c626996d49e7b315eb7dce10"
# ==========================================

config_content = {
    "models-dir": real_model_path,
    "device-mode": "cuda",
    "table-config": {
        "model": "TableMaster",
        "device": "cuda",
        "table_max_len": 480
    },
    "layout-config": {
        "model": "layoutlmv3",
        "device": "cuda"
    },
    "formula-config": {
        "mfr_batch_size": 1,
        "device": "cuda"
    }
}

# 写入配置文件到用户主目录
config_path = Path.home() / "magic-pdf.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_content, f, indent=4)

print(f"✅ 配置文件已生成: {config_path}")
print(f"📂 模型根目录指向: {real_model_path}")


