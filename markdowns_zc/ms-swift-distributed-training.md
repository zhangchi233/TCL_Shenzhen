# 如何运行 ms-swift + Megatron 进行模型训练

## 1. 环境准备

### 方案 1：使用现有的 Docker 镜像
1. 从注册表拉取 Docker 镜像：
   - 确保宿主机已安装 NVIDIA Container Toolkit。
   - 运行 `docker pull` 拉取指定镜像。

---

## 2. 分布式训练指南

### 方案 1：为分布式训练准备模型格式
在 ms-swift 中，模型格式主要分为两类：

1. **HuggingFace (HF) 格式**：
   - 最通用的预训练模型格式。
   - **适用场景**：通过数据并行（Data Parallelism）进行训练。
   - **核心提示**：如果您使用 DeepSpeed ZeRO 1/2/3 或 FSDP，它们都属于数据并行后端。在单节点上通过 DeepSpeed/FSDP 拆分模型通常已足够，无需转换格式，且通信开销更小。

2. **Megatron 格式**：
   - **适用场景**：如果您需要运行流水线并行（PP）、张量并行（TP）或专家并行（EP），必须将 HF 格式转换为 Megatron 格式。
   - **MoE 模型推荐**：训练 MoE 模型时，强烈建议开启专家并行以减少 GPU 气泡时间，此时必须进行转换。

#### 模型格式转换步骤
- **从 HuggingFace 转换为 Megatron**：
```bash
#!/usr/bin/env bash
set -e

swift export \
    --model /mnt/storage/models/OpenGVLab/InternVL3_5-241B-A28B \
    --output_dir /mnt/storage/MLLM/zc/megatron_output/InternVL3_5-241B-A28B
```

- **训练后转回 HuggingFace 格式**：
```bash
swift export \
    --mcore_model megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx \
    --test_convert_precision true
```

*注意：* 使用 LoRA 训练时，Megatron 存储的是 LoRA 权重。需先合并权重再导出：
```bash
# Megatron 导出 (LoRA 合并)
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron export \
    --adapter_load megatron_output/Qwen2.5-7B-Instruct/vx-xxx/checkpoint-xxx
```

---

### 方案 2：运行分布式训练命令示例

#### A. 使用 DeepSpeed/FSDP 后端 (无需转换格式)
```bash
# 示例：8 * 70GiB 显存配置
IMAGE_MAX_TOKEN_NUM=1024 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /mnt/storage/models/Qwen3/Qwen3_VL235B-thinking \
    --strict False
```

| 参数 | 说明 |
| :--- | :--- |
| `IMAGE_MAX_TOKEN_NUM` | 图像输入的最大 Token 数量 |
| `CUDA_VISIBLE_DEVICES` | 指定参与训练的 GPU 编号 |
| `swift sft` | 启动 Swift 监督微调的命令 |
| `--model` | 预训练模型的存储路径 |
| `--train_type` | 训练类型（例如：lora） |
| `--torch_dtype` | 训练使用的数值精度（如 bfloat16） |
| `--attn_impl` | 注意力机制实现（建议用 flash_attn） |
| `--gradient_checkpointing` | 是否启用梯度检查点（见下方说明） |

#### B. 使用 Megatron 后端 (需转换格式)
```bash
# 示例：双节点分布式训练
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=4096 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nnodes=2 \
nproc_per_node=8 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_ADDR=10.10.9.97 \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
megatron sft \
    --load /mnt/storage/MLLM/zc/megatron_output/Qwen3_VL235B-thinking/v3-20250919-184145-mcore \
    --attention_backend flash
```

---

### 技术要点说明：梯度检查点 (Gradient Checkpointing)
参数 `--gradient_checkpointing true` 能有效**降低显存消耗**。
- **原理**：在前向传播时不保存中间激活值，而在反向传播时重新计算。
- **权衡**：以额外的计算时间换取巨大的显存节省，使在有限资源下训练大型模型成为可能。