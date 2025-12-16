# vllm serve --config /mnt/workspace/MLLM/zc/zyf/vllm_config.yaml --allowed-local-media-path /mnt/workspace/MLLM/zc/tclreasoning/data/testoutput/dataset --tensor-parallel-size 4 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# lmdeploy serve api_server /mnt/storage/models/OpenGVLab/InternVL3_5-241B-A28B --server-port 8848 --tp 8 --backend pytorch 


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /mnt/storage/models/OpenGVLab/InternVL3_5-241B-A28B \
#   --served-model-name InternVL3_5-241B-A28B \
#   --dtype bfloat16 \
#   --port 8848 \
#   --gpu_memory_utilization 0.95 \
#   --tensor-parallel-size 8 \
#   --pipeline-parallel-size 1 \
#   --trust_remote_code

python -m vllm.entrypoints.openai.api_server \
  --model /mnt/storage/models/Qwen3/Qwen3-VL-thinking \
  --served-model-name Qwen3-VL-thinking \
  --tensor-parallel-size 8 \
  --port 8848 \
  --max_model_len 32000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.97 

# python -m vllm.entrypoints.openai.api_server --served-model-name U_Ground_v1_72B --model /mnt/storage/zc/zyf/models/U_Ground_v1 --dtype float16 \
#   --tensor-parallel-size 4 \
#   --port 8848 \
#   --gpu-memory-utilization 0.90 --chat-template internvl2_5

# python -m vllm.entrypoints.openai.api_server \
#   --model /mnt/storage/models/Qwen3/Qwen3_VL235B-thinking \
#   --served-model-name Qwen3_VL235B-thinking \
#   --tensor-parallel-size 8 \
#   --enable-expert-parallel \
#   --port 8848 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 9000