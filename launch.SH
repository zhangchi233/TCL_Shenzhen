model_name=/mnt/storage/models/Qwen3/Qwen3-4B
dataset_name=/mnt/data/MLLM/zc/TCL_Shenzhen/dataset


accelerate launch train_qwen3_nsga.py --loss_type nsga\
    --report_to tensorboard\
    --model_name_or_path $model_name\
    --dataset_name $dataset_name --output_dir ./nsgarl_output \
    --num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --evaluation_strategy steps --eval_steps 500 --save_steps 500 --logging_steps 5 \
    --learning_rate 2e-5 --fp16 --seed 42 --use_wandb False --use_vllm True --vllm_device cuda \
    --vllm_gpu_memory_utilization 0.9 --vllm_dtype float16 --vllm_max_model_len 8192 \
    --vllm_enable_prefix_caching True --vllm_tensor_parallel_size 1
