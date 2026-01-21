model_name=/mnt/storage/models/Qwen3/Qwen3-4B
dataset_name=/mnt/data/MLLM/zc/TCL_Shenzhen/dataset


accelerate launch train_qwen3_nsga.py --loss_type nsga\
    --model_name $model_name\
    --dataset_name $dataset_name --output_dir ./nsgarl_output \
   