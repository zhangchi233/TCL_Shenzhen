#!/bin/bash
# Test script for custom reward with GSM8K dataset
# Uses conditional probability - uniform baseline as reward

set -x

# ============= DATA PATHS =============
# UPDATE THESE PATHS TO YOUR GSM8K DATA LOCATION
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

# Verify data exists
if [ ! -f "$gsm8k_train_path" ]; then
    echo "ERROR: Training data not found at $gsm8k_train_path"
    echo "Please update gsm8k_train_path in this script"
    exit 1
fi

if [ ! -f "$gsm8k_test_path" ]; then
    echo "ERROR: Test data not found at $gsm8k_test_path"
    echo "Please update gsm8k_test_path in this script"
    exit 1
fi

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# ============= MODEL & TRAINING CONFIG =============
# Using a smaller model for testing (adjust as needed)
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"  # or your preferred model

# Small batch sizes for testing
TRAIN_BATCH_SIZE=64        # Number of prompts per batch
N_ROLLOUTS=4               # Number of responses per prompt
MINI_BATCH_SIZE=64         # Mini-batch size for actor updates (64 = 16 prompts * 4 rollouts)

# Training parameters
TOTAL_EPOCHS=5             # Short test run
TEST_FREQ=1                # Validate every epoch
SAVE_FREQ=5                # Save checkpoint frequency

# GPU configuration (adjust based on your setup)
N_GPUS_PER_NODE=1          # Number of GPUs you have
NNODES=1

# ============= RUN TRAINING =============
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='test_custom_reward' \
    trainer.experiment_name='gsm8k_conditional_prob_reward' \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    custom_reward_function.path=verl/trainer/ppo/dummy_reward.py \
    custom_reward_function.name=dummy_reward \
    use_custom_reward_with_logprobs=True \
    trainer.default_local_dir=checkpoints/test_custom_reward_gsm8k $@

echo ""
echo "============================================"
echo "Training completed!"
echo "Checkpoints saved to: checkpoints/test_custom_reward_gsm8k"
echo "============================================"
