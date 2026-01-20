#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NSGA-II GRPO Training Script for Qwen3-4B on DAPO Math Dataset

使用 ParetoGRPOTrainer 训练 Qwen3-4B 模型，支持以下 loss_type:
- grpo: 标准 GRPO
- nsga: NSGA-II 算法进行 token 级别筛选
"""

import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from custom_trainer import ParetoGRPOTrainer

# ==================== 配置参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-4B with NSGA-II GRPO on DAPO Math")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base",
                        help="模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_nsga_grpo",
                        help="输出目录")
    
    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default="BytedTsinghua-SIA/DAPO-Math-17k",
                        help="数据集名称")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数 (用于调试)")
    
    # 训练配置
    parser.add_argument("--loss_type", type=str, default="nsga",
                        choices=["grpo", "nsga", "bnpo", "dr_grpo", "dapo"],
                        help="损失函数类型")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每个设备的训练 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="学习率")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="最大生成长度")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 生成的样本数")
    
    # NSGA 配置
    parser.add_argument("--nsga_keep_ratio", type=float, default=0.5,
                        help="NSGA 保留的 token 比例")
    parser.add_argument("--nsga_crowding_weight", type=float, default=0.5,
                        help="NSGA 拥挤度权重")
    parser.add_argument("--nsga_use_abs_advantage", action="store_true", default=True,
                        help="是否使用 |advantage| 作为目标")
    
    # GRPO 配置
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL penalty 系数")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip epsilon")
    
    # 其他配置
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="模型保存步数")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用 Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="nsga-grpo",
                        help="WandB 项目名")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="是否使用 bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="是否使用梯度检查点")
    
    return parser.parse_args()


# ==================== 奖励函数 ====================

def math_reward_function(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    数学问题奖励函数
    
    评估生成的答案是否正确。这里使用简化版本，
    实际应用中可以接入更复杂的数学验证器。
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # 简化版: 检查是否包含数字答案
        # 实际应用中应该使用 sympy 或其他数学验证工具
        reward = 0.0
        
        # 检查是否有合理的输出格式
        if "answer" in completion.lower() or "=" in completion:
            reward += 0.3
        
        # 检查是否有数学表达式
        if any(op in completion for op in ["+", "-", "*", "/", "^", "sqrt"]):
            reward += 0.2
        
        # 检查是否有最终答案 (boxed 格式)
        if "\\boxed" in completion or "答案" in completion:
            reward += 0.5
        
        rewards.append(reward)
    
    return rewards


def accuracy_reward(prompts: list, completions: list, ground_truths: list = None, **kwargs) -> list[float]:
    """
    基于正确率的奖励函数
    
    如果数据集提供了 ground truth，则进行精确匹配。
    """
    if ground_truths is None:
        return math_reward_function(prompts, completions)
    
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # 提取数字答案
        import re
        
        # 尝试从 completion 中提取答案
        completion_nums = re.findall(r'-?\d+\.?\d*', completion)
        gt_nums = re.findall(r'-?\d+\.?\d*', str(gt))
        
        if completion_nums and gt_nums:
            # 比较最后一个数字 (通常是最终答案)
            try:
                pred = float(completion_nums[-1])
                target = float(gt_nums[-1])
                if abs(pred - target) < 1e-6:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    return rewards


# ==================== 数据预处理 ====================

def prepare_dataset(dataset_name: str, tokenizer, max_samples: int = None):
    """
    加载并预处理 DAPO Math 数据集
    """
    print(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Failed to load {dataset_name}, trying alternative...")
        # 备选数据集
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    
    def format_prompt(example):
        """将数据格式化为对话格式"""
        # 根据数据集结构调整字段名
        if "prompt" in example:
            question = example["prompt"]
        elif "question" in example:
            question = example["question"]
        elif "problem" in example:
            question = example["problem"]
        else:
            question = str(example.get("input", ""))
        
        # 构建 prompt
        prompt = f"请解决以下数学问题，给出详细的解题步骤和最终答案。\n\n问题: {question}\n\n解答:"
        
        return {"prompt": prompt}
    
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    return dataset


# ==================== 主训练函数 ====================

def main():
    args = parse_args()
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # 加载 tokenizer
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"Loading model from {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 使用 Flash Attention 2
    )
    
    # 准备数据集
    train_dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        max_samples=args.max_samples,
    )
    
    # 配置 GRPO
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        
        # 训练参数
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        
        # GRPO 特定参数
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        
        # 优化参数
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        
        # 日志和保存
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # 其他
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        
        # 生成参数
        temperature=0.7,
        top_p=0.9,
    )
    
    # 创建 Trainer
    print(f"Creating ParetoGRPOTrainer with loss_type={args.loss_type}")
    trainer = ParetoGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=math_reward_function,
        
        # NSGA 参数
        nsga_keep_ratio=args.nsga_keep_ratio,
        nsga_crowding_weight=args.nsga_crowding_weight,
        nsga_use_abs_advantage=args.nsga_use_abs_advantage,
    )
    
    # 开始训练
    print("=" * 50)
    print(f"Starting training with {args.loss_type} loss")
    print(f"  - Model: {args.model_name}")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Epochs: {args.num_train_epochs}")
    print(f"  - Batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    if args.loss_type == "nsga":
        print(f"  - NSGA keep ratio: {args.nsga_keep_ratio}")
        print(f"  - NSGA crowding weight: {args.nsga_crowding_weight}")
    print("=" * 50)
    
    trainer.train()
    
    # 保存最终模型
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
