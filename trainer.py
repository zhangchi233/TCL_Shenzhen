import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset

# === 关键引入：导入我们在 Step 2 写的自定义 Trainer ===
from custom_trainer import ParetoPPOTrainer

def train():
    # 1. 配置参数
    config = PPOConfig(
        model_name="gpt2",          # 演示用小模型，方便快速调试
        learning_rate=1.41e-5,
        batch_size=16,              # 如果显存不够，改小这个
        mini_batch_size=4,          # PPO 更新时的微批次
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
    )

    # 2. 加载模型和分词器
    # AutoModelForCausalLMWithValueHead 会自动在模型顶层加一个 Value Head 用于 RL
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # GPT2 需要手动设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 准备奖励模型 (Reward Model)
    # 这里我们用一个现成的情感分析 pipeline 作为 Reward Model
    # 目标：让模型生成的句子尽可能 Positive
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=0 if torch.cuda.is_available() else -1)

    # 4. 准备数据
    # 使用 IMDB 数据集的前 100 条作为 Prompt 演示
    dataset = load_dataset("imdb", split="train")
    
    # 简单的 filter：只取长一点的句子，截取前几个词作为 Prompt
    def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
        ds = load_dataset(dataset_name, split="train")
        ds = ds.filter(lambda x: len(x["text"]) > 200, batched=False)
        input_size = list(range(input_min_text_length, input_max_text_length))
        
        def tokenize(sample):
            # 随机截取一段作为 Prompt
            sample["input_ids"] = tokenizer.encode(sample["text"])[: input_max_text_length]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    dataset = build_dataset(config)

    # 5. 初始化我们的 ParetoPPOTrainer
    # 注意：这里使用的是我们自定义的类，而不是 trl 原生的 PPOTrainer
    trainer = ParetoPPOTrainer(
        config=config,
        model=model,
        ref_model=None, # trl 会自动复制一个 ref_model
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0]),
    )

    # 6. 开始训练循环
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20, # 生成长度
    }

    print("=== 开始训练: Pareto-PPO ===")
    
    # 模拟 10 个 Step 的训练
    for epoch, batch in tqdm(enumerate(trainer.dataloader)):
        if epoch >= 10: break # 演示只跑 10 步
        query_tensors = batch["input_ids"]

        # A. 生成回复 (Rollout)
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        
        # 解码生成的文本用于计算 Reward
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        batch["query"] = [tokenizer.decode(q.squeeze()) for q in query_tensors]
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        # B. 计算 Reward
        # pipe 输出: [{'label': 'POSITIVE', 'score': 0.9}, ...]
        pipe_outputs = sentiment_pipe(texts, top_k=None, function_to_apply="none")
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs] # 取 Positive 的分数作为 Reward
        
        # C. PPO 更新 Step (这里会调用我们在 custom_trainer 中写的 loss 函数)
        # 我们的 Pareto 筛选逻辑会在 trainer.step 内部触发
        stats = trainer.step(query_tensors, response_tensors, rewards)
        
        # D. 打印关键日志
        # 重点观察 'pareto/keep_ratio'
        if epoch % 2 == 0:
            print(f"\nStep {epoch}:")
            print(f"  Total Loss: {stats['ppo/loss/total']:.4f}")
            print(f"  Mean Reward: {stats['ppo/mean_scores']:.4f}")
            # 下面这个指标显示了 NSGA-II 到底过滤了多少 Token
            if 'pareto/keep_ratio' in stats:
                print(f"  Pareto Keep Ratio: {stats['pareto/keep_ratio']:.2%}")
            print("-" * 30)

    # 保存模型
    trainer.save_pretrained("./pareto_gpt2_model")
    print("训练完成！模型已保存。")

if __name__ == "__main__":
    train()