import torch
import torch.nn as nn
from trl import PPOTrainer
from pareto_utils import calculate_pareto_mask

class ParetoPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def loss(
        self,
        old_logprobs,
        values,
        rewards,
        logits,
        vpreds,
        logprobs,
        mask,
        advantages,
        return_stats=False,
    ):
        """
        重写 PPO 的 Loss 计算函数，植入 NSGA-II 筛选逻辑。
        """
        
        # 1. === 创新点开始：计算 Token 级别的 Entropy ===
        # logits shape: [Batch, Seq_Len, Vocab]
        probs = torch.softmax(logits, dim=-1)
        # 计算熵: -sum(p * log(p))
        # 加上 1e-8 防止 log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) 
        # entropy shape 现在是 [Batch, Seq_Len]
        
        # 2. === 创新点核心：计算 Pareto Mask ===
        # 我们需要展平 tensor 来进行比较，或者按 batch 处理
        # pareto_utils 中的函数期望的是展平后的 vector
        # 注意：advantages 和 entropy 的形状必须对齐
        
        # 为了防止维度对齐问题，我们先取 slice。
        # PPO 训练时 logits 包含了 prompt+response，但 advantages 通常只对应 response 部分。
        # trl 的逻辑中，传入的 advantages 已经对齐了 response 长度。
        # 我们需要截取 entropy 的后半部分（Response 部分）
        
        gen_len = advantages.shape[1]
        resp_entropy = entropy[:, -gen_len:]
        
        # 展平输入到我们的筛选器
        flat_adv = advantages.flatten()
        flat_ent = resp_entropy.flatten()
        
        # 调用我们在 Step 1 写的工具
        pareto_keep_mask = calculate_pareto_mask(flat_adv, flat_ent)
        
        # 恢复形状 [Batch, Seq_Len]
        pareto_keep_mask = pareto_keep_mask.view_as(advantages)
        
        # 3. === 创新点应用：修改 Advantages ===
        # 策略：如果一个 Token 被支配（mask=0），我们将其 Advantage 置为 0。
        # 这样 PPO Loss (ratio * adv) 就会变成 0，模型不会学习这个 Token。
        # 同时我们保留原始 mask (padding mask) 的作用
        
        filtered_advantages = advantages * pareto_keep_mask
        
        # === 创新点结束，以下是标准 PPO Loss 计算逻辑 ===
        
        # 计算 PPO ratio
        # logprobs 和 old_logprobs 都是 [Batch, Seq_Len]
        ratio = torch.exp(logprobs - old_logprobs)
        
        # PPO Clipped Loss
        pg_losses = -filtered_advantages * ratio
        pg_losses2 = -filtered_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
        
        # 取最大值（因为前面加了负号，实际上是取 min）
        pg_loss = torch.max(pg_losses, pg_losses2).mean()
        
        # Value Function Loss (这部分不需要 Pareto 过滤，Critic 应该学习所有真实情况)
        vpreds_clipped = values + torch.clamp(
            vpreds - values, -self.config.cliprange_value, self.config.cliprange_value
        )
        vf_losses1 = (vpreds - rewards) ** 2
        vf_losses2 = (vpreds_clipped - rewards) ** 2
        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
        
        # 总 Loss
        loss = pg_loss + self.config.vf_coef * vf_loss
        
        # 统计数据（可选，用于 WandB 监控）
        stats = {}
        if return_stats:
            stats = {
                "loss/policy": pg_loss.item(),
                "loss/value": vf_loss.item(),
                "pareto/keep_ratio": pareto_keep_mask.mean().item(), # 监控有多少比例的 token 被保留了
                "pareto/avg_entropy": flat_ent[pareto_keep_mask.bool().flatten()].mean().item(), # 保留样本的平均熵
                "pareto/avg_adv": flat_adv[pareto_keep_mask.bool().flatten()].mean().item(), # 保留样本的平均优势
            }

        return (loss, stats) if return_stats else loss