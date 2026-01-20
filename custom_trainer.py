import torch
import torch.nn as nn
from trl import PPOTrainer, GRPOTrainer
from typing import Optional, Dict
from pareto_utils import (
    calculate_pareto_mask,
    calculate_pareto_mask_with_crowding,
    calculate_entropy_binned_mask_with_crowding,
    calculate_soft_pareto_mask,
    calculate_adaptive_threshold_mask,
    calculate_crowding_distance_vectorized,
)


def calculate_nsga_group_mask(
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    group_size: int,
    keep_ratio: float = 0.5,
    crowding_weight: float = 0.5,
) -> torch.Tensor:
    """
    在每个 group 内应用 NSGA-II 算法进行 token 筛选。
    
    NSGA-II 核心思想:
    1. 非支配排序: 将解分成不同的 Pareto 前沿层级
    2. 拥挤度计算: 在同一层级内，优先保留分布更稀疏的解
    
    Args:
        advantages: [B, T] 每个 token 的优势值
        entropy: [B, T] 每个 token 的熵值
        group_size: 每个 group 包含的样本数 (num_generations)
        keep_ratio: 每个 group 内保留的 token 比例
        crowding_weight: 拥挤度在排序中的权重
        
    Returns:
        mask: [B, T] 保留掩码，1 表示保留，0 表示过滤
    """
    B, T = advantages.shape
    device = advantages.device
    
    # 确保 B 可以被 group_size 整除
    num_groups = B // group_size
    if num_groups == 0:
        # 如果 batch 太小，将整个 batch 作为一个 group
        num_groups = 1
        group_size = B
    
    # 初始化结果 mask
    result_mask = torch.zeros(B, T, device=device, dtype=torch.float32)
    
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, B)
        
        # 获取当前 group 的数据
        group_adv = advantages[start_idx:end_idx]  # [group_size, T]
        group_ent = entropy[start_idx:end_idx]  # [group_size, T]
        
        # 将 group 内所有 token 展平
        flat_adv = group_adv.reshape(-1)  # [group_size * T]
        flat_ent = group_ent.reshape(-1)  # [group_size * T]
        
        n_tokens = flat_adv.numel()
        
        # 构建目标矩阵: [N, 2] - 我们同时最大化 advantage 和 entropy
        objectives = torch.stack([flat_adv, flat_ent], dim=1)
        
        # === NSGA-II Step 1: 非支配排序 ===
        # 计算 Pareto 等级 (简化: 只计算第一前沿)
        pareto_front = calculate_pareto_mask(flat_adv, flat_ent)
        
        # === NSGA-II Step 2: 拥挤度计算 ===
        crowding_dist = calculate_crowding_distance_vectorized(objectives)
        
        # 处理 inf 值
        crowding_dist = torch.where(
            torch.isinf(crowding_dist),
            torch.full_like(crowding_dist, crowding_dist[~torch.isinf(crowding_dist)].max() * 2 if (~torch.isinf(crowding_dist)).any() else 1.0),
            crowding_dist
        )
        
        # 归一化拥挤度到 [0, 1]
        if crowding_dist.max() > crowding_dist.min():
            crowding_dist_norm = (crowding_dist - crowding_dist.min()) / (crowding_dist.max() - crowding_dist.min() + 1e-8)
        else:
            crowding_dist_norm = torch.ones_like(crowding_dist)
        
        # === NSGA-II Step 3: 组合排序 ===
        # 排序优先级: Pareto前沿等级 > 拥挤度
        # 分数 = pareto_rank * large_const + crowding_distance
        combined_score = pareto_front * 1e6 + crowding_dist_norm * crowding_weight
        
        # Top-K 筛选
        k = max(1, int(n_tokens * keep_ratio))
        _, top_k_indices = torch.topk(combined_score, k)
        
        # 创建当前 group 的 mask
        group_mask_flat = torch.zeros(n_tokens, device=device, dtype=torch.float32)
        group_mask_flat[top_k_indices] = 1.0
        
        # 恢复形状并存入结果
        group_mask = group_mask_flat.view(end_idx - start_idx, T)
        result_mask[start_idx:end_idx] = group_mask
    
    return result_mask


class ParetoPPOTrainer(PPOTrainer):
    """
    在标准 PPO 基础上增加基于熵/优势的可配置筛选机制。

    参数:
    - loss_type: 'ppo'（默认，不筛选）、'pareto_crowding'、'entropy_binned'、'soft_pareto'、'adaptive'
    - pareto_kwargs: 传递给底层筛选函数的参数字典·
    """

    def __init__(self, *args, loss_type: str = "ppo", pareto_kwargs: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pareto_kwargs = pareto_kwargs or {}

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
        return_stats: bool = False,
    ):
        """
        重写 PPO 的 Loss 计算函数，并在其中可选地应用基于熵/优势的筛选策略。

        注意：
        - advantages, logprobs, old_logprobs, vpreds 的形状应为 [B, L]
        - logits 的形状为 [B, L, V]
        - 本实现会把 response 部分（advantages 对应部分）与 logits 的末尾对齐
        """

        # 1) 计算 token 级别熵：[-sum p log p]
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [B, L]

        # 2) 对齐 response 部分的熵（默认假设 advantages 对应 logits 的末端）
        try:
            gen_len = advantages.shape[1]
            resp_entropy = entropy[:, -gen_len:]
        except Exception:
            resp_entropy = entropy

        flat_adv = advantages.reshape(-1)
        flat_ent = resp_entropy.reshape(-1)

        # 初始化 mask 与拥挤度
        pareto_keep_mask = torch.ones_like(flat_adv, dtype=torch.float32, device=flat_adv.device)
        crowding_distance = torch.zeros_like(flat_adv, dtype=torch.float32, device=flat_adv.device)

        # 3) 根据 loss_type 调用不同筛选策略
        try:
            if self.loss_type == "pareto_crowding":
                keep_ratio = float(self.pareto_kwargs.get("keep_ratio", 0.5))
                crowding_weight = float(self.pareto_kwargs.get("crowding_weight", 0.3))
                mask_flat, cd = calculate_pareto_mask_with_crowding(flat_adv, flat_ent, keep_ratio=keep_ratio, crowding_weight=crowding_weight)
                pareto_keep_mask = mask_flat.to(flat_adv.device).float()
                crowding_distance = cd.to(flat_adv.device).float()

            elif self.loss_type == "entropy_binned":
                num_bins = int(self.pareto_kwargs.get("num_bins", 5))
                top_k_ratio = float(self.pareto_kwargs.get("top_k_ratio", 0.5))
                use_crowding = bool(self.pareto_kwargs.get("use_crowding", True))
                mask_flat, cd = calculate_entropy_binned_mask_with_crowding(flat_adv, flat_ent, num_bins=num_bins, top_k_ratio=top_k_ratio, use_crowding=use_crowding)
                pareto_keep_mask = mask_flat.to(flat_adv.device).float()
                crowding_distance = cd.to(flat_adv.device).float()

            elif self.loss_type == "soft_pareto":
                alpha = float(self.pareto_kwargs.get("alpha", 0.5))
                keep_ratio = float(self.pareto_kwargs.get("keep_ratio", 0.5))
                mask_flat = calculate_soft_pareto_mask(flat_adv, flat_ent, alpha=alpha, keep_ratio=keep_ratio)
                pareto_keep_mask = mask_flat.to(flat_adv.device).float()

            elif self.loss_type == "adaptive":
                low_pct = float(self.pareto_kwargs.get("low_entropy_adv_percentile", 70.0))
                high_pct = float(self.pareto_kwargs.get("high_entropy_adv_percentile", 30.0))
                ent_pct = float(self.pareto_kwargs.get("entropy_threshold_percentile", 50.0))
                mask_flat = calculate_adaptive_threshold_mask(flat_adv, flat_ent, low_entropy_adv_percentile=low_pct, high_entropy_adv_percentile=high_pct, entropy_threshold_percentile=ent_pct)
                pareto_keep_mask = mask_flat.to(flat_adv.device).float()

            else:
                # 'ppo' 或未知类型 -> 不筛选
                pareto_keep_mask = torch.ones_like(flat_adv, dtype=torch.float32, device=flat_adv.device)
        except Exception:
            # 如果任何函数发生异常，退回不筛选
            pareto_keep_mask = torch.ones_like(flat_adv, dtype=torch.float32, device=flat_adv.device)
            crowding_distance = torch.zeros_like(flat_adv, dtype=torch.float32, device=flat_adv.device)

        # 4) 恢复为 [B, L] 形状的 mask
        try:
            pareto_keep_mask_2d = pareto_keep_mask.view_as(advantages)
        except Exception:
            pareto_keep_mask_2d = pareto_keep_mask.reshape(advantages.shape)

        # 5) 应用筛选：将被过滤 token 的 advantage 置为 0
        filtered_advantages = advantages * pareto_keep_mask_2d

        # 6) 标准 PPO 损失计算
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -filtered_advantages * ratio
        pg_losses2 = -filtered_advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
        pg_loss = torch.max(pg_losses, pg_losses2).mean()

        # Value function loss（不对 Critic 做 Pareto 过滤）
        vpreds_clipped = values + torch.clamp(vpreds - values, -self.config.cliprange_value, self.config.cliprange_value)
        vf_losses1 = (vpreds - rewards) ** 2
        vf_losses2 = (vpreds_clipped - rewards) ** 2
        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

        loss = pg_loss + self.config.vf_coef * vf_loss

        # 7) 返回统计信息
        stats = {}
        if return_stats:
            try:
                kept_bool = pareto_keep_mask.bool()
                avg_ent = float(flat_ent[kept_bool].mean().item()) if kept_bool.any() else float('nan')
                avg_adv = float(flat_adv[kept_bool].mean().item()) if kept_bool.any() else float('nan')
                keep_ratio_actual = float(pareto_keep_mask.mean().item())
            except Exception:
                avg_ent = float('nan')
                avg_adv = float('nan')
                keep_ratio_actual = float(pareto_keep_mask.mean().item())

            stats = {
                "loss/policy": pg_loss.item(),
                "loss/value": vf_loss.item(),
                "pareto/keep_ratio": keep_ratio_actual,
                "pareto/avg_entropy": avg_ent,
                "pareto/avg_adv": avg_adv,
                "pareto/loss_type": self.loss_type,
            }

        return (loss, stats) if return_stats else loss


class ParetoGRPOTrainer(GRPOTrainer):
    """
    继承自 GRPOTrainer，支持 NSGA-II 算法进行 token 级别筛选。
    
    当 loss_type == "nsga" 时，在每个 generation group 内应用 NSGA-II:
    - 目标1: 最大化 |advantage| (利用高信息量样本)
    - 目标2: 最大化 entropy (保持探索多样性)
    
    使用 Pareto 非支配排序 + 拥挤度距离来选择 tokens。
    
    参数:
    - nsga_keep_ratio: NSGA 筛选保留的 token 比例 (默认 0.5)
    - nsga_crowding_weight: 拥挤度权重 (默认 0.5)
    - nsga_use_abs_advantage: 是否使用 |advantage| 作为目标 (默认 True)
    """
    
    def __init__(
        self,
        *args,
        nsga_keep_ratio: float = 0.5,
        nsga_crowding_weight: float = 0.5,
        nsga_use_abs_advantage: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.nsga_keep_ratio = nsga_keep_ratio
        self.nsga_crowding_weight = nsga_crowding_weight
        self.nsga_use_abs_advantage = nsga_use_abs_advantage
        
    def _compute_loss(self, model, inputs):
        """
        重写 _compute_loss 方法，支持 NSGA-II token 筛选。
        """
        # 获取基础输入
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]

        # 计算 per_token_logps 和 entropy
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # 高熵 mask (保持与原实现兼容)
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # 获取 advantages
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # 获取 old_per_token_logps
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Off-policy mask
        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        # 计算 log ratio 和 importance weights
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # ========== NSGA-II Token Filtering ==========
        nsga_mask = None
        if self.loss_type == "nsga":
            # 准备 NSGA 的目标
            # 目标1: advantage (或 |advantage|)
            if self.nsga_use_abs_advantage:
                nsga_adv = advantages.abs()
            else:
                nsga_adv = advantages
            
            # 扩展 advantages 到 token 维度 (如果需要)
            if nsga_adv.shape[1] == 1:
                nsga_adv = nsga_adv.expand(-1, entropies.shape[1])
            
            # 目标2: entropy
            nsga_ent = entropies
            
            # 计算 NSGA mask
            # num_generations 通常可以从 args 获取，如果没有就使用整个 batch
            group_size = getattr(self.args, 'num_generations', inputs.get('num_generations', 1))
            if group_size is None or group_size <= 0:
                group_size = 1
                
            nsga_mask = calculate_nsga_group_mask(
                advantages=nsga_adv,
                entropy=nsga_ent,
                group_size=group_size,
                keep_ratio=self.nsga_keep_ratio,
                crowding_weight=self.nsga_crowding_weight,
            )
            
            # 结合 completion_mask
            nsga_mask = nsga_mask * mask.float()

        # 计算 per_token_loss (使用基础 GRPO 公式)
        if self.loss_type == "nsga":
            # NSGA 模式下使用 GRPO 的 loss 公式
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 应用各种 masks
        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # 应用 NSGA mask
        if nsga_mask is not None:
            per_token_loss = per_token_loss * nsga_mask
            # 更新 mask 用于后续 loss 计算
            effective_mask = nsga_mask
        else:
            effective_mask = mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # 计算最终 loss
        if self.loss_type in ["grpo", "sapo", "nsga"]:
            # NSGA 使用与 GRPO 相同的归一化方式
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # ========== Logging ==========
        mode = "train" if self.model.training else "eval"
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # NSGA 特定的 metrics
        if self.loss_type == "nsga" and nsga_mask is not None:
            nsga_keep_ratio = nsga_mask.sum() / mask.sum().clamp(min=1.0)
            self._metrics[mode]["nsga/keep_ratio"].append(
                self.accelerator.gather(nsga_keep_ratio).nanmean().item()
            )
            # 记录被选中 token 的平均熵和优势
            selected_entropy = (entropies * nsga_mask).sum() / nsga_mask.sum().clamp(min=1.0)
            self._metrics[mode]["nsga/selected_entropy"].append(
                self.accelerator.gather(selected_entropy).nanmean().item()
            )

        # Clip ratio metrics (与原实现保持一致)
        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "nsga"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(self._nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(self._nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    @staticmethod
    def _nanmin(tensor):
        """Safe nanmin that handles empty tensors."""
        valid = tensor[~torch.isnan(tensor)]
        if valid.numel() == 0:
            return torch.tensor(float('nan'))
        return valid.min()

    @staticmethod
    def _nanmax(tensor):
        """Safe nanmax that handles empty tensors."""
        valid = tensor[~torch.isnan(tensor)]
        if valid.numel() == 0:
            return torch.tensor(float('nan'))
        return valid.max()
