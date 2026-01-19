import torch
from typing import Optional, Tuple


def calculate_crowding_distance(
    objectives: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算拥挤度 (Crowding Distance)，用于衡量解在目标空间中的分布密度。
    
    拥挤度越大，说明该解周围越"空旷"，多样性越好，应该优先保留。
    
    Args:
        objectives: [N, M] 目标函数矩阵，N 个解，M 个目标
        mask: [N] 可选，只对 mask=1 的点计算拥挤度
        
    Returns:
        crowding_distance: [N] 每个点的拥挤度
    """
    N, M = objectives.shape
    device = objectives.device
    
    crowding_distance = torch.zeros(N, device=device, dtype=torch.float32)
    
    if mask is not None:
        valid_indices = torch.where(mask > 0)[0]
        if valid_indices.numel() == 0:
            return crowding_distance
    else:
        valid_indices = torch.arange(N, device=device)
    
    n_valid = valid_indices.numel()
    
    if n_valid <= 2:
        # 边界情况：点太少，全部设为无穷大
        crowding_distance[valid_indices] = float('inf')
        return crowding_distance
    
    valid_objectives = objectives[valid_indices]  # [n_valid, M]
    cd_local = torch.zeros(n_valid, device=device, dtype=torch.float32)
    
    for m in range(M):
        # 按第 m 个目标排序
        sorted_local_idx = torch.argsort(valid_objectives[:, m])
        sorted_obj_m = valid_objectives[sorted_local_idx, m]
        
        # 目标范围 (用于归一化)
        obj_range = sorted_obj_m[-1] - sorted_obj_m[0]
        if obj_range < 1e-10:
            continue  # 该目标无区分度
        
        # 边界点设为无穷大
        cd_local[sorted_local_idx[0]] = float('inf')
        cd_local[sorted_local_idx[-1]] = float('inf')
        
        # 中间点：累加邻居距离
        for i in range(1, n_valid - 1):
            dist = (sorted_obj_m[i + 1] - sorted_obj_m[i - 1]) / obj_range
            cd_local[sorted_local_idx[i]] += dist
    
    crowding_distance[valid_indices] = cd_local
    return crowding_distance


def calculate_crowding_distance_vectorized(
    objectives: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    拥挤度的向量化实现 (更高效)。
    
    Args:
        objectives: [N, M] 目标函数矩阵
        mask: [N] 可选掩码
        
    Returns:
        crowding_distance: [N]
    """
    N, M = objectives.shape
    device = objectives.device
    
    crowding_distance = torch.zeros(N, device=device, dtype=torch.float32)
    
    if mask is not None:
        valid_indices = torch.where(mask > 0)[0]
        if valid_indices.numel() == 0:
            return crowding_distance
    else:
        valid_indices = torch.arange(N, device=device)
    
    n_valid = valid_indices.numel()
    
    if n_valid <= 2:
        crowding_distance[valid_indices] = float('inf')
        return crowding_distance
    
    valid_objectives = objectives[valid_indices]  # [n_valid, M]
    cd_local = torch.zeros(n_valid, device=device, dtype=torch.float32)
    
    for m in range(M):
        sorted_local_idx = torch.argsort(valid_objectives[:, m])
        sorted_obj_m = valid_objectives[sorted_local_idx, m]
        # the shape of sorted_obj_m is [n_valid]
        obj_range = sorted_obj_m[-1] - sorted_obj_m[0]
        if obj_range < 1e-10:
            continue
        
        # 向量化计算中间点距离
        distances = torch.zeros(n_valid, device=device)
        distances[0] = float('inf')
        distances[-1] = float('inf')
        distances[1:-1] = (sorted_obj_m[2:] - sorted_obj_m[:-2]) / obj_range
        # the distances is equal to distances[i] = (sorted_obj_m[i + 1] - sorted_obj_m[i - 1]) / obj_range
        # 反向映射到原始顺序
        inv_idx = torch.argsort(sorted_local_idx)
        cd_local += distances[inv_idx]
    
    crowding_distance[valid_indices] = cd_local
    return crowding_distance


def calculate_pareto_mask_with_crowding(
    advantages: torch.Tensor, 
    entropy: torch.Tensor,
    keep_ratio: float = 0.5,
    crowding_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    结合 Pareto 前沿 + 拥挤度的筛选策略。
    
    流程:
    1. 计算 Pareto 前沿 (非支配解)
    2. 计算拥挤度
    3. 根据 (Pareto等级, 拥挤度) 进行排序筛选
    
    Args:
        advantages: [N] 优势函数 (目标1: 最大化)
        entropy:    [N] 熵 (目标2: 最大化，鼓励探索)
        keep_ratio: 保留比例
        crowding_weight: 拥挤度在最终排序中的权重
        
    Returns:
        mask: [N] 保留掩码
        crowding_distance: [N] 拥挤度值
    """
    N = advantages.shape[0]
    device = advantages.device
    
    # 构建目标矩阵 [N, 2]
    objectives = torch.stack([advantages, entropy], dim=1)
    
    # === Step 1: 计算 Pareto 等级 (简化版: 只区分前沿/非前沿) ===
    pareto_mask = calculate_pareto_mask(advantages, entropy)
    
    # === Step 2: 计算拥挤度 ===
    crowding_distance = calculate_crowding_distance_vectorized(objectives, mask=pareto_mask)
    # crowding_distance algorithm is:
    # 1. For each objective, sort the points.
    # 2. For each point, calculate the normalized distance between its two neighbors.
    # 3. Sum the distances over all objectives to get the final crowding distance.
    
    # === Step 3: 组合排序 ===
    # 优先级: Pareto前沿 > 拥挤度大
    # 分数 = pareto_rank * large_const + crowding_distance
    combined_score = pareto_mask * 1e6 + crowding_distance * crowding_weight
    
    # 处理 inf 值
    combined_score = torch.where(
        torch.isinf(combined_score),
        torch.full_like(combined_score, 1e8),
        combined_score
    )
    
    # Top-K 筛选
    k = max(1, int(N * keep_ratio))
    _, top_k_indices = torch.topk(combined_score, k)
    
    keep_mask = torch.zeros(N, device=device, dtype=torch.float32)
    keep_mask[top_k_indices] = 1.0
    
    return keep_mask, crowding_distance


def calculate_pareto_mask(advantages: torch.Tensor, entropy: torch.Tensor, domination_threshold: float = 0.0):
    """
    原始的 Pareto 支配掩码计算（保留用于对比）
    """
    points = torch.stack([advantages, entropy], dim=1)
    N = points.shape[0]
    keep_mask = torch.ones(N, device=advantages.device, dtype=torch.float32)
    sorted_idx = torch.argsort(advantages, descending=True)
    sorted_ent = entropy[sorted_idx]
    current_max_ent = -float('inf')
    is_dominated_ordered = torch.zeros(N, dtype=torch.bool, device=advantages.device)
    for i in range(N):
        if sorted_ent[i] < current_max_ent:
            is_dominated_ordered[i] = True
        else:
            current_max_ent = sorted_ent[i]
    is_dominated = torch.zeros(N, dtype=torch.bool, device=advantages.device)
    is_dominated[sorted_idx] = is_dominated_ordered
    keep_mask = (~is_dominated).float()
    return keep_mask


def calculate_entropy_binned_mask_with_crowding(
    advantages: torch.Tensor, 
    entropy: torch.Tensor,
    num_bins: int = 5,
    top_k_ratio: float = 0.5,
    use_crowding: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Entropy 分箱 + 拥挤度增强版。
    
    在每个 bin 内，不仅考虑 advantage 排名，还考虑拥挤度以保持多样性。
    
    Args:
        advantages: [N] 优势函数
        entropy:    [N] 熵
        num_bins:   分箱数量
        top_k_ratio: 每个 bin 保留比例
        use_crowding: 是否使用拥挤度
        
    Returns:
        mask: [N]
        crowding_distance: [N]
    """
    N = advantages.shape[0]
    device = advantages.device
    
    keep_mask = torch.zeros(N, device=device, dtype=torch.float32)
    crowding_distance = torch.zeros(N, device=device, dtype=torch.float32)
    
    # 计算全局拥挤度
    if use_crowding:
        objectives = torch.stack([advantages, entropy], dim=1)
        crowding_distance = calculate_crowding_distance_vectorized(objectives)
    
    # 分箱
    entropy_quantiles = torch.linspace(0, 1, num_bins + 1, device=device)[1:-1]
    ent_boundaries = torch.quantile(entropy, entropy_quantiles)
    bin_ids = torch.bucketize(entropy, ent_boundaries)
    
    for bin_id in range(num_bins):
        bin_mask = (bin_ids == bin_id)
        bin_indices = torch.where(bin_mask)[0]
        
        if bin_indices.numel() == 0:
            continue
        
        bin_advantages = advantages[bin_indices]
        bin_crowding = crowding_distance[bin_indices]
        
        # 动态调整保留比例
        dynamic_ratio = top_k_ratio * (1 + 0.15 * bin_id)
        dynamic_ratio = min(dynamic_ratio, 1.0)
        k = max(1, int(bin_indices.numel() * dynamic_ratio))
        
        if use_crowding:
            # 归一化后组合
            norm_adv = (bin_advantages - bin_advantages.min()) / (bin_advantages.max() - bin_advantages.min() + 1e-8)
            norm_cd = bin_crowding / (bin_crowding.max() + 1e-8)
            norm_cd = torch.where(torch.isinf(norm_cd), torch.ones_like(norm_cd), norm_cd)
            
            combined_score = norm_adv + 0.3 * norm_cd
            _, top_k_local_indices = torch.topk(combined_score, k)
        else:
            _, top_k_local_indices = torch.topk(bin_advantages, k)
        
        top_k_global_indices = bin_indices[top_k_local_indices]
        keep_mask[top_k_global_indices] = 1.0
    
    return keep_mask, crowding_distance


def calculate_adaptive_threshold_mask(
    advantages: torch.Tensor, 
    entropy: torch.Tensor,
    low_entropy_adv_percentile: float = 70.0,
    high_entropy_adv_percentile: float = 30.0,
    entropy_threshold_percentile: float = 50.0,
):
    """
    简化版：基于熵的二分法 + 动态 Advantage 百分位阈值。
    """
    N = advantages.shape[0]
    device = advantages.device
    
    entropy_threshold = torch.quantile(entropy, entropy_threshold_percentile / 100.0)
    
    low_entropy_mask = entropy <= entropy_threshold
    high_entropy_mask = entropy > entropy_threshold
    
    keep_mask = torch.zeros(N, device=device, dtype=torch.float32)
    
    if low_entropy_mask.any():
        low_ent_adv = advantages[low_entropy_mask]
        low_ent_threshold = torch.quantile(low_ent_adv, low_entropy_adv_percentile / 100.0)
        keep_mask[low_entropy_mask & (advantages >= low_ent_threshold)] = 1.0
    
    if high_entropy_mask.any():
        high_ent_adv = advantages[high_entropy_mask]
        high_ent_threshold = torch.quantile(high_ent_adv, high_entropy_adv_percentile / 100.0)
        keep_mask[high_entropy_mask & (advantages >= high_ent_threshold)] = 1.0
    
    return keep_mask


def calculate_soft_pareto_mask(
    advantages: torch.Tensor, 
    entropy: torch.Tensor,
    alpha: float = 0.5,
    keep_ratio: float = 0.3,
):
    """
    软 Pareto：使用加权组合分数进行筛选。
    """
    N = advantages.shape[0]
    device = advantages.device
    
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_adv = normalize(advantages)
    norm_ent = normalize(entropy)
    
    combined_score = norm_adv + alpha * norm_ent
    
    k = max(1, int(N * keep_ratio))
    _, top_k_indices = torch.topk(combined_score, k)
    
    keep_mask = torch.zeros(N, device=device, dtype=torch.float32)
    keep_mask[top_k_indices] = 1.0
    
    return keep_mask


def calculate_pareto_mask_soft_with_crowding(
    advantages: torch.Tensor, 
    entropy: torch.Tensor,
    alpha: float = 0.5,
    keep_ratio: float = 0.3,
    crowding_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    软 Pareto + 拥挤度增强版。
    """
    N = advantages.shape[0]
    device = advantages.device
    
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_adv = normalize(advantages)
    norm_ent = normalize(entropy)
    
    combined_score = (1-alpha) * norm_adv + alpha * norm_ent
    
    # 计算拥挤度
    objectives = torch.stack([advantages, entropy], dim=1)
    crowding_distance = calculate_crowding_distance_vectorized(objectives)
    
    # 综合分数
    final_score = combined_score + crowding_distance * crowding_weight
    
    k = max(1, int(N * keep_ratio))
    _, top_k_indices = torch.topk(final_score, k)
    
    keep_mask = torch.zeros(N, device=device, dtype=torch.float32)
    keep_mask[top_k_indices] = 1.0
    
    return keep_mask, crowding_distance


def visualize_pareto_front(
    advantages: torch.Tensor, 
    entropy: torch.Tensor, 
    mask: torch.Tensor, 
    title: str = "Pareto Front",
    crowding_distance: Optional[torch.Tensor] = None,
):
    """
    可视化筛选结果，支持拥挤度颜色映射。
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if crowding_distance is not None else 1, figsize=(14 if crowding_distance is not None else 10, 6))
    
    if crowding_distance is None:
        axes = [axes]
    
    # 左图：保留/过滤
    ax1 = axes[0]
    filtered_mask = mask == 0
    ax1.scatter(
        entropy[filtered_mask].cpu().numpy(), 
        advantages[filtered_mask].cpu().numpy(), 
        c='lightgray', alpha=0.5, label='Filtered', s=10
    )
    
    kept_mask = mask == 1
    ax1.scatter(
        entropy[kept_mask].cpu().numpy(), 
        advantages[kept_mask].cpu().numpy(), 
        c='red', label='Kept', s=20
    )
    
    ax1.set_xlabel('Entropy (Uncertainty)')
    ax1.set_ylabel('Advantages (Reward Signal)')
    ax1.set_title(f'{title}\nKept: {kept_mask.sum().item()} / {len(mask)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：拥挤度热力图
    if crowding_distance is not None:
        ax2 = axes[1]
        cd_plot = crowding_distance.clone()
        cd_plot = torch.where(torch.isinf(cd_plot), cd_plot.max() * 1.5, cd_plot)
        
        scatter = ax2.scatter(
            entropy[kept_mask].cpu().numpy(), 
            advantages[kept_mask].cpu().numpy(), 
            c=cd_plot[kept_mask].cpu().numpy(),
            cmap='viridis', s=30, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax2, label='Crowding Distance')
        ax2.set_xlabel('Entropy')
        ax2.set_ylabel('Advantages')
        ax2.set_title('Crowding Distance Heatmap (Kept Points)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace('/', '_')}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    adv = torch.randn(16 * 512)
    ent = torch.abs(torch.randn(16 * 512))

    print("=" * 50)
    print("方法 1: 原始 Pareto 支配")
    mask1 = calculate_pareto_mask(adv, ent)
    print(f"保留点数: {mask1.sum().item()}")
    visualize_pareto_front(adv, ent, mask1, "Original Pareto Dominance")

    print("=" * 50)
    print("方法 2: Pareto + 拥挤度")
    mask2, cd2 = calculate_pareto_mask_with_crowding(adv, ent, keep_ratio=0.4)
    print(f"保留点数: {mask2.sum().item()}")
    visualize_pareto_front(adv, ent, mask2, "Pareto with Crowding Distance", crowding_distance=cd2)

    print("=" * 50)
    print("方法 3: Entropy 分箱 + 拥挤度")
    mask3, cd3 = calculate_entropy_binned_mask_with_crowding(adv, ent, num_bins=5, top_k_ratio=0.3, use_crowding=True)
    print(f"保留点数: {mask3.sum().item()}")
    visualize_pareto_front(adv, ent, mask3, "Entropy-Binned with Crowding", crowding_distance=cd3)

    print("=" * 50)
    print("方法 4: 自适应阈值 (二分法)")
    mask4 = calculate_adaptive_threshold_mask(adv, ent)
    print(f"保留点数: {mask4.sum().item()}")
    visualize_pareto_front(adv, ent, mask4, "Adaptive Threshold")

    print("=" * 50)
    print("方法 5: 软 Pareto (加权组合)")
    mask5 = calculate_soft_pareto_mask(adv, ent, alpha=0.5, keep_ratio=0.4)
    print(f"保留点数: {mask5.sum().item()}")
    visualize_pareto_front(adv, ent, mask5, "Soft Pareto Weighted")