import torch

def calculate_crowding_distance(points: torch.Tensor):
    """
    计算拥挤度距离 (Crowding Distance)。
    
    Args:
        points: [N, M] 张量，N是点数，M是目标数 (这里通常是2: Adv, Ent)
        
    Returns:
        distances: [N] 每个点的拥挤度距离。边界点为无穷大。
    """
    N, M = points.shape
    if N == 0:
        return torch.tensor([], device=points.device)
    
    distances = torch.zeros(N, device=points.device)
    
    # 针对每个目标分别计算
    for m in range(M):
        # 1. 对当前目标的值进行排序
        # values: 排序后的值, indices: 原索引
        sorted_values, sorted_indices = torch.sort(points[:, m])
        
        # 2. 边界点赋予无穷大距离 (保留极值点)
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # 3. 计算中间点的距离
        # norm = max - min
        norm = sorted_values[-1] - sorted_values[0]
        if norm == 0:
            norm = 1e-8 # 防止除以0
            
        # distance[i] += (val[i+1] - val[i-1]) / norm
        # 利用切片操作进行向量化计算
        # sorted_values[2:] 是 i+1
        # sorted_values[:-2] 是 i-1
        diffs = (sorted_values[2:] - sorted_values[:-2]) / norm
        
        # 将计算出的距离累加回对应的原索引位置
        # sorted_indices[1:-1] 是中间点的原索引
        distances[sorted_indices[1:-1]] += diffs
        
    return distances

def calculate_pareto_mask(advantages: torch.Tensor, entropy: torch.Tensor, retention_ratio: float = 0.8):
    """
    向量化计算 Pareto 支配掩码，并结合拥挤度距离进行二次筛选。
    
    Args:
        advantages: [Batch*Seq]
        entropy:    [Batch*Seq]
        retention_ratio: float, (0, 1.0]。在 Pareto Front 上保留多少比例的点。
                         如果为 1.0，则保留所有非支配点（不使用拥挤度剪枝）。
        
    Returns:
        mask: [Batch*Seq] 0.0 or 1.0
    """
    # 1. 数据准备
    N = advantages.shape[0]
    points = torch.stack([advantages, entropy], dim=1) # [N, 2]
    
    # 2. 计算 Pareto Front (非支配排序 Level 1)
    # 之前的快速筛选逻辑
    sorted_idx = torch.argsort(advantages, descending=True)
    sorted_ent = entropy[sorted_idx]
    
    current_max_ent = -float('inf')
    is_dominated_ordered = torch.zeros(N, dtype=torch.bool, device=advantages.device)
    
    # O(N) 扫描
    # 注意：这里如果 tensor 很大，可以使用 torch.jit.script 加速循环，或者用 cuda kernel
    # 但 Python 循环对于几千个 token 通常也够快
    for i in range(N):
        if sorted_ent[i] < current_max_ent:
            is_dominated_ordered[i] = True
        else:
            current_max_ent = sorted_ent[i]
            
    # 映射回原顺序
    is_dominated = torch.zeros(N, dtype=torch.bool, device=advantages.device)
    is_dominated[sorted_idx] = is_dominated_ordered
    
    # 此时 mask 中 True 代表是被支配的(坏点)，False 代表是 Front 上的(好点)
    front_mask = ~is_dominated 
    
    # 3. === 新增：拥挤度距离计算 ===
    # 我们只对 Front 上的点计算拥挤度，不需要对被支配的点计算
    
    # 获取 Front 上点的索引
    front_indices = torch.nonzero(front_mask).squeeze()
    
    # 如果 Front 上没点或者点太少，直接返回
    if front_indices.numel() <= 2:
        return front_mask.float()
    
    # 如果不需要剪枝，直接返回
    if retention_ratio >= 1.0:
        return front_mask.float()
        
    # 提取 Front 上的点的值 [K, 2]
    front_points = points[front_indices]
    
    # 计算拥挤度 [K]
    crowding_dists = calculate_crowding_distance(front_points)
    
    # 4. === 基于拥挤度的二次筛选 ===
    # 我们希望保留距离大的点（稀疏的），去掉距离小的点（密集的）
    
    num_to_keep = int(front_indices.numel() * retention_ratio)
    num_to_keep = max(num_to_keep, 2) # 至少保留2个端点
    
    # 选出拥挤度最大的 top k 个索引
    # sort descending
    _, keep_topk_indices = torch.topk(crowding_dists, num_to_keep)
    
    # 这些 indices 是相对于 front_indices 的索引
    # 我们需要映射回全局索引
    final_keep_indices = front_indices[keep_topk_indices]
    
    # 生成最终 Mask
    final_mask = torch.zeros(N, device=advantages.device, dtype=torch.float32)
    final_mask[final_keep_indices] = 1.0
    
    return final_mask