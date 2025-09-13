#!/usr/bin/env python3
"""
测试门观测空间修改后的功能 - 包括张量化优化
"""
import torch
import time

def quaternion_to_rotation_matrix_vectorized(quat):
    """
    向量化的四元数转旋转矩阵函数
    Args:
        quat: [batch_size, 4] - quaternion (x, y, z, w)
    Returns:
        R: [batch_size, 3, 3] - rotation matrices
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)], dim=1),
        torch.stack([  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)], dim=1),
        torch.stack([  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)], dim=1),
    ], dim=1)
    
    return R

def compute_gate_corners_vectorized(gate_pos, gate_rot, gate_width=4.0, gate_height=3.0, relative_to_pos=None):
    """
    向量化的门角点计算函数
    """
    device = gate_pos.device
    half_w = gate_width * 0.5
    half_h = gate_height * 0.5
    
    # Define local corner offsets
    local_corners = torch.tensor([
        [0.0, -half_w, -half_h],  # bottom-left
        [0.0, -half_w,  half_h],  # top-left
        [0.0,  half_w, -half_h],  # bottom-right
        [0.0,  half_w,  half_h],  # top-right
    ], device=device, dtype=torch.float32)
    
    # Convert quaternion to rotation matrix (vectorized)
    R = quaternion_to_rotation_matrix_vectorized(gate_rot)  # [batch_size, 3, 3]
    
    # Transform corners to world frame (vectorized)
    corners_world = torch.einsum('bij,kj->bki', R, local_corners) + gate_pos.unsqueeze(1)  # [batch_size, 4, 3]
    
    # Make relative if requested
    if relative_to_pos is not None:
        corners_world = corners_world - relative_to_pos.unsqueeze(1)
        
    return corners_world

def compute_gate_corners_loop(gate_pos, gate_rot, gate_width=4.0, gate_height=3.0, relative_to_pos=None):
    """
    原始循环版本的门角点计算函数
    """
    device = gate_pos.device
    batch_size = gate_pos.shape[0]
    half_w = gate_width * 0.5
    half_h = gate_height * 0.5
    
    local_corners = torch.tensor([
        [0.0, -half_w, -half_h],
        [0.0, -half_w,  half_h],
        [0.0,  half_w, -half_h],
        [0.0,  half_w,  half_h],
    ], device=device, dtype=torch.float32)
    
    corners_world = []
    for i in range(batch_size):
        quat = gate_rot[i]
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        
        R = torch.tensor([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], device=device, dtype=torch.float32)
        
        corners_world_i = (R @ local_corners.T).T + gate_pos[i:i+1]
        
        if relative_to_pos is not None:
            corners_world_i = corners_world_i - relative_to_pos[i:i+1]
            
        corners_world.append(corners_world_i)
        
    return torch.stack(corners_world, dim=0)

def test_vectorized_vs_loop():
    """测试向量化版本与循环版本的一致性和性能"""
    device = torch.device("cpu")
    
    # 测试数据：多个环境，多个门
    batch_size = 100
    gate_pos = torch.randn(batch_size, 3, device=device) * 10
    
    # 生成随机的标准化四元数
    gate_rot = torch.randn(batch_size, 4, device=device)
    gate_rot = gate_rot / torch.norm(gate_rot, dim=1, keepdim=True)  # 标准化四元数
    
    # 无人机位置
    drone_pos = torch.randn(batch_size, 3, device=device) * 5
    
    print(f"Testing with batch_size = {batch_size}")
    
    # 测试一致性
    print("\n1. Testing consistency...")
    
    # 计算全局角点
    corners_vec = compute_gate_corners_vectorized(gate_pos, gate_rot)
    corners_loop = compute_gate_corners_loop(gate_pos, gate_rot)
    
    print(f"Global corners - Max difference: {torch.max(torch.abs(corners_vec - corners_loop)).item():.2e}")
    print(f"Global corners - Are close: {torch.allclose(corners_vec, corners_loop, atol=1e-5)}")
    
    # 计算相对角点
    corners_rel_vec = compute_gate_corners_vectorized(gate_pos, gate_rot, relative_to_pos=drone_pos)
    corners_rel_loop = compute_gate_corners_loop(gate_pos, gate_rot, relative_to_pos=drone_pos)
    
    print(f"Relative corners - Max difference: {torch.max(torch.abs(corners_rel_vec - corners_rel_loop)).item():.2e}")
    print(f"Relative corners - Are close: {torch.allclose(corners_rel_vec, corners_rel_loop, atol=1e-5)}")
    
    # 测试性能
    print("\n2. Testing performance...")
    
    num_runs = 1000
    
    # 向量化版本
    start_time = time.time()
    for _ in range(num_runs):
        _ = compute_gate_corners_vectorized(gate_pos, gate_rot, relative_to_pos=drone_pos)
    vec_time = time.time() - start_time
    
    # 循环版本
    start_time = time.time()
    for _ in range(num_runs):
        _ = compute_gate_corners_loop(gate_pos, gate_rot, relative_to_pos=drone_pos)
    loop_time = time.time() - start_time
    
    print(f"Vectorized version: {vec_time:.4f}s ({num_runs} runs)")
    print(f"Loop version: {loop_time:.4f}s ({num_runs} runs)")
    print(f"Speedup: {loop_time/vec_time:.2f}x")
    
    # 验证观测空间维度
    print("\n3. Testing observation dimensions...")
    corners_flat = corners_vec[0].flatten()  # 12维
    lin_vel = torch.tensor([0.1, 0.2, 0.3], device=device)  # 3维
    ang_vel = torch.tensor([0.01, 0.02, 0.03], device=device)  # 3维
    
    gate_obs = torch.cat([corners_flat, lin_vel, ang_vel])  # 18维
    print(f"Gate observation shape: {gate_obs.shape}")
    
    return torch.allclose(corners_vec, corners_loop, atol=1e-5) and gate_obs.shape[0] == 18

if __name__ == "__main__":
    success = test_vectorized_vs_loop()
    print(f"\nOverall test {'PASSED' if success else 'FAILED'}")
