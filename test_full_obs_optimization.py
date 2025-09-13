#!/usr/bin/env python3
"""
测试完整观测计算的性能优化 - 包括所有向量化改进
"""
import torch
import time

def simulate_compute_obs_original():
    """
    模拟原始循环版本的观测计算
    """
    # 模拟参数
    num_envs = 100
    gate_count = 5
    drone_n = 4
    device = torch.device("cpu")
    
    # 模拟数据
    current_gate_idx = torch.randint(0, gate_count, (num_envs,), device=device)
    gate_positions = torch.randn(num_envs, gate_count, 3, device=device)
    gate_rotations = torch.randn(num_envs, gate_count, 4, device=device)
    gate_rotations = gate_rotations / torch.norm(gate_rotations, dim=-1, keepdim=True)
    gate_velocities = torch.randn(num_envs, gate_count, 3, device=device)
    gate_angular_velocities = torch.randn(num_envs, gate_count, 3, device=device)
    drone_pos = torch.randn(num_envs, drone_n, 3, device=device)
    end_positions = torch.randn(num_envs, drone_n, 3, device=device)
    
    gate_width, gate_height = 4.0, 3.0
    
    def compute_gate_corners_original(gate_pos, gate_rot):
        """原始循环版本"""
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
            corners_world.append(corners_world_i)
            
        return torch.stack(corners_world, dim=0)
    
    start_time = time.time()
    
    # 1. 门信息计算 - 原始循环版本
    formation_center = drone_pos.mean(dim=1, keepdim=True)
    gate_info = torch.zeros(num_envs, 1, 18, device=device)
    
    for env_idx in range(num_envs):
        gate_idx = current_gate_idx[env_idx]
        if gate_idx < gate_count:
            gate_pos = gate_positions[env_idx, gate_idx]
            gate_rot = gate_rotations[env_idx, gate_idx]
            gate_lin_vel = gate_velocities[env_idx, gate_idx]
            gate_ang_vel = gate_angular_velocities[env_idx, gate_idx]
            
            corners = compute_gate_corners_original(
                gate_pos.unsqueeze(0), 
                gate_rot.unsqueeze(0)
            )
            corners_flat = corners[0].flatten()
            
            gate_info[env_idx, 0] = torch.cat([
                corners_flat, gate_lin_vel, gate_ang_vel
            ])
    
    # 2. 终点信息计算 - 原始循环版本
    end_center = end_positions.mean(dim=1)
    endpoint_info = torch.zeros(num_envs, 1, 4, device=device)
    
    for env_idx in range(num_envs):
        center_pos = formation_center[env_idx, 0]
        end_pos = end_center[env_idx]
        relative_end_pos = end_pos - center_pos
        distance_to_endpoint = torch.norm(relative_end_pos)
        endpoint_info[env_idx, 0] = torch.cat([
            relative_end_pos, distance_to_endpoint.unsqueeze(0)
        ])
    
    # 3. 个体观测 - 原始循环版本
    for i in range(drone_n):
        gate_info_relative = gate_info.clone()
        for env_idx in range(num_envs):
            if gate_info_relative[env_idx, 0, 0] != 0:
                corners_global = gate_info_relative[env_idx, 0, :12].view(4, 3)
                corners_relative = corners_global - drone_pos[env_idx, i]
                gate_info_relative[env_idx, 0, :12] = corners_relative.flatten()
    
    # 4. 中央观测 - 原始循环版本
    all_gate_info = torch.zeros(num_envs, gate_count, 18, device=device)
    for i in range(gate_count):
        gate_pos = gate_positions[:, i]
        gate_rot = gate_rotations[:, i]
        gate_lin_vel = gate_velocities[:, i]
        gate_ang_vel = gate_angular_velocities[:, i]
        
        corners = compute_gate_corners_original(gate_pos, gate_rot)
        corners_flat = corners.view(num_envs, -1)
        
        all_gate_info[:, i, :12] = corners_flat
        all_gate_info[:, i, 12:15] = gate_lin_vel
        all_gate_info[:, i, 15:18] = gate_ang_vel
    
    return time.time() - start_time

def simulate_compute_obs_vectorized():
    """
    模拟向量化版本的观测计算
    """
    # 模拟参数
    num_envs = 100
    gate_count = 5
    drone_n = 4
    device = torch.device("cpu")
    
    # 模拟数据
    current_gate_idx = torch.randint(0, gate_count, (num_envs,), device=device)
    gate_positions = torch.randn(num_envs, gate_count, 3, device=device)
    gate_rotations = torch.randn(num_envs, gate_count, 4, device=device)
    gate_rotations = gate_rotations / torch.norm(gate_rotations, dim=-1, keepdim=True)
    gate_velocities = torch.randn(num_envs, gate_count, 3, device=device)
    gate_angular_velocities = torch.randn(num_envs, gate_count, 3, device=device)
    drone_pos = torch.randn(num_envs, drone_n, 3, device=device)
    end_positions = torch.randn(num_envs, drone_n, 3, device=device)
    
    gate_width, gate_height = 4.0, 3.0
    
    def compute_gate_corners_vectorized(gate_pos, gate_rot):
        """向量化版本"""
        half_w = gate_width * 0.5
        half_h = gate_height * 0.5
        
        local_corners = torch.tensor([
            [0.0, -half_w, -half_h],
            [0.0, -half_w,  half_h],
            [0.0,  half_w, -half_h],
            [0.0,  half_w,  half_h],
        ], device=device, dtype=torch.float32)
        
        x, y, z, w = gate_rot[:, 0], gate_rot[:, 1], gate_rot[:, 2], gate_rot[:, 3]
        
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)], dim=1),
            torch.stack([  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)], dim=1),
            torch.stack([  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)], dim=1),
        ], dim=1)
        
        corners_world = torch.einsum('bij,kj->bki', R, local_corners) + gate_pos.unsqueeze(1)
        return corners_world
    
    start_time = time.time()
    
    # 1. 门信息计算 - 向量化版本
    formation_center = drone_pos.mean(dim=1, keepdim=True)
    gate_info = torch.zeros(num_envs, 1, 18, device=device)
    
    valid_mask = current_gate_idx < gate_count
    if valid_mask.any():
        valid_env_indices = torch.where(valid_mask)[0]
        valid_gate_indices = current_gate_idx[valid_mask]
        
        gate_pos_batch = gate_positions[valid_env_indices, valid_gate_indices]
        gate_rot_batch = gate_rotations[valid_env_indices, valid_gate_indices]
        gate_lin_vel_batch = gate_velocities[valid_env_indices, valid_gate_indices]
        gate_ang_vel_batch = gate_angular_velocities[valid_env_indices, valid_gate_indices]
        
        corners_batch = compute_gate_corners_vectorized(gate_pos_batch, gate_rot_batch)
        corners_flat_batch = corners_batch.reshape(len(valid_env_indices), -1)
        
        gate_info_batch = torch.cat([
            corners_flat_batch, gate_lin_vel_batch, gate_ang_vel_batch
        ], dim=-1)
        
        gate_info[valid_env_indices, 0] = gate_info_batch
    
    # 2. 终点信息计算 - 向量化版本
    end_center = end_positions.mean(dim=1)
    center_pos = formation_center[:, 0]
    relative_end_pos = end_center - center_pos
    distance_to_endpoint = torch.norm(relative_end_pos, dim=-1)
    
    endpoint_info = torch.zeros(num_envs, 1, 4, device=device)
    endpoint_info[:, 0, :3] = relative_end_pos
    endpoint_info[:, 0, 3] = distance_to_endpoint
    
    # 3. 个体观测 - 向量化版本
    for i in range(drone_n):
        gate_info_relative = gate_info.clone()
        valid_gate_mask = gate_info_relative[:, 0, 0] != 0
        
        if valid_gate_mask.any():
            valid_env_indices = torch.where(valid_gate_mask)[0]
            corners_global_batch = gate_info_relative[valid_env_indices, 0, :12].reshape(-1, 4, 3)
            drone_pos_batch = drone_pos[valid_env_indices, i]
            corners_relative_batch = corners_global_batch - drone_pos_batch.unsqueeze(1)
            gate_info_relative[valid_env_indices, 0, :12] = corners_relative_batch.reshape(-1, 12)
    
    # 4. 中央观测 - 向量化版本
    batch_size = num_envs * gate_count
    gate_pos_flat = gate_positions.view(batch_size, 3)
    gate_rot_flat = gate_rotations.view(batch_size, 4)
    gate_lin_vel_flat = gate_velocities.view(batch_size, 3)
    gate_ang_vel_flat = gate_angular_velocities.view(batch_size, 3)
    
    corners_all = compute_gate_corners_vectorized(gate_pos_flat, gate_rot_flat)
    corners_flat_all = corners_all.reshape(batch_size, -1)
    corners_reshaped = corners_flat_all.reshape(num_envs, gate_count, 12)
    
    all_gate_info = torch.zeros(num_envs, gate_count, 18, device=device)
    all_gate_info[:, :, :12] = corners_reshaped
    all_gate_info[:, :, 12:15] = gate_velocities
    all_gate_info[:, :, 15:18] = gate_angular_velocities
    
    return time.time() - start_time

def test_full_optimization():
    """测试完整观测计算的性能优化"""
    print("Testing full observation computation optimization...")
    print(f"Configuration: num_envs=100, gate_count=5, drone_n=4")
    
    num_runs = 100
    
    print(f"\nRunning {num_runs} iterations...")
    
    # 测试原始版本
    original_times = []
    for _ in range(num_runs):
        original_times.append(simulate_compute_obs_original())
    
    # 测试向量化版本
    vectorized_times = []
    for _ in range(num_runs):
        vectorized_times.append(simulate_compute_obs_vectorized())
    
    original_avg = sum(original_times) / len(original_times)
    vectorized_avg = sum(vectorized_times) / len(vectorized_times)
    
    print(f"\nResults:")
    print(f"Original (loop-based) average time: {original_avg:.4f}s")
    print(f"Vectorized average time: {vectorized_avg:.4f}s")
    print(f"Speedup: {original_avg/vectorized_avg:.2f}x")
    
    # 计算总体性能提升
    total_original = sum(original_times)
    total_vectorized = sum(vectorized_times)
    
    print(f"\nTotal time comparison:")
    print(f"Original total time: {total_original:.4f}s")
    print(f"Vectorized total time: {total_vectorized:.4f}s")
    print(f"Time saved: {total_original - total_vectorized:.4f}s ({((total_original - total_vectorized) / total_original * 100):.1f}%)")
    
    return vectorized_avg < original_avg

if __name__ == "__main__":
    success = test_full_optimization()
    print(f"\nOptimization test {'PASSED' if success else 'FAILED'}")
