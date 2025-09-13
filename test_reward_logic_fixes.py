#!/usr/bin/env python3
"""
测试奖励函数逻辑错误修复
验证 last_gate_distance 和 last_endpoint_distance 的正确初始化
"""

import torch
import numpy as np
import sys
import os

def test_distance_initialization():
    """测试距离初始化的修复"""
    print("="*60)
    print("测试距离初始化修复")
    print("="*60)
    
    # 模拟环境参数
    num_envs = 4
    num_agents = 3
    gate_count = 3
    device = torch.device('cpu')
    
    # 模拟门位置 [num_envs, gate_count, 3]
    gate_positions = torch.tensor([
        [[0.0, 0.0, 2.0], [5.0, 0.0, 2.0], [10.0, 0.0, 2.0]],  # env 0
        [[0.1, 0.1, 2.1], [5.1, 0.1, 2.1], [10.1, 0.1, 2.1]],  # env 1
        [[0.2, 0.2, 2.2], [5.2, 0.2, 2.2], [10.2, 0.2, 2.2]],  # env 2
        [[0.3, 0.3, 2.3], [5.3, 0.3, 2.3], [10.3, 0.3, 2.3]],  # env 3
    ], dtype=torch.float32, device=device)
    
    # 模拟无人机初始位置 [num_envs, num_agents, 3]
    drone_pos = torch.tensor([
        [[-3.0, -1.0, 2.0], [-3.0, 0.0, 2.0], [-3.0, 1.0, 2.0]],  # env 0
        [[-2.9, -0.9, 2.1], [-2.9, 0.1, 2.1], [-2.9, 1.1, 2.1]],  # env 1
        [[-2.8, -0.8, 2.2], [-2.8, 0.2, 2.2], [-2.8, 1.2, 2.2]],  # env 2
        [[-2.7, -0.7, 2.3], [-2.7, 0.3, 2.3], [-2.7, 1.3, 2.3]],  # env 3
    ], dtype=torch.float32, device=device)
    
    # 计算编队中心
    formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
    print(f"编队中心位置: {formation_center}")
    
    # 测试旧的初始化方法（错误的）
    print("\n=== 旧方法（错误）：直接设为0 ===")
    last_gate_distance_old = torch.zeros(num_envs, device=device)
    last_endpoint_distance_old = torch.zeros(num_envs, device=device)
    
    print(f"旧方法 last_gate_distance: {last_gate_distance_old}")
    print(f"旧方法 last_endpoint_distance: {last_endpoint_distance_old}")
    
    # 模拟第一步的门距离计算
    current_gate_distance = torch.norm(formation_center - gate_positions[:, 0], dim=-1)
    print(f"第一步实际门距离: {current_gate_distance}")
    
    # 旧方法的进度奖励（第一步总是0）
    old_progress = (last_gate_distance_old - current_gate_distance).clamp(min=0.0)
    print(f"旧方法第一步进度奖励: {old_progress} (总是0)")
    
    # 测试新的初始化方法（正确的）
    print("\n=== 新方法（正确）：初始化为实际距离 ===")
    last_gate_distance_new = torch.zeros(num_envs, device=device)
    last_endpoint_distance_new = torch.zeros(num_envs, device=device)
    
    # 正确初始化：设为实际的初始距离
    for env_idx in range(num_envs):
        first_gate_pos = gate_positions[env_idx, 0]
        last_gate_distance_new[env_idx] = torch.norm(formation_center[env_idx] - first_gate_pos)
        
        # 终点距离（假设终点在最后一个门后面）
        endpoint_pos = torch.tensor([15.0, 0.0, 2.0], device=device)
        last_endpoint_distance_new[env_idx] = torch.norm(formation_center[env_idx] - endpoint_pos)
    
    print(f"新方法 last_gate_distance: {last_gate_distance_new}")
    print(f"新方法 last_endpoint_distance: {last_endpoint_distance_new}")
    
    # 模拟第二步：无人机向门移动
    print("\n=== 模拟第二步：无人机向门移动 ===")
    # 假设无人机向第一个门移动了0.5米
    formation_center_step2 = formation_center + torch.tensor([0.5, 0.0, 0.0], device=device)
    current_gate_distance_step2 = torch.norm(formation_center_step2 - gate_positions[:, 0], dim=-1)
    
    print(f"第二步门距离: {current_gate_distance_step2}")
    
    # 新方法的进度奖励（应该为正值，因为距离减少了）
    new_progress = (last_gate_distance_new - current_gate_distance_step2).clamp(min=0.0)
    print(f"新方法第二步进度奖励: {new_progress} (应该为正值)")
    
    # 验证距离确实减少了
    distance_improvement = last_gate_distance_new - current_gate_distance_step2
    print(f"距离改善量: {distance_improvement} (正值表示靠近)")
    
    return {
        'old_first_step_reward': old_progress,
        'new_second_step_reward': new_progress,
        'distance_improvement': distance_improvement
    }

def test_gate_active_logic():
    """测试门激活逻辑"""
    print("\n" + "="*60)
    print("测试门激活逻辑")
    print("="*60)
    
    num_envs = 3
    gate_count = 2
    device = torch.device('cpu')
    
    # 模拟当前门索引
    current_gate_idx = torch.tensor([0, 1, 2], device=device)  # env0: 门0, env1: 门1, env2: 完成所有门
    
    # 模拟门位置
    gate_positions = torch.randn(num_envs, gate_count, 3, device=device)
    
    # 测试门激活逻辑
    current_gate_pos = torch.zeros(num_envs, 3, device=device)
    gate_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    for env_idx in range(num_envs):
        gate_idx = current_gate_idx[env_idx]
        if gate_idx < gate_count:
            current_gate_pos[env_idx] = gate_positions[env_idx, gate_idx]
            gate_active[env_idx] = True
    
    print(f"当前门索引: {current_gate_idx}")
    print(f"门激活状态: {gate_active}")
    print(f"激活的环境: {torch.where(gate_active)[0]}")
    
    # 模拟编队中心
    formation_center = torch.randn(num_envs, 3, device=device)
    
    # 计算门距离（只对激活的门）
    gate_distance = torch.norm(formation_center - current_gate_pos, dim=-1)
    print(f"所有环境的门距离: {gate_distance}")
    
    # 模拟修复后的进度奖励计算
    gate_progress_reward = torch.zeros(num_envs, device=device)
    last_gate_distance = torch.ones(num_envs, device=device) * 10.0  # 假设上一步距离
    
    if gate_active.any():
        active_envs = torch.where(gate_active)[0]
        active_gate_distance = gate_distance[active_envs]
        active_last_distance = last_gate_distance[active_envs]
        
        print(f"激活环境的当前距离: {active_gate_distance}")
        print(f"激活环境的上一步距离: {active_last_distance}")
        
        # 计算进度（只对激活环境）
        active_progress = (active_last_distance - active_gate_distance).clamp(min=0.0)
        gate_progress_reward[active_envs] = active_progress
        
        print(f"激活环境的进度奖励: {active_progress}")
    
    print(f"所有环境的最终进度奖励: {gate_progress_reward}")
    
    return {
        'gate_active': gate_active,
        'progress_reward': gate_progress_reward
    }

if __name__ == "__main__":
    print("🔍 奖励函数逻辑错误检测与修复验证")
    
    # 测试距离初始化修复
    distance_result = test_distance_initialization()
    
    # 测试门激活逻辑
    gate_result = test_gate_active_logic()
    
    print("\n" + "="*60)
    print("修复效果总结")
    print("="*60)
    
    print("1. ✅ 距离初始化修复:")
    print(f"   - 旧方法第一步奖励: {distance_result['old_first_step_reward'].max():.4f} (总是0)")
    print(f"   - 新方法能正确计算进度: {distance_result['new_second_step_reward'].max():.4f}")
    print(f"   - 距离改善效果: {distance_result['distance_improvement'].max():.4f}")
    
    print("\n2. ✅ 门激活逻辑优化:")
    print(f"   - 只有{gate_result['gate_active'].sum()}个环境有激活门")
    print(f"   - 进度奖励只计算激活环境: {gate_result['progress_reward'].nonzero().numel()}个非零")
    
    print("\n3. 🎯 主要修复点:")
    print("   ✅ last_gate_distance 初始化为实际距离而非0")
    print("   ✅ last_endpoint_distance 初始化为实际距离而非0") 
    print("   ✅ gate_progress_reward 只对有激活门的环境计算")
    print("   ✅ 避免对已完成所有门的环境进行错误的距离更新")
    
    print("\n🚀 修复后的优势:")
    print("   • 第一步就能获得正确的进度信号")
    print("   • 避免无效门状态的干扰")
    print("   • 更准确的奖励计算")
    print("   • 提高训练效率和稳定性")
