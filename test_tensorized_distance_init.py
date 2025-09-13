#!/usr/bin/env python3
"""
测试张量化的距离初始化实现
验证去除循环后的正确性和性能
"""

import torch
import time

def test_tensorized_distance_initialization():
    """测试张量化的距离初始化"""
    print("="*60)
    print("测试张量化距离初始化")
    print("="*60)
    
    # 模拟环境参数
    num_envs = 8
    num_agents = 4
    gate_count = 3
    device = torch.device('cpu')
    
    # 模拟要重置的环境ID
    env_ids = torch.tensor([1, 3, 5, 7], device=device)  # 重置部分环境
    print(f"要重置的环境ID: {env_ids}")
    
    # 模拟门位置 [num_envs, gate_count, 3]
    gate_positions = torch.randn(num_envs, gate_count, 3, device=device) * 5
    gate_positions[:, :, 2] = 2.0  # 固定Z高度
    print(f"门位置 shape: {gate_positions.shape}")
    
    # 模拟无人机位置 [num_envs, num_agents, 3]
    drone_pos = torch.randn(num_envs, num_agents, 3, device=device) * 2
    drone_pos[:, :, 2] = 2.0  # 固定Z高度
    formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
    print(f"编队中心 shape: {formation_center.shape}")
    
    # 模拟终点
    last_gate_x = 15.0
    end_center = torch.tensor([last_gate_x, 0.0, 2.0], device=device)
    
    print(f"终点位置: {end_center}")
    
    # =====================
    # 旧方法：使用循环
    # =====================
    print("\n=== 旧方法：使用循环 ===")
    
    last_gate_distance_old = torch.zeros(num_envs, device=device)
    last_endpoint_distance_old = torch.zeros(num_envs, device=device)
    
    start_time = time.time()
    
    # 门距离初始化（循环版本）
    for i, env_idx in enumerate(env_ids):
        if gate_count > 0:
            first_gate_pos = gate_positions[env_idx, 0]
            last_gate_distance_old[env_idx] = torch.norm(formation_center[env_idx] - first_gate_pos)
        else:
            last_gate_distance_old[env_idx] = 0
    
    # 终点距离初始化（循环版本）
    for i, env_idx in enumerate(env_ids):
        last_endpoint_distance_old[env_idx] = torch.norm(formation_center[env_idx] - end_center)
    
    old_time = time.time() - start_time
    
    print(f"循环版本计算时间: {old_time*1000:.4f} ms")
    print(f"重置环境的门距离: {last_gate_distance_old[env_ids]}")
    print(f"重置环境的终点距离: {last_endpoint_distance_old[env_ids]}")
    
    # =====================
    # 新方法：张量化操作
    # =====================
    print("\n=== 新方法：张量化操作 ===")
    
    last_gate_distance_new = torch.zeros(num_envs, device=device)
    last_endpoint_distance_new = torch.zeros(num_envs, device=device)
    
    start_time = time.time()
    
    # 门距离初始化（张量化版本）
    if gate_count > 0:
        # 获取所有重置环境的第一个门位置 [len(env_ids), 3]
        first_gate_positions = gate_positions[env_ids, 0]  
        # 计算对应环境的编队中心到第一个门的距离 [len(env_ids)]
        gate_distances = torch.norm(formation_center[env_ids] - first_gate_positions, dim=-1)
        last_gate_distance_new[env_ids] = gate_distances
    else:
        last_gate_distance_new[env_ids] = 0
    
    # 终点距离初始化（张量化版本）
    # 计算所有重置环境的编队中心到终点的距离 [len(env_ids)]
    endpoint_distances = torch.norm(formation_center[env_ids] - end_center, dim=-1)
    last_endpoint_distance_new[env_ids] = endpoint_distances
    
    new_time = time.time() - start_time
    
    print(f"张量化版本计算时间: {new_time*1000:.4f} ms")
    print(f"重置环境的门距离: {last_gate_distance_new[env_ids]}")
    print(f"重置环境的终点距离: {last_endpoint_distance_new[env_ids]}")
    
    # =====================
    # 结果验证
    # =====================
    print("\n=== 结果验证 ===")
    
    # 验证门距离计算的正确性
    gate_distance_diff = torch.abs(last_gate_distance_old[env_ids] - last_gate_distance_new[env_ids])
    gate_distance_max_diff = gate_distance_diff.max()
    
    # 验证终点距离计算的正确性
    endpoint_distance_diff = torch.abs(last_endpoint_distance_old[env_ids] - last_endpoint_distance_new[env_ids])
    endpoint_distance_max_diff = endpoint_distance_diff.max()
    
    print(f"门距离最大差异: {gate_distance_max_diff:.8f}")
    print(f"终点距离最大差异: {endpoint_distance_max_diff:.8f}")
    
    # 检查是否完全一致
    gate_identical = torch.allclose(last_gate_distance_old[env_ids], last_gate_distance_new[env_ids], atol=1e-6)
    endpoint_identical = torch.allclose(last_endpoint_distance_old[env_ids], last_endpoint_distance_new[env_ids], atol=1e-6)
    
    print(f"门距离计算一致性: {'✅' if gate_identical else '❌'}")
    print(f"终点距离计算一致性: {'✅' if endpoint_identical else '❌'}")
    
    # 性能提升
    speedup = old_time / new_time if new_time > 0 else float('inf')
    print(f"性能提升: {speedup:.2f}x")
    
    return {
        'old_time': old_time,
        'new_time': new_time,
        'speedup': speedup,
        'gate_identical': gate_identical,
        'endpoint_identical': endpoint_identical,
        'gate_max_diff': gate_distance_max_diff,
        'endpoint_max_diff': endpoint_distance_max_diff
    }

def test_large_scale_performance():
    """测试大规模下的性能对比"""
    print("\n" + "="*60)
    print("大规模性能测试")
    print("="*60)
    
    test_configs = [
        (32, 8, 4),    # 32环境, 8个agent, 4个门
        (128, 12, 6),  # 128环境, 12个agent, 6个门
        (512, 16, 8),  # 512环境, 16个agent, 8个门
    ]
    
    results = []
    
    for num_envs, num_agents, gate_count in test_configs:
        print(f"\n测试配置: {num_envs} 环境, {num_agents} agents, {gate_count} 门")
        
        device = torch.device('cpu')
        
        # 模拟重置一半的环境
        env_ids = torch.arange(0, num_envs, 2, device=device)  
        
        # 生成测试数据
        gate_positions = torch.randn(num_envs, gate_count, 3, device=device) * 10
        drone_pos = torch.randn(num_envs, num_agents, 3, device=device) * 5
        formation_center = drone_pos.mean(dim=1)
        end_center = torch.tensor([20.0, 0.0, 2.0], device=device)
        
        # 循环版本测试
        start_time = time.time()
        for _ in range(50):  # 重复50次取平均
            last_gate_distance_old = torch.zeros(num_envs, device=device)
            last_endpoint_distance_old = torch.zeros(num_envs, device=device)
            
            for i, env_idx in enumerate(env_ids):
                first_gate_pos = gate_positions[env_idx, 0]
                last_gate_distance_old[env_idx] = torch.norm(formation_center[env_idx] - first_gate_pos)
                last_endpoint_distance_old[env_idx] = torch.norm(formation_center[env_idx] - end_center)
        
        old_time = (time.time() - start_time) / 50
        
        # 张量化版本测试
        start_time = time.time()
        for _ in range(50):  # 重复50次取平均
            last_gate_distance_new = torch.zeros(num_envs, device=device)
            last_endpoint_distance_new = torch.zeros(num_envs, device=device)
            
            first_gate_positions = gate_positions[env_ids, 0]
            gate_distances = torch.norm(formation_center[env_ids] - first_gate_positions, dim=-1)
            last_gate_distance_new[env_ids] = gate_distances
            
            endpoint_distances = torch.norm(formation_center[env_ids] - end_center, dim=-1)
            last_endpoint_distance_new[env_ids] = endpoint_distances
        
        new_time = (time.time() - start_time) / 50
        
        speedup = old_time / new_time
        
        print(f"  循环版本: {old_time*1000:.4f} ms")
        print(f"  张量版本: {new_time*1000:.4f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        results.append({
            'config': f"{num_envs}x{num_agents}x{gate_count}",
            'old_time': old_time*1000,
            'new_time': new_time*1000,
            'speedup': speedup
        })
    
    print(f"\n{'配置':<15} {'循环版本(ms)':<12} {'张量版本(ms)':<12} {'加速比':<8}")
    print("-" * 55)
    for result in results:
        print(f"{result['config']:<15} {result['old_time']:<12.4f} {result['new_time']:<12.4f} {result['speedup']:<8.2f}x")

if __name__ == "__main__":
    print("🚀 张量化距离初始化测试")
    
    # 基本功能测试
    basic_result = test_tensorized_distance_initialization()
    
    # 大规模性能测试
    test_large_scale_performance()
    
    print("\n" + "="*60)
    print("张量化优化总结")
    print("="*60)
    
    print("1. ✅ 功能正确性:")
    print(f"   - 门距离计算: {'完全一致' if basic_result['gate_identical'] else '存在差异'}")
    print(f"   - 终点距离计算: {'完全一致' if basic_result['endpoint_identical'] else '存在差异'}")
    
    print("\n2. ✅ 性能提升:")
    print(f"   - 基本测试加速比: {basic_result['speedup']:.2f}x")
    print(f"   - 计算时间减少: {(1-basic_result['new_time']/basic_result['old_time'])*100:.1f}%")
    
    print("\n3. 🎯 优化要点:")
    print("   ✅ 用张量索引替代循环: gate_positions[env_ids, 0]")
    print("   ✅ 批量距离计算: torch.norm(..., dim=-1)")
    print("   ✅ 批量赋值: tensor[env_ids] = values")
    print("   ✅ 消除Python循环开销")
    
    print("\n4. 🚀 实际收益:")
    print("   • 代码更简洁易读")
    print("   • 计算效率显著提升")
    print("   • 更好的GPU/并行支持")
    print("   • 减少内存访问次数")
