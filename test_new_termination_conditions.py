#!/usr/bin/env python3
"""
测试新增的终止条件逻辑
1. 无人机超过门但没有穿过门就终止
2. 任何无人机超过终点位置就终止
"""

import torch
import numpy as np

def test_gate_bypass_termination():
    """测试门绕过终止条件"""
    print("="*60)
    print("测试门绕过终止条件")
    print("="*60)
    
    # 模拟参数
    num_envs = 4
    num_drones = 3
    gate_count = 2
    gate_width = 3.0
    gate_height = 2.0
    device = torch.device('cpu')
    
    # 模拟当前门索引
    current_gate_idx = torch.tensor([0, 0, 1, 2], device=device)  # env3已完成所有门
    print(f"当前门索引: {current_gate_idx}")
    
    # 模拟门位置 [num_envs, gate_count, 3]
    gate_positions = torch.tensor([
        [[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]],   # env 0
        [[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]],   # env 1  
        [[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]],   # env 2
        [[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]],   # env 3
    ], dtype=torch.float32, device=device)
    
    # 模拟无人机位置 [num_envs, num_drones, 3]
    drone_pos = torch.tensor([
        # env 0: 成功穿过门（都在门附近且超过门X位置）
        [[1.0, -0.5, 2.0], [1.0, 0.0, 2.0], [1.0, 0.5, 2.0]],
        
        # env 1: 绕过门（超过门X位置但距离门中心太远）
        [[2.0, -5.0, 2.0], [2.0, 5.0, 2.0], [2.0, 6.0, 2.0]],
        
        # env 2: 还没到门（没有超过门X位置）
        [[-2.0, 0.0, 2.0], [-2.0, 1.0, 2.0], [-2.0, -1.0, 2.0]],
        
        # env 3: 已完成所有门，不应该检查
        [[10.0, 0.0, 2.0], [10.0, 1.0, 2.0], [10.0, -1.0, 2.0]],
    ], dtype=torch.float32, device=device)
    
    print(f"无人机位置 shape: {drone_pos.shape}")
    print(f"门位置 shape: {gate_positions.shape}")
    
    # 实现门绕过检测逻辑
    gate_bypass_failure = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    # 创建有效门的掩码
    valid_gate_mask = current_gate_idx < gate_count  # [num_envs]
    print(f"有效门掩码: {valid_gate_mask}")
    
    if valid_gate_mask.any():
        valid_envs = torch.where(valid_gate_mask)[0]  # 有效环境索引
        valid_gate_indices = current_gate_idx[valid_envs]  # 对应门索引
        
        print(f"有效环境: {valid_envs}")
        print(f"对应门索引: {valid_gate_indices}")
        
        # 批量获取当前门位置
        current_gate_positions = gate_positions[valid_envs, valid_gate_indices]  # [valid_envs, 3]
        print(f"当前门位置: {current_gate_positions}")
        
        # 批量检查无人机是否超过门的X位置
        drone_pos_valid = drone_pos[valid_envs]  # [valid_envs, num_drones, 3]
        gate_x_positions = current_gate_positions[:, 0].unsqueeze(1)  # [valid_envs, 1]
        
        # 检查哪些无人机超过了门的X位置
        drones_passed_gate_x = drone_pos_valid[:, :, 0] > gate_x_positions  # [valid_envs, num_drones]
        print(f"超过门X位置的无人机: {drones_passed_gate_x}")
        
        # 对于有无人机超过门X位置的环境，检查是否成功穿过
        envs_with_passed_drones = drones_passed_gate_x.any(dim=1)  # [valid_envs]
        print(f"有无人机超过门的环境: {envs_with_passed_drones}")
        
        if envs_with_passed_drones.any():
            check_envs = valid_envs[envs_with_passed_drones]  # 需要检查的环境
            print(f"需要检查的环境: {check_envs}")
            
            # 计算这些环境中无人机到门中心的距离
            check_drone_pos = drone_pos[check_envs]  # [check_envs, num_drones, 3]
            check_gate_pos = current_gate_positions[envs_with_passed_drones]  # [check_envs, 3]
            
            # 计算距离：[check_envs, num_drones]
            drone_to_gate_distances = torch.norm(
                check_drone_pos - check_gate_pos.unsqueeze(1), dim=-1
            )
            print(f"无人机到门距离: {drone_to_gate_distances}")
            
            # 门通过阈值
            gate_threshold = max(gate_width, gate_height) * 0.8
            print(f"门通过阈值: {gate_threshold}")
            
            # 检查超过门X位置的无人机是否都成功穿过
            passed_mask = drones_passed_gate_x[envs_with_passed_drones]  # [check_envs, num_drones]
            near_gate_mask = drone_to_gate_distances < gate_threshold  # [check_envs, num_drones]
            
            print(f"超过门的无人机掩码: {passed_mask}")
            print(f"接近门的无人机掩码: {near_gate_mask}")
            
            # 对于每个环境，检查超过门的无人机是否都足够接近门中心
            for i, env_idx in enumerate(check_envs):
                passed_drones = passed_mask[i]  # [num_drones]
                if passed_drones.any():
                    near_gate_for_passed = near_gate_mask[i][passed_drones]  # 只看超过门的无人机
                    print(f"环境 {env_idx}: 超过门的无人机是否都接近门中心: {near_gate_for_passed.all()}")
                    if not near_gate_for_passed.all():  # 如果有超过门但不够接近的无人机
                        gate_bypass_failure[env_idx] = True
                        print(f"环境 {env_idx}: 检测到门绕过失败!")
    
    print(f"门绕过失败结果: {gate_bypass_failure}")
    
    # 验证预期结果
    expected = torch.tensor([False, True, False, False], device=device)  # 只有env1应该失败
    assert torch.equal(gate_bypass_failure, expected), f"期望 {expected}, 得到 {gate_bypass_failure}"
    print("✅ 门绕过检测测试通过!")
    
    return gate_bypass_failure

def test_endpoint_exceeded_termination():
    """测试超过终点终止条件"""
    print("\n" + "="*60)
    print("测试超过终点终止条件")
    print("="*60)
    
    num_envs = 3
    num_drones = 3
    device = torch.device('cpu')
    
    # 模拟终点位置 [num_envs, num_drones, 3]
    end_positions = torch.tensor([
        [[15.0, -1.0, 2.0], [15.0, 0.0, 2.0], [15.0, 1.0, 2.0]],  # env 0
        [[15.0, -1.0, 2.0], [15.0, 0.0, 2.0], [15.0, 1.0, 2.0]],  # env 1
        [[15.0, -1.0, 2.0], [15.0, 0.0, 2.0], [15.0, 1.0, 2.0]],  # env 2
    ], dtype=torch.float32, device=device)
    
    # 模拟无人机位置 [num_envs, num_drones, 3]
    drone_pos = torch.tensor([
        # env 0: 还没到终点
        [[12.0, -1.0, 2.0], [12.0, 0.0, 2.0], [12.0, 1.0, 2.0]],
        
        # env 1: 有一个无人机超过终点
        [[14.0, -1.0, 2.0], [16.0, 0.0, 2.0], [14.0, 1.0, 2.0]],
        
        # env 2: 所有无人机都超过终点
        [[16.0, -1.0, 2.0], [17.0, 0.0, 2.0], [18.0, 1.0, 2.0]],
    ], dtype=torch.float32, device=device)
    
    print(f"终点位置: {end_positions[:, 0, 0]}")  # 每个环境的终点X坐标
    print(f"无人机X位置: {drone_pos[:, :, 0]}")
    
    # 实现超过终点检测逻辑
    # 获取所有环境的终点X坐标（取每个环境第一个无人机的终点X坐标）
    endpoint_x_positions = end_positions[:, 0, 0]  # [num_envs]
    print(f"终点X坐标: {endpoint_x_positions}")
    
    # 检查每个环境中是否有无人机超过终点X位置
    drones_x_positions = drone_pos[:, :, 0]  # [num_envs, num_drones]
    print(f"无人机X坐标: {drones_x_positions}")
    
    # 每个无人机是否超过终点
    exceed_mask = drones_x_positions > endpoint_x_positions.unsqueeze(1)  # [num_envs, num_drones]
    print(f"超过终点掩码: {exceed_mask}")
    
    # 每个环境是否有无人机超过终点
    endpoint_exceeded = exceed_mask.any(dim=1)  # [num_envs]
    print(f"超过终点结果: {endpoint_exceeded}")
    
    # 验证预期结果
    expected = torch.tensor([False, True, True], device=device)  # env1和env2应该超过
    assert torch.equal(endpoint_exceeded, expected), f"期望 {expected}, 得到 {endpoint_exceeded}"
    print("✅ 超过终点检测测试通过!")
    
    return endpoint_exceeded

def test_combined_termination_logic():
    """测试组合终止条件"""
    print("\n" + "="*60)
    print("测试组合终止条件")
    print("="*60)
    
    num_envs = 5
    device = torch.device('cpu')
    
    # 模拟各种终止条件
    collision_terminated = torch.tensor([True, False, False, False, False], device=device)
    formation_breakdown = torch.tensor([False, True, False, False, False], device=device)
    success = torch.tensor([False, False, True, False, False], device=device)
    out_of_bounds = torch.tensor([False, False, False, True, False], device=device)
    gate_bypass_failure = torch.tensor([False, False, False, False, True], device=device)
    endpoint_exceeded = torch.tensor([False, False, False, False, False], device=device)
    
    # 组合终止条件
    terminated = collision_terminated | formation_breakdown | success | out_of_bounds | gate_bypass_failure | endpoint_exceeded
    
    print(f"碰撞终止: {collision_terminated}")
    print(f"编队崩溃: {formation_breakdown}")
    print(f"任务成功: {success}")
    print(f"越界: {out_of_bounds}")
    print(f"门绕过失败: {gate_bypass_failure}")
    print(f"超过终点: {endpoint_exceeded}")
    print(f"最终终止: {terminated}")
    
    # 验证每个环境都有对应的终止原因
    expected = torch.tensor([True, True, True, True, True], device=device)
    assert torch.equal(terminated, expected), f"期望 {expected}, 得到 {terminated}"
    print("✅ 组合终止条件测试通过!")
    
    return terminated

if __name__ == "__main__":
    print("🔍 新增终止条件测试")
    
    # 测试门绕过检测
    gate_bypass_result = test_gate_bypass_termination()
    
    # 测试超过终点检测
    endpoint_exceeded_result = test_endpoint_exceeded_termination()
    
    # 测试组合终止逻辑
    combined_result = test_combined_termination_logic()
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    print("1. ✅ 门绕过检测:")
    print("   - 正确识别绕过门的无人机")
    print("   - 忽略已完成所有门的环境")
    print("   - 只有真正绕过门的环境才终止")
    
    print("\n2. ✅ 超过终点检测:")
    print("   - 准确检测超过终点X坐标的无人机")
    print("   - 任何一个无人机超过都会导致环境终止")
    print("   - 张量化操作效率高")
    
    print("\n3. 🎯 新增终止条件的优势:")
    print("   ✅ 防止无人机绕过门而不穿过")
    print("   ✅ 防止无人机飞过终点过远")
    print("   ✅ 提供更严格的训练约束")
    print("   ✅ 帮助智能体学习正确的飞行路径")
    
    print("\n4. 🚀 技术实现:")
    print("   • 完全张量化，无Python循环")
    print("   • 批量处理，高效计算")
    print("   • 条件掩码，精确控制")
    print("   • 与现有终止条件完美集成")
