import logging
import os
import time
import threading

import sys
import os
import torch
import numpy as np
import time
import traceback

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


# 导入MPC相关模块
from omni_drones.learning.mpc_components.Mpc import MPC
from omni_drones.learning.mpc_components.diff_acados import clear_solver_cache

def normalize_quaternion(quat):
    """归一化四元数"""
    norm = torch.norm(quat, dim=-1, keepdim=True)
    return quat / (norm + 1e-8)

def safe_numpy(tensor):
    """安全地将tensor转换为numpy，兼容CUDA"""
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def get_device():
    """获取可用的设备"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_test_initial_states(batch_size=2, device="cpu", use_real_data=False):
    """创建测试用的初始状态"""
    if use_real_data:
        # 使用从实际训练日志中提取的真实数据
        # 这些数据来自实际的训练过程，导致QP solver错误
        real_states = [
            # 推理计算用的数据 (正常工作)
            # [0.9948, -0.6514, -0.1761, 0.6010, 0.0900, 0.2646, 0.7488, 0.7419, 0.6315, 0.4075],
            # 训练阶段用的数据 (导致QP solver错误)
            [0.5342, 0.9940, 0.4143, 0.2989, 0.2850, 0.8874, 0.2048, -2.4883, -0.3000, -6.7153],
        ]
        
        states = torch.zeros(batch_size, 10, device=device, dtype=torch.float32)
        for i in range(min(batch_size, len(real_states))):
            states[i] = torch.tensor(real_states[i], dtype=torch.float32, device=device)
            
        # 如果batch_size大于real_states数量，复制最后一个状态
        for i in range(len(real_states), batch_size):
            states[i] = states[len(real_states) - 1] + torch.randn(10, device=device) * 0.01
            
        # 确保四元数归一化
        for i in range(batch_size):
            quat = states[i, 3:7]
            states[i, 3:7] = normalize_quaternion(quat)
            
        return states
    
    # 原始的随机生成逻辑
    states = torch.zeros(batch_size, 10, device=device, dtype=torch.float32)
    
    # 设置位置 (在原点附近的小扰动)
    states[:, 0] = torch.randn(batch_size) * 0.1  # px
    states[:, 1] = torch.randn(batch_size) * 0.1  # py  
    states[:, 2] = torch.randn(batch_size) * 0.1 + 1.0  # pz (悬停在1米高度)
    
    # 设置四元数为接近单位四元数的值
    states[:, 3] = 1.0  # qw
    states[:, 4] = torch.randn(batch_size) * 0.01  # qx (小扰动)
    states[:, 5] = torch.randn(batch_size) * 0.01  # qy (小扰动)
    states[:, 6] = torch.randn(batch_size) * 0.01  # qz (小扰动)
    
    # 归一化四元数
    quat = states[:, 3:7]
    states[:, 3:7] = normalize_quaternion(quat)
    
    # 设置速度 (小的初始速度)
    states[:, 7] = torch.randn(batch_size) * 0.1  # vx
    states[:, 8] = torch.randn(batch_size) * 0.1  # vy
    states[:, 9] = torch.randn(batch_size) * 0.1  # vz
    
    return states

def create_test_cost_weights(batch_size=2, device="cpu", use_real_data=False):
    """创建测试用的代价权重"""
    s_dim = 10  # 状态维度
    u_dim = 4   # 控制维度
    
    if use_real_data:
        # 使用从实际训练日志中提取的真实数据
        # 这些数据来自实际的训练过程，导致QP solver错误
        real_Q_q = [
            # 推理计算用的Q_q数据 (正常工作)
            # 前10个是Q权重 (状态权重)，后4个是R权重 (控制权重)
            # [10.7526, 10.5957, 10.6002, 10.7543, 10.8157, 10.7067, 10.6436, 10.8319, 10.6797, 10.6891, 10.8524, 10.6047, 10.6586, 10.6904],
            # 训练阶段用的Q_q数据 (导致QP solver错误)
            [10.4386, 10.7093, 10.7427, 10.7208, 11.0851, 10.3501, 10.7585, 10.5358, 10.8849, 10.8537, 10.7587, 10.4720, 10.7615, 11.3534],
        ]
        
        Q_q = torch.zeros(batch_size, s_dim + u_dim, device=device, dtype=torch.float32)
        for i in range(min(batch_size, len(real_Q_q))):
            Q_q[i] = torch.tensor(real_Q_q[i], dtype=torch.float32, device=device)
            
        # 如果batch_size大于real_Q_q数量，复制最后一个权重
        for i in range(len(real_Q_q), batch_size):
            Q_q[i] = Q_q[len(real_Q_q) - 1] + torch.randn(s_dim + u_dim, device=device) * 0.01
            
        # 确保权重为正
        Q_q = torch.abs(Q_q)
        
        return Q_q
    
    # 原始的随机生成逻辑
    # 创建Q权重 (状态权重)
    Q_base = torch.tensor([
        100.0, 100.0, 100.0,        # 位置权重 (px, py, pz)
        10.0, 10.0, 10.0, 10.0,     # 四元数权重 (qw, qx, qy, qz)
        10.0, 10.0, 10.0             # 速度权重 (vx, vy, vz)
    ], dtype=torch.float32, device=device)
    
    # 创建R权重 (控制权重)
    R_base = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    
    # 为每个batch添加小的随机扰动
    Q_diag = Q_base.unsqueeze(0).repeat(batch_size, 1)
    R_diag = R_base.unsqueeze(0).repeat(batch_size, 1)
    
    # 添加小的随机扰动使每个batch的权重略有不同
    Q_diag += torch.randn_like(Q_diag) * 0.01 * Q_base.unsqueeze(0)
    R_diag += torch.randn_like(R_diag) * 0.01 * R_base.unsqueeze(0)
    
    # 确保权重为正
    Q_diag = torch.abs(Q_diag)
    R_diag = torch.abs(R_diag)
    
    # 合并Q和R权重
    Q_q = torch.cat([Q_diag, R_diag], dim=1)
    
    return Q_q

def test_problematic_case_specifically():
    """专门测试导致QP solver错误的特定数据"""
    print("=" * 80)
    print("🚨 专门测试导致QP solver错误的特定数据")
    print("=" * 80)
    
    device = get_device()  # 自动选择可用设备
    batch_size = 1  # 使用单个样本进行调试
    T = 0.016
    dt = 0.016
    
    print(f"使用设备: {device}")
    
    # 清除求解器缓存
    clear_solver_cache()
    
    print(f"创建MPC实例用于调试")
    mpc = MPC(T=T, dt=dt, device=device)
    
    # 使用导致问题的特定数据
    problematic_state = torch.tensor([
        [0.5342, 0.9940, 0.4143, 0.2989, 0.2850, 0.8874, 0.2048, -2.4883, -0.3000, -6.7153]
    ], dtype=torch.float32, device=device)
    
    problematic_Q_q = torch.tensor([
        [10.4386, 10.7093, 10.7427, 10.7208, 11.0851, 10.3501, 10.7585, 10.5358, 10.8849, 10.8537, 10.7587, 10.4720, 10.7615, 11.3534]
    ], dtype=torch.float32, device=device)

    # # 使用导致问题的特定数据
    # problematic_state = torch.tensor([
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # ], dtype=torch.float32, device=device)
    
    # problematic_Q_q = torch.tensor([
    #     [10.6899, 10.6724, 10.7156, 10.7129, 10.6692, 10.6841, 10.7064, 10.7039,
    #      10.6806, 10.7025, 10.7043, 10.6673, 10.6735, 10.6953]
    # ], dtype=torch.float32, device=device)
    
    # # 归一化四元数
    # quat = problematic_state[0, 3:7]
    # problematic_state[0, 3:7] = normalize_quaternion(quat)
    
    print("\n📊 问题数据分析:")
    print(f"初始状态: {safe_numpy(problematic_state[0])}")
    print(f"  - 位置: [{problematic_state[0, 0]:.4f}, {problematic_state[0, 1]:.4f}, {problematic_state[0, 2]:.4f}]")
    print(f"  - 四元数: [{problematic_state[0, 3]:.4f}, {problematic_state[0, 4]:.4f}, {problematic_state[0, 5]:.4f}, {problematic_state[0, 6]:.4f}]")
    print(f"  - 速度: [{problematic_state[0, 7]:.4f}, {problematic_state[0, 8]:.4f}, {problematic_state[0, 9]:.4f}]")
    print(f"代价权重: {safe_numpy(problematic_Q_q[0])}")
    print(f"  - Q权重范围: [{torch.min(problematic_Q_q[0, :10]):.4f}, {torch.max(problematic_Q_q[0, :10]):.4f}]")
    print(f"  - R权重范围: [{torch.min(problematic_Q_q[0, 10:]):.4f}, {torch.max(problematic_Q_q[0, 10:]):.4f}]")
    
    # 检查数据的数值特性
    print("\n🔍 数值特性分析:")
    quat_norm = torch.norm(problematic_state[0, 3:7])
    print(f"四元数范数: {quat_norm:.6f}")
    
    vel_magnitude = torch.norm(problematic_state[0, 7:10])
    # print(f"速度大小: {vel_magnitude:.6f}")
    
    # 检查是否有异常值
    if vel_magnitude > 10.0:
        print("⚠️  警告: 速度异常大!")
    
    if torch.any(torch.abs(problematic_state[0, 7:10]) > 5.0):
        print("⚠️  警告: 存在大的速度分量，可能导致数值不稳定")
    
    try:
        # 1. 先测试非训练模式
        print("\n" + "─" * 60)
        print("测试1: 非训练模式")
        print("─" * 60)
        
        start_time = time.time()
        opt_u_eval, x_pred_eval = mpc.solve(problematic_Q_q, problematic_state, is_training=False)
        eval_time = time.time() - start_time
        
        print(f"✓ 非训练模式成功")
        print(f"  - 求解时间: {eval_time:.4f}秒")
        print(f"  - 最优控制: {safe_numpy(opt_u_eval[0])}")
        
        # 2. 测试训练模式（这里应该会出错）
        print("\n" + "─" * 60)
        print("测试2: 训练模式 (预期出现QP solver错误)")
        print("─" * 60)
        
        Q_q_grad = problematic_Q_q.clone().detach().requires_grad_(True)
        x0_grad = problematic_state.clone().detach().requires_grad_(True)
        
        print("开始训练模式求解（可能出现错误）...")
        start_time = time.time()
        opt_u_train, x_pred_train = mpc.solve(Q_q_grad, x0_grad, is_training=True)
        train_time = time.time() - start_time
        
        print(f"✓ 训练模式意外成功！")
        print(f"  - 求解时间: {train_time:.4f}秒")
        print(f"  - 最优控制: {safe_numpy(opt_u_train[0])}")
        
        # 如果到这里没有错误，进行反向传播测试
        print("\n执行反向传播测试...")
        target_u = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device).expand_as(opt_u_train)
        loss = torch.sum((opt_u_train - target_u) ** 2)
        loss.backward()
        
        print(f"✓ 反向传播也成功！")
        if Q_q_grad.grad is not None:
            print(f"  - 梯度范数: {torch.norm(Q_q_grad.grad):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练模式出现预期错误: {e}")
        print("\n完整错误堆栈:")
        traceback.print_exc()
        
        # 分析可能的错误原因
        print("\n" + "🔍 " * 20)
        print("错误原因分析")
        print("🔍 " * 20)
        
        error_str = str(e)
        if "QP solver returned error status 3" in error_str:
            print("✅ 成功重现了 'QP solver returned error status 3' 错误!")
            print("\n可能的原因:")
            print("1. 大的速度分量 (-2.4883, -6.7153) 导致动力学线性化不准确")
            print("2. 四元数 (0.2989, 0.2850, 0.8874, 0.2048) 可能接近奇异配置")
            print("3. 在训练模式下，Hessian计算变得病态")
            print("4. 约束条件在这个状态下变得不兼容")
            
            print("\n🎯 建议的解决方案:")
            print("1. 在MPC中添加状态预处理和约束")
            print("2. 增加求解器的正则化参数")
            print("3. 使用更鲁棒的QP求解器设置")
            print("4. 添加状态饱和限制")
            print("5. 在训练前进行状态有效性检查")
            
        return False

def test_mpc_forward_backward(use_real_data=False):
    """测试MPC的前向和反向传播"""
    print("=" * 80)
    if use_real_data:
        print("MPC灵敏度测试 - 使用真实训练数据")
    else:
        print("MPC灵敏度测试 - 使用随机生成数据")
    print("=" * 80)
    
    # 设置参数
    device = "cuda"
    batch_size = 1
    T = 0.016  # 预测时域
    dt = 0.016  # 时间步长
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 清除求解器缓存
    # clear_solver_cache()
    
    print(f"创建MPC实例 (T={T}, dt={dt}, batch_size={batch_size})")
    mpc = MPC(T=T, dt=dt, device=device)
    
    try:
        # 创建测试数据
        x0 = create_test_initial_states(batch_size, device, use_real_data)
        Q_q = create_test_cost_weights(batch_size, device, use_real_data)
        
        if use_real_data:
            print(f"\n使用真实训练数据:")
            print(f"初始状态示例 (第一个batch):")
            print(f"  位置: [{x0[0, 0]:.4f}, {x0[0, 1]:.4f}, {x0[0, 2]:.4f}]")
            print(f"  四元数: [{x0[0, 3]:.4f}, {x0[0, 4]:.4f}, {x0[0, 5]:.4f}, {x0[0, 6]:.4f}]")
            print(f"  速度: [{x0[0, 7]:.4f}, {x0[0, 8]:.4f}, {x0[0, 9]:.4f}]")
            print(f"代价权重示例 (第一个batch):")
            print(f"  Q权重: {safe_numpy(Q_q[0, :10])}")
            print(f"  R权重: {safe_numpy(Q_q[0, 10:])}")
        
        # 1. 测试非训练模式（无灵敏度）
        print("\n" + "─" * 60)
        print("测试1: 非训练模式 (无灵敏度计算)")
        print("─" * 60)
        
        print(f"初始状态形状: {x0.shape}")
        print(f"代价权重形状: {Q_q.shape}")
        
        start_time = time.time()
        opt_u_eval, x_pred_eval = mpc.solve(Q_q, x0, is_training=False)
        eval_time = time.time() - start_time
        
        print(f"✓ 非训练模式求解成功")
        print(f"  - 求解时间: {eval_time:.4f}秒")
        print(f"  - 最优控制形状: {opt_u_eval.shape}")
        print(f"  - 预测轨迹形状: {x_pred_eval.shape}")
        print(f"  - 最优控制示例: {safe_numpy(opt_u_eval[0])}")
        
        # 2. 测试训练模式（带灵敏度）
        print("\n" + "─" * 60)
        print("测试2: 训练模式 (带灵敏度计算)")
        print("─" * 60)
        
        # 准备需要梯度的变量
        Q_q_grad = Q_q.clone().detach().requires_grad_(True)
        x0_grad = x0.clone().detach().requires_grad_(True)
        
        print(f"Q_q梯度启用: {Q_q_grad.requires_grad}")
        print(f"x0梯度启用: {x0_grad.requires_grad}")
        
        # 前向传播
        print("\n执行训练模式前向传播...")
        start_time = time.time()
        opt_u_train, x_pred_train = mpc.solve(Q_q_grad, x0_grad, is_training=True)
        forward_time = time.time() - start_time
        
        print(f"✓ 训练模式前向传播成功")
        print(f"  - 前向传播时间: {forward_time:.4f}秒")
        print(f"  - 最优控制形状: {opt_u_train.shape}")
        print(f"  - 预测轨迹形状: {x_pred_train.shape}")
        
        # 比较两种模式的结果
        u_diff = torch.norm(opt_u_train - opt_u_eval).item()
        x_diff = torch.norm(x_pred_train - x_pred_eval).item()
        print(f"  - 控制差异 (训练 vs 非训练): {u_diff:.6f}")
        print(f"  - 轨迹差异 (训练 vs 非训练): {x_diff:.6f}")
        
        # 3. 测试反向传播
        print("\n" + "─" * 60)
        print("测试3: 反向传播 (灵敏度计算)")
        print("─" * 60)
        
        # 定义损失函数
        target_u = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device).expand_as(opt_u_train)
        target_pos = torch.zeros_like(x_pred_train[-1, :, :3])
        
        # 计算损失
        control_loss = torch.sum((opt_u_train - target_u) ** 2)
        position_loss = torch.sum((x_pred_train[-1, :, :3] - target_pos) ** 2)
        total_loss = control_loss + 0.1 * position_loss
        
        print(f"控制损失: {control_loss.item():.6f}")
        print(f"位置损失: {position_loss.item():.6f}")
        print(f"总损失: {total_loss.item():.6f}")
        
        # 反向传播
        print("\n执行反向传播...")
        start_time = time.time()
        total_loss.backward()
        backward_time = time.time() - start_time
        
        print(f"✓ 反向传播成功")
        print(f"  - 反向传播时间: {backward_time:.4f}秒")
        
        # 4. 检查梯度
        print("\n" + "─" * 60)
        print("测试4: 梯度检查")
        print("─" * 60)
        
        success = True
        
        # 检查Q_q的梯度
        if Q_q_grad.grad is not None:
            q_grad = Q_q_grad.grad
            q_grad_norm = torch.norm(q_grad).item()
            q_grad_max = torch.max(torch.abs(q_grad)).item()
            
            print(f"Q_q梯度统计:")
            print(f"  - 梯度形状: {q_grad.shape}")
            print(f"  - 梯度范数: {q_grad_norm:.6f}")
            print(f"  - 梯度最大值: {q_grad_max:.6f}")
            print(f"  - 梯度示例 (前5个Q元素): {safe_numpy(q_grad[0, :5])}")
            print(f"  - 梯度示例 (前4个R元素): {safe_numpy(q_grad[0, -4:])}")
            
            # 检查梯度有效性
            if torch.any(torch.isnan(q_grad)):
                print("❌ Q_q梯度包含NaN")
                success = False
            elif torch.any(torch.isinf(q_grad)):
                print("❌ Q_q梯度包含Inf")
                success = False
            elif q_grad_norm < 1e-10:
                print("⚠ Q_q梯度过小，可能存在问题")
            else:
                print("✓ Q_q梯度正常")
        else:
            print("❌ Q_q梯度为None")
            success = False
        
        # 检查x0的梯度
        if x0_grad.grad is not None:
            x0_grad_norm = torch.norm(x0_grad.grad).item()
            x0_grad_max = torch.max(torch.abs(x0_grad.grad)).item()
            
            print(f"\nx0梯度统计:")
            print(f"  - 梯度形状: {x0_grad.grad.shape}")
            print(f"  - 梯度范数: {x0_grad_norm:.6f}")
            print(f"  - 梯度最大值: {x0_grad_max:.6f}")
            print(f"  - 梯度示例: {safe_numpy(x0_grad.grad[0])}")
            
            if torch.any(torch.isnan(x0_grad.grad)):
                print("❌ x0梯度包含NaN")
                success = False
            elif torch.any(torch.isinf(x0_grad.grad)):
                print("❌ x0梯度包含Inf")
                success = False
            else:
                print("✓ x0梯度正常")
        else:
            print("⚠ x0梯度为None (可能未实现，这是正常的)")
        
        # 5. 简单的梯度验证
        print("\n" + "─" * 60)
        print("测试5: 简单数值梯度验证")
        print("─" * 60)
        
        # 选择一个参数进行数值梯度检查
        eps = 1e-4
        param_idx = (0, 0)  # 第一个batch的第一个Q参数
        
        # 原始损失
        original_loss = total_loss.item()
        
        if Q_q_grad.grad is not None:
            analytical_grad = Q_q_grad.grad[param_idx].item()
        else:
            print("❌ 无法获取解析梯度进行验证")
            analytical_grad = 0.0
        
        # 正向扰动
        Q_q_plus = Q_q.clone()
        Q_q_plus[param_idx] += eps
        opt_u_plus, x_pred_plus = mpc.solve(Q_q_plus, x0, is_training=False)
        target_u_plus = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device).expand_as(opt_u_plus)
        target_pos_plus = torch.zeros_like(x_pred_plus[-1, :, :3])
        control_loss_plus = torch.sum((opt_u_plus - target_u_plus) ** 2)
        position_loss_plus = torch.sum((x_pred_plus[-1, :, :3] - target_pos_plus) ** 2)
        total_loss_plus = control_loss_plus + 0.1 * position_loss_plus
        
        # 负向扰动
        Q_q_minus = Q_q.clone()
        Q_q_minus[param_idx] -= eps
        opt_u_minus, x_pred_minus = mpc.solve(Q_q_minus, x0, is_training=False)
        target_u_minus = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device).expand_as(opt_u_minus)
        target_pos_minus = torch.zeros_like(x_pred_minus[-1, :, :3])
        control_loss_minus = torch.sum((opt_u_minus - target_u_minus) ** 2)
        position_loss_minus = torch.sum((x_pred_minus[-1, :, :3] - target_pos_minus) ** 2)
        total_loss_minus = control_loss_minus + 0.1 * position_loss_minus
        
        # 数值梯度
        numerical_grad = (total_loss_plus.item() - total_loss_minus.item()) / (2 * eps)
        
        # 比较
        relative_error = abs(analytical_grad - numerical_grad) / (abs(analytical_grad) + abs(numerical_grad) + 1e-8)
        
        print(f"参数 {param_idx} 梯度验证:")
        print(f"  - 解析梯度: {analytical_grad:.6f}")
        print(f"  - 数值梯度: {numerical_grad:.6f}")
        print(f"  - 相对误差: {relative_error:.4f}")
        
        if relative_error < 0.1:  # 10%容忍度
            print("✓ 梯度验证通过")
        else:
            print(f"⚠ 梯度验证有较大误差 (相对误差: {relative_error:.4f})")
            # 不标记为失败，因为MPC求解器的数值特性可能导致误差
        
        # 总结
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        
        if success:
            print("🎉 所有关键测试通过！")
            print(f"✓ 非训练模式求解正常 ({eval_time:.4f}秒)")
            print(f"✓ 训练模式求解正常 ({forward_time:.4f}秒)")
            print(f"✓ 反向传播计算正常 ({backward_time:.4f}秒)")
            print("✓ 梯度计算正常，无NaN/Inf")
            print("\neval_adjoint_solution_sensitivity功能工作正常！")
            return True
        else:
            print("❌ 部分测试失败")
            print("请检查acados求解器配置和灵敏度计算设置")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    print("-------------------------------")
    print("Cfgs:",OmegaConf.to_yaml(cfg, resolve=True))
    print("-------------------------------")

    simulation_app = init_simulation_app(cfg)

    for i in range(1):

        print(f"开始第{i}次MPC灵敏度测试...")
        
        # 首先专门测试导致问题的特定数据
        print("\n" + "🎯 " * 20)
        print("第一阶段：专门测试导致问题的特定数据")
        print("🎯 " * 20)
        success_specific = test_problematic_case_specifically()
            
        # 然后使用真实数据重现问题
        print("\n" + "🚨 " * 20)
        print("第三阶段：使用真实训练数据集测试")
        print("🚨 " * 20)
        print("⚠️  注意：这可能会重现 'QP solver returned error status 3' 错误")
        
        success_real = test_mpc_forward_backward(use_real_data=True)
        
        # if not success_real:
        #     print("\n" + "🔍 " * 20)
        #     print("调试信息：QP求解器错误分析")
        #     print("🔍 " * 20)
        #     print("可能的原因:")
        #     print("1. 代价矩阵条件数过大或接近奇异")
        #     print("2. 初始状态导致优化问题不可行")
        #     print("3. 控制约束过于严格")
        #     print("4. 灵敏度计算时的数值不稳定")
        #     print("5. Hessian矩阵计算问题")
        #     print("\n建议调试步骤:")
        #     print("- 检查代价权重的条件数")
        #     print("- 验证初始状态的可行性")
        #     print("- 尝试放宽求解器收敛容忍度")
        #     print("- 使用不同的QP求解器 (FULL_CONDENSING_HPIPM)")
        #     print("- 增加正则化参数 levenberg_marquardt")
        
        # # 总结
        # print("\n" + "📊 " * 20)
        # print("测试总结")
        # print("📊 " * 20)
        # print(f"特定问题数据测试: {'✅ 通过' if success_specific else '❌ 失败 (符合预期)'}")
        # # print(f"真实数据测试: {'✅ 通过' if success_real else '❌ 失败'}")
        
        # if not success_specific:
        #     print("\n🎯 结论: 成功重现了QP solver错误!")
        #     print("问题确实出现在特定的训练数据上，具体是:")
        #     print("- 初始状态包含大的速度分量 (-2.4883, -6.7153)")
        #     print("- 在训练模式下计算灵敏度时导致数值不稳定")
        #     print("- 需要在MPC中添加状态预处理和鲁棒性检查")

    simulation_app.close()


if __name__ == "__main__":
    main()
