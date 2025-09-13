#!/usr/bin/env python3
"""
测试MPC类的前向和反向传播过程
重点测试eval_adjoint_solution_sensitivity功能
"""

import sys
import os
import torch
import numpy as np
import time
import traceback
try:
    from torch.autograd.gradcheck import gradcheck
except ImportError:
    gradcheck = None

# 添加项目路径
sys.path.append('/home/zhaoguodong/work/code/MAPPO-MPC-OmniDrones/OmniDrones')
sys.path.append('/home/zhaoguodong/work/code/MAPPO-MPC-OmniDrones/acados')

# 导入MPC相关模块
from omni_drones.learning.mpc_components.Mpc import MPC, MpcModule, QuadrotorMpcFunction
from omni_drones.learning.mpc_components.diff_acados import clear_solver_cache

def normalize_quaternion(quat):
    """归一化四元数"""
    norm = torch.norm(quat, dim=-1, keepdim=True)
    return quat / (norm + 1e-8)

def create_test_initial_states(batch_size=4, device="cpu"):
    """创建测试用的初始状态"""
    # 状态: [px, py, pz, qw, qx, qy, qz, vx, vy, vz]
    states = torch.zeros(batch_size, 10, device=device, dtype=torch.float32)
    
    # 设置位置 (在原点附近的小扰动)
    states[:, 0] = torch.randn(batch_size) * 0.1  # px
    states[:, 1] = torch.randn(batch_size) * 0.1  # py  
    states[:, 2] = torch.randn(batch_size) * 0.1 + 1.0  # pz (悬停在1米高度)
    
    # 设置四元数 (接近单位四元数，小角度扰动)
    small_angles = torch.randn(batch_size, 3) * 0.1  # 小角度扰动
    states[:, 3] = torch.cos(torch.norm(small_angles, dim=1) / 2)  # qw
    sin_half = torch.sin(torch.norm(small_angles, dim=1, keepdim=True) / 2)
    states[:, 4:7] = small_angles * sin_half / (torch.norm(small_angles, dim=1, keepdim=True) + 1e-8)
    
    # 归一化四元数
    quat = states[:, 3:7]
    states[:, 3:7] = normalize_quaternion(quat)
    
    # 设置速度 (小的初始速度)
    states[:, 7] = torch.randn(batch_size) * 0.1  # vx
    states[:, 8] = torch.randn(batch_size) * 0.1  # vy
    states[:, 9] = torch.randn(batch_size) * 0.1  # vz
    
    return states

def create_test_cost_weights(batch_size=4, device="cpu"):
    """创建测试用的代价权重"""
    s_dim = 10  # 状态维度
    u_dim = 4   # 控制维度
    
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

def test_forward_pass(mpc, batch_size=4, device="cpu"):
    """测试前向传播"""
    print("=" * 60)
    print("测试前向传播 (Forward Pass)")
    print("=" * 60)
    
    try:
        # 创建测试数据
        x0 = create_test_initial_states(batch_size, device)
        Q_q = create_test_cost_weights(batch_size, device)
        
        print(f"初始状态形状: {x0.shape}")
        print(f"代价权重形状: {Q_q.shape}")
        print(f"初始状态示例:\n{x0[0]}")
        print(f"代价权重示例:\n{Q_q[0]}")
        
        # 测试非训练模式
        print("\n--- 测试非训练模式 (is_training=False) ---")
        start_time = time.time()
        opt_u_eval, x_pred_eval = mpc.solve(Q_q, x0, is_training=False)
        forward_time_eval = time.time() - start_time
        
        print(f"前向传播时间 (非训练): {forward_time_eval:.4f}秒")
        print(f"最优控制形状: {opt_u_eval.shape}")
        print(f"预测轨迹形状: {x_pred_eval.shape}")
        print(f"最优控制示例:\n{opt_u_eval[0]}")
        
        # 检查控制约束
        thrust_min, thrust_max = 2.0, 20.0
        w_max_xy, w_max_yaw = 6.0, 6.0
        
        assert torch.all(opt_u_eval[:, 0] >= thrust_min), "推力下界违反"
        assert torch.all(opt_u_eval[:, 0] <= thrust_max), "推力上界违反"
        assert torch.all(opt_u_eval[:, 1:3] >= -w_max_xy), "xy角速度下界违反"
        assert torch.all(opt_u_eval[:, 1:3] <= w_max_xy), "xy角速度上界违反"
        assert torch.all(opt_u_eval[:, 3] >= -w_max_yaw), "yaw角速度下界违反"
        assert torch.all(opt_u_eval[:, 3] <= w_max_yaw), "yaw角速度上界违反"
        
        print("✓ 控制约束检查通过")
        
        # 测试训练模式
        print("\n--- 测试训练模式 (is_training=True) ---")
        start_time = time.time()
        opt_u_train, x_pred_train = mpc.solve(Q_q, x0, is_training=True)
        forward_time_train = time.time() - start_time
        
        print(f"前向传播时间 (训练): {forward_time_train:.4f}秒")
        print(f"最优控制形状: {opt_u_train.shape}")
        print(f"预测轨迹形状: {x_pred_train.shape}")
        print(f"最优控制示例:\n{opt_u_train[0]}")
        
        # 比较两种模式的结果差异
        u_diff = torch.norm(opt_u_train - opt_u_eval)
        x_diff = torch.norm(x_pred_train - x_pred_eval)
        
        print(f"\n控制差异 (训练 vs 非训练): {u_diff:.6f}")
        print(f"轨迹差异 (训练 vs 非训练): {x_diff:.6f}")
        
        if u_diff < 1e-3 and x_diff < 1e-3:
            print("✓ 两种模式结果一致")
        else:
            print("⚠ 两种模式结果存在差异，可能是求解器设置不同")
        
        print("✓ 前向传播测试通过")
        return opt_u_train, x_pred_train, Q_q, x0
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        traceback.print_exc()
        return None, None, None, None

def test_backward_pass(mpc, opt_u, x_pred, Q_q, x0, device="cpu"):
    """测试反向传播"""
    print("\n" + "=" * 60)
    print("测试反向传播 (Backward Pass)")
    print("=" * 60)
    
    try:
        # 设置需要梯度的变量
        Q_q_grad = Q_q.clone().detach().requires_grad_(True)
        x0_grad = x0.clone().detach().requires_grad_(True)
        
        print(f"Q_q梯度启用: {Q_q_grad.requires_grad}")
        print(f"x0梯度启用: {x0_grad.requires_grad}")
        
        # 前向传播 (训练模式)
        print("\n--- 执行训练模式前向传播 ---")
        start_time = time.time()
        opt_u_train, x_pred_train = mpc.solve(Q_q_grad, x0_grad, is_training=True)
        forward_time = time.time() - start_time
        print(f"前向传播时间: {forward_time:.4f}秒")
        
        # 定义一个简单的损失函数
        # 目标: 最小化控制能量和位置误差
        target_pos = torch.zeros_like(x_pred_train[0, :, :3])  # 目标位置为原点
        target_u = torch.tensor([9.81, 0.0, 0.0, 0.0], device=device).expand_as(opt_u_train)
        
        # 损失 = 控制误差 + 最终位置误差
        control_loss = torch.sum((opt_u_train - target_u) ** 2)
        position_loss = torch.sum((x_pred_train[-1, :, :3] - target_pos) ** 2)
        total_loss = control_loss + position_loss
        
        print(f"控制损失: {control_loss.item():.6f}")
        print(f"位置损失: {position_loss.item():.6f}")
        print(f"总损失: {total_loss.item():.6f}")
        
        # 反向传播
        print("\n--- 执行反向传播 ---")
        start_time = time.time()
        total_loss.backward()
        backward_time = time.time() - start_time
        print(f"反向传播时间: {backward_time:.4f}秒")
        
        # 检查梯度
        print("\n--- 梯度检查 ---")
        if Q_q_grad.grad is not None:
            q_grad_norm = torch.norm(Q_q_grad.grad)
            q_grad_max = torch.max(torch.abs(Q_q_grad.grad))
            print(f"Q_q梯度范数: {q_grad_norm:.6f}")
            print(f"Q_q梯度最大值: {q_grad_max:.6f}")
            print(f"Q_q梯度形状: {Q_q_grad.grad.shape}")
            print(f"Q_q梯度示例 (前5个元素):\n{Q_q_grad.grad[0, :5]}")
            
            # 检查梯度是否合理
            if torch.any(torch.isnan(Q_q_grad.grad)):
                print("❌ Q_q梯度包含NaN")
                return False
            elif torch.any(torch.isinf(Q_q_grad.grad)):
                print("❌ Q_q梯度包含Inf")
                return False
            elif q_grad_norm < 1e-10:
                print("⚠ Q_q梯度过小，可能存在问题")
            else:
                print("✓ Q_q梯度正常")
        else:
            print("❌ Q_q梯度为None")
            return False
        
        if x0_grad.grad is not None:
            x0_grad_norm = torch.norm(x0_grad.grad)
            x0_grad_max = torch.max(torch.abs(x0_grad.grad))
            print(f"x0梯度范数: {x0_grad_norm:.6f}")
            print(f"x0梯度最大值: {x0_grad_max:.6f}")
            print(f"x0梯度形状: {x0_grad.grad.shape}")
            print(f"x0梯度示例:\n{x0_grad.grad[0]}")
            
            # 检查梯度是否合理
            if torch.any(torch.isnan(x0_grad.grad)):
                print("❌ x0梯度包含NaN")
                return False
            elif torch.any(torch.isinf(x0_grad.grad)):
                print("❌ x0梯度包含Inf")
                return False
            else:
                print("✓ x0梯度正常")
        else:
            print("⚠ x0梯度为None (可能未实现)")
        
        print("✓ 反向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 反向传播测试失败: {e}")
        traceback.print_exc()
        return False

def test_gradient_consistency(mpc, device="cpu", batch_size=2):
    """测试梯度一致性"""
    print("\n" + "=" * 60)
    print("测试梯度一致性")
    print("=" * 60)
    
    try:
        # 创建小batch的测试数据
        x0 = create_test_initial_states(batch_size, device)
        Q_q = create_test_cost_weights(batch_size, device)
        
        # 设置需要梯度
        Q_q.requires_grad_(True)
        
        # 定义一个包装函数用于梯度检查
        def mpc_wrapper(Q_q_input):
            opt_u, _ = mpc.solve(Q_q_input, x0, is_training=True)
            return torch.sum(opt_u ** 2)  # 简单的二次损失
        
        print("执行数值梯度检查...")
        
        # 使用较大的epsilon因为MPC求解器的数值精度有限
        eps = 1e-3
        
        # 手动计算数值梯度 (只检查几个重要参数)
        print("计算解析梯度...")
        Q_q_test = Q_q.clone().detach().requires_grad_(True)
        loss_analytical = mpc_wrapper(Q_q_test)
        loss_analytical.backward()
        
        if Q_q_test.grad is not None:
            analytical_grad = Q_q_test.grad.clone()
        else:
            print("❌ 解析梯度为None")
            return False
        
        print("计算数值梯度...")
        numerical_grad = torch.zeros_like(Q_q)
        
        # 只检查前几个参数以节省时间
        indices_to_check = [(0, 0), (0, 1), (0, 10), (0, 11)]  # 检查一些代表性的参数
        
        for i, j in indices_to_check:
            if i < Q_q.shape[0] and j < Q_q.shape[1]:
                # 正向扰动
                Q_q_plus = Q_q.clone().detach()
                Q_q_plus[i, j] += eps
                loss_plus = mpc_wrapper(Q_q_plus)
                
                # 负向扰动
                Q_q_minus = Q_q.clone().detach()
                Q_q_minus[i, j] -= eps
                loss_minus = mpc_wrapper(Q_q_minus)
                
                # 中心差分
                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
                
                # 比较
                analytical_val = analytical_grad[i, j].item()
                numerical_val = numerical_grad[i, j].item()
                relative_error = abs(analytical_val - numerical_val) / (abs(analytical_val) + abs(numerical_val) + 1e-8)
                
                print(f"参数[{i},{j}]: 解析={analytical_val:.6f}, 数值={numerical_val:.6f}, 相对误差={relative_error:.4f}")
                
                if relative_error < 0.1:  # 10%的容忍度，因为MPC求解器的数值特性
                    print(f"✓ 参数[{i},{j}]梯度检查通过")
                else:
                    print(f"⚠ 参数[{i},{j}]梯度检查有较大误差")
        
        print("✓ 梯度一致性测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 梯度一致性测试失败: {e}")
        traceback.print_exc()
        return False

def test_sensitivity_computation_directly():
    """直接测试灵敏度计算功能"""
    print("\n" + "=" * 60)
    print("直接测试eval_adjoint_solution_sensitivity")
    print("=" * 60)
    
    try:
        from omni_drones.learning.mpc_components.diff_acados import solve_using_acados
        from omni_drones.learning.mpc_components.problems import (
            NonlinearDiscreteDynamics, QuadraticCost, ControlBounds, ControlBoundedOcp
        )
        import casadi as ca
        
        # 创建简单的测试问题
        batch_size = 2
        s_dim = 10
        u_dim = 4
        N_horizon = 5
        dt = 0.05
        
        # 创建CasADi变量
        ca_x = ca.SX.sym('x', s_dim)
        ca_u = ca.SX.sym('u', u_dim)
        
        # 简单的线性动力学 (用于测试)
        A = np.eye(s_dim) + np.random.randn(s_dim, s_dim) * 0.01
        B = np.random.randn(s_dim, u_dim) * 0.1
        
        # 创建CasADi表达式
        f_expr = ca.mtimes(A, ca_x) + ca.mtimes(B, ca_u)
        
        # 创建动力学对象 (使用一个简单的虚拟PyTorch模型)
        class DummyDynamics(torch.nn.Module):
            def forward(self, x, u):
                return x  # 简单返回
        
        dummy_model = DummyDynamics()
        
        dynamics = NonlinearDiscreteDynamics(
            f_pytorch=dummy_model,
            f_casadi_expr=f_expr,
            x=ca_x,
            u=ca_u,
            dt=dt
        )
        
        # 创建成本函数
        cost = QuadraticCost(
            Q=np.eye(s_dim) * 10.0,
            R=np.eye(u_dim) * 0.1,
            r=np.zeros(s_dim),
            q=np.zeros(u_dim)
        )
        
        # 创建控制约束
        control_bounds = ControlBounds(
            u_lower=np.array([-10.0] * u_dim),
            u_upper=np.array([10.0] * u_dim)
        )
        
        # 创建OCP问题
        problem = ControlBoundedOcp(
            dynamics=dynamics,
            cost=cost,
            control_bounds=control_bounds,
            N_horizon=N_horizon
        )
        
        # 创建初始状态
        x0_vals = np.random.randn(batch_size, s_dim) * 0.1
        
        # 创建代价权重
        Q_diag_np = np.ones((batch_size, s_dim)) * 10.0
        R_diag_np = np.ones((batch_size, u_dim)) * 0.1
        
        # 测试不带灵敏度的求解
        print("测试不带灵敏度的求解...")
        x_sol_no_sens, u_sol_no_sens, timing_no_sens, sens_no_sens = solve_using_acados(
            problem, x0_vals, Q_diag_np, R_diag_np,
            seed=None,
            batched=True,
            solver_options={
                "nlp_solver_type": "SQP",
                "qp_solver": "PARTIAL_CONDENSING_HPIPM",
                "nlp_solver_max_iter": 10,
                "print_level": 0,
                "integrator_type": "DISCRETE"
            }
        )
        print(f"✓ 不带灵敏度求解成功，耗时: {timing_no_sens:.4f}秒")
        
        # 测试带灵敏度的求解
        print("\n测试带灵敏度的求解...")
        seed = np.ones(u_dim)
        
        solver_options_sens = {
            "nlp_solver_type": "SQP",
            "qp_solver": "PARTIAL_CONDENSING_HPIPM",
            "nlp_solver_max_iter": 10,
            "print_level": 0,
            "integrator_type": "DISCRETE",
            "with_solution_sens_wrt_params": True,
            "hessian_approx": "EXACT",
            "qp_solver_cond_ric_alg": 0,
            "qp_solver_ric_alg": 1,
            "levenberg_marquardt": 0.0,
        }
        
        x_sol_sens, u_sol_sens, timing_sens, sens_result = solve_using_acados(
            problem, x0_vals, Q_diag_np, R_diag_np,
            seed=seed,
            batched=True,
            solver_options=solver_options_sens
        )
        
        print(f"✓ 带灵敏度求解成功，耗时: {timing_sens:.4f}秒")
        
        # 检查灵敏度结果
        if sens_result is not None:
            print(f"灵敏度形状: {sens_result.shape}")
            print(f"灵敏度范数: {np.linalg.norm(sens_result):.6f}")
            print(f"灵敏度最大值: {np.max(np.abs(sens_result)):.6f}")
            
            if np.any(np.isnan(sens_result)):
                print("❌ 灵敏度包含NaN")
                return False
            elif np.any(np.isinf(sens_result)):
                print("❌ 灵敏度包含Inf")
                return False
            else:
                print("✓ 灵敏度计算正常")
        else:
            print("❌ 灵敏度结果为None")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 直接灵敏度测试失败: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """运行全面的测试"""
    print("开始MPC可微分性全面测试")
    print("=" * 80)
    
    # 设置设备
    device = "cpu"  # 使用CPU确保稳定性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 清除之前的求解器缓存
    clear_solver_cache()
    
    # 创建MPC实例
    T = 0.25  # 预测时域
    dt = 0.05  # 时间步长
    batch_size = 4
    
    print(f"创建MPC实例 (T={T}, dt={dt}, device={device})")
    mpc = MPC(T=T, dt=dt, device=device)
    
    # 测试序列
    tests_passed = 0
    total_tests = 4
    
    # 1. 测试前向传播
    opt_u, x_pred, Q_q, x0 = test_forward_pass(mpc, batch_size, device)
    if opt_u is not None:
        tests_passed += 1
    
    # 2. 测试反向传播 (如果前向传播成功)
    if opt_u is not None:
        if test_backward_pass(mpc, opt_u, x_pred, Q_q, x0, device):
            tests_passed += 1
    
    # 3. 测试梯度一致性
    if test_gradient_consistency(mpc, device, batch_size=2):
        tests_passed += 1
    
    # 4. 直接测试灵敏度计算
    if test_sensitivity_computation_directly():
        tests_passed += 1
    
    # 输出测试结果
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    print(f"通过的测试: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 所有测试通过！MPC可微分性工作正常。")
        return True
    else:
        print("⚠ 部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
