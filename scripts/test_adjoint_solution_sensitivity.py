import torch
import torch.nn as nn
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import time
import matplotlib.pyplot as plt
import os
import shutil

# --- 1. 定义神经网络 (PyTorch) ---
class CostNet(nn.Module):
    """一个简单的全连接网络，用于学习代价函数"""
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(CostNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs),
            # 使用 softplus 确保代价值为正
            nn.Softplus()
        )

    def forward(self, x):
        return self.network(x)

# --- 2. 定义系统动力学模型 (Acados) ---
def get_integrator_model() -> AcadosModel:
    """定义一个双积分器模型"""
    model = AcadosModel()
    model.name = 'double_integrator_nn_cost'

    # 状态: [position, velocity]
    p = ca.SX.sym('p')
    v = ca.SX.sym('v')
    x = ca.vertcat(p, v)
    model.x = x
    model.xdot = ca.SX.sym('xdot', 2, 1)

    # 控制: [force]
    u = ca.SX.sym('u')
    model.u = u

    # 神经网络权重将作为参数 p 传入
    # casadi符号的维度将在后面根据网络大小动态设置
    model.p = ca.SX.sym('p', 0, 0) 

    # 动态方程: xdot = [v, u/m] (假设质量 m=1.0)
    f_expl = ca.vertcat(v, u)
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl

    return model

# --- 3. 连接 PyTorch 和 CasADi 的桥梁 ---
def setup_casadi_nn_function(pytorch_net: nn.Module):
    """
    将一个PyTorch网络包装成一个可供Acados使用的CasADi函数。
    这通过在CasADi中用符号变量手动重建网络的前向传播过程来实现。
    """
    # 获取网络权重和偏置的初始值 (作为numpy数组)
    weights_and_biases = [param.data.numpy() for param in pytorch_net.parameters()]
    
    # 扁平化所有权重和偏置，创建一个大的参数向量
    flat_params = np.concatenate([p.flatten() for p in weights_and_biases])
    n_params = len(flat_params)
    
    # 定义CasADi的符号输入 (状态x, 控制u)
    cas_x = ca.SX.sym('x', pytorch_net.network[0].in_features - 1)
    cas_u = ca.SX.sym('u', 1)
    cas_xu = ca.vertcat(cas_x, cas_u)
    
    # 定义CasADi的符号参数 (代表网络的所有权重和偏置)
    cas_p = ca.SX.sym('p', n_params)
    
    # --- 在 CasADi 中重建网络的前向传播 ---
    current_out = cas_xu
    param_start_idx = 0
    
    # Layer 1: Linear + Tanh
    w1_shape = weights_and_biases[0].shape
    w1_size = w1_shape[0] * w1_shape[1]
    w1 = ca.reshape(cas_p[param_start_idx : param_start_idx + w1_size], w1_shape[1], w1_shape[0]).T
    param_start_idx += w1_size
    b1_shape = weights_and_biases[1].shape
    b1_size = b1_shape[0]
    b1 = cas_p[param_start_idx : param_start_idx + b1_size]
    param_start_idx += b1_size
    current_out = ca.tanh(ca.mtimes(w1, current_out) + b1)
    
    # Layer 2: Linear
    w2_shape = weights_and_biases[2].shape
    w2_size = w2_shape[0] * w2_shape[1]
    w2 = ca.reshape(cas_p[param_start_idx : param_start_idx + w2_size], w2_shape[1], w2_shape[0]).T
    param_start_idx += w2_size
    b2_shape = weights_and_biases[3].shape
    b2_size = b2_shape[0]
    b2 = cas_p[param_start_idx : param_start_idx + b2_size]
    current_out = ca.mtimes(w2, current_out) + b2

    # Output activation: Softplus
    nn_output = ca.log(1 + ca.exp(current_out))
    
    # 创建一个CasADi函数
    # 输入: [状态x, 控制u, 参数p (网络权重)]
    # 输出: 标量代价值
    nn_cost_func = ca.Function('nn_cost', [cas_x, cas_u, cas_p], [nn_output])
    
    return nn_cost_func, n_params

# --- 4. 设置并创建 Acados OCP 求解器 ---
def setup_acados_ocp(x0, N, tf, nn_cost_func, n_params):
    """配置并创建Acados OCP求解器"""
    ocp = AcadosOcp()
    ocp.model = get_integrator_model()
    
    # --- 维度设置 ---
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    ocp.dims.N = N
    # 兼容新版本 API: 在 solver_options 中设置 N_horizon
    ocp.solver_options.N_horizon = N
    
    # use global parameters for network weights
    ocp.dims.np_global = n_params
    ocp.model.p_global = ca.SX.sym('p_global', n_params)
    # default global param values
    ocp.p_global_values = np.zeros(n_params)

    # --- 代价函数设置 ---
    # 使用外部代价函数 (EXTERNAL)，由我们自己定义的NN函数提供
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL' # 终端代价也使用同一个NN
    
    # 将CasADi包装的NN函数赋给模型
    ocp.model.cost_expr_ext_cost = nn_cost_func(ocp.model.x, ocp.model.u, ocp.model.p_global)
    # 终端代价中，我们假设控制输入为0
    ocp.model.cost_expr_ext_cost_e = nn_cost_func(ocp.model.x, ca.SX.zeros(nu,1), ocp.model.p_global)
    
    # --- 约束设置 ---
    ocp.constraints.x0 = x0
    
    # --- 求解器选项 ---
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # use discrete-time dynamics to support parameter sensitivities
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP is more robust for this kind of problem
    ocp.solver_options.tf = tf

    # *********** 关键: 启用参数灵敏度计算 ***********
    # 启用全局参数灵敏度计算
    ocp.solver_options.with_solution_sens_wrt_params = True

    # 创建求解器
    json_file = 'acados_ocp_nn_cost.json'
    acados_solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True)
    
    return acados_solver

def cleanup_acados_files():
    """清理acados生成的中间文件"""
    if os.path.exists('c_generated_code'):
        shutil.rmtree('c_generated_code')
    for f in ['acados_ocp_nn_cost.json', 'acados_ocp_double_integrator_nn_cost.so']:
        if os.path.exists(f):
            os.remove(f)

# --- 5. 主训练循环 ---
def main():
    """主函数，执行训练过程"""
    # --- 超参数 ---
    N_horizon = 20           # MPC视界长度
    Tf = 2.0                 # 预测总时长
    N_sim = 5000               # 训练迭代次数
    x0 = np.array([3.0, 0.0])# 初始状态 [pos=3, vel=0]
    learning_rate = 0.005    # 学习率

    # --- 1. 初始化神经网络和优化器 ---
    nx = 2 # 状态维度
    nu = 1 # 控制维度
    net = CostNet(n_inputs=nx + nu, n_hidden=16, n_outputs=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # --- 2. 包装NN并获取参数数量 ---
    print("Setting up CasADi function from PyTorch network...")
    nn_casadi_func, n_params = setup_casadi_nn_function(net)
    print(f"Network has {n_params} parameters (weights + biases).")

    # --- 3. 创建Acados求解器 ---
    print("Setting up Acados OCP solver...")
    solver = setup_acados_ocp(x0, N_horizon, Tf, nn_casadi_func, n_params)
    
    # --- 4. 训练循环 ---
    cost_history = []
    print("\nStarting training loop...")
    start_time = time.time()
    
    for i in range(N_sim):
        # a. 将当前PyTorch网络权重设置给Acados求解器作为参数 `p`
        # set global parameters for this solve
        current_weights_flat = np.concatenate([p.data.numpy().flatten() for p in net.parameters()])
        solver.set_p_global_and_precompute_dependencies(current_weights_flat)

        # b. 求解OCP
        status = solver.solve()
        
        # c. 获取总成本 (用于记录和打印)
        total_cost = solver.get_cost()
        cost_history.append(total_cost)
        print(f"Iter {i:3d}/{N_sim} | Total MPC Cost: {total_cost:8.4f} | Solver Status: {status}")

        if status != 0:
            print(f"Warning: Acados solver returned status {status}. The OCP might be infeasible.")
            # 即使求解失败，有时仍能获得梯度信息，可以尝试继续

        # c1. 获取第0步的前向灵敏度 du0/dp
        # 需要在 solver 创建时启用 sens_forw=True
        sens = solver.eval_solution_sensitivity(
            stages=[0],
            with_respect_to="p_global",
            return_sens_x=False,
            return_sens_u=True,
            sanity_checks=False
        )
        sens_u0 = sens["sens_u"][0]
        print(f"Forward du0/dp shape: {sens_u0.shape}")

        # d. 计算总成本关于网络权重的梯度 (dCost/dp)
        # 使用伴随灵敏度接口 eval_adjoint_solution_sensitivity
        grad_p = solver.eval_adjoint_solution_sensitivity(
            seed_x=[],
            seed_u=[(0, np.ones((nu, 1)))],  # unit seed for stage 0 control input
            with_respect_to="p_global",
            sanity_checks=False
        )
        print(f"Adjoint dCost/dp shape: {grad_p.shape}")

        # e. 使用梯度更新PyTorch网络权重
        optimizer.zero_grad() # 清除旧梯度
        
        # 将acados计算出的梯度手动赋给PyTorch网络的各个参数
        param_start_idx = 0
        for param in net.parameters():
            num_param_elems = param.numel()
            # 从扁平的梯度向量中取出对应部分
            param_grad_flat = grad_p[param_start_idx : param_start_idx + num_param_elems]
            # 恢复形状并转换为torch张量
            param_grad = torch.from_numpy(param_grad_flat.reshape(param.shape)).float()
            
            # 赋值梯度
            param.grad = param_grad
            param_start_idx += num_param_elems

        optimizer.step() # PyTorch优化器根据梯度更新权重
        
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    
    # --- 6. 清理并可视化结果 ---
    cleanup_acados_files()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Training Iteration')
    plt.ylabel('Total MPC Cost')
    plt.title('Cost Evolution during Training via Differentiable MPC')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()