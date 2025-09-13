"""
Standard MPC for Passing through a dynamic gate using acados
"""
import time
import torch
import numpy as np
import casadi as ca
from torch import nn
from typing import Optional, Union, Dict, Any
import multiprocessing
from torch.autograd import Function

# Import acados solver
from .diff_acados import solve_using_acados
from .problems import NonlinearDiscreteDynamics, QuadraticCost, ControlBounds, ControlBoundedOcp


class QuadrotorMpcFunction(Function):
    """
    自定义PyTorch函数，实现可微分的MPC求解器
    支持批处理和梯度计算
    """
    
    @staticmethod
    def forward(ctx, x0, Q_q, dynamics_expr, ca_x, ca_u,
                thrust_min, thrust_max, w_max_xy, w_max_yaw,
                T, dt, quad_dynamics, gz=9.81, is_training=False):
        """
        前向传播：求解MPC问题
        
        Args:
            ctx: PyTorch上下文，用于保存反向传播需要的数据
            x0: 初始状态 [batch_size, s_dim]
            Q_diag: 状态二次权重对角线元素 [s_dim]
            R_diag: 控制二次权重对角线元素 [u_dim]
            q: 线性控制权重 [u_dim]
            r: 线性状态权重 [s_dim]
            dynamics_expr: CasADi动力学表达式
            ca_x, ca_u: CasADi符号变量
            thrust_min, thrust_max: 推力约束
            w_max_xy, w_max_yaw: 角速度约束
            T: 预测时域
            dt: 时间步长
            quad_dynamics: 四旋翼动力学模型
            gz: 重力加速度
            is_training: 是否计算灵敏度
            
        Returns:
            u_pred: 最优控制动作 [batch_size, u_dim]
            x_pred: 预测轨迹 [horizon+1, batch_size, s_dim]
        """

        # 开始计时
        # total_start_time = time.time()

        # 转换为NumPy数组
        x0_np = x0.detach().cpu().numpy()

        Q_q_np = Q_q.detach().cpu().numpy()

        # print("Q_diag is: ", Q_diag)
        # Q_diag_np = Q_diag.detach().cpu().numpy()
        # # print("R_diag is: ", R_diag)
        # R_diag_np = R_diag.detach().cpu().numpy()
        # q_np = q.detach().cpu().numpy()
        # r_np = r.detach().cpu().numpy()
        # print("x0_np is: ", x0_np.shape)
        # print("Q_diag_np is: ", Q_diag_np.shape)
        # print("R_diag_np is: ", R_diag_np.shape)
            
        # 计算批量大小和维度
        batch_size = x0_np.shape[0]
        # print("batch size and is_training is:", batch_size, is_training)

        # Handle dimensions correctly:
        # If Q_diag is [batch_size, n_state], maintain the batch dimension
        # Otherwise assume it's [n_state]

        s_dim = x0_np.shape[1]  # For batch-specific Q
        u_dim = int(Q_q_np.shape[1] / 2 - s_dim)  # For batch-specific R
        # print("s_dim is: ", s_dim)
        # print("u_dim is: ", u_dim)

        N_horizon = int(T/dt)
        # assert N_horizon == Q_diag_np.shape[1], "Q_diag must have the same horizon length as T/dt"
        
        # 设置控制约束
        u_lower = np.array([thrust_min, -w_max_xy, -w_max_xy, -w_max_yaw])
        u_upper = np.array([thrust_max, w_max_xy, w_max_xy, w_max_yaw])
        
        # 创建动力学模型
        start_time = time.time()
        dynamics = NonlinearDiscreteDynamics(
            f_pytorch=quad_dynamics,  # PyTorch模型
            f_casadi_expr=dynamics_expr,  # CasADi表达式
            x=ca_x,
            u=ca_u,
            dt=dt
        )
        # print(f"Dynamics creation time: {time.time() - start_time:.4f} seconds")

        # 创建成本函数
        start_time = time.time()

        # Batch-specific cost
        # For ACADOS solver, we'll pass the raw diagonal values and let solve_using_acados handle them
        # cost = QuadraticCost(
        #     Q=np.zeros((N_horizon * s_dim, N_horizon * s_dim)),  # Pass batch-specific Q diagonals directly
        #     R=np.zeros((N_horizon * u_dim, N_horizon * u_dim)),  # Pass batch-specific R diagonals directly
        #     r=np.zeros((N_horizon * s_dim)),
        #     q=np.zeros((N_horizon * u_dim))
        # )

        cost = QuadraticCost(
            Q=np.zeros((s_dim, s_dim)),  # 批量特定的Q对角线
            R=np.zeros((u_dim, u_dim)),  # 批量特定的R对角线
            r=np.zeros((s_dim)),  # 状态权重
            q=np.zeros((u_dim))   # 控制权重
        )
        
        # 创建控制约束
        control_bounds = ControlBounds(
            u_lower=u_lower,
            u_upper=u_upper
        )
        
        # 创建最优控制问题
        problem = ControlBoundedOcp(
            dynamics=dynamics,
            cost=cost,
            control_bounds=control_bounds,
            N_horizon=N_horizon
        )

        # print(f"OCP problem creation time: {time.time() - start_time:.4f} seconds")
        # 创建均匀分布的时间步数组
        # time_steps_array = np.ones(N_horizon) * dt
        # # 确保 qp_solver_cond_N 不大于 N_horizon
        # qp_cond_N = min(3, max(1, N_horizon - 1))  # 设置为比N小的值，但至少是1
        


        # 优化求解器配置
        solver_options = {
            # "qp_solver": "FULL_CONDENSING_QPOASES", 
            "nlp_solver_type": "SQP", 
            "qp_solver": "PARTIAL_CONDENSING_HPIPM",
            # "nlp_solver_type": "SQP_RTI", 
            "qp_solver_iter_max": 50,               # 减少最大迭代次数
            "qp_solver_warm_start": 0,               # 启用热启动
            "nlp_solver_max_iter": 100,              # 减少SQP迭代次数
            "print_level": 0,                      # 禁用打印
            "sim_method_num_steps": 1,             # 最小积分步数
            "sim_method_num_stages": 2,            # 使用RK2而不是RK4
            "levenberg_marquardt": 0.1,            # 增大正则化参数提高稳定性和速度
            "with_batch_functionality": True, # 启用批处理功能
            "tol": 1e-3,
            "integrator_type": "DISCRETE"
        }
        
        
        # 如果需要计算灵敏度，添加相关设置
        if is_training:
            solver_options.update({
                "with_solution_sens_wrt_params": True,
                # "with_value_sens_wrt_params": True,
                "hessian_approx": "EXACT",
                "qp_solver_cond_ric_alg": 0,
                "qp_solver_ric_alg": 1,
                "levenberg_marquardt": 0.0,
                "solution_sens_qp_t_lam_min": 1e-7,
            })
            # solver_options.update({
            #     "with_solution_sens_wrt_params": True,
            #     "hessian_approx": "EXACT",

            #     # 强化稳定性的关键选项（不影响“精确灵敏度”判定）
            #     "regularize_method": "PROJECT",
            #     "with_adaptive_levenberg_marquardt": True,
            #     "levenberg_marquardt": 1e-2,

            #     # 采用更稳健的全压缩 QP 与 funnel 线搜索
            #     "qp_solver": "FULL_CONDENSING_HPIPM",
            #     "globalization": "FUNNEL_L1PEN_LINESEARCH",

            #     # QP/SQP 上限略放宽，给线搜索空间
            #     "qp_solver_iter_max": 50,
            #     "nlp_solver_max_iter": 15,

            #     # 其余 EXACT 相关配置保留
            #     "qp_solver_cond_ric_alg": 0,
            #     "qp_solver_ric_alg": 1,

            #     # 可选收敛阈值（略放宽避免过早失败）
            #     "nlp_solver_tol_stat": 1e-4,
            #     "nlp_solver_tol_eq": 1e-4,
            #     "nlp_solver_tol_ineq": 1e-4,
            #     "nlp_solver_tol_comp": 1e-4,
            # })
            seed = np.ones(u_dim)  # 设置种子用于灵敏度计算
        else:
            seed = None
            
        # 使用batch模式当批次大小大于1且不需要灵敏度
        # use_batch = batch_size > 1 and not is_training
        use_batch = True
        num_threads = min(batch_size, multiprocessing.cpu_count()) * 2 if use_batch else 1
        # print("num threads is: ", num_threads)
        
        # try:
        start_time = time.time()
        # 调用solve_using_acados求解MPC问题，开启求解器重
        # print("solving with acados...")
        x_sol, u_sol, timing, sensitivities = solve_using_acados(
            problem,
            x0_np,
            Q_q_np,
            seed=seed,
            batched=use_batch,
            num_threads=num_threads,
            solver_options=solver_options,
            verify_kkt_residuals=False,
            vebose_solver_creation=True,
            # reuse_solver=True if not is_training else False # 开启求解器重用
            reuse_solver=True,  # 开启求解器重用
        )
        # print("time is:", timing)
        # print("total time used of solving with acados:", time.time() - start_time)
        # 将结果转换为PyTorch张量
        x_pred = torch.tensor(x_sol, dtype=torch.float32, device=x0.device)
        u_pred = torch.tensor(u_sol, dtype=torch.float32, device=x0.device)
        # print("u_pred is: ", u_pred)
        # print("x_pred is: ", x_pred)
        
        # 提取第一个控制作为最优控制
        opt_u = u_pred[0]
        
        # 保存上下文以便在backward中使用
        # Convert sensitivities from numpy to torch tensor if needed
        if sensitivities is not None and isinstance(sensitivities, np.ndarray):
            sensitivities = torch.tensor(sensitivities, dtype=torch.float32, device=x0.device)
        ctx.sensitivities = sensitivities
        ctx.s_dim = s_dim
        ctx.u_dim = u_dim
        ctx.batch_size = batch_size
        # ctx.save_for_backward(Q_diag, R_diag,)
        # print("total time used of solving with acados:", time.time() - total_start_time)
        # 返回结果
        return opt_u, x_pred
            
        # except Exception as e:
        #     raise RuntimeError(f"MPC solver error: {e}")
        #     # 如果求解失败，返回默认控制（重力平衡）
        #     default_u = torch.tensor([gz, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(batch_size, 1)
        #     default_x = x0.unsqueeze(0).repeat(N_horizon+1, 1, 1)
        #     return default_u, default_x
    
    @staticmethod
    def backward(ctx, grad_u, grad_x):
        """
        反向传播：计算MPC参数的梯度
        
        Args:
            ctx: 保存的上下文
            grad_u: 最优控制的梯度 [batch_size, u_dim]
            grad_x: 预测轨迹的梯度 [horizon+1, batch_size, s_dim] (currently unused)
            
        Returns:
            各输入参数的梯度
        """

        # assert hasattr(ctx, 'saved_tensors') and len(ctx.saved_tensors) != 0
        # print("------------------------begin backward-----------------")
        # 从上下文中恢复数据
        # Q_diag_param, R_diag_param = ctx.saved_tensors
        sensitivities = ctx.sensitivities
        s_dim = ctx.s_dim
        u_dim = ctx.u_dim
        batch_size = ctx.batch_size
        
        # 初始化梯度
        # grad_x0 will be zero as we don't have sensitivities w.r.t. x0 from this setup
        grad_x0 = torch.zeros(batch_size, s_dim, device=grad_u.device)
        
        # grad_Q and grad_R should match the shape of Q_diag_param and R_diag_param
        grad_Q = torch.zeros(batch_size, s_dim, device=grad_u.device)
        grad_R = torch.zeros(batch_size, u_dim, device=grad_u.device)

        grad_q = torch.zeros(batch_size, s_dim, device=grad_u.device)
        grad_r = torch.zeros(batch_size, u_dim, device=grad_u.device)

        # 如果没有有效的灵敏度，返回零梯度
        # if sensitivities is None:
        #     print("警告: sensitivities_flat 为 None 或为空。返回零梯度。")
        #     return (grad_x0, grad_Q, grad_R) + (None,) * 12
        assert sensitivities is not None, "Sensitivities must not be None"

        grad_Q_q = torch.einsum("bkj,bk->bj", sensitivities, grad_u)

        # print("grad_Q is: ", grad_Q)
        # print("grad_R is: ", grad_R)
        # print("grad_u is: ", grad_u)
        # print("grad_u shape:", grad_u.shape)
        # print("grad_Q_q is:", grad_Q_q)
        # print("grad_Q_q shape is:", grad_Q_q.shape)

        return (grad_x0, grad_Q_q) + (None,) * 12
class MpcModule(nn.Module):
    """
    可微分MPC模块，继承自torch.nn.Module
    可以作为神经网络的一部分进行端到端训练
    """

    def __init__(self, T, dt, mpc_x_dim, action_dim, device="cpu", gz=9.81,mass=0.716):
        """
        初始化MPC模块
        
        Args:
            T: 预测时域长度
            dt: 时间步长
            device: 计算设备
            gz: 重力加速度
        """
        super(MpcModule, self).__init__()
        self.device = torch.device(device)
        
        # 状态和控制维度
        self.quad_s0_dim = 10  # (px, py, pz, qw, qx, qy, qz, vx, vy, vz)
        self.s_dim = mpc_x_dim
        self.u_dim = action_dim

        # 时间常数
        self.T = T
        self.dt = dt
        self.N = int(self.T/self.dt)

        # 重力加速度
        self.gz = gz
        self.mass = mass

        # 四旋翼约束
        self.w_max_yaw = 6.0
        self.w_max_xy = 6.0
        self.thrust_min = 2.0
        self.thrust_max = 20.0
        
        # 创建可学习的代价权重参数
        # 初始值参考了原来的Q和R矩阵
        # Q_init = torch.tensor([
        #     100.0, 100.0, 100.0,   # delta_x, delta_y, delta_z
        #     10.0, 10.0, 10.0, 10.0, # delta_qw, delta_qx, delta_qy, delta_qz  
        #     10.0, 10.0, 10.0        # delta_vx, delta_vy, delta_vz
        # ], dtype=torch.float32, device=self.device)
        
        # R_init = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32, device=self.device)
        
        # 注册为可训练参数
        # self.Q_diag = nn.Parameter(Q_init)
        # self.R_diag = nn.Parameter(R_init)
        
        # 初始状态和控制
        # self.register_buffer("quad_s0", torch.tensor(
        #     [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        #     device=self.device))
        
        # self.register_buffer("quad_u0", torch.tensor(
        #     [9.81, 0.0, 0.0, 0.0], 
        #     device=self.device))
        
        # 初始化四旋翼动力学
        self._init_quad_dynamics()
        
        # 创建CasADi符号变量 - 用于acados接口
        self._create_casadi_model()
        
        # 获取系统的CPU核心数，用于并行计算
        try:
            self.num_threads = multiprocessing.cpu_count()
        except:
            self.num_threads = 4  # 默认值
    
    def clear_solver_cache(self):
        """清除求解器缓存，释放内存"""
        from .diff_acados import clear_solver_cache
        clear_solver_cache()

    def _create_casadi_model(self):
        """创建CasADi模型并通过RK4预离散化"""
        # 创建状态和控制变量
        self.ca_x = ca.SX.sym('x', self.s_dim)
        self.ca_u = ca.SX.sym('u', self.u_dim)
        
        # 提取状态分量
        px, py, pz = self.ca_x[0], self.ca_x[1], self.ca_x[2]
        qw, qx, qy, qz = self.ca_x[3], self.ca_x[4], self.ca_x[5], self.ca_x[6]
        vx, vy, vz = self.ca_x[7], self.ca_x[8], self.ca_x[9]
        
        # 提取控制分量
        thrust, wx, wy, wz = self.ca_u[0] / self.mass, self.ca_u[1], self.ca_u[2], self.ca_u[3]
        
        # 四元数导数
        quat_derivs_qw = -0.5 * (wx * qx + wy * qy + wz * qz)
        quat_derivs_qx = 0.5 * (wx * qw + wz * qy - wy * qz)
        quat_derivs_qy = 0.5 * (wy * qw - wz * qx + wx * qz)
        quat_derivs_qz = 0.5 * (wz * qw + wy * qx - wx * qy)
        
        # 速度导数
        vel_derivs_x = 2 * (qw * qy + qx * qz) * thrust
        vel_derivs_y = 2 * (qy * qz - qw * qx) * thrust
        vel_derivs_z = (qw**2 - qx**2 - qy**2 + qz**2) * thrust - self.gz
        
        # 位置导数 = 速度
        pos_derivs_x = vx
        pos_derivs_y = vy
        pos_derivs_z = vz
        
        # 定义连续时间动力学方程
        f_expl = ca.vertcat(
            pos_derivs_x, pos_derivs_y, pos_derivs_z,
            quat_derivs_qw, quat_derivs_qx, quat_derivs_qy, quat_derivs_qz,
            vel_derivs_x, vel_derivs_y, vel_derivs_z
        )
        
        # 保存连续时间动力学表达式（用于兼容性）
        self.ca_dynamics_expr = f_expl
        
        # 使用RK4方法离散化动力学
        # 创建ODE函数
        ode = ca.Function('ode', [self.ca_x[0:self.quad_s0_dim], self.ca_u], [f_expl])
        
        # 应用RK4积分
        dt = self.dt
        k1 = ode(self.ca_x[0:self.quad_s0_dim], self.ca_u)
        k2 = ode(self.ca_x[0:self.quad_s0_dim] + dt/2 * k1, self.ca_u)
        k3 = ode(self.ca_x[0:self.quad_s0_dim] + dt/2 * k2, self.ca_u)
        k4 = ode(self.ca_x[0:self.quad_s0_dim] + dt * k3, self.ca_u)
        
        # 计算RK4离散状态更新
        x_next = self.ca_x[0:self.quad_s0_dim] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # 四元数归一化（CasADi符号版本）
        quat_next = x_next[3:7]
        quat_norm = ca.sqrt(quat_next[0]**2 + quat_next[1]**2 + quat_next[2]**2 + quat_next[3]**2 + 1e-12)
        quat_normalized = quat_next / quat_norm
        
        # 构建归一化后的下一个状态
        self.ca_x_next = ca.vertcat(
            x_next[0:3],
            quat_normalized,
            x_next[7:10],
            self.ca_x[self.quad_s0_dim:]
        )
        
        # 保存离散时间动力学模型，用于ACADOS
        self.ca_discrete_dynamics = self.ca_x_next


    def _init_quad_dynamics(self):
        """初始化四旋翼动力学"""
        self.quad_dynamics = QuadrotorDynamics(self.gz, dt=self.dt).to(self.device)
    
    def forward(self, quad_s0, Q_q=None, is_training=False):
        """
        前向传播：求解MPC问题
        
        Args:
            quad_s0: 四旋翼当前状态 [batch_size, s_dim]
            Q_q: 代价矩阵权重，如果为None则使用模块的参数
            is_training: 是否计算灵敏度，用于训练
            
        Returns:
            opt_u: 最优控制动作
            x_pred: 预测轨迹
        """
        assert Q_q is not None 
        # 如果传入了Q_q，从中提取Q和R
        batch_size = Q_q.size(0)
        n_state = self.s_dim
        n_ctrl = self.u_dim
        n_sc = n_state + n_ctrl
        
        # Q_diag = Q_q[:, :n_state]
        # R_diag = Q_q[:, n_state:n_state+n_ctrl]
        # q = Q_q[:, n_state+n_ctrl:2*n_state+n_ctrl]
        # r = Q_q[:, 2*n_state+n_ctrl:]



        # 调用自定义函数求解MPC问题
        start_time = time.time()
        # print("quad_s0 is: ", quad_s0)
        # print("is_training is: ", is_training)
        # print("Running QuadrotorMpcFunction with Q_diag:", Q_diag, "R_diag:", R_diag)
        # print("R_diag  before running QuadrotorMpcFunction:", R_diag)
        opt_u, x_pred = QuadrotorMpcFunction.apply(
            quad_s0, Q_q,
            self.ca_discrete_dynamics, 
            self.ca_x, self.ca_u,
            self.thrust_min, self.thrust_max, self.w_max_xy, self.w_max_yaw,
            self.T, self.dt, self.quad_dynamics, self.gz, is_training
        )
        # print(f"Time taken for MPC computation: {time.time() - start_time:.4f} seconds")
        # if is_training == True:
        # print("--------------------opt_u shape:--------------------:",opt_u.shape)
        # print("quad_s0 is: ", quad_s0, quad_s0.dtype)
        # print("quad s0 0 is: ", quad_s0[0])
        # print("Q_q is: ", Q_q, Q_q.dtype)
        # print("opt_u after solve is: ", opt_u)
        # print("opt_u 0 after solve is: ", opt_u[0])
        # 确保返回的控制值在有效范围内
        opt_u = torch.clamp(
            opt_u.to(self.device),
            min=torch.tensor([self.thrust_min, -self.w_max_xy, -self.w_max_xy, -self.w_max_yaw], device=self.device),
            max=torch.tensor([self.thrust_max, self.w_max_xy, self.w_max_xy, self.w_max_yaw], device=self.device)
        )
        return opt_u, x_pred


# 导出API，保持与原始MPC类兼容
class MPC(nn.Module):
    """
    Nonlinear MPC using acados for quadrotor control
    """
    def __init__(self, T, dt, mpc_x_dim, action_dim, device="cpu"):
        """
        Nonlinear MPC for quadrotor control
        """
        super(MPC, self).__init__()
        # device = "cpu"
        self.mpc_module = MpcModule(T, dt, mpc_x_dim, action_dim, device)
        
        # 导出需要的属性，保持与原始API兼容
        self._s_dim = self.mpc_module.s_dim
        self._u_dim = self.mpc_module.u_dim
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)
        self._gz = self.mpc_module.gz
        self._w_max_yaw = self.mpc_module.w_max_yaw
        self._w_max_xy = self.mpc_module.w_max_xy
        self._thrust_min = self.mpc_module.thrust_min
        self._thrust_max = self.mpc_module.thrust_max
        self.device = self.mpc_module.device
        self.num_threads = self.mpc_module.num_threads

        import shutil, os
        shutil.rmtree("need_sense_c_generated_code", ignore_errors=True)
        shutil.rmtree("no_need_sense_c_generated_code", ignore_errors=True)
        shutil.rmtree("c_generated_code", ignore_errors=True)
        # os.remove("acados_ocp_nlp_no_sens.json")
        # os.remove("acados_ocp_nlp_sens.json")
        # os.remove("acados_ocp_nlp.json")

    def solve(self, Q_q, quad_s0, is_training=False):
        """
        求解MPC问题，使用简化的二次代价函数 J = x^T Q x + u^T R u
        
        Args:
            Q_q: 代价矩阵，shape为[batch_size, N*(s_dim+u_dim)]，包含所有时间步的状态和控制权重
            quad_s0: 四旋翼当前状态 [s_dim]
                
        Returns:
            opt_u: 最优控制动作
            x_pred: 预测轨迹
        """
        return self.mpc_module(quad_s0, Q_q, is_training)
    
    def forward(self, Q_q, quad_s0, is_training=False):
        """
        与solve相同，用于PyTorch模块兼容
        """
        return self.solve(Q_q, quad_s0, is_training)

class QuadrotorDynamics(nn.Module):
    """
    四旋翼动力学模型
    实现基于四元数的四旋翼动力学，适用于MPC优化
    """
    
    def __init__(self, gz=9.81, dt=0.05):
        """
        初始化四旋翼动力学模型
        
        Args:
            gz: 重力加速度
            dt: 时间步长
        """
        super().__init__()
        self.register_buffer("gz", torch.tensor(gz, dtype=torch.float32))
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))

    def forward(self, x, u):
        """
        执行四旋翼动力学的前向传播
        
        Args:
            x: 当前状态
            u: 控制输入
        
        Returns:
            x_next: 下一个状态
        """
        # 确保输入张量至少是二维的
        x_orig_shape = x.shape
        u_orig_shape = u.shape
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批次维度
        if u.dim() == 1:
            u = u.unsqueeze(0)  # 添加批次维度
            
        # RK4 积分
        k1 = self.state_derivatives(x, u, self.gz)
        k2 = self.state_derivatives(x + 0.5 * self.dt * k1, u, self.gz)
        k3 = self.state_derivatives(x + 0.5 * self.dt * k2, u, self.gz)
        k4 = self.state_derivatives(x + self.dt * k3, u, self.gz)
        
        # 使用可微分操作更新状态
        x_next_unnorm = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 四元数归一化
        quat = x_next_unnorm[:, 3:7]
        quat_norm = torch.sqrt(torch.sum(quat**2, dim=1, keepdim=True) + 1e-12)
        quat_normalized = quat / quat_norm
        
        # 构建最终状态向量
        x_next = torch.cat([
            x_next_unnorm[:, :3],
            quat_normalized,
            x_next_unnorm[:, 7:10]
        ], dim=1)
        
        # 恢复原始形状（如果需要）
        if len(x_orig_shape) == 1:
            x_next = x_next.squeeze(0)
        
        return x_next

    @torch.jit.script
    def state_derivatives(x_state, u_state, gz):
        """
        计算状态导数
        
        Args:
            x_state: 当前状态
            u_state: 控制输入
            gz: 重力加速度
            
        Returns:
            状态导数
        """
        # 确保输入张量至少是二维的
        x_orig_shape = x_state.shape
        u_orig_shape = u_state.shape
        
        if x_state.dim() == 1:
            x_state = x_state.unsqueeze(0)  # 添加批次维度
        if u_state.dim() == 1:
            u_state = u_state.unsqueeze(0)  # 添加批次维度
            
        # 使用 narrow 替代切片，保持梯度
        pos_derivs = x_state.narrow(1, 7, 3)  # 位置导数是速度
        
        # 使用 narrow 获取四元数
        quat = x_state.narrow(1, 3, 4)
        
        # 解包控制量，保持梯度
        u_padded = u_state
        if u_padded.size(1) < 4:  # 确保控制向量至少有4个元素
            padding = torch.zeros(u_padded.size(0), 4 - u_padded.size(1), device=u_padded.device)
            u_padded = torch.cat([u_padded, padding], dim=1)
            
        thrust_s, wx_s, wy_s, wz_s = torch.unbind(u_padded, dim=1)
        thrust_s = thrust_s.view(-1, 1)
        wx_s = wx_s.view(-1, 1)
        wy_s = wy_s.view(-1, 1)
        wz_s = wz_s.view(-1, 1)
        
        # 解包四元数，保持梯度
        qw_s = quat.narrow(1, 0, 1)
        qx_s = quat.narrow(1, 1, 1)
        qy_s = quat.narrow(1, 2, 1)
        qz_s = quat.narrow(1, 3, 1)
        
        # 计算四元数导数
        quat_derivs_qw = -wx_s * qx_s - wy_s * qy_s - wz_s * qz_s
        quat_derivs_qx = wx_s * qw_s + wz_s * qy_s - wy_s * qz_s
        quat_derivs_qy = wy_s * qw_s - wz_s * qx_s + wx_s * qz_s
        quat_derivs_qz = wz_s * qw_s + wy_s * qx_s - wx_s * qy_s
        
        quat_derivs = torch.cat([
            quat_derivs_qw, quat_derivs_qx, quat_derivs_qy, quat_derivs_qz
        ], dim=1) * 0.5
        
        # 计算速度导数
        vel_derivs_x = 2 * (qw_s * qy_s + qx_s * qz_s) * thrust_s
        vel_derivs_y = 2 * (qy_s * qz_s - qw_s * qx_s) * thrust_s
        vel_derivs_z = (qw_s**2 - qx_s**2 - qy_s**2 + qz_s**2) * thrust_s - gz.view(1, 1)
        
        vel_derivs = torch.cat([vel_derivs_x, vel_derivs_y, vel_derivs_z], dim=1)
        
        # 合并所有导数
        derivs = torch.cat([pos_derivs, quat_derivs, vel_derivs], dim=1)
        
        # 恢复原始形状（如果需要）
        if len(x_orig_shape) == 1:
            derivs = derivs.squeeze(0)
            
        return derivs
    
    def grad_input(self, x, u):
        """
        计算动力学相对于输入的雅可比矩阵 (Jacobian)，使用完全向量化的计算
        
        Args:
            x: 当前状态 [batch_size, n_state]
            u: 控制输入 [batch_size, n_ctrl]
            
        Returns:
            R: 状态对状态的雅可比矩阵 [batch_size, n_state, n_state]
            S: 状态对控制的雅可比矩阵 [batch_size, n_state, n_ctrl]
        """
        # 确保输入张量至少是二维的
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批次维度
        if u.dim() == 1:
            u = u.unsqueeze(0)  # 添加批次维度
        
        batch_size = x.size(0)
        n_state = 10  # 状态维度: (px, py, pz, qw, qx, qy, qz, vx, vy, vz)
        n_ctrl = 4    # 控制维度: (c_thrust, wx, wy, wz)
        
        # 初始化雅可比矩阵
        R = torch.zeros(batch_size, n_state, n_state, device=x.device)
        S = torch.zeros(batch_size, n_state, n_ctrl, device=x.device)
        
        # 从状态中提取各个部分
        pos = x[:, :3]  # 位置: px, py, pz
        quat = x[:, 3:7]  # 四元数: qw, qx, qy, qz
        vel = x[:, 7:10]  # 速度: vx, vy, vz
        
        # 分解四元数
        qw = quat[:, 0].unsqueeze(1)
        qx = quat[:, 1].unsqueeze(1)
        qy = quat[:, 2].unsqueeze(1)
        qz = quat[:, 3].unsqueeze(1)
        
        # 分解控制输入
        thrust = u[:, 0].unsqueeze(1)
        wx = u[:, 1].unsqueeze(1)
        wy = u[:, 2].unsqueeze(1)
        wz = u[:, 3].unsqueeze(1)
        
        # ------ 计算状态对状态的雅可比矩阵 R ------
        
        # 位置对速度的导数 (∂pos/∂vel) - RK4积分下的影响
        # 为每个批次创建一个单位矩阵，并按位置乘以dt
        for i in range(3):
            # 批量设置 ∂position_i/∂velocity_i = dt
            R[:, i, i+7] = self.dt
        
        # 四元数对四元数的导数 (∂quat/∂quat)
        # qw对四元数的导数
        R[:, 3, 3] = 1.0 - 0.5 * self.dt * (wx**2 + wy**2 + wz**2).squeeze(1)
        R[:, 3, 4] = -0.5 * self.dt * wx.squeeze(1)
        R[:, 3, 5] = -0.5 * self.dt * wy.squeeze(1)
        R[:, 3, 6] = -0.5 * self.dt * wz.squeeze(1)
        
        # qx对四元数的导数
        R[:, 4, 3] = 0.5 * self.dt * wx.squeeze(1)
        R[:, 4, 4] = 1.0 - 0.5 * self.dt * (wy**2 + wz**2).squeeze(1)
        R[:, 4, 5] = 0.5 * self.dt * wz.squeeze(1)
        R[:, 4, 6] = -0.5 * self.dt * wy.squeeze(1)
        
        # qy对四元数的导数
        R[:, 5, 3] = 0.5 * self.dt * wy.squeeze(1)
        R[:, 5, 4] = -0.5 * self.dt * wz.squeeze(1)
        R[:, 5, 5] = 1.0 - 0.5 * self.dt * (wx**2 + wz**2).squeeze(1)
        R[:, 5, 6] = 0.5 * self.dt * wx.squeeze(1)
        
        # qz对四元数的导数
        R[:, 6, 3] = 0.5 * self.dt * wz.squeeze(1)
        R[:, 6, 4] = 0.5 * self.dt * wy.squeeze(1)
        R[:, 6, 5] = -0.5 * self.dt * wx.squeeze(1)
        R[:, 6, 6] = 1.0 - 0.5 * self.dt * (wx**2 + wy**2).squeeze(1)
        
        # 速度对四元数的导数 (∂vel/∂quat) - 考虑推力变换
        # vx对四元数的导数
        R[:, 7, 3] = 2 * self.dt * qy.squeeze(1) * thrust.squeeze(1)
        R[:, 7, 4] = 2 * self.dt * qz.squeeze(1) * thrust.squeeze(1)
        R[:, 7, 5] = 2 * self.dt * qw.squeeze(1) * thrust.squeeze(1)
        R[:, 7, 6] = 2 * self.dt * qx.squeeze(1) * thrust.squeeze(1)
        
        # vy对四元数的导数
        R[:, 8, 3] = -2 * self.dt * qx.squeeze(1) * thrust.squeeze(1)
        R[:, 8, 4] = -2 * self.dt * qw.squeeze(1) * thrust.squeeze(1)
        R[:, 8, 5] = 2 * self.dt * qz.squeeze(1) * thrust.squeeze(1)
        R[:, 8, 6] = 2 * self.dt * qy.squeeze(1) * thrust.squeeze(1)
        
        # vz对四元数的导数
        R[:, 9, 3] = 2 * self.dt * qw.squeeze(1) * thrust.squeeze(1)
        R[:, 9, 4] = -2 * self.dt * qx.squeeze(1) * thrust.squeeze(1)
        R[:, 9, 5] = -2 * self.dt * qy.squeeze(1) * thrust.squeeze(1)
        R[:, 9, 6] = 2 * self.dt * qz.squeeze(1) * thrust.squeeze(1)
        
        # ------ 计算状态对控制的雅可比矩阵 S ------
        
        # 四元数对角速度的导数 (∂quat/∂w)
        # qw对角速度的导数
        S[:, 3, 1] = -0.5 * self.dt * qx.squeeze(1)
        S[:, 3, 2] = -0.5 * self.dt * qy.squeeze(1)
        S[:, 3, 3] = -0.5 * self.dt * qz.squeeze(1)
        
        # qx对角速度的导数
        S[:, 4, 1] = 0.5 * self.dt * qw.squeeze(1)
        S[:, 4, 2] = -0.5 * self.dt * qz.squeeze(1)
        S[:, 4, 3] = 0.5 * self.dt * qy.squeeze(1)
        
        # qy对角速度的导数
        S[:, 5, 1] = 0.5 * self.dt * qz.squeeze(1)
        S[:, 5, 2] = 0.5 * self.dt * qw.squeeze(1)
        S[:, 5, 3] = -0.5 * self.dt * qx.squeeze(1)
        
        # qz对角速度的导数
        S[:, 6, 1] = -0.5 * self.dt * qy.squeeze(1)
        S[:, 6, 2] = 0.5 * self.dt * qx.squeeze(1)
        S[:, 6, 3] = 0.5 * self.dt * qw.squeeze(1)
        
        # 速度对推力的导数 (∂vel/∂thrust)
        # vx对推力的导数
        S[:, 7, 0] = 2 * self.dt * (qw * qy + qx * qz).squeeze(1)
        
        # vy对推力的导数
        S[:, 8, 0] = 2 * self.dt * (qy * qz - qw * qx).squeeze(1)
        
        # vz对推力的导数
        S[:, 9, 0] = self.dt * (qw**2 - qx**2 - qy**2 + qz**2).squeeze(1)
        
        return R, S
