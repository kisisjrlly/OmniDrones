"""
可微分MPC求解器使用acados后端
支持线性和非线性问题，以及批处理求解
"""
from typing import Optional, Union, Dict, Any, Tuple
import os
import hashlib
import casadi as ca
import numpy as np
import scipy.linalg
import time
import multiprocessing

from timeit import default_timer as timer
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosOcpBatchSolver

# from acados_template.acados_ocp_solver import AcadosSolverOptions # Example if it's here

from .problems import ControlBoundedLqrProblem, ControlBoundedOcp


# 全局求解器缓存
_SOLVER_CACHE = {}
_BATCH_SOLVER_CACHE = {}

def clear_solver_cache():
    """清除所有求解器缓存"""
    _SOLVER_CACHE.clear()
    _BATCH_SOLVER_CACHE.clear()
    print("求解器缓存已清空")

def casadi_flatten(x: ca.SX) -> ca.SX:
    """将CasADi矩阵展平为列向量"""
    s = x.shape
    return x.reshape((s[0]*s[1], 1))

def create_ocp_from_problem(problem, x0_vals, seed=None, solver_options=None):
    """从问题定义创建AcadosOcp对象"""
    nx, nu = problem.nx, problem.nu
    nxu = nx + nu
    N_horizon = problem.N_horizon
    
    # 创建acados OCP模型
    ocp = AcadosOcp()
    
    # 设置模型和动力学
    # model = ocp.model
    ocp.model.name = 'ocp_problem'
    
    # 处理线性与非线性动力学
    is_linear = isinstance(problem, ControlBoundedLqrProblem)
    
    # 非线性动力学设置
    ocp.model.x = problem.dynamics.x
    ocp.model.u = problem.dynamics.u

    ocp.dims.N = N_horizon

    ocp.model.disc_dyn_expr = problem.dynamics.f_casadi_expr

    # 代价：阶段代价 EXTERNAL   
    ocp.cost.cost_type = 'EXTERNAL'
    H_mat = ca.SX.sym('H', nxu, nxu)
    h_vec = ca.SX.sym('h', nxu) # Linear cost term
    xu = ca.vertcat(ocp.model.x, ocp.model.u)
    # ocp.model.cost_expr_ext_cost = ca.mtimes([xu.T, H_mat, xu])
    ocp.model.cost_expr_ext_cost = 0.5 * ca.mtimes([xu.T, H_mat, xu]) + ca.mtimes([xu.T, h_vec])

    # # # 终端代价：EXTERNAL（仅用 H 的 x-x 子块）
    # ocp.cost.cost_type_e = 'EXTERNAL'
    # H_xx = H_mat[0:nx, 0:nx]
    # xN = ocp.model.x
    # ocp.model.cost_expr_ext_cost_e = ca.mtimes([xN.T, H_xx, xN])

    ocp.model.p_global = ca.vertcat(casadi_flatten(H_mat), h_vec)
    H_mat_val = scipy.linalg.block_diag(problem.cost.Q, problem.cost.R)
    h_vec_val = np.concatenate([problem.cost.q, problem.cost.r])
    ocp.p_global_values = np.concatenate([H_mat_val.flatten(order='F'), h_vec_val])
    print("ocp.p_global_values shape:", ocp.p_global_values.shape)

    ocp.constraints.lbu = problem.control_bounds.u_lower
    ocp.constraints.ubu = problem.control_bounds.u_upper
    ocp.constraints.idxbu = np.arange(nu)

    # 初始状态
    ocp.constraints.x0 = x0_vals[0,:]
    
    # 求解器选项
    ocp.solver_options.tf = N_horizon * problem.dynamics.dt
    ocp.solver_options.N_horizon = N_horizon
    
    # 应用用户选项
    if solver_options is not None:
        for key, value in solver_options.items():
            try:
                setattr(ocp.solver_options, key, value)
            except Exception as e:
                print(f"警告: 无法设置求解器选项 {key}: {e}")
    
    return ocp

def get_solver_key(problem, solver_options, batched, n_batch, num_threads, need_sens=False, has_batch_cost=False):
    """为求解器缓存生成唯一键 - 包含灵敏度信息"""
    # 创建关键求解器属性的字符串表示
    key_parts = [
        f"nx={problem.nx}",
        f"nu={problem.nu}",
        f"N_horizon={problem.N_horizon}",
        f"batched={batched}",
        f"need_sens={need_sens}",  # 添加灵敏度需求标识
        f"has_batch_cost={has_batch_cost}",  # 添加批量特定成本标识
    ]
    
    if batched:
        key_parts.extend([
            f"n_batch={n_batch}",
            f"num_threads={num_threads}"
        ])
    
    # 添加重要的求解器选项
    if solver_options is not None:
        for k, v in sorted(solver_options.items()):
            if k in ['nlp_solver_type', 'qp_solver', 'integrator_type', 'hessian_approx', 'qp_solver_cond_ric_alg']:
                key_parts.append(f"{k}={v}")
    
    # 添加关于动力学类型的信息
    key_parts.append(f"is_linear={isinstance(problem, ControlBoundedLqrProblem)}")
    
    # 创建唯一字符串并哈希它
    key_str = "|".join(str(part) for part in key_parts)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

# 在 solve_using_acados 函数开始处添加
def check_initial_state(x0_vals):
    """检查初始状态的数值稳定性"""
    for i, x0 in enumerate(x0_vals):
        # 检查是否有NaN或Inf
        if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
            raise ValueError(f"Initial state {i} contains NaN or Inf values: {x0}")
        
        # 检查四元数归一化
        quat = x0[3:7]
        quat_norm = np.linalg.norm(quat)
        if abs(quat_norm - 1.0) > 1e-3:
            print(f"Warning: Quaternion in initial state {i} not normalized: norm = {quat_norm}")
            x0[3:7] = quat / quat_norm
    
    return x0_vals

def solve_using_acados(problem: Union[ControlBoundedLqrProblem, ControlBoundedOcp], x0_vals, Q_q_np,
                      seed: Optional[np.ndarray]=None, batched=False,
                      verify_kkt_residuals=False, vebose_solver_creation=False,
                      num_threads=None, solver_options: Optional[Dict[str, Any]]=None,
                      reuse_solver=True):
    """
    Use acados to solve the optimization control problem, supporting linear and nonlinear problems
    
    Args:
        problem: control problem definition (linear or nonlinear)
        x0_vals: initial state values [batch_size, nx]
        seed: seed vector for sensitivity computation
        batched: whether to use batch processing mode
        verify_kkt_residuals: whether to verify KKT residuals
        vebose_solver_creation: whether to output detailed solver creation info
        num_threads: number of threads for batch mode
        solver_options: solver options dictionary
        reuse_solver: whether to reuse previously created solver
        
    Returns:
        x_batch_sol: state trajectory [N_horizon+1, batch_size, nx]
        u_batch_sol: control trajectory [N_horizon+1, batch_size, nu]
        timing: solve time
        du_dp_adj_batch: sensitivities (if computed)
    """
    # Parameter validation
    if verify_kkt_residuals and batched:
        raise NotImplementedError("Batch mode does not support KKT residual verification")
    if batched and num_threads is None:
        num_threads = 1  # Default thread count to 1 to avoid OpenMP warnings

    MAX_BATCH_SIZE = 4096
    
    # Extract problem parameters
    nx, nu = problem.nx, problem.nu
    N_horizon = problem.N_horizon

    # n_batch = x0_vals.shape[0] if seed is None else MAX_BATCH_SIZE
    n_batch = x0_vals.shape[0]
    
    # Check if we have batch-specific cost matrices
    has_batch_cost = False
    
    # Check if problem.cost.Q is batch-specific (has a batch dimension)

    assert len(Q_q_np.shape) == 2 and Q_q_np.shape[0] == n_batch, f"Expected batch-specific Q matrix for batch size {n_batch}, and len(Q_q_np.shape) == 2, got shape {Q_q_np.shape}"
    # Q is provided as batch-specific matrices [batch_size, nx, nx]
    has_batch_cost = True

    # print("here1")
    # assert len(R_diag_np.shape) == 2 and R_diag_np.shape[0] == n_batch, f"Expected batch-specific R matrix for batch size {n_batch}, and len(R_diag_np.shape) == 2, got shape {R_diag_np.shape}"
    # R is provided as batch-specific matrices [batch_size, nu, nu]
    has_batch_cost = True
    
    # Initialize sensitivity variables
    du_dp_adj_batch = None
    
    # Set up sensitivity options if needed
    need_sens = seed is not None
    if need_sens:
        assert solver_options is not None, "Solver options must be provided for sensitivity computation"
    
    # Generate solver cache key (if reusing)
    if reuse_solver:
        # Include information about whether we have batch-specific costs in the key
        solver_key = get_solver_key(problem, solver_options, batched, n_batch, num_threads, need_sens, has_batch_cost)
    # print("here2")
    # Create or retrieve batch solver
    if reuse_solver:
        # Check cache
        if solver_key in _BATCH_SOLVER_CACHE:
            solver, ocp = _BATCH_SOLVER_CACHE[solver_key]
        else:
            print("Creating new batch solver and caching")
            # Create OCP object
            ocp = create_ocp_from_problem(problem, x0_vals, seed, solver_options)
            print("ocp.solver_options,hessian_approx is: ", ocp.solver_options.hessian_approx)
            if need_sens:
                print("need sense to create acados solver")
                ocp.code_export_directory = "need_sense_c_generated_code"
                solver = AcadosOcpBatchSolver(
                    ocp, 
                    json_file = "acados_ocp_nlp_sens.json",
                    N_batch_max=n_batch, 
                    verbose=vebose_solver_creation, 
                    num_threads_in_batch_solve=num_threads if num_threads else 1
                )
            else:
                print("no need sense to create acados solver")
                ocp.code_export_directory = "no_need_sense_c_generated_code"
                solver = AcadosOcpBatchSolver(
                    ocp, 
                    json_file = "acados_ocp_nlp_no_sens.json",
                    N_batch_max=n_batch, 
                    verbose=vebose_solver_creation,
                    num_threads_in_batch_solve=num_threads if num_threads else 1
                )
            _BATCH_SOLVER_CACHE[solver_key] = (solver, ocp)
            # print("here4")
    else:
        # print("here6")
        # Always create new solver
        ocp = create_ocp_from_problem(problem, x0_vals, seed, solver_options)
        
        solver = AcadosOcpBatchSolver(
            ocp, 
            N_batch_max=n_batch, 
            verbose=vebose_solver_creation,
            num_threads_in_batch_solve=num_threads if num_threads else 1
        )
    
    # Start timing
    time_start = timer()
    
    # Initialize result arrays
    x_batch_sol = np.zeros((n_batch, N_horizon+1, nx
                            ))
    u_batch_sol = np.zeros((n_batch, N_horizon, nu))

    # Prepare for sensitivity computation if needed
    # if need_sens:
    #     seed_u_batch = np.tile(seed, (n_batch, 1, 1))
    #     seed_u_batch = seed_u_batch.transpose((0, 2, 1))
    #     n_seed = 1
        
    #     try:
    #         np_global = ocp.dims.np_global
    #         du_dp_adj_batch = np.zeros((n_seed, np_global))
    #     except Exception as e:
    #         print(f"Warning: Cannot determine parameter dimensions, sensitivity calculation may fail: {e}")
    #         du_dp_adj_batch = np.array([])

        # print("x0_vals shape is: ", x0_vals.shape)
        # print("Q_diag_np: ", Q_diag_np)
        # print("R_diag_np: ", R_diag_np)

    u_stage_guess = np.array([7.02396, 0.0, 0.0, 0.0])

    for j in range(n_batch):
        solver.ocp_solvers[j].set(0, 'lbx', x0_vals[j,:])
        solver.ocp_solvers[j].set(0, 'ubx', x0_vals[j,:])
        solver.ocp_solvers[j].set(0, 'u', u_stage_guess)

        # Create batch-specific cost matrices for this instance
        # Make sure the dimensions are correct - Q_diag_np[j] should be a vector of length nx
        Q_diag_values = Q_q_np[j][:nx]  # Flatten to 1D array
        R_diag_values = Q_q_np[j][nx:nx + nu]  # Flatten to 1D array
        q_values = Q_q_np[j][nx + nu:nx + nu + nx]  # Flatten to 1D array
        r_values = Q_q_np[j][nx + nu + nx:]  # Flatten to 1D array

        # Create diagonal matrices with correct dimensions
        batch_Q = np.zeros((nx, nx))
        np.fill_diagonal(batch_Q, Q_diag_values)
        
        batch_R = np.zeros((nu, nu))
        np.fill_diagonal(batch_R, R_diag_values)
        
        # Set cost for each stage
        # for stage in range(N_horizon + 1):
        # Create properly sized block diagonal matrix
        batch_W = np.zeros((nx+nu, nx+nu))
        batch_W[:nx, :nx] = batch_Q
        batch_W[nx:, nx:] = batch_R
        
        # Pass the parameter to the solver
        # solver.ocp_solvers[j].set(stage, "p", batch_W.flatten())

        # 验证矩阵是正定的
        try:
            np.linalg.cholesky(batch_W)
        except np.linalg.LinAlgError:
            raise ValueError(f"Warning: Cost matrix for batch {j} is not positive definite")

        # solver.ocp_solvers[j].set_p_global_and_precompute_dependencies(batch_W.flatten(order='F'))
        h_vec_val = np.concatenate([q_values, r_values])
        p_global_val = np.concatenate([batch_W.flatten(order='F'), h_vec_val])
        solver.ocp_solvers[j].set_p_global_and_precompute_dependencies(p_global_val)
            

    # Solve batch
    # print("here20")
    # solver.eval_params_jac(n_batch)
    solver.setup_qp_matrices_and_factorize(n_batch)
    status = solver.solve()
    if status != 0 and status is not None:
        print(f"Warning: Batch solve returned non-zero status: {status}")
    # for ocp_solver in solver.ocp_solvers:
    #     ocp_solver.print_statistics()
    #     print("qp stats:", ocp_solver.get_stats('qp_stat'))
    #     print("qp_diagnostics():", ocp_solver.get_stats('statistics'))
    # Get solutions for all batches
    x_batch_sol = solver.get_flat('x').reshape((n_batch, N_horizon+1, nx))
    u_batch_sol = solver.get_flat('u').reshape((n_batch, N_horizon, nu))
    # print("here22")
    # Fill in the last control action (typically unused)
    u_batch_sol = np.pad(u_batch_sol, ((0,0), (0,1), (0,0)), mode='edge')

    # Compute sensitivities in batch mode if needed
    if need_sens:
        # try:
        # print("here23")
        single_seed = np.eye(nu)
        seed_vec = np.repeat(single_seed[np.newaxis, :, :], n_batch, axis=0)
        p_sens_ = solver.eval_adjoint_solution_sensitivity(
            seed_x=None, 
            seed_u=[(0, seed_vec)], 
            with_respect_to="p_global",
            sanity_checks=True
        )

        # p_sens_ = np.array(
        #     [
        #         s.eval_solution_sensitivity(
        #             stages=0,
        #             with_respect_to="p_global",
        #             return_sens_u=True,
        #             return_sens_x=False,
        #             sanity_checks=False
        #         )["sens_u"]
        #         for s in solver.ocp_solvers[:n_batch]
        #     ]
        # ).reshape(n_batch, nu, (nx+nu) * (nx+nu))  # type:ignore
        # print("p_sens_ is: ", p_sens_)
        # print("p_sens_  shape is: ", p_sens_.shape)
        # print("p_sens_[..., :-1] shape is: ", p_sens_[..., :-1].shape)
        p_sens_H = p_sens_[..., :-nx-nu].reshape(n_batch, nu, nx+nu, nx+nu)
        # print("p_sens_ is: ", p_sens_)
        du_dp_adj_batch = np.concatenate([np.diagonal(p_sens_H, axis1=2, axis2=3), p_sens_[..., -nx-nu:].reshape(n_batch, nu, nx+nu)], axis=-1)
        # print("du_dp_adj_batch is: ", du_dp_adj_batch)
        # print("du_dp_adj_batch shape is: ", du_dp_adj_batch.shape)


    # Calculate total time
    timing = timer() - time_start
    x_batch_sol = x_batch_sol.transpose((1,0,2))
    u_batch_sol = u_batch_sol.transpose((1,0,2))

    return x_batch_sol, u_batch_sol, timing, du_dp_adj_batch