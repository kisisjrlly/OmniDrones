# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D
from torch.func import vmap
import math
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion, quat_axis
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni.isaac.core.prims import XFormPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, BoundedTensorSpec

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw
from pxr import UsdGeom, UsdPhysics, PhysxSchema

from omni_drones.controllers import (
    LeePositionController,
    AttitudeController,
    RateController
)

# from omni_drones.envs.platform.utils import create_frame

# Formation configurations - 以Y轴为对称轴，无人机编队横向分布，沿X轴飞行
# 编队配置：[x偏移, y偏移, z偏移]，x=0表示编队中心，y轴左右分布

TIGHT_FORMATION = [
    [0, 0, 0],      # 中心无人机
    [0, 1.2, 0],    # 右侧无人机 (+Y方向)
    [0, -1.2, 0],   # 左侧无人机 (-Y方向)
    [0, 2.4, 0],    # 右侧外围无人机
    [0, -2.4, 0],   # 左侧外围无人机
]

# 3架无人机的紧密编队 - 横向一字排开
TIGHT_FORMATION_3 = [
    [0, 0, 0],      # 中心无人机
    [0, 1.5, 0],    # 右侧无人机 (+Y方向)
    [0, -1.5, 0],   # 左侧无人机 (-Y方向)
]

# 5架无人机的宽松编队 - 横向分布更开
WIDE_FORMATION = [
    [0, 0, 0],      # 中心无人机
    [0, 2.0, 0],    # 右侧无人机
    [0, -2.0, 0],   # 左侧无人机
    [0, 4.0, 0],    # 右侧外围无人机
    [0, -4.0, 0],   # 左侧外围无人机
]

# V字型编队 - 以编队中心为顶点，向后展开
V_FORMATIO_5 = [
    [0, 0, 0],      # 领头无人机 (V字顶点)
    [-1.2, 1.2, 0], # 右后无人机
    [-1.2, -1.2, 0], # 左后无人机
    [-2.4, 2.4, 0], # 右后外围无人机
    [-2.4, -2.4, 0], # 左后外围无人机
]

# V字型编队 - 以编队中心为顶点，向后展开
V_FORMATION_3 = [
    [0, 0, 0],      # 领头无人机 (V字顶点)
    [-1.2, 1.2, 0], # 右后无人机
    [-1.2, -1.2, 0], # 左后无人机
]

# 雁形编队 - 斜线编队
WEDGE_FORMATION = [
    [0, 0, 0],      # 领头无人机
    [-0.8, 1.0, 0], # 右后无人机
    [-1.6, 2.0, 0], # 右后外围无人机
    [-0.8, -1.0, 0], # 左后无人机
    [-1.6, -2.0, 0], # 左后外围无人机
]

FORMATIONS = {
    "tight": TIGHT_FORMATION,
    "wide": WIDE_FORMATION,
    "v_shape": V_FORMATIO_5,
    "tight_3": TIGHT_FORMATION_3,  # 3架无人机选项
    "v_shape_3": V_FORMATION_3,  # 3架无人机的V字型编队
    "wedge": WEDGE_FORMATION,      # 雁形编队
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationGateTraversal(IsaacEnv):
    """
    Multi-drone formation maintenance and dynamic gate traversal environment.
    
    This environment combines formation control with gate traversal tasks. Multiple drones 
    must maintain formation while navigating through a series of dynamic gates that move 
    and rotate in space.

    ## Observation

    - `obs_self`: The state of the drone (position, orientation, velocity)
    - `obs_others`: The relative states of other drones in the formation
    - `gate_info`: Information about the current target gate (position, orientation, size, velocity)
    - `formation_target`: Target formation relative positions
    - `time_encoding`: Time encoding for episode progress

    ## Reward

    - `formation`: Reward for maintaining formation shape
    - `gate_progress`: Reward for progressing towards and through gates
    - `endpoint_progress`: Reward for progressing towards final endpoint positions
    - `collision_avoidance`: Penalty for getting too close to other drones
    - `gate_traversal`: Bonus reward for successfully passing through gates

    The total reward combines formation maintenance, gate traversal, and endpoint progress.

    ## Episode End

    The episode terminates when:
    - Any drone crashes
    - Drones get too close to each other
    - Formation deviates too much from target
    - All gates are successfully traversed (success)

    ## Config
    """
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.formation_tolerance = cfg.task.formation_tolerance
        self.gate_count = cfg.task.gate_count
        self.gate_spacing = cfg.task.gate_spacing
        # Support both old and new gate size parameters
        if hasattr(cfg.task, 'gate_width') and hasattr(cfg.task, 'gate_height'):
            self.gate_width = cfg.task.gate_width
            self.gate_height = cfg.task.gate_height
        else:
            # Fallback to old square gate_size parameter
            self.gate_width = cfg.task.gate_size
            self.gate_height = cfg.task.gate_size
        self.gate_radius = cfg.task.gate_radius
        self.gate_movement_speed = cfg.task.gate_movement_speed
        
        # Direct Learning Configuration (No Curriculum Learning)
        # Use default reward weights from config
        # self.velocity_reward_weight = cfg.task.velocity_reward_weight
        self.uprightness_reward_weight = cfg.task.uprightness_reward_weight
        # self.survival_reward_weight = cfg.task.survival_reward_weight
        # self.effort_reward_weight = cfg.task.effort_reward_weight
        self.formation_reward_weight = cfg.task.formation_reward_weight
        self.gate_reward_weight = cfg.task.gate_reward_weight
        self.endpoint_progress_reward_weight = cfg.task.endpoint_progress_reward_weight
        
        self.collision_penalty_weight = cfg.task.collision_penalty_weight
        self.reward_action_smoothness_weight = getattr(cfg.task, 'reward_action_smoothness_weight', 0.0)
        
        # Formation scale parameter for adjusting tightness
        self.formation_scale = getattr(cfg.task, 'formation_scale', 1.0)
        
        # Position parameters - 需要在 super().__init__() 之前定义，因为 _design_scene() 会用到
        self.start_x = cfg.task.start_x  # 起始X位置，在第一个门之前
        self.end_x_offset = - cfg.task.start_x  # 终点X位置偏移，在最后一个门之后

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        # Create view for kinematic control of gate movements across all environments
        self.gate_view = XFormPrimView(
            prim_paths_expr="/World/envs/env_*/Gate_*",
            name="gate_view",
            reset_xform_properties=False
        )
        # Initialize gate view after scene creation
        self.gate_view.initialize()

        # Initialize DebugDraw for trajectory visualization
        self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # Trajectory visualization buffers
        self.drone_trajectories = []  # List of trajectory points for each drone
        self.trajectory_max_length = 100  # Maximum trajectory length to display
        self.visualization_enabled = True  # Flag to enable/disable visualization
        
        # Drone trajectory history for each environment and drone
        self.drone_trajectory_history = torch.zeros(
            self.num_envs, self.drone.n, self.trajectory_max_length, 3, device=self.device
        )
        self.trajectory_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Start and end position markers
        self.start_positions = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.end_positions = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)

        # 初始化门状态跟踪张量
        # 重要：所有位置、速度、旋转都是基于门的中心点（Xform父对象位置）
        # 门的4个组件（2个立柱+2个横梁）相对于这个中心点进行偏移
        self.gate_positions = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device)         # 门中心位置 [环境数, 门数, XYZ坐标]
        self.gate_velocities = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device)        # 门中心速度 [环境数, 门数, XYZ速度]
        self.gate_rotations = torch.zeros(self.num_envs, self.gate_count, 4, device=self.device)         # 门旋转四元数 [环境数, 门数, XYZW四元数]
        self.gate_angular_velocities = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device) # 门角速度 [环境数, 门数, XYZ角速度]
        
        # Gate traversal tracking
        self.current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Reward tracking
        self.last_formation_error = torch.zeros(self.num_envs, device=self.device)
        self.last_gate_distance = torch.zeros(self.num_envs, device=self.device)
        # Change to track individual drone distances to endpoints
        self.last_endpoint_distances = torch.zeros(self.num_envs, self.drone.n, device=self.device)

        # 生成起始位置的候选点 - 在起始区域内随机分布
        self.cells = (
            make_cells([self.start_x - 2, -3, 0.5], [self.start_x + 2, 3, 2.5], [0.8, 0.8, 0.4])
            .flatten(0, -2)
            .to(self.device)
        )
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.], device=self.device) * torch.pi
        )

        # Formation and gate tracking
        self.formation_center_target = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Additional tracking variables for comprehensive stats
        self.prev_drone_velocities = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.drone.n, 4, device=self.device)
        self.gate_traversal_times = torch.zeros(self.num_envs, device=self.device)
        self.path_lengths = torch.zeros(self.num_envs, device=self.device)
        self.energy_consumption = torch.zeros(self.num_envs, device=self.device)
        self.near_collision_counter = torch.zeros(self.num_envs, device=self.device)
        self.ground_collision_counter = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_time = torch.zeros(self.num_envs, device=self.device)
        self.behavior_history = torch.zeros(self.num_envs, 10, device=self.device)  # Track behavioral patterns
        
        # Consistency and learning progress tracking
        self.recent_rewards = torch.zeros(self.num_envs, 20, device=self.device)  # Track recent rewards for consistency
        self.reward_buffer_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.exploration_counter = torch.zeros(self.num_envs, device=self.device)
        self.novelty_buffer = torch.zeros(self.num_envs, 5, 3, device=self.device)  # Track position novelty

    def _design_scene(self) -> Optional[List[str]]:
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller, self.device
        )

        scene_utils.design_scene()

        # Set up formation
        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                formation, device=self.device
            ).float()
        else:
            raise ValueError(f"Invalid target formation {formation}")

        # Apply formation scale parameter
        self.formation = self.formation * self.formation_scale

        # Spawn drones in formation at starting position
        # 设置无人机初始生成位置在起始区域
        start_pos = torch.tensor([self.start_x, 0.0, 2.0], device=self.device, dtype=torch.float32)
        spawn_positions = self.formation + start_pos
        self.drone.spawn(translations=spawn_positions)

        # Create dynamic gates
        self._create_dynamic_gates()

        return ["/World/defaultGroundPlane"]

    def _create_dynamic_gates(self):
        """
        Create a series of dynamic gates that move and rotate as unified objects.
        
        门的结构说明：
        - 每个门由4个部分组成：左柱子、右柱子、上横梁、下横梁
        - 所有部分都是 FixedCylinder 对象，作为 Xform 父对象的子对象
        - self.gate_positions 存储的是门的中心点位置（Xform父对象的位置）
        - 各个柱子和横梁相对于门中心点的偏移位置创建
        """
        self.gate_prims = []  # 存储门的原语信息列表
        
        for i in range(self.gate_count):
            # 门位置设置：从 X=0 开始，每隔 gate_spacing 放置一个门
            # 门位于起点(X=-8)和终点(X=+8)之间
            # 重新设计门的X位置分布，让门均匀分布在飞行路径中间
            total_flight_distance = 2 * (-self.start_x) # 从起点到终点的总距离 (-8 到 +8)
            usable_distance = total_flight_distance - 2.0  # 留出起点和终点的缓冲区域
            
            if self.gate_count == 1:
                # 单个门放在中间
                base_x = 0.0
            else:
                # 多个门平均分布
                gate_spacing_calculated = usable_distance / (self.gate_count - 1) if self.gate_count > 1 else 0
                base_x = -usable_distance/2 + i * gate_spacing_calculated
            
            base_y = 0.0                          # 门在Y轴方向居中
            base_z = 1 + self.gate_height / 2   # 门中心高度：地面偏移1米 + 门高度的一半

            # 创建门的统一变换对象（Xform），作为所有门组件的父对象
            # 这个Xform对象的位置就是 self.gate_positions 中存储的门中心位置
            gate_path = f"/World/envs/env_0/Gate_{i}"  # USD场景图中的路径
            if not prim_utils.is_prim_path_valid(gate_path):
                # 创建主要的门变换对象（这将是可移动的父对象）
                gate_prim = prim_utils.create_prim(gate_path, "Xform")
                
                # 设置门的初始位置（世界坐标）
                gate_xform = UsdGeom.Xformable(gate_prim)
                gate_xform.ClearXformOpOrder()  # 清除之前的变换操作
                translate_op = gate_xform.AddTranslateOp()  # 添加平移操作
                translate_op.Set((base_x, base_y, base_z))  # 设置门中心位置
                
                # 创建门组件作为子对象（它们将跟随父对象移动）
                # 门的几何参数
                post_height = self.gate_height    # 立柱高度 = 门的高度
                
                # 计算各个组件在世界坐标系中的绝对位置
                # 注意：FixedCylinder的position参数是世界坐标系的绝对位置，不是相对偏移
                
                # 左立柱 - 使用 FixedCylinder 作为子对象
                left_post_path = f"{gate_path}/left_post"
                left_post_world_pos = [base_x, base_y - self.gate_width/2, base_z]  # 世界坐标：门中心 + Y轴负偏移
                left_post = objects.FixedCylinder(
                    prim_path=left_post_path,
                    name=f"left_post_{i}",
                    position=left_post_world_pos,              # 世界坐标系绝对位置
                    orientation=[0.0, 0.0, 0.0, 1.0],         # 四元数：无旋转
                    radius=self.gate_radius,                        # 柱子半径
                    height=post_height,                        # 柱子高度
                    color=np.array([0.8, 0.2, 0.2]),         # RGB颜色：红色
                )

                # 右立柱 - 使用 FixedCylinder 作为子对象  
                right_post_path = f"{gate_path}/right_post"
                right_post_world_pos = [base_x, base_y + self.gate_width/2, base_z]  # 世界坐标：门中心 + Y轴正偏移
                right_post = objects.FixedCylinder(
                    prim_path=right_post_path,
                    name=f"right_post_{i}",
                    position=right_post_world_pos,             # 世界坐标系绝对位置
                    orientation=[0.0, 0.0, 0.0, 1.0],         # 四元数：无旋转
                    radius=self.gate_radius,                        # 柱子半径
                    height=post_height,                        # 柱子高度
                    color=np.array([0.8, 0.2, 0.2]),         # RGB颜色：红色
                )
                
                # 上横梁（水平圆柱体）- 使用 FixedCylinder 作为子对象
                top_bar_path = f"{gate_path}/top_bar"
                top_bar_world_pos = [base_x, base_y, base_z + self.gate_height/2]  # 世界坐标：门中心 + Z轴正偏移
                top_bar = objects.FixedCylinder(
                    prim_path=top_bar_path,
                    name=f"top_bar_{i}",
                    position=top_bar_world_pos,                # 世界坐标系绝对位置
                    orientation=[0.0, 0.0, 0.7071, 0.7071],  # 四元数：绕Z轴旋转90度（横向放置）
                    radius=self.gate_radius,                        # 横梁半径（与立柱相同）
                    height=self.gate_width,                    # 横梁长度 = 门宽度
                    color=np.array([0.8, 0.2, 0.2]),         # RGB颜色：红色
                )
                
                # 下横梁（水平圆柱体）- 使用 FixedCylinder 作为子对象
                bottom_bar_path = f"{gate_path}/bottom_bar"
                bottom_bar_world_pos = [base_x, base_y, base_z - self.gate_height/2]  # 世界坐标：门中心 + Z轴负偏移
                bottom_bar = objects.FixedCylinder(
                    prim_path=bottom_bar_path,
                    name=f"bottom_bar_{i}",
                    position=bottom_bar_world_pos,             # 世界坐标系绝对位置
                    orientation=[0.0, 0.0, 0.7071, 0.7071],  # 四元数：绕Z轴旋转90度（横向放置）
                    radius=self.gate_radius,                        # 横梁半径（与立柱相同）
                    height=self.gate_width,                    # 横梁长度 = 门宽度
                    color=np.array([0.8, 0.2, 0.2]),         # RGB颜色：红色
                )
                
                # 存储门信息用于跟踪和控制
                # 这个字典包含了门的所有组件引用，便于后续操作
                self.gate_prims.append({
                    'gate_prim': gate_prim,        # 门的主Xform对象（父变换）
                    'gate_path': gate_path,        # USD场景图路径
                    'left_post': left_post,        # 左立柱对象引用
                    'right_post': right_post,      # 右立柱对象引用
                    'top_bar': top_bar,            # 上横梁对象引用
                    'bottom_bar': bottom_bar,      # 下横梁对象引用
                })

        print(f"Created {len(self.gate_prims)} dynamic gates")


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        # Gate information: 4 corner positions (4*3=12) + linear velocity (3) + angular velocity (3) = 18 dimensions
        gate_info_dim = 4 * 3 + 3 + 3  # 18 dimensions
        # Gate information for central obs: 4 corner positions (4*3=12) + linear velocity (3) + angular velocity (3) = 18 dimensions  
        gate_central_dim = 4 * 3 + 3 + 3  # 18 dimensions
        # Formation target: position error from ideal formation position (3 dimensions)
        formation_target_dim = 3  # 改为3维：当前位置到理想位置的误差向量
        # Endpoint information: relative position to final target (3) + distance to endpoint (1) = 4 dimensions
        endpoint_info_dim = 4

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim)),
            "gate_info": UnboundedContinuousTensorSpec((1, gate_info_dim)),
            "formation_target": UnboundedContinuousTensorSpec((1, formation_target_dim)),
            "endpoint_info": UnboundedContinuousTensorSpec((1, endpoint_info_dim)),
        }).to(self.device)
        
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
            "gates": UnboundedContinuousTensorSpec((self.gate_count, gate_central_dim)),
            "formation": UnboundedContinuousTensorSpec((self.drone.n, 3)),
        }).to(self.device)
        
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)

        # {'agents': {'action': torch.Size([1024, 3, 4]),
        #             'observation': {'endpoint_info': torch.Size([1024, 3, 1, 4]),
        #                             'formation_target': torch.Size([1024, 3, 1, 3]),
        #                             'gate_info': torch.Size([1024, 3, 1, 18]),
        #                             'obs_others': torch.Size([1024, 3, 2, 23]),
        #                             'obs_self': torch.Size([1024, 3, 1, 23])},
        #             'observation_central': {'drones': torch.Size([1024, 3, 23]),
        #                                     'formation': torch.Size([1024, 3, 3]),
        #                                     'gates': torch.Size([1024, 1, 18])}},
        
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents","action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )
        
        # Comprehensive stats for monitoring RL training progress
        stats_spec = CompositeSpec({
            # Basic episode metrics
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            
            # Basic reward components
            "reward_uprightness": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
            
            # Formation reward components
            "reward_formation": UnboundedContinuousTensorSpec(1),
            "reward_cohesion": UnboundedContinuousTensorSpec(1),
            
            # Gate traversal reward components
            "reward_gate_progress": UnboundedContinuousTensorSpec(1),
            "reward_gate_traversal": UnboundedContinuousTensorSpec(1),
            
            # Endpoint reward components
            "reward_endpoint_progress": UnboundedContinuousTensorSpec(1),
            
            # Safety reward components
            "reward_collision_penalty": UnboundedContinuousTensorSpec(1),
            "reward_soft_collision_penalty": UnboundedContinuousTensorSpec(1),
            
            # Task completion reward components
            "reward_completion": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # 设置无人机起始位置 - 在门的一侧（X轴负方向）
        # 起始位置基于编队配置和随机化
        n_envs = len(env_ids)
        
        # 基础起始位置：编队中心位于起始区域
        base_start_pos = torch.tensor([self.start_x, 0.0, 2.0], device=self.device, dtype=torch.float32)
        
        # 添加小量随机化
        pos_noise = torch.randn(n_envs, 3, device=self.device) * 0.5
        pos_noise[:, 0] = torch.clamp(pos_noise[:, 0], -1.0, 1.0)  # X方向噪声限制在±1米
        
        # 计算每个环境的编队中心位置
        formation_centers = base_start_pos + pos_noise

        formation_centers = base_start_pos.unsqueeze(0)

        # print("formation centers shape:", formation_centers.shape)
        
        # 应用编队配置偏移 (已经包含formation_scale)
        formation_offset = self.formation.unsqueeze(0).expand(n_envs, -1, -1)
        pos = formation_centers.unsqueeze(1) + formation_offset
        
        # 随机朝向（主要是yaw角度）
        rpy = torch.zeros(n_envs, self.drone.n, 3, device=self.device)
        rpy[:, :, 2] = torch.rand(n_envs, self.drone.n, device=self.device) * 0.4 - 0.2  # yaw: ±0.2弧度
        rot = euler_to_quaternion(rpy)
        
        # 设置无人机位置和朝向
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(torch.zeros_like(self.drone.get_velocities()[env_ids]), env_ids)

        # 存储轨迹可视化的起点位置
        self.start_positions[env_ids] = pos + self.envs_positions[env_ids].unsqueeze(1)
        
        # 计算终点位置 - 在门的另一侧（X轴正方向）
        # 终点位置：最后一个门位置 + 偏移量
        if self.gate_count == 1:
            last_gate_x = 0.0 + self.end_x_offset
        else:
            # 计算最后一个门的X位置
            total_flight_distance = 16.0
            usable_distance = total_flight_distance - 4.0
            gate_spacing_calculated = usable_distance / (self.gate_count - 1)
            last_gate_x = -usable_distance/2 + (self.gate_count - 1) * gate_spacing_calculated
            last_gate_x += self.end_x_offset
        
        # 终点编队中心
        end_center = torch.tensor([last_gate_x, 0.0, 2.0], device=self.device)
        # 保持编队形状到达终点 (formation_offset已经包含scale)
        self.end_positions[env_ids] = (end_center.unsqueeze(0).unsqueeze(1) + formation_offset + 
                                     self.envs_positions[env_ids].unsqueeze(1))
        
        # 重置轨迹历史
        self.drone_trajectory_history[env_ids] = 0
        self.trajectory_step_count[env_ids] = 0

        # Reset gate tracking
        self.current_gate_idx[env_ids] = 0
        self.gates_passed[env_ids] = 0
        
        # Reset gate dynamics
        # self._reset_gate_dynamics(env_ids)
        
        # Reset reward tracking
        self.last_formation_error[env_ids] = 0
        
        # Initialize distance tracking with actual distances (not zero)
        # Get current drone positions after reset
        drone_pos, _ = self.drone.get_world_poses()
        formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
        
        # Initialize gate distance to first gate for each environment - 张量化版本

        # 获取所有重置环境的第一个门位置 [len(env_ids), 3]
        first_gate_positions = self.gate_positions[env_ids, 0]  
        # 计算对应环境的编队中心到第一个门的距离 [len(env_ids)]
        gate_distances = torch.norm(formation_center[env_ids] - first_gate_positions, dim=-1)
        self.last_gate_distance[env_ids] = gate_distances

        
        # Initialize endpoint distances - 张量化版本，计算每个无人机到对应终点的距离
        # drone_pos shape: [len(env_ids), n_drones, 3]
        # self.end_positions shape: [num_envs, n_drones, 3]
        drone_pos_reset = drone_pos[env_ids]  # [len(env_ids), n_drones, 3]
        end_positions_reset = self.end_positions[env_ids]  # [len(env_ids), n_drones, 3]
        # 计算每个无人机到其对应终点的距离 [len(env_ids), n_drones]
        individual_endpoint_distances = torch.norm(drone_pos_reset - end_positions_reset, dim=-1)
        self.last_endpoint_distances[env_ids] = individual_endpoint_distances
        
        # Reset comprehensive stats
        # Basic episode metrics
        self.stats["return"][env_ids] = 0

        # Basic reward components
        self.stats["reward_uprightness"][env_ids] = 0
        self.stats["reward_action_smoothness"][env_ids] = 0
        
        # Formation reward components
        self.stats["reward_formation"][env_ids] = 0
        self.stats["reward_cohesion"][env_ids] = 0
        
        # Safety reward components
        self.stats["reward_collision_penalty"][env_ids] = 0
        self.stats["reward_soft_collision_penalty"][env_ids] = 0
        
        # Task completion reward components
        self.stats["reward_completion"][env_ids] = 0
        
        # Endpoint reward components

        self.stats["reward_endpoint_progress"][env_ids] = 0
        self.stats["reward_completion"][env_ids] = 0

        
        # Reset additional tracking variables
        self.prev_drone_velocities[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.gate_traversal_times[env_ids] = 0
        self.energy_consumption[env_ids] = 0
        self.near_collision_counter[env_ids] = 0
        self.ground_collision_counter[env_ids] = 0
        self.episode_start_time[env_ids] = self.progress_buf[env_ids].float()
        self.behavior_history[env_ids] = 0
        self.recent_rewards[env_ids] = 0
        self.reward_buffer_ptr[env_ids] = 0
        self.exploration_counter[env_ids] = 0
        self.novelty_buffer[env_ids] = 0

    def _reset_gate_dynamics(self, env_ids: torch.Tensor):
        """
        重置门的位置、旋转和动力学状态。
        
        新的门位置策略：
        - 门位于起点(X=-8)和终点(X=+8)之间
        - 门均匀分布在飞行路径上
        - 门保持垂直于飞行方向（门面朝向X轴）
        
        Args:
            env_ids: 需要重置的环境ID列表
        """
        for i in range(self.gate_count):
            # 计算门的基础中心位置
            total_flight_distance = 16.0  # 从起点到终点的总距离
            usable_distance = total_flight_distance - 4.0  # 留出起点和终点的缓冲区域
            
            if self.gate_count == 1:
                base_x = 0.0  # 单个门放在中间
            else:
                gate_spacing_calculated = usable_distance / (self.gate_count - 1)
                base_x = -usable_distance/2 + i * gate_spacing_calculated
            
            base_y = 0.0  # 门在Y轴方向居中
            base_z = 1 + self.gate_height / 2  # 门中心高度

            # 添加轻微的随机化扰动
            pos_noise = torch.randn(len(env_ids), 3, device=self.device) * 0.3
            pos_noise[:, 0] = 0  # X轴位置保持固定，确保门的顺序正确
            pos_noise[:, 1] = torch.clamp(pos_noise[:, 1], -1.0, 1.0)  # Y轴扰动限制
            pos_noise[:, 2] = torch.clamp(pos_noise[:, 2], -0.5, 0.5)  # Z轴扰动限制
            
            # 设置门中心的最终位置
            self.gate_positions[env_ids, i] = torch.tensor([base_x, base_y, base_z], device=self.device, dtype=torch.float32) + pos_noise
            
            # 设置门的初始速度（动态运动）
            self.gate_velocities[env_ids, i] = torch.randn(len(env_ids), 3, device=self.device) * self.gate_movement_speed
            self.gate_velocities[env_ids, i, 0] = 0  # X方向不移动，保持门的顺序
            
            # 设置门的初始旋转（垂直于飞行方向）
            self.gate_rotations[env_ids, i] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).expand(len(env_ids), -1)
            
            # 设置门的角速度（缓慢振荡旋转）
            self.gate_angular_velocities[env_ids, i] = torch.randn(len(env_ids), 3, device=self.device) * 0.05
        
        # Set initial gate positions using the view
        if hasattr(self, 'gate_view') and self.gate_view is not None:
            try:
                # Prepare positions and orientations for reset environments
                reset_positions = self.gate_positions[env_ids].reshape(-1, 3)  # [len(env_ids) * gate_count, 3]
                reset_orientations = self.gate_rotations[env_ids].reshape(-1, 4)  # [len(env_ids) * gate_count, 4]
                
                # Create environment indices for gates in reset environments
                reset_env_indices = torch.repeat_interleave(
                    env_ids,
                    self.gate_count
                )
                
                # Update gate positions using the view
                self.gate_view.set_world_poses(
                    positions=reset_positions,
                    orientations=reset_orientations,
                    env_indices=reset_env_indices
                )
                
            except Exception as e:
                # Fallback to direct prim manipulation only for env_0 if it's being reset
                if 0 in env_ids:
                    for i in range(self.gate_count):
                        gate_path = f"/World/envs/env_0/Gate_{i}"
                        if prim_utils.is_prim_path_valid(gate_path):
                            gate_pos = self.gate_positions[0, i].cpu().numpy()
                            # Set position using USD
                            prim = prim_utils.get_prim_at_path(gate_path)
                            if prim:
                                xform = UsdGeom.Xformable(prim)
                                if xform:
                                    xform.ClearXformOpOrder()
                                    translate_op = xform.AddTranslateOp()
                                    translate_op.Set(tuple(gate_pos))

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # Store actions for stats calculations
        self.actions = actions.clone()
        # self._update_gate_dynamics()

        # Get current drone state
        drone_state = self.drone.get_state()[..., :13]
        
        # Use the controller to convert high-level actions to rotor commands
        # Depending on controller type, process the actions accordingly
        if isinstance(self.controller, LeePositionController):
            # For position controller: actions are [target_pos, target_yaw]
            current_pos, _ = self.drone.get_world_poses()
            target_pos = actions[..., :3] + current_pos  # Relative position target
            target_yaw = actions[..., 3:4] * torch.pi  # Scale to radians
            rotor_commands = self.controller.compute(
                drone_state, 
                target_pos=target_pos,
                target_yaw=target_yaw
            )
        elif isinstance(self.controller, AttitudeController):
            # For attitude controller: actions are [thrust, yaw_rate, roll, pitch]
            target_thrust = ((actions[..., 0:1] + 1) / 2).clip(0.) * self.controller.max_thrusts.sum(-1)
            target_yaw_rate = actions[..., 1:2] * torch.pi
            target_roll = actions[..., 2:3] * torch.pi/4  # Scale to reasonable range
            target_pitch = actions[..., 3:4] * torch.pi/4  # Scale to reasonable range
            rotor_commands = self.controller(
                drone_state,
                target_thrust=target_thrust,
                target_yaw_rate=target_yaw_rate,
                target_roll=target_roll,
                target_pitch=target_pitch
            )
        elif isinstance(self.controller, RateController):
            # For rate controller: actions are [rate_x, rate_y, rate_z, thrust]
            target_rate = actions[..., :3] * torch.pi  # Scale to radians
            target_thrust = ((actions[..., 3:4] + 1) / 2).clip(0.) * self.controller.max_thrusts.sum(-1)
            rotor_commands = self.controller(
                drone_state,
                target_rate=target_rate,
                target_thrust=target_thrust
            )
        else:
            # Default: pass through actions directly
            rotor_commands = actions
        
        # Handle NaN values
        torch.nan_to_num_(rotor_commands, 0.)
        self.effort = self.drone.apply_action(rotor_commands)


    def _update_gate_dynamics(self):
        """
        实时更新门的位置和旋转状态（每个时间步调用）。
        
        门的运动模式：
        1. Y轴方向：正弦波运动（左右摆动）
        2. Z轴方向：余弦波运动（上下移动）  
        3. X轴方向：固定不动（保持推进顺序）
        4. 旋转：绕Z轴缓慢振荡旋转
        
        所有运动都基于 self.gate_positions 中存储的门中心点位置进行。
        """
        dt = self.dt  # 时间步长
        
        for i in range(self.gate_count):
            # 计算基于时间的门速度（正弦和余弦函数实现周期性运动）
            # Y轴速度：正弦波，频率2.0，幅度由gate_movement_speed控制
            self.gate_velocities[:, i, 1] = torch.sin(self.progress_buf.float() * dt * 0.1) * self.gate_movement_speed
            # Z轴速度：余弦波，频率1.5，幅度减少到60%
            self.gate_velocities[:, i, 2] = torch.cos(self.progress_buf.float() * dt * 0.1) * self.gate_movement_speed * 0.6
            
            # 根据速度更新门中心位置
            self.gate_positions[:, i] += self.gate_velocities[:, i] * dt
            
            # 添加额外的振荡运动（叠加在基础运动上）
            t = self.progress_buf.float() * dt  # 当前时间
            oscillation = torch.stack([
                torch.zeros_like(t),           # X方向：无振荡（保持固定）
                torch.sin(t * 2.0) * 0.5,      # Y方向：正弦振荡，幅度0.5米
                torch.cos(t * 1.5) * 0.3,      # Z方向：余弦振荡，幅度0.3米
            ], dim=-1)
            self.gate_positions[:, i] += oscillation  # 将振荡加到门中心位置上
            
            # 限制门中心的运动范围，防止门移动得太远
            self.gate_positions[:, i, 1] = torch.clamp(self.gate_positions[:, i, 1], -3.0, 3.0)  # Y轴：±3米范围
            # Z轴约束：门中心最低高度 = 1 + gate_height/2，最高高度可以稍微高一些
            min_center_height = 1 + self.gate_height / 2   # 确保下横梁不低于地面
            max_center_height = min_center_height + 2.0       # 允许向上移动2米
            self.gate_positions[:, i, 2] = torch.clamp(self.gate_positions[:, i, 2], min_center_height, max_center_height)
            
            # 更新门的旋转（绕Z轴缓慢振荡）
            rotation_angle = torch.sin(t * 0.5) * 0.2  # 旋转角度：±0.2弧度（约±11.5度）
            self.gate_rotations[:, i] = torch.stack([
                torch.zeros_like(rotation_angle),         # X轴旋转分量：0
                torch.zeros_like(rotation_angle),         # Y轴旋转分量：0
                torch.sin(rotation_angle / 2),            # Z轴旋转分量：sin(θ/2)
                torch.cos(rotation_angle / 2)             # W分量：cos(θ/2)
            ], dim=-1)  # 构造绕Z轴旋转的四元数
        
        # Apply positions and rotations to all gate objects using the view
        if hasattr(self, 'gate_view') and self.gate_view is not None:
            for i in range(self.gate_count):
                gate_path = f"/World/envs/env_0/Gate_{i}"
                if prim_utils.is_prim_path_valid(gate_path):
                    prim = prim_utils.get_prim_at_path(gate_path)
                    if prim:
                        gate_pos = self.gate_positions[0, i].cpu().numpy()
                        gate_rot = self.gate_rotations[0, i].cpu().numpy()
                        # Set transform using USD
                        xform = UsdGeom.Xformable(prim)
                        if xform:
                            # Clear previous transforms and set new position
                            xform.ClearXformOpOrder()
                            translate_op = xform.AddTranslateOp()
                            translate_op.Set(tuple(gate_pos))
                            
                            # Set rotation if significant
                            if abs(gate_rot[2]) > 1e-6:  # Only set rotation if there's actual rotation
                                rotate_op = xform.AddRotateZOp()
                                rotate_op.Set(math.degrees(2 * math.asin(gate_rot[2])))



    def _compute_state_and_obs(self):
        obs = self._compute_obs()
        
        # Update trajectory visualization
        self._update_trajectory_visualization()
        
        self._tensordict.update(obs)
        return self._tensordict

    def _compute_gate_corners(self, gate_pos, gate_rot, relative_to_pos=None):
        """
        Compute the 4 corner positions of a gate.
        
        Args:
            gate_pos: Gate center position [batch_size, 3]
            gate_rot: Gate rotation quaternion [batch_size, 4] 
            relative_to_pos: If provided, return positions relative to this point [batch_size, 3]
            
        Returns:
            corners: [batch_size, 4, 3] - 4 corner positions
        """
        batch_size = gate_pos.shape[0]
        half_w = self.gate_width * 0.5
        half_h = self.gate_height * 0.5
        
        # Define local corner offsets (gate frame: X forward, Y right, Z up)
        local_corners = torch.tensor([
            [0.0, -half_w, -half_h],  # bottom-left
            [0.0, -half_w,  half_h],  # top-left
            [0.0,  half_w, -half_h],  # bottom-right
            [0.0,  half_w,  half_h],  # top-right
        ], device=self.device, dtype=torch.float32)  # [4, 3]
        
        # Convert quaternion to rotation matrix (vectorized)
        # gate_rot: [batch_size, 4] - quaternion (x, y, z, w)
        x, y, z, w = gate_rot[:, 0], gate_rot[:, 1], gate_rot[:, 2], gate_rot[:, 3]  # [batch_size] each
        
        # Compute rotation matrices for all quaternions at once [batch_size, 3, 3]
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)], dim=1),
            torch.stack([  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)], dim=1),
            torch.stack([  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)], dim=1),
        ], dim=1)  # [batch_size, 3, 3]
        
        # Transform corners to world frame (vectorized)
        # local_corners: [4, 3], R: [batch_size, 3, 3]
        # Use einsum for batch matrix multiplication: [batch_size, 4, 3] = [batch_size, 3, 3] @ [4, 3].T
        corners_world = torch.einsum('bij,kj->bki', R, local_corners) + gate_pos.unsqueeze(1)  # [batch_size, 4, 3]
        
        # Make relative if requested
        if relative_to_pos is not None:
            corners_world = corners_world - relative_to_pos.unsqueeze(1)  # [batch_size, 4, 3]
            
        return corners_world  # [batch_size, 4, 3]

    def _compute_obs(self):
        drone_pos, drone_rot = self.drone.get_world_poses()
        drone_vel = self.drone.get_velocities()
        drone_state = self.drone.get_state()
        
        # 获取当前目标门的信息 - 改为18维：4个角点(12)+线速度(3)+角速度(3) - 向量化版本
        current_gates = self.current_gate_idx
        gate_info = torch.zeros(self.num_envs, 1, 18, device=self.device)  # 新门信息：18维特征
        
        # 计算编队中心
        formation_center = drone_pos.mean(dim=1, keepdim=True)
        
        # 向量化处理：为有效门计算信息
        valid_mask = current_gates < self.gate_count  # [num_envs] - 哪些环境有有效的门
        
        if valid_mask.any():
            valid_env_indices = torch.where(valid_mask)[0]  # 有效环境的索引
            valid_gate_indices = current_gates[valid_mask]  # 对应的门索引
            
            # 批量提取门的状态信息
            gate_pos_batch = self.gate_positions[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            gate_rot_batch = self.gate_rotations[valid_env_indices, valid_gate_indices]  # [valid_envs, 4]
            gate_lin_vel_batch = self.gate_velocities[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            gate_ang_vel_batch = self.gate_angular_velocities[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            
            # 批量计算4个角点位置
            corners_batch = self._compute_gate_corners(gate_pos_batch, gate_rot_batch)  # [valid_envs, 4, 3]
            corners_flat_batch = corners_batch.reshape(len(valid_env_indices), -1)  # [valid_envs, 12]
            
            # 批量组装门信息：角点(12) + 线速度(3) + 角速度(3) = 18维
            gate_info_batch = torch.cat([
                corners_flat_batch,    # 4个角点全局坐标 (12维)
                gate_lin_vel_batch,    # 线速度 (3维)
                gate_ang_vel_batch     # 角速度 (3维)
            ], dim=-1)  # [valid_envs, 18]
            
            # 将批量计算的结果写入对应位置
            gate_info[valid_env_indices, 0] = gate_info_batch
        
        # Compute formation targets (relative positions)
        target_formation_global = self.formation.unsqueeze(0).expand(self.num_envs, -1, -1) + formation_center
        
        # 计算终点信息 - 向量化版本
        endpoint_info = torch.zeros(self.num_envs, 1, 4, device=self.device)  # 终点信息：4维特征
        
        # 计算编队中心的最终目标位置 - 向量化处理
        end_center = self.end_positions.mean(dim=1)  # [num_envs, 3]
        center_pos = formation_center[:, 0]  # 当前编队中心 [num_envs, 3]
        
        # 批量计算相对位置向量 (3维)
        relative_end_pos = end_center - center_pos  # [num_envs, 3]
        
        # 批量计算到终点的距离 (1维)
        distance_to_endpoint = torch.norm(relative_end_pos, dim=-1)  # [num_envs]
        
        # 批量组装终点信息：相对位置(3) + 距离(1) = 4维
        endpoint_info[:, 0, :3] = relative_end_pos  # 相对终点位置 (3维): [dx, dy, dz]
        endpoint_info[:, 0, 3] = distance_to_endpoint  # 终点距离 (1维)
        
        # Individual agent observations
        obs_self_list = []
        obs_others_list = []
        gate_info_list = []
        formation_target_list = []
        endpoint_info_list = []
        
        for i in range(self.drone.n):
            # Self observation
            obs_self = drone_state[:, i:i+1]
            
            if self.time_encoding:
                t = (self.progress_buf.float() / self.max_episode_length).unsqueeze(-1)
                time_encoding = torch.stack([
                    t, torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t), 
                    torch.sin(4 * torch.pi * t)
                ], dim=-1)
                obs_self = torch.cat([obs_self, time_encoding], dim=-1)
            
            # Others observation (relative states)
            other_indices = [j for j in range(self.drone.n) if j != i]
            obs_others = drone_state[:, other_indices].clone()
            
            # Gate info relative to current agent (convert global corners to relative) - 向量化版本
            gate_info_relative = gate_info.clone()
            
            # 检查哪些环境有有效的门信息（第一个角点的X坐标不为0作为标识）
            valid_gate_mask = gate_info_relative[:, 0, 0] != 0  # [num_envs]
            
            if valid_gate_mask.any():
                valid_env_indices = torch.where(valid_gate_mask)[0]  # 有效环境索引
                
                # 批量提取和处理角点信息
                corners_global_batch = gate_info_relative[valid_env_indices, 0, :12].reshape(-1, 4, 3)  # [valid_envs, 4, 3]
                drone_pos_batch = drone_pos[valid_env_indices, i]  # [valid_envs, 3]
                
                # 批量转换为相对坐标
                corners_relative_batch = corners_global_batch - drone_pos_batch.unsqueeze(1)  # [valid_envs, 4, 3]
                
                # 将转换后的相对坐标写回
                gate_info_relative[valid_env_indices, 0, :12] = corners_relative_batch.reshape(-1, 12)
                # 线速度和角速度保持不变 (gate_info_relative[valid_envs, 0, 12:])
            
            # Formation target: current position error from ideal formation position
            formation_target = drone_pos[:, i] - target_formation_global[:, i]  # [num_envs, 3]
            formation_target_flat = formation_target.unsqueeze(1)  # [num_envs, 1, 3]
            
            obs_self_list.append(obs_self)
            obs_others_list.append(obs_others)
            gate_info_list.append(gate_info_relative)
            formation_target_list.append(formation_target_flat)
            endpoint_info_list.append(endpoint_info)
        
        # print("endpoint_info list:", endpoint_info_list)
        # print("formation_target list:", formation_target_list)
        # print("gate info list:", gate_info_list)
        # print("obs others list:", obs_others_list)
        # print("obs self list:", obs_self_list)

        # Stack observations to create proper tensor structure
        obs_self_stacked = torch.stack(obs_self_list, dim=1)  # [num_envs, num_agents, 1, obs_dim]
        obs_others_stacked = torch.stack(obs_others_list, dim=1)  # [num_envs, num_agents, num_others, obs_dim]
        gate_info_stacked = torch.stack(gate_info_list, dim=1)  # [num_envs, num_agents, 1, gate_dim]
        formation_target_stacked = torch.stack(formation_target_list, dim=1)  # [num_envs, num_agents, 1, formation_dim]
        endpoint_info_stacked = torch.stack(endpoint_info_list, dim=1)  # [num_envs, num_agents, 1, endpoint_dim]
        
        # Central observation - 使用18维门信息：4个全局角点(12)+线速度(3)+角速度(3) - 向量化版本
        all_gate_info = torch.zeros(self.num_envs, self.gate_count, 18, device=self.device)
        
        # 批量处理所有门的信息
        gate_pos_all = self.gate_positions  # [num_envs, gate_count, 3]
        gate_rot_all = self.gate_rotations  # [num_envs, gate_count, 4]
        gate_lin_vel_all = self.gate_velocities  # [num_envs, gate_count, 3]
        gate_ang_vel_all = self.gate_angular_velocities  # [num_envs, gate_count, 3]
        
        # 重塑数据以进行批量计算：将 [num_envs, gate_count, ...] 变为 [num_envs * gate_count, ...]
        batch_size = self.num_envs * self.gate_count
        gate_pos_flat = gate_pos_all.reshape(batch_size, 3)
        gate_rot_flat = gate_rot_all.reshape(batch_size, 4)
        gate_lin_vel_flat = gate_lin_vel_all.reshape(batch_size, 3)
        gate_ang_vel_flat = gate_ang_vel_all.reshape(batch_size, 3)
        
        # 批量计算所有门的4个角点全局坐标
        corners_all = self._compute_gate_corners(gate_pos_flat, gate_rot_flat)  # [batch_size, 4, 3]
        corners_flat_all = corners_all.reshape(batch_size, -1)  # [batch_size, 12]
        
        # 重塑回原始维度
        corners_reshaped = corners_flat_all.reshape(self.num_envs, self.gate_count, 12)
        
        # 批量组装门信息：角点(12) + 线速度(3) + 角速度(3) = 18维
        all_gate_info[:, :, :12] = corners_reshaped      # 4个角点全局坐标 (12维)
        all_gate_info[:, :, 12:15] = gate_lin_vel_all    # 线速度 (3维)
        all_gate_info[:, :, 15:18] = gate_ang_vel_all    # 角速度 (3维)
        
        obs_central = {
            "drones": drone_state,
            "gates": all_gate_info,
            "formation": target_formation_global,
        }

        # print("obs_self_stacked:", obs_self_stacked)
        # print("obs_others_stacked:", obs_others_stacked)
        # print("gate_info_stacked:", gate_info_stacked)
        # print("formation_target_stacked:", formation_target_stacked)
        
        return TensorDict({
            "agents": {
                "observation": {
                    "obs_self": obs_self_stacked,
                    "obs_others": obs_others_stacked,
                    "gate_info": gate_info_stacked,
                    "formation_target": formation_target_stacked,
                    "endpoint_info": endpoint_info_stacked,
                },
                "observation_central": obs_central,
            },
            "stats": self.stats.clone(),
        }, batch_size=self.batch_size)

    def _compute_reward_and_done(self):
        drone_pos, drone_rot = self.drone.get_world_poses()
        drone_vel = self.drone.get_velocities()
        
        # ====================
        # DIRECT LEARNING REWARD FUNCTION
        # Multi-objective reward combining all task requirements
        # ====================
        
        # 1. Basic Flight Stability Rewards (inspired by Forest.py)
        
        # 1.1 Velocity toward current objective reward (gate first, then endpoint)
        end_center = self.end_positions.mean(dim=1)  # [num_envs, 3]
        formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
        
        # Determine current objective: gate if available, otherwise endpoint (vectorized)
        # Check which environments have valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count  # [num_envs]
        
        # Get current gate positions for all environments using advanced indexing
        # For environments with valid gates, use gate position; for others, use endpoint
        env_indices = torch.arange(self.num_envs, device=self.device)
        gate_positions_current = self.gate_positions[env_indices, self.current_gate_idx.clamp(0, self.gate_count-1)]  # [num_envs, 3]
        
        # Select between gate position and endpoint based on valid gate mask
        current_objective = torch.where(
            valid_gate_mask.unsqueeze(-1),  # [num_envs, 1]
            gate_positions_current,         # [num_envs, 3] - gate positions
            end_center                      # [num_envs, 3] - endpoint positions
        )
        
        target_direction = current_objective - formation_center
        distance_to_target = torch.norm(target_direction, dim=-1, keepdim=True)
        target_direction = target_direction / (distance_to_target.clamp_min(1e-6))
        
        avg_velocity = drone_vel[..., :3].mean(dim=1)  # [num_envs, 3]
        velocity_toward_target = (avg_velocity * target_direction).sum(dim=-1).clamp(max=2.0)
        
        # 1.2 Uprightness reward - maintaining stable orientation
        drone_up = quat_axis(drone_rot, axis=2)[..., 2].mean(dim=-1)  # Average Z-component
        reward_uprightness = torch.square((drone_up + 1) / 2) * self.uprightness_reward_weight
        
        # 1.5 Action smoothness reward
        action_diff = torch.norm(self.actions - self.prev_actions, dim=-1)
        reward_action_smoothness = torch.exp(-action_diff.mean(dim=-1)) * self.reward_action_smoothness_weight
        
        # 2. Formation Maintenance Reward
        
        # 2.1 Formation shape maintenance using Hausdorff distance (inspired by Formation.py)
        formation_center_expanded = formation_center.unsqueeze(1)
        target_formation_global = self.formation.unsqueeze(0).expand(self.num_envs, -1, -1) + formation_center_expanded
        
        # Calculate formation cost using Hausdorff distance
        formation_cost = cost_formation_hausdorff(drone_pos, target_formation_global)[:, 0]  # [num_envs]
        
        # Convert cost to reward using the same method as Formation.py
        reward_formation = 1 / (1 + torch.square(formation_cost * 1.6)) * self.formation_reward_weight
        
        # 2.2 Formation cohesion bonus - reward for keeping drones close together
        pairwise_distances = torch.cdist(drone_pos, drone_pos)
        pairwise_distances = pairwise_distances + torch.eye(self.drone.n, device=self.device) * 1000
        avg_inter_drone_distance = pairwise_distances.mean(dim=(-2, -1))
        reward_cohesion = torch.exp(-avg_inter_drone_distance * 0.3) * 0.5
        
        # 3. Gate Traversal Rewards
        
        # 3.1 Gate approach reward (vectorized)
        # Check which environments have valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count  # [num_envs]
        
        # Get current gate positions using advanced indexing
        env_indices = torch.arange(self.num_envs, device=self.device)
        current_gate_pos = self.gate_positions[env_indices, self.current_gate_idx.clamp(0, self.gate_count-1)]  # [num_envs, 3]
        gate_active = valid_gate_mask  # [num_envs]
        
        # Distance to current gate
        gate_distance = torch.norm(formation_center - current_gate_pos, dim=-1)
        
        # Progressive approach reward: exponential decay with distance
        reward_gate_approach = torch.where(
            gate_active,
            gate_distance * self.gate_reward_weight,
            torch.zeros_like(gate_distance)
        )
        
        # 3.2 Gate alignment reward - reward for approaching gate from correct direction
        gate_direction = current_gate_pos - formation_center
        gate_direction_norm = gate_direction / (torch.norm(gate_direction, dim=-1, keepdim=True).clamp_min(1e-6))
        formation_velocity_norm = avg_velocity / (torch.norm(avg_velocity, dim=-1, keepdim=True).clamp_min(1e-6))
        
        gate_alignment = torch.sum(formation_velocity_norm * gate_direction_norm, dim=-1)
        reward_gate_alignment = torch.where(
            gate_active,
            torch.clamp(gate_alignment, 0.0, 1.0) * self.gate_reward_weight * 0.3,
            torch.zeros_like(gate_alignment)
        )
        
        # 3.3 Gate progress reward based on distance improvement
        # Only calculate and update for environments with active gates
        reward_gate_progress = torch.zeros(self.num_envs, device=self.device)
        if gate_active.any():
            active_envs = torch.where(gate_active)[0]
            active_gate_distance = gate_distance[active_envs]
            active_last_distance = self.last_gate_distance[active_envs]
            
            # Calculate progress only for active environments
            active_progress = active_last_distance - active_gate_distance
            # print("gate active envs:", gate_active)
            # print("active envs:", active_envs)
            # print("Active gate progress:", active_progress)
            # print("self gate reward weight:", self.gate_reward_weight)
            reward_gate_progress[active_envs] = active_progress * self.gate_reward_weight
            
            # Update last distance only for active environments
            self.last_gate_distance[active_envs] = active_gate_distance
        
        # 4. Endpoint Progress Rewards
        
        # 4.1 Calculate individual drone distances to their target endpoints
        individual_endpoint_distances = torch.norm(drone_pos - self.end_positions, dim=-1)  # [num_envs, n_drones]
        
        # 4.2 Distance-based endpoint reward using sum of individual distances
        total_endpoint_distance = individual_endpoint_distances.sum(dim=-1)  # [num_envs]
        reward_endpoint_distance = torch.exp(-total_endpoint_distance * 0.01) * self.endpoint_progress_reward_weight
        
        # 4.3 Velocity toward endpoint reward (keep existing formation center approach for velocity)
        end_direction = end_center - formation_center
        end_direction_norm = end_direction / (torch.norm(end_direction, dim=-1, keepdim=True).clamp_min(1e-6))
        velocity_toward_endpoint = torch.sum(avg_velocity * end_direction_norm, dim=-1).clamp(min=0.0)
        reward_endpoint_velocity = velocity_toward_endpoint * self.endpoint_progress_reward_weight
        
        # 4.4 Endpoint progress reward based on sum of individual distance improvements
        last_total_distance = self.last_endpoint_distances.sum(dim=-1)  # [num_envs]
        endpoint_progress = last_total_distance - total_endpoint_distance  # [num_envs]
        reward_endpoint_progress = endpoint_progress * self.endpoint_progress_reward_weight 
        self.last_endpoint_distances = individual_endpoint_distances.clone()
        
        # 4.5 Formation endpoint alignment - all drones should reach their target positions
        reward_endpoint_formation = torch.exp(-individual_endpoint_distances.mean(dim=-1) * 0.3) * self.endpoint_progress_reward_weight * 0.3
        
        # 5. Safety and Collision Avoidance
        
        # 5.1 Inter-drone collision penalty
        min_distances = pairwise_distances.min(dim=-1)[0].min(dim=-1)[0]
        reward_collision_penalty = torch.where(
            min_distances < self.safe_distance,
            -self.collision_penalty_weight,
            torch.zeros_like(min_distances)
        )
        
        # 5.2 Soft collision avoidance - gradual penalty as drones get closer
        reward_soft_collision_penalty = torch.where(
            min_distances < self.safe_distance * 2.0,
            -0.5 * torch.exp(-(min_distances - self.safe_distance)),
            torch.zeros_like(min_distances)
        )
        
        # 6. Task Completion Bonuses
        
        # 6.1 Gate traversal bonus (vectorized)
        reward_gate_traversal = torch.zeros(self.num_envs, device=self.device)
        
        # Check gate traversal for environments with valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count
        if valid_gate_mask.any():
            # Get relevant data for environments with valid gates
            valid_envs = torch.where(valid_gate_mask)[0]
            valid_gate_indices = self.current_gate_idx[valid_envs]
            valid_gate_pos = self.gate_positions[valid_envs, valid_gate_indices]  # [valid_envs, 3]
            valid_center_pos = formation_center[valid_envs]  # [valid_envs, 3]
            valid_gate_distance = gate_distance[valid_envs]  # [valid_envs]
            
            # Check if formation center has passed through the gate
            gate_threshold = max(self.gate_width, self.gate_height)
            passed_x = valid_center_pos[:, 0] > valid_gate_pos[:, 0]  # [valid_envs]
            close_enough = valid_gate_distance < gate_threshold  # [valid_envs]
            traversal_mask = passed_x & close_enough  # [valid_envs]
            
            if traversal_mask.any():
                traversed_envs = valid_envs[traversal_mask]
                reward_gate_traversal[traversed_envs] = 10.0 * self.gate_reward_weight
                
                # Update gate indices and counts
                self.current_gate_idx[traversed_envs] = torch.clamp(
                    self.current_gate_idx[traversed_envs] + 1, 
                    max=self.gate_count - 1
                )
                self.gates_passed[traversed_envs] += 1
        
        # 6.2 Task completion bonus - extra reward for completing all gates
        reward_completion = torch.where(
            self.gates_passed >= self.gate_count,
            20.0,
            torch.zeros_like(self.gates_passed, dtype=torch.float32)
        )
        
        
        # 8. TOTAL REWARD COMBINATION
        total_reward = (
            # Basic flight rewards
            reward_uprightness  + reward_action_smoothness +
            
            # Formation rewards (now using improved Hausdorff distance method)
            reward_formation + reward_cohesion +
            
            # Gate traversal rewards
            # reward_gate_progress + reward_gate_traversal +
            reward_gate_traversal + 
            
            # Endpoint progress rewards
            reward_endpoint_progress +
            
            # Safety penalties
            # reward_collision_penalty + reward_soft_collision_penalty +
            
            # Completion bonuses
            reward_completion
        )
        
        # Expand reward to all drones (shared reward)
        reward = total_reward.unsqueeze(-1).unsqueeze(-1).expand(-1, self.drone.n, 1)
        
        # ====================
        # TERMINATION CONDITIONS
        # ====================
        
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Critical collision
        collision_terminated = min_distances < 0.5
        
        # Formation breakdown - more lenient threshold for direct learning
        # Use formation_cost as a measure of formation breakdown
        formation_breakdown = formation_cost > (self.formation_tolerance * 2.0)
        
        # Task success - all gates passed
        success = self.gates_passed >= self.gate_count
        
        # Out of bounds
        max_height = 6.0
        min_height = 0.2
        out_of_bounds = (drone_pos[..., 2] > max_height) | (drone_pos[..., 2] < min_height)
        out_of_bounds = out_of_bounds.any(dim=-1)
        
        # 新增终止条件1: 无人机超过门但没有穿过门就终止 - 正确的坐标系版本
        gate_bypass_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 创建有效门的掩码
        valid_gate_mask = self.current_gate_idx < self.gate_count  # [num_envs]
        
        if valid_gate_mask.any():
            valid_envs = torch.where(valid_gate_mask)[0]  # 有效环境索引
            valid_gate_indices = self.current_gate_idx[valid_envs]  # 对应门索引
            
            # 获取门的相对位置并转换为全局位置（这是关键修复）
            gate_local_positions = self.gate_positions[valid_envs, valid_gate_indices]  # [valid_envs, 3]
            env_offsets = self.envs_positions[valid_envs]  # [valid_envs, 3]
            gate_global_positions = gate_local_positions + env_offsets  # [valid_envs, 3]
            
            # 无人机位置（已经是全局坐标）
            drone_pos_valid = drone_pos[valid_envs]  # [valid_envs, num_drones, 3]
            
            # 检查无人机是否超过门的X位置（全局坐标系）
            gate_x_global = gate_global_positions[:, 0].unsqueeze(1)  # [valid_envs, 1]
            
            # 需要明显超过门的X位置才算"超过"，避免边界情况
            buffer_distance = 0.0  # 1米缓冲区就足够了
            drones_passed_gate_x = drone_pos_valid[:, :, 0] > (gate_x_global + buffer_distance)  # [valid_envs, num_drones]
            
            # 对于有无人机超过门X位置的环境，检查是否成功穿过
            envs_with_passed_drones = drones_passed_gate_x.any(dim=1)  # [valid_envs]
            
            if envs_with_passed_drones.any():
                check_env_mask = envs_with_passed_drones  # [valid_envs] - 布尔掩码
                check_envs = valid_envs[check_env_mask]  # 需要检查的环境索引
                
                # 计算无人机到门中心的距离（使用全局坐标）
                check_drone_pos = drone_pos_valid[check_env_mask]  # [check_envs, num_drones, 3]
                check_gate_pos = gate_global_positions[check_env_mask]  # [check_envs, 3]
                
                # 计算距离：[check_envs, num_drones]
                drone_to_gate_distances = torch.norm(
                    check_drone_pos - check_gate_pos.unsqueeze(1), dim=-1
                )
                
                # 门通过阈值 - 合理的阈值
                gate_threshold = max(self.gate_width, self.gate_height)  # 稍微宽松一些
                
                # 检查超过门X位置的无人机是否都成功穿过
                passed_mask = drones_passed_gate_x[check_env_mask]  # [check_envs, num_drones]
                near_gate_mask = drone_to_gate_distances < gate_threshold  # [check_envs, num_drones]
                
                # 逐环境检查：如果有无人机超过门但距离门中心太远，则判断为失败
                for i, env_idx in enumerate(check_envs):
                    passed_drones_in_env = passed_mask[i]  # [num_drones]
                    if passed_drones_in_env.any():
                        # 检查超过门的无人机是否都接近门中心
                        passed_and_near = passed_drones_in_env & near_gate_mask[i]  # [num_drones]
                        # 如果有超过门的无人机不在门附近，则判断为绕过失败
                        if not torch.equal(passed_drones_in_env, passed_and_near):
                            gate_bypass_failure[env_idx] = True
        
        # 新增终止条件2: 无人机超过终点过远就终止 - 合理版本
        endpoint_exceeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 使用正确的终点位置计算
        endpoint_positions = self.end_positions[:, 0, :]  # [num_envs, 3] 取第一个无人机的终点作为参考
        
        # 计算所有无人机到各自环境终点的距离
        distances_to_endpoint = torch.norm(
            drone_pos - endpoint_positions.unsqueeze(1), dim=-1
        )  # [num_envs, num_drones]
        
        # 设置合理的阈值：15米
        endpoint_threshold = 0.5
        
        # 检查是否有任何无人机超过阈值距离
        any_drone_too_far = (distances_to_endpoint > endpoint_threshold).any(dim=1)  # [num_envs]
        
        # 同时检查无人机是否真的向前移动了（避免在起始位置就判断为超出）
        # 检查无人机是否超过了终点的X位置
        drone_x_positions = drone_pos[:, :, 0]  # [num_envs, num_drones]
        endpoint_x_positions = endpoint_positions[:, 0].unsqueeze(1)  # [num_envs, 1]
        any_drone_passed_endpoint_x = (drone_x_positions > endpoint_x_positions + 0.5).any(dim=1)  # [num_envs] 0.5米缓冲

        # 只有当无人机既超过了终点X位置又距离终点过远时才终止
        endpoint_exceeded = any_drone_too_far & any_drone_passed_endpoint_x

        # terminated = collision_terminated | formation_breakdown | success | out_of_bounds | gate_bypass_failure | endpoint_exceeded
        terminated = success | out_of_bounds | endpoint_exceeded

        # ==================== STATS UPDATE ====================
        truncated = self.progress_buf >= self.max_episode_length
        completion_rate = self.gates_passed.float() / self.gate_count
        drone_speeds = torch.norm(drone_vel[..., :3], dim=-1)
        
        # Basic episode metrics
        self.stats["return"] += total_reward.unsqueeze(-1).expand(-1, self.drone.n)

        
        # Reward component tracking (current step values)
        self.stats["reward_uprightness"] += reward_uprightness.unsqueeze(-1)
        self.stats["reward_action_smoothness"] += reward_action_smoothness.unsqueeze(-1)
        self.stats["reward_formation"] += reward_formation.unsqueeze(-1)
        self.stats["reward_cohesion"] += reward_cohesion.unsqueeze(-1)
        self.stats["reward_endpoint_progress"] += reward_endpoint_progress.unsqueeze(-1)
        self.stats["reward_gate_progress"] += reward_gate_progress.unsqueeze(-1)
        
        # Safety reward components
        self.stats["reward_collision_penalty"] += reward_collision_penalty.unsqueeze(-1)
        self.stats["reward_soft_collision_penalty"] += reward_soft_collision_penalty.unsqueeze(-1)
        
        # Task completion reward components
        self.stats["reward_completion"] += reward_completion.unsqueeze(-1)

        # Update previous state for next iteration
        self.prev_drone_velocities = drone_vel[..., :3].clone()
        if hasattr(self, 'actions'):
            self.prev_actions = self.actions.clone()

        terminated = terminated.unsqueeze(-1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
        return TensorDict(
            {
                "agents": {"reward": reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
    
    def _update_trajectory_visualization(self):
        """Update drone trajectory visualization with start points, end points, and flight paths."""
        if not self.visualization_enabled or not self._should_render(0):
            return
            
        # Clear previous visualization
        self.draw.clear_lines()
        
        # Get current drone positions
        drone_pos, _ = self.drone.get_world_poses()
        
        # Update trajectory history for each environment
        for env_idx in range(self.num_envs):
            step_idx = int(self.trajectory_step_count[env_idx] % self.trajectory_max_length)
            self.drone_trajectory_history[env_idx, :, step_idx] = drone_pos[env_idx]
            
        self.trajectory_step_count += 1
        
        # Visualize only the central environment for clarity
        # Use the environment closest to the origin (defaultGroundPlane center)
        central_env_idx = self.central_env_idx
        
        # Draw start positions (green spheres)
        start_pos_list = self.start_positions[central_env_idx].cpu().tolist()
        for i, pos in enumerate(start_pos_list):
            # Draw start point as a small cross
            offset = 0.1
            cross_points_0 = [
                [pos[0] - offset, pos[1], pos[2]],
                [pos[0], pos[1] - offset, pos[2]],
                [pos[0], pos[1], pos[2] - offset]
            ]
            cross_points_1 = [
                [pos[0] + offset, pos[1], pos[2]],
                [pos[0], pos[1] + offset, pos[2]],
                [pos[0], pos[1], pos[2] + offset]
            ]
            
            # Green color for start positions
            colors = [(0, 1, 0, 1)] * len(cross_points_0)  # Green
            sizes = [2.0] * len(cross_points_0)
            
            self.draw.draw_lines(cross_points_0, cross_points_1, colors, sizes)
        
        # Draw end positions (red spheres)
        end_pos_list = self.end_positions[central_env_idx].cpu().tolist()
        for i, pos in enumerate(end_pos_list):
            # Draw end point as a small cross
            offset = 0.1
            cross_points_0 = [
                [pos[0] - offset, pos[1], pos[2]],
                [pos[0], pos[1] - offset, pos[2]],
                [pos[0], pos[1], pos[2] - offset]
            ]
            cross_points_1 = [
                [pos[0] + offset, pos[1], pos[2]],
                [pos[0], pos[1] + offset, pos[2]],
                [pos[0], pos[1], pos[2] + offset]
            ]
            
            # Red color for end positions
            colors = [(1, 0, 0, 1)] * len(cross_points_0)  # Red
            sizes = [2.0] * len(cross_points_0)
            
            self.draw.draw_lines(cross_points_0, cross_points_1, colors, sizes)
        
        # Draw gate positions (yellow frames)
        gate_pos_list = self.gate_positions[central_env_idx].cpu().tolist()
        for gate_pos in gate_pos_list:
            # Draw gate frame
            gate_height = self.gate_height
            gate_width = self.gate_width
            
            # Gate corners
            corners = [
                [gate_pos[0], gate_pos[1] - gate_width/2, gate_pos[2] - gate_height/2],
                [gate_pos[0], gate_pos[1] + gate_width/2, gate_pos[2] - gate_height/2],
                [gate_pos[0], gate_pos[1] + gate_width/2, gate_pos[2] + gate_height/2],
                [gate_pos[0], gate_pos[1] - gate_width/2, gate_pos[2] + gate_height/2],
            ]
            
            # Draw gate frame as connected lines
            frame_lines_0 = [
                corners[0], corners[1], corners[2], corners[3]
            ]
            frame_lines_1 = [
                corners[1], corners[2], corners[3], corners[0]
            ]
            
            # Yellow color for gates
            colors = [(1, 1, 0, 1)] * len(frame_lines_0)  # Yellow
            sizes = [3.0] * len(frame_lines_0)
            
            self.draw.draw_lines(frame_lines_0, frame_lines_1, colors, sizes)
        
        # Draw drone trajectories (blue lines)
        current_step = int(self.trajectory_step_count[central_env_idx].item())
        if current_step > 1:
            trajectory_length = min(current_step, self.trajectory_max_length)
            
            for drone_idx in range(self.drone.n):
                # Get trajectory points for this drone
                drone_trajectory = self.drone_trajectory_history[central_env_idx, drone_idx]
                
                # Extract valid trajectory points
                valid_points = []
                for step in range(trajectory_length - 1):
                    point_idx = (current_step - trajectory_length + step) % self.trajectory_max_length
                    pos = drone_trajectory[point_idx].cpu().tolist()
                    valid_points.append(pos)
                
                if len(valid_points) > 1:
                    # Create line segments
                    point_list_0 = valid_points[:-1]
                    point_list_1 = valid_points[1:]
                    
                    # Different color for each drone
                    color = [0.2 + 0.3 * drone_idx, 0.5, 1.0, 0.8]  # Blue gradient
                    colors = [color] * len(point_list_0)
                    sizes = [1.5] * len(point_list_0)
                    
                    self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)


def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    """
    Calculate Hausdorff distance-based formation cost.
    
    Args:
        p: Current positions [batch_size, n_drones, 3]
        desired_p: Desired formation positions [batch_size, n_drones, 3] or [n_drones, 3]
        
    Returns:
        cost: Formation cost [batch_size, 1]
    """
    # Center both formations at their centroids
    p_centered = p - p.mean(-2, keepdim=True)
    if desired_p.dim() == 2:
        # If desired_p is 2D, expand to match batch size
        desired_p = desired_p.unsqueeze(0).expand(p.shape[0], -1, -1)
    desired_p_centered = desired_p - desired_p.mean(-2, keepdim=True)
    
    # Calculate bidirectional Hausdorff distance
    cost = torch.max(
        directed_hausdorff(p_centered, desired_p_centered), 
        directed_hausdorff(desired_p_centered, p_centered)
    )
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate directed Hausdorff distance.
    
    Args:
        p: Source points [batch_size, n, 3]
        q: Target points [batch_size, m, 3]
        
    Returns:
        distance: Directed Hausdorff distance [batch_size]
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d
