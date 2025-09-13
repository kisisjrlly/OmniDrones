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


import torch
import torch.distributions as D
from torch.func import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

import omni_drones.utils.kit as kit_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.scene import design_scene
from omni_drones.utils.torch import euler_to_quaternion

from .utils import OveractuatedPlatform, PlatformCfg

from omni_drones.controllers import (
    LeePositionController,
    AttitudeController,
    RateController
)

class PlatformHover(IsaacEnv):
    r"""
    A cooperative control task where a group of `k` UAVs are connected together by a
    rigid frame to form an overactuated platform. Each individual UAV, attached
    by a 2-DoF passive gimbal joint, acts as a thrust generator.
    The goal for the agents is to make the platform hover at a reference pose
    (position and attitude).

    ## Observation

    The observation is a `CompositeSpec` containing the following items:

    - `obs_self` (1, \*): The state of each UAV observed by itself, containing its kinematic
      information with the position being relative to the frame center, and an one-hot
      identity indicating the UAV's index.
    - `obs_others` (k-1, \*): The observed states of other agents.
    - `obs_frame`:
      - `state_frame`: (1, \*): The state of the frame.
      - `rpos` (3): The relative position of the platform to the reference positions.
      - `time_encoding` (optional): The time encoding, which is a 4-dimensional
        vector encoding the current progress of the episode.

    ## Reward

    - `reward_pose`: The reward for the pose error between the platform and
      the reference (position and orientation).
    - `reward_up`: The reward for the alignment of the platform's up vector and
      the reference up vector.
    - `reward_spin`: Reward computed from the spin of the drone to discourage spinning.
    - `reward_effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pose} + r_\text{pose} * (r_\text{up} + r_\text{spin}) + r_\text{effort}
    ```

    ## Config

    | Parameter               | Type  | Default       | Description |
    | ----------------------- | ----- | ------------- | ----------- |
    | `drone_model`           | str   | "hummingbird" |             |
    | `num_drones`            | int   | 4             |             |
    | `arm_length`            | float | 0.85          |             |
    | `reward_distance_scale` | float | 1.2           |             |
    | `time_encoding`         | bool  | True          |             |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding

        self.num_drones = cfg.task.num_drones
        self.arm_length = cfg.task.arm_length
        self.joint_damping  = cfg.task.joint_damping
        super().__init__(cfg, headless)

        self.platform.initialize()

        self.target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        ).initialize()

        self.init_vels = torch.zeros_like(self.platform.get_velocities())
        self.init_joint_pos = self.platform.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.platform.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-3., -3., 1.5], device=self.device),
            torch.tensor([3.0, 3.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.0], device=self.device),
            torch.tensor([0.2, 0.2, 2.0], device=self.device)
        )
        self.target_pos = torch.tensor([0., 0., 2.25], device=self.device)
        self.target_heading =  torch.zeros(self.num_envs, 3, device=self.device)
        self.target_up = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)

        self.alpha = 0.8

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller, self.device
        )

        platform_cfg = PlatformCfg(
            num_drones=self.num_drones,
            arm_length=self.arm_length,
            joint_damping=self.joint_damping
        )
        self.platform = OveractuatedPlatform(
            cfg=platform_cfg,
            drone=self.drone,
        )
        self.platform.spawn(
            translations=[0., 0., 2.],
            enable_collision=True
        )

        # for visualization
        target_prim_path = self.platform._create_frame(
            "/World/envs/env_0/target",
            enable_collision=False
        ).GetPath().pathString
        kit_utils.set_rigid_body_properties(target_prim_path, disable_gravity=True)

        design_scene()
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 25
        if self.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, drone_state_dim + 3 +self.drone.n)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim + 3 + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            },
        }).expand(self.num_envs).to(self.device)
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
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        platform_pos = self.init_pos_dist.sample(env_ids.shape) + self.envs_positions[env_ids]
        platform_rpy = self.init_rpy_dist.sample(env_ids.shape)
        platform_rot = euler_to_quaternion(platform_rpy)
        platform_heading = torch_utils.quat_axis(platform_rot, 0)
        platform_up = torch_utils.quat_axis(platform_rot,  2)
        self.platform.set_world_poses(platform_pos, platform_rot, env_indices=env_ids)
        self.platform.set_velocities(self.init_vels[env_ids], env_ids)

        self.platform.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.platform.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        target_rpy = self.target_rpy_dist.sample(env_ids.shape)
        target_rot = euler_to_quaternion(target_rpy)
        target_heading = torch_utils.quat_axis(target_rot, 0)
        target_up = torch_utils.quat_axis(target_rot, 2)
        self.target_heading[env_ids] = target_heading
        self.target_up[env_ids] = target_up

        self.target_vis.set_world_poses(
            self.target_pos + self.envs_positions[env_ids],
            orientations=target_rot,
            env_indices=env_ids
        )

        self.stats[env_ids] = 0.
        distance = torch.cat([
            self.target_pos - platform_pos,
            target_heading - platform_heading,
            target_up - platform_up
        ], dim=-1).norm(dim=-1, keepdim=True)
        self.last_distance[env_ids] = distance

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # print("actions:", actions)
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
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)

        self.platform_state = self.platform.get_state()

        self.target_platform_rpos = self.target_pos - self.platform.pos
        self.target_platform_rheading = self.target_heading.unsqueeze(1) - self.platform.heading
        self.target_platform_rup = self.target_up.unsqueeze(1) - self.platform.up
        self.target_platform_rpose = torch.cat([
            self.target_platform_rpos,
            self.target_platform_rheading,
            self.target_platform_rup
        ], dim=-1)

        platform_drone_rpos = self.platform.pos - self.drone_states[..., :3]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            platform_state = torch.cat([
                self.target_platform_rpose, # 9
                self.platform_state[..., 3:],
                t.expand(-1, self.time_encoding_dim).unsqueeze(1)
            ], dim=-1) # [num_envs, 1, 25+time_encoding_dim]
        else:
            platform_state = torch.cat([
                self.target_platform_rpose, # 9
                self.platform_state[..., 3:]
            ], dim=-1) # [num_envs, 1, 25]

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        # obs["obs_self"] = torch.cat(
        #     [-platform_drone_rpos, self.drone_states[..., 3:], identity], dim=-1
        # ).unsqueeze(2)

        # print("self.drone_states.shape:", self.drone_states.shape)
        # print("self.platform.pos.shape:", self.platform.pos.shape)
        # print("identity.shape:", identity.shape)
        obs["obs_self"] = torch.cat(
            [self.drone_states, -platform_drone_rpos, identity], dim=-1
        ).unsqueeze(2)
        obs["obs_others"] = torch.cat(
            [self.drone_rpos, vmap(others)(self.drone_states[..., 3:13])], dim=-1
        )
        obs["state_frame"] = platform_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["obs_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = platform_state                # [num_envs, 1, platform_state_dim]

        self.pos_error = torch.norm(self.target_platform_rpos, dim=-1)
        self.heading_alignment = torch.sum(self.platform.heading * self.target_heading.unsqueeze(1), dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "observation_central": state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        platform_vels = self.platform.get_velocities()

        distance = torch.norm(self.target_platform_rpose, dim=-1)

        # reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        reward_pose = torch.exp(- self.reward_distance_scale * distance)

        up = torch.sum(self.platform.up * self.target_up.unsqueeze(1), dim=-1)
        reward_up = torch.square((up + 1) / 2)

        spinnage = platform_vels[:, -3:].abs().sum(-1)
        reward_spin = 1. / (1 + torch.square(spinnage))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference).mean(-1, keepdim=True)

        self.last_distance[:] = distance
        assert reward_pose.shape == reward_up.shape == reward_action_smoothness.shape

        reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        reward[:] = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
        )

        misbehave = (
            (self.drone_states[..., 2] < 0.2).any(-1, keepdim=True)
            | (distance > 5.0)
        )
        hasnan = torch.isnan(self.drone_states).any(-1)

        terminated = misbehave | hasnan.any(-1, keepdim=True)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["pos_error"].lerp_(self.pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(self.heading_alignment, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference.mean(-1, True), (1-self.alpha))

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )


# 完整obs["agents"]["observation"]展平后的字段注释（98维）
# 展平顺序：obs_self -> obs_others -> state_frame
#
# === obs_self 部分 (30维) ===
# [0:23]  : 无人机完整状态 (drone_state_dim = self.drone.state_spec.shape.numel()，通常为23)
#           - [0:3]   位置 [px, py, pz]
#           - [3:7]   四元数姿态 [qw, qx, qy, qz]
#           - [7:10]  线速度 [vx, vy, vz]
#           - [10:13] 角速度 [wx, wy, wz]
#           - [13:16] 航向向量 [hx, hy, hz]
#           - [16:19] 向上向量 [ux, uy, uz]
#           - [19:23] 其它无人机状态字段（与具体无人机模型相关，4维）
# [23:26] : 平台中心位置 [platform_px, platform_py, platform_pz]
# [26:30] : 无人机身份编码 (one-hot, 4维)
#
# === obs_others 部分 (3×13=39维) ===
# 其他3个无人机的观测信息，每个13维（按顺序展平）
# [30:43]  : 其他无人机1的信息 [relative_pos(3), drone_state[3:13](10)]
#           - [30:33]  与当前无人机的相对位置 [rel_px, rel_py, rel_pz]
#           - [33:43]  其他无人机状态片段 [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
# [43:56]  : 其他无人机2的信息（结构同上）
# [56:69]  : 其他无人机3的信息（结构同上）
#
# === state_frame 部分 (29维) ===
# [69:98]  : 平台状态信息 (29维)
#           根据代码：target_platform_rpose (9) + platform_state[3:] (16) + time_encoding (4) = 29
#           - [69:78]  目标平台相对姿态 target_platform_rpose (9维)
#             - [69:72] 目标相对位置 [target_rel_px, target_rel_py, target_rel_pz]
#             - [72:75] 目标相对航向 [target_rel_hx, target_rel_hy, target_rel_hz]
#             - [75:78] 目标相对向上 [target_rel_ux, target_rel_uy, target_rel_uz]
#           - [78:94]  平台状态 platform_state[3:] (16维)
#             - [78:82] 平台四元数 [platform_qw, platform_qx, platform_qy, platform_qz]
#             - [82:85] 平台线速度 [platform_vx, platform_vy, platform_vz]
#             - [85:88] 平台角速度 [platform_wx, platform_wy, platform_wz]
#             - [88:91] 平台航向向量 [platform_hx, platform_hy, platform_hz]
#             - [91:94] 平台向上向量 [platform_ux, platform_uy, platform_uz]
#           - [94:98]  时间编码 time_encoding (4维)
#             - [94]    归一化时间进度 t = progress_buf / max_episode_length
#             - [95]    sin(2πt)
#             - [96]    cos(2πt)
#             - [97]    sin(4πt)
#
# 总维度验证：30 + 39 + 29 = 98 ✓
# 说明：drone_state_dim 在代码中由 self.drone.state_spec.shape.numel() 决定（参见 `_set_specs`），
# 若无人机模型变更，请同步更新此注释。
# 参见实现：[`PlatformHover._set_specs`](OmniDrones/omni_drones/envs/platform/platform_hover.py)，
# 无人机模型定义：[`MultirotorBase` 在 OmniDrones/omni_drones/robots/drone.py](OmniDrones/omni_drones/robots/drone.py).
