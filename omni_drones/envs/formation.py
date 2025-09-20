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

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

from omni_drones.controllers import (
    LeePositionController,
    AttitudeController,
    RateController
)

REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [0, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

FORMATIONS = {
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class Formation(IsaacEnv):
    """
    This is a formation control task. The goal is to control the drone to form a
    regular polygon formation. The reward is the negative of the formation cost.

    ## Observation

    - `obs_self`: the relative position, velocity, and orientation of the drone
    - `obs_others`: the relative position, velocity, and orientation of other drones

    ## Reward

    - `formation`: the negative of the formation cost.
    - `pos`: the negative of the distance to the target position.
    - `heading`: the negative of the heading error.

    ## Episode End

    The episode terminates when any of the following conditions are met:

    - The drone crashes.
    - The minimum distance between any two drones is less than a threshold.

    or is truncated when it reaches the maximum length.

    ## Config
    """
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance

        super().__init__(cfg, headless)

        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        # initial state distribution
        self.cells = (
            make_cells([-2, -2, 0.5], [2, 2, 2], [0.5, 0.5, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_pos = self.target_pos.expand(self.num_envs, 1, 3)
        self.target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_heading[..., 0] = -1

        self.alpha = 0.8

        # self.last_cost_l = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_cost_pos = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller, self.device
        )

        scene_utils.design_scene()

        self.target_pos = torch.tensor([0.0, 0.0, 1.5], device=self.device)

        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                self.cfg.task.formation, device=self.device
            )
        else:
            raise ValueError(f"Invalid target formation {formation}")

        self.formation = self.formation + self.target_pos
        # self.formation_L = laplacian(self.formation)

        self.drone.spawn(translations=self.formation)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)),
        }).to(self.device)
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
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
            "drone",
            self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents","action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )
         # additional infos & buffers
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            "episode_len": UnboundedContinuousTensorSpec(1),
            # "cost_laplacian": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            "cost_hausdorff": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        pos = vmap(sample_from_grid, randomness="different")(
            self.cells.expand(len(env_ids), *self.cells.shape), n=self.drone.n
        ) + self.envs_positions[env_ids].unsqueeze(1)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)

        print("pos shape is:", pos.shape)
        # print("self.formation shape is:", self.formation.shape)
        self.last_cost_h[env_ids] = vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )
        # self.last_cost_l[env_ids] = vmap(cost_formation_laplacian)(
        #     pos, desired_p=self.formation
        # )
        com_pos = (pos - self.envs_positions[env_ids].unsqueeze(1)).mean(1, keepdim=True)
        self.last_cost_pos[env_ids] = torch.square(
            com_pos - self.target_pos[env_ids]
        ).sum(2)

        self.stats[env_ids] = 0.

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

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        pos = self.drone.pos
        self.drone_states[..., :3] = self.target_pos - pos

        obs_self = [self.drone_states]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        obs_self = torch.cat(obs_self, dim=-1)

        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))
        relative_pos = vmap(off_diag)(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            vmap(others)(self.drone_states[..., 3:13])
        ], dim=-1)

        obs = TensorDict({
            "obs_self": obs_self.unsqueeze(2),
            "obs_others": obs_others,
        }, [self.num_envs, self.drone.n])

        state = TensorDict({"drones": self.drone_states}, self.batch_size)

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
        # cost_l = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L)
        pos = self.drone.pos

        cost_h = cost_formation_hausdorff(pos, desired_p=self.formation)

        distance = torch.norm(pos.mean(-2, keepdim=True) - self.target_pos, dim=-1)

        reward_formation =  1 / (1 + torch.square(cost_h * 1.6))
        # reward_pos = 1 / (1 + cost_pos)

        # reward_formation = torch.exp(- cost_h * 1.6)
        reward_pos = torch.exp(- distance)
        reward_heading = self.drone.heading[..., 0].mean(-1, True)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)
        reward = (
            reward_separation * (
                reward_formation
                + reward_formation * (reward_pos + reward_heading)
                + 0.4 * reward_pos
            )
        )

        # self.last_cost_l[:] = cost_l
        self.last_cost_h[:] = cost_h
        self.last_cost_pos[:] = torch.square(distance)

        misbehave = (pos[..., 2] < 0.2).any(-1, keepdim=True) | (separation < 0.23)
        hasnan = torch.isnan(self.drone_states).any(-1)

        terminated = misbehave | hasnan.any(-1, keepdim=True)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        # self.stats["cost_laplacian"] -= cost_l
        self.stats["cost_hausdorff"].lerp_(cost_h, (1-self.alpha))
        self.stats["pos_error"].lerp_(distance, (1-self.alpha))

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(1).expand(-1, self.drone.n, 1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )


def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    symmetric normalized laplacian

    p: (n, dim)
    """
    assert p.dim() == 2
    A = torch.cdist(p, p)
    D = torch.sum(A, dim=-1)
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = D - A
    return L

# @vmap
def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d
