"""
正确的MPC-MAPPO实现，基于参考代码架构
MPC作为Actor网络的一部分，神经网络输出Q和R矩阵
遵循mappo_new.py的架构模式，使用__call__方法和TorchRL组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import Normal, Independent
from typing import Dict, Tuple, Any, Optional, Union
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data import CompositeSpec, TensorSpec, BoundedTensorSpec

# TorchRL imports for compatibility
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
    make_functional,
    TensorDictParams,
    EnsembleModule as _EnsembleModule
)
from torchrl.modules import ProbabilisticActor
from torch._functorch.apis import vmap
from einops.layers.torch import Rearrange

# from omni_drones.learning.utils.gae import GAE
from .ppo.common import GAE, make_mlp
from .utils.valuenorm import ValueNorm1
from .modules.distributions import IndependentNormal


from .mpc_components import MPC

class MPCACTLayer(nn.Module):
    """MPC增强的Actor层，类似参考代码中的Actor"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        mpc_horizon: int = 10,
        mpc_dt: float = 0.1,
        device: Union[torch.device, str] = "cpu",
        **mpc_kwargs
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # MPC配置
        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt

        # 神经网络：输出Q和R矩阵的对角元素
        # Q矩阵对角元素：状态权重 (10维)
        # R矩阵对角元素：控制权重 (4维)
        self.q_r_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 14),  # 10 (Q) + 4 (R) = 14
            nn.Softplus()  # 确保权重为正
        )
        
        # 初始化MPC求解器
        self.mpc = MPC(T=mpc_horizon * mpc_dt, dt=mpc_dt, device=str(self.device))


    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，直接返回分布参数（遵循TorchRL模式）
        
        Args:
            observation: 观测状态 [batch_size, obs_dim]
            
        Returns:
            loc: 动作分布的均值 [batch_size, action_dim]
            scale: 动作分布的标准差 [batch_size, action_dim]
        """
        obs = observation.to(self.device)
        batch_size = obs.shape[0]
        
        # 神经网络输出Q和R矩阵权重
        q_r_weights = self.q_r_net(obs)  # [batch_size, 14]
        
        # 提取状态的前10维作为MPC初始状态
        # 假设obs包含：[px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...]
        assert obs.shape[-1] >= 10, "Observation must have at least 10 dimensions for MPC initial state."
        quad_s0 = obs[..., :10]  # [batch_size, 10]
        # print("quad_s0 shape in MPCACTLayer:", quad_s0.shape)
        
        with torch.no_grad():
            mpc_actions, _ = self.mpc.solve(q_r_weights, quad_s0, is_evaluated=False)
        
        # 使用MPC动作作为分布的均值，添加固定方差用于探索
        loc = mpc_actions
        scale = torch.full_like(loc, 0.1)  # 固定标准差
        return loc, scale
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        评估动作，用于训练
        
        Args:
            obs: 观测状态 [batch_size, obs_dim]
            actions: 动作 [batch_size, action_dim]
            
        Returns:
            action_log_probs: 动作对数概率 [batch_size]
            entropy: 动作分布熵 [batch_size]
            action_dist: 动作分布
        """
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        
        # 神经网络输出Q和R矩阵权重
        q_r_weights = self.q_r_net(obs)
        
        # 提取状态的前10维作为MPC初始状态

        quad_s0 = obs[:, :10]

        
        # 计算MPC动作
        mpc_actions, _ = self.mpc.solve(q_r_weights, quad_s0, is_evaluated=True)
        
        # 使用MPC动作作为分布均值
        action_mean = mpc_actions
        action_std = torch.full_like(action_mean, 0.1)
        
        action_dist = Independent(Normal(action_mean, action_std), 1)
        action_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return action_log_probs, entropy, action_dist


class EnsembleModule(_EnsembleModule):
    """
    MPC-MAPPO的EnsembleModule，为每个agent提供独立的参数
    基于mappo_new.py的实现
    """

    def __init__(self, module: TensorDictModuleBase, num_copies: int):
        super(_EnsembleModule, self).__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.num_copies = num_copies

        params_td = make_functional(module).expand(num_copies).to_tensordict()
        self.module = module
        self.vmapped_forward = vmap(self.module, (1, 0), 1)
        self.params_td = TensorDictParams(params_td)

    def forward(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.in_keys)
        tensordict.batch_size = [tensordict.shape[0], self.num_copies]
        print("tensordict in EnsembleModule:", tensordict)
        print("tensordict batch size in EnsembleModule:", tensordict.batch_size)
        print("tensordict shape in EnsembleModule:", tensordict.shape)
        print("params_td shape in EnsembleModule:", self.params_td.shape)
        return self.vmapped_forward(tensordict, self.params_td)


def init_(module):
    """权重初始化函数"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0.)


class MPCCritic(nn.Module):
    """价值函数网络"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MPCMAPPO:
    """
    MPC-MAPPO实现，遵循mappo_new.py的架构模式
    支持不共享Actor的逻辑
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        observation_spec: CompositeSpec,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        
        # PPO配置
        self.entropy_coef = cfg.get("entropy_coef", 0.001)
        self.clip_param = cfg.get("clip_param", 0.1)
        self.ppo_epochs = cfg.get("ppo_epochs", 15)
        self.num_minibatches = cfg.get("num_minibatches", 1)
        
        # 损失函数配置
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        
        # GAE
        self.gae = GAE(0.99, 0.95)
        
        # 从action_spec提取agent数量和动作维度
        if not action_spec.ndim >= 2:
            raise ValueError("Please use PPOPolicy for single-agent environments.")
        
        self.num_agents, self.action_dim = action_spec.shape[-2:]
        
        # 构建网络
        self._build_networks()
        
        # 构建优化器
        self._build_optimizers()
        
        # Value normalization
        self.value_norm = ValueNorm1(input_shape=1).to(self.device)
    
    def _build_networks(self):
        """构建Actor和Critic网络，支持不共享Actor"""
        # 获取观测维度
        obs_dim = self._get_obs_dim()
        
        # 创建fake input用于初始化
        fake_input = self.observation_spec.zero()
        
        # 构建MPC Actor模块
        mpc_actor_layer = MPCACTLayer(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            mpc_horizon=self.cfg.get("mpc_horizon", 10),
            mpc_dt=self.cfg.get("mpc_dt", 0.1),
            device=self.device
        )
        
        # 将MPC Actor包装为TensorDictModule
        actor_module = TensorDictModule(
            mpc_actor_layer,
            in_keys=[("agents", "observation")],
            out_keys=["loc", "scale"]
        ).to(self.device)
        
        print("fake input:", fake_input)
        # 初始化actor
        actor_module(fake_input)
        print("actor_module initialized with fake input:", actor_module)
        
        # 根据配置决定是否共享Actor参数
        if not self.cfg.get("share_actor", False):
            # 不共享：使用EnsembleModule为每个agent创建独立参数
            print("not sharing actor parameters, using EnsembleModule")
            actor_module = EnsembleModule(actor_module, self.num_agents)
        else:
            # 共享参数：应用初始化
            print("sharing actor parameters, applying init_")
            actor_module.apply(init_)
        
        # 创建ProbabilisticActor
        self.actor = ProbabilisticActor(
            module=actor_module,  # type: ignore
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        # 构建Critic网络
        self.critic = TensorDictModule(
            nn.Sequential(
                make_mlp([512, 256], nn.LeakyReLU),
                nn.LazyLinear(self.num_agents),
                Rearrange("... -> ... 1")
            ),
            [("agents", "observation_central")], ["state_value"]
        ).to(self.device)
        
        # 初始化critic
        self.critic(fake_input)
        self.critic.apply(init_)
    
    def _get_obs_dim(self) -> int:
        """计算观测维度"""
        if isinstance(self.observation_spec, CompositeSpec):
            if ("agents", "observation") in self.observation_spec.keys(True):
                obs_spec = self.observation_spec[("agents", "observation")]
                if isinstance(obs_spec, CompositeSpec):
                    obs_dim = 0
                    for key in obs_spec.keys():
                        spec = obs_spec[key]
                        if hasattr(spec, 'shape'):
                            obs_dim += spec.shape.numel() // spec.shape[0] if len(spec.shape) > 1 else spec.shape[-1]
                else:
                    obs_dim = obs_spec.shape[-1]
            elif "observation" in self.observation_spec.keys():
                obs_dim = self.observation_spec["observation"].shape[-1]
            else:
                obs_dim = sum([spec.shape[-1] for spec in self.observation_spec.values()])
        else:
            obs_dim = self.observation_spec.shape[-1]
        
        return obs_dim
    
    def _build_optimizers(self):
        """构建优化器"""
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
    
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """前向传播，遵循mappo_new.py的模式"""
        tensordict.update(self.actor(tensordict))
        self.critic(tensordict)
        return tensordict
    
    def train_op(self, tensordict: TensorDict) -> Dict[str, float]:
        """训练操作，遵循mappo_new.py的模式"""
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]
        dones = einops.repeat(dones, "t n 1 -> t n a 1", a=self.num_agents)
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.ppo_epochs):
            batch = make_batch(tensordict, self.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        # 正确处理infos的聚合
        if infos:
            infos_dict = {}
            for key in infos[0].keys():
                infos_dict[key] = torch.stack([info[key] for info in infos]).mean().item()
            return infos_dict
        else:
            return {}

    def _update(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        """执行单个mini-batch的更新"""
        tensordict = tensordict[~tensordict["is_init"].squeeze(1)]
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy().mean()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * entropy

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }


def make_batch(tensordict: TensorDict, num_minibatches: int):
    """创建mini-batches，遵循mappo_new.py的实现"""
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
