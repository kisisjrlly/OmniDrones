"""
正确的MPC-MAPPO实现，基于参考代码架构
MPC作为Actor网络的一部分，神经网络输出Q和R矩阵
遵循mappo_new.py的架构模式，使用__call__方法和TorchRL组件
"""

import time
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
        self.epsilon = 1e-4
        # self.epsilon = 10.0

        # 神经网络：输出Q和R矩阵的对角元素
        # Q矩阵对角元素：状态权重 (10维)
        # R矩阵对角元素：控制权重 (4维)
        # self.mpc_obs_indices = torch.tensor(list(range(94-23,94-23+10)) + list(range(25,25+3)) + list(range(25+23,25+23+3)) + list(range(7,7+18)) + list(range(4,7)) + list(range(0,4)), dtype=torch.long, device=self.device)
        self.mpc_obs_indices = torch.tensor(list(range(39,39+10)) + list(range(0,3)) + list(range(13,16)) + list(range(26,29)) + list(range(69,72)), dtype=torch.long)

        self.q_r_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            # nn.ReLU(),
            nn.Mish(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_dim, 14),  # 10 (Q) + 4 (R) = 14
            nn.Linear(hidden_dim, (self.mpc_obs_indices.shape[0] + 4) * 2),
            nn.Softplus()  # 确保权重为正
        )
        
        # 初始化MPC求解器
        self.mpc = MPC(T=mpc_horizon * mpc_dt, dt=mpc_dt, mpc_x_dim=self.mpc_obs_indices.shape[0], action_dim=action_dim, device=str(self.device))
        # self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.actor_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.scale_mapping = torch.exp


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
        # print("obs in MPCACTLayer:", obs)
        # print("endpoint_info in MPCACTLayer:", obs[..., 0:4])
        # print("formation_target in MPCACTLayer:", obs[..., 4:7])
        # print("gate_info in MPCACTLayer:", obs[..., 7:25])
        # print("obs others in MPCACTLayer:", obs[..., 25:71])
        # print("obs self in MPCACTLayer:", obs[..., 71:94])
        # print("obs shape in MPCACTLayer:", obs.shape)
        batch_size = obs.shape[0]

        # 神经网络输出Q和R矩阵权重
        q_r_weights_raw = self.q_r_net(obs) + self.epsilon  # [batch_size, 14]
        # platform hover
        # indices = torch.tensor(list(range(0,10)) + list(range(23,26)) + list(range(30,33)) + list(range(43,46)) + list(range(56,59)) + list(range(69,72)), dtype=torch.long)
        
        bs_mpc_x = obs[..., self.mpc_obs_indices]
        mpc_actions, _ = self.mpc.solve(q_r_weights_raw, bs_mpc_x, self.training)
        # print("MPC actions in MPCACTLayer:", mpc_actions)
        
        # 使用MPC动作作为分布的均值，添加固定方差用于探索
        loc = mpc_actions
        # scale = torch.full_like(loc, 0.0)  # 固定标准差
        scale = self.actor_std.expand_as(loc)
        scale = self.scale_mapping(scale)
        return loc, scale
    

class EnsembleModule(_EnsembleModule):
    """
    MPC-MAPPO的EnsembleModule，为每个agent提供独立的参数
    替换vmap实现为基于循环的实现，以解决与torch.autograd.Function的兼容性问题
    """

    def __init__(self, module: TensorDictModuleBase, num_copies: int):
        super(_EnsembleModule, self).__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.num_copies = num_copies

        # 创建functional module并扩展参数
        params_td = make_functional(module).expand(num_copies).to_tensordict()
        self.module = module
        self.params_td = TensorDictParams(params_td)

    def forward(self, tensordict: TensorDict):
        """
        基于循环的前向传播实现，替代vmap以避免与torch.autograd.Function的冲突
        """
        tensordict = tensordict.select(*self.in_keys)
        original_batch_size = tensordict.shape[0]
        
        # 准备输出结果字典
        output_dict = {}
        
        # 对每个agent逐个处理
        outputs_list = []
        for agent_idx in range(self.num_copies):
            # 为当前agent准备输入数据
            agent_input = tensordict.clone()
            
            # 从输入中提取当前agent的数据
            # 假设输入格式为 [batch_size, num_agents, ...]
            for key in agent_input.keys(True):
                if agent_input[key].ndim >= 2 and agent_input[key].shape[1] == self.num_copies:
                    # 选择当前agent的数据：[batch_size, agent_dim, ...]
                    agent_input[key] = agent_input[key][:, agent_idx]
            
            # 设置batch大小为单个agent
            agent_input.batch_size = [original_batch_size]
            
            # 获取当前agent的参数
            agent_params = self.params_td[agent_idx]
            
            # 使用functional API调用模块
            agent_output = self.module(agent_input, agent_params)
            outputs_list.append(agent_output)
        
        # 合并所有agent的输出
        # 将结果重新组织为 [batch_size, num_agents, ...]
        for key in self.out_keys:
            if key in outputs_list[0].keys():
                # 收集所有agent对应key的输出
                key_outputs = [output[key] for output in outputs_list]
                # 沿着agent维度堆叠
                stacked_output = torch.stack(key_outputs, dim=1)
                output_dict[key] = stacked_output
        
        # 创建输出TensorDict
        output_td = TensorDict(output_dict, batch_size=[original_batch_size, self.num_copies])
        
        return output_td


def init_(module):
    """权重初始化函数"""
    # if isinstance(module, nn.Linear):
    #     nn.init.orthogonal_(module.weight, gain=1)
    #     nn.init.constant_(module.bias, 0.)
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        print(f"init weight of {module}")
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        nn.init.constant_(module.bias, 0)


class MPCCritic(nn.Module):
    """价值函数网络"""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MPCMAPPO():
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
        # super().__init__()
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

        # 计算固定的训练batch size
        self.fixed_training_batch_size = self._calculate_fixed_batch_size(cfg)
        print(f"Fixed training batch size: {self.fixed_training_batch_size}")

    def _calculate_fixed_batch_size(self, cfg: DictConfig) -> int:
        """根据配置计算固定的训练batch size"""
        # 从配置中获取参数
        num_envs = getattr(cfg, 'num_envs', 1024)
        train_every = getattr(cfg, 'train_every', 64)
        num_minibatches = getattr(cfg, 'num_minibatches', 16)

        print("num envs: ", num_envs)
        print("train every: ", train_every)
        print("num minibatches: ", num_minibatches)
        # 如果直接指定了固定batch size，使用它
        if hasattr(cfg, 'fixed_training_batch_size'):
            return cfg.fixed_training_batch_size
        
        # 否则根据rollout参数计算
        total_rollout_size = num_envs * train_every  # 1024 * 64 = 65536
        calculated_batch_size = total_rollout_size // num_minibatches  # 65536 // 16 = 4096
        
        # 考虑到is_init过滤可能减少样本数量，使用保守估计
        conservative_batch_size = int(calculated_batch_size * 0.95)  # 减少5%作为缓冲
        
        return conservative_batch_size
    
    def _build_networks(self):
        """构建Actor和Critic网络，支持不共享Actor"""
        # 获取观测维度
        obs_dim = self._get_obs_dim()
        
        # 构建MPC Actor模块
        mpc_actor_layer = MPCACTLayer(
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            mpc_horizon=self.cfg.mpc_config.get("horizon", 10),
            mpc_dt=self.cfg.mpc_config.get("dt", 0.016),
            device=self.device
        )
        
        # 将MPC Actor包装为TensorDictModule
        actor_module = TensorDictModule(
            mpc_actor_layer,
            in_keys=[("agents", "observation")],
            out_keys=["loc", "scale"]
        ).to(self.device)
        
        # 根据配置决定是否共享Actor参数，并相应地创建fake input进行初始化
        if not self.cfg.get("share_actor", False):
            # 不共享：为EnsembleModule创建单agent的fake input进行初始化
            print("not sharing actor parameters, using EnsembleModule")
            # 创建单个agent的fake input，维度为 [batch_size, obs_dim]
            fake_input_single_agent = TensorDict({
                ("agents", "observation"): torch.ones(1, obs_dim, device=self.device)
            }, batch_size=[1], device=self.device)
            print("fake input for single agent:", fake_input_single_agent)
            # 使用单agent输入初始化
            actor_module(fake_input_single_agent)
            print("actor_module initialized with single agent fake input")
            actor_module = EnsembleModule(actor_module, self.num_agents)
        else:
            # 共享参数：使用多agent的fake input进行初始化
            print("sharing actor parameters, applying init_")
            fake_input = self.observation_spec.zero()
            print("fake input for shared actor:", fake_input)
            actor_module(fake_input)
            print("actor_module initialized with shared fake input")
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
                make_mlp([512, 256], nn.Mish),
                nn.LazyLinear(self.num_agents),
                Rearrange("... -> ... 1")
            ),
            [("agents", "observation_central")], ["state_value"]
        ).to(self.device)
        
        # 初始化critic (需要使用多agent的fake input)
        fake_input_for_critic = self.observation_spec.zero()
        self.critic(fake_input_for_critic)
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
        """
        前向传播，用于数据收集阶段（推理模式）
        在此模式下，MPC计算不需要梯度以提高效率和避免内存问题
        """
        # print("MPCMAPPO.__call__ called (inference mode)")
        start_time = time.time()
        
        # 设置为推理模式：禁用梯度计算以提高效率
        with torch.no_grad():
            # 确保actor模块在推理模式
            self.actor.eval()
            self.critic.eval()
            # print("tensordict before actor:", tensordict)
            # 执行actor前向传播（MPC计算在此处进行，无梯度）
            tensordict.update(self.actor(tensordict))
            # print("tensordict after actor:", tensordict)

            # 计算critic值（用于后续的GAE计算）
            self.critic(tensordict)
        # print("MPCMAPPO.__call__ completed in {:.4f} seconds".format(time.time() - start_time))
        return tensordict
    
    def train_op(self, tensordict: TensorDict) -> Dict[str, float]:
        """
        训练操作，遵循mappo_new.py的模式
        在此模式下，确保MPC计算启用梯度以支持可微分优化
        """
        print("MPCMAPPO.train_op called (training mode)")
        
        # 确保模型在训练模式（启用梯度计算）
        self.actor.train()
        self.critic.train()
        
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

        # print("tensordict:", tensordict)

        start_time_epoch = time.time()
        infos = []
        for epoch in range(self.ppo_epochs):
            batch = make_batch(tensordict, 
                               self.num_minibatches,
                               target_batch_size=self.fixed_training_batch_size)
            for minibatch in batch:
                start_time_minibatch = time.time()
                infos.append(self._update(minibatch))
                # print(f"Epoch {epoch+1}/{self.ppo_epochs}, minibatch train updated in {time.time() - start_time_minibatch:.4f} seconds")
        # print(f"Epochs {self.ppo_epochs}, total train updated in {time.time() - start_time_epoch:.4f} seconds")

        # 正确处理infos的聚合
        if infos:
            infos_dict = {}
            for key in infos[0].keys():
                infos_dict[key] = torch.stack([info[key] for info in infos]).mean().item()
            return infos_dict
        else:
            return {}

    def _update(self, tensordict: TensorDict) -> Dict[str, torch.Tensor]:
        """执行单个mini-batch的更新，batch size现在是固定的"""
        # 不再过滤 is_init，因为在 make_batch 中已经处理了
        # tensordict = tensordict[~tensordict["is_init"].squeeze(1)]  # 移除这一行
        
        # print(f"Training with fixed batch size: {tensordict.shape[0]}")
        
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

    def state_dict(self):
        """返回模型状态字典，用于保存检查点"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'value_norm': self.value_norm.state_dict() if hasattr(self.value_norm, 'state_dict') else None
        }
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典，用于恢复检查点"""
        try:
            self.actor.load_state_dict(state_dict['actor'])
            self.critic.load_state_dict(state_dict['critic'])
            self.actor_opt.load_state_dict(state_dict['actor_opt'])
            self.critic_opt.load_state_dict(state_dict['critic_opt'])
            
            if state_dict.get('value_norm') is not None and hasattr(self.value_norm, 'load_state_dict'):
                self.value_norm.load_state_dict(state_dict['value_norm'])
            
            print("Successfully loaded MPCMAPPO checkpoint")
        except Exception as e:
            print(f"Error loading MPCMAPPO checkpoint: {e}")
            raise


def make_batch(tensordict: TensorDict, num_minibatches: int, target_batch_size: int = None):
    """创建真正固定大小的mini-batches，确保训练时batch size不变"""
    start_time = time.time()
    tensordict = tensordict.reshape(-1)
    
    # 第一步：过滤掉 is_init=True 的样本
    if "is_init" in tensordict.keys():
        valid_mask = ~tensordict["is_init"].squeeze(-1)
        valid_tensordict = tensordict[valid_mask]
        # print(f"Filtered {(~valid_mask).sum().item()} is_init samples, remaining: {valid_tensordict.shape[0]}")
    else:
        valid_tensordict = tensordict
    
    total_valid_samples = valid_tensordict.shape[0]
    
    # 第二步：计算或使用指定的目标batch size
    if target_batch_size is None:
        target_batch_size = total_valid_samples // num_minibatches
    
    total_target_samples = target_batch_size * num_minibatches
    
    # 第三步：处理样本数量不足的情况
    if total_valid_samples < total_target_samples:
        # 方案A：通过重复采样到目标数量
        repeat_factor = (total_target_samples + total_valid_samples - 1) // total_valid_samples
        
        # 使用 torch.cat 来重复数据
        expanded_data_list = [valid_tensordict] * repeat_factor
        expanded_data = torch.cat(expanded_data_list, dim=0)
        
        # 随机选择目标数量的样本
        indices = torch.randperm(expanded_data.shape[0], device=expanded_data.device)[:total_target_samples]
        final_tensordict = expanded_data[indices]
        
        # print(f"Padded with repetition: {total_valid_samples} -> {total_target_samples}")
        
    elif total_valid_samples > total_target_samples:
        # 方案B：随机选择目标数量的样本
        indices = torch.randperm(total_valid_samples, device=valid_tensordict.device)[:total_target_samples]
        final_tensordict = valid_tensordict[indices]
        
        # print(f"Randomly sampled: {total_valid_samples} -> {total_target_samples}")
        
    else:
        # 刚好相等
        final_tensordict = valid_tensordict
        # print(f"Perfect match: {total_target_samples} samples")
    
    # 第四步：随机打乱并分成固定大小的batches
    perm = torch.randperm(total_target_samples, device=final_tensordict.device)
    final_tensordict = final_tensordict[perm]
    
    # 重新整形为固定大小的batches
    final_tensordict = final_tensordict.reshape(num_minibatches, target_batch_size)
    
    # print(f"Created {num_minibatches} batches of size {target_batch_size} each ,used time: {time.time() - start_time}")
    
    for i in range(num_minibatches):
        yield final_tensordict[i]