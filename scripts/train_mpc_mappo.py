#!/usr/bin/env python3

"""
Training script for MPC-enhanced MAPPO in OmniDrones.
Integrates Model Predictive Control with Multi-Agent Proximal Policy Optimization
for improved drone formation control performance.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import wandb
from tensordict import TensorDict

# Add OmniDrones to path
sys.path.append(str(Path(__file__).parent.parent))

from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.learning.mpc_mappo import MPCMAPPO
from omni_drones.utils.torchrl.env import AgentSpec
from omni_drones.utils.wandb import init_wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPCMAPPOTrainer:
    """Trainer class for MPC-enhanced MAPPO algorithm."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Get agent specification
        self.agent_spec = AgentSpec(
            name="quadrotor",
            action_spec=self.env.action_spec["agents"],
            observation_spec=self.env.observation_spec["agents"],
            reward_spec=self.env.reward_spec["agents"],
        )
        
        # Initialize algorithm
        self.algorithm = MPCMAPPO(
            cfg=cfg.algorithm.mpc_mappo,
            agent_spec=self.agent_spec,
            device=str(self.device),
            mpc_config=cfg.algorithm.mpc_mappo.mpc_config,
        )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'mpc_solve_times': [],
            'formation_success_rate': [],
            'gate_completion_rate': [],
        }
        
        # Setup logging
        self._setup_logging()
        
    def _create_environment(self) -> IsaacEnv:
        """Create and configure the training environment."""
        logger.info("Creating training environment...")
        
        # Environment configuration
        env_cfg = self.cfg.task
        
        # Create environment using OmniDrones factory
        from omni_drones.envs import env_factory
        
        env = env_factory.make(
            task=env_cfg.task,
            cfg=env_cfg,
            headless=self.cfg.headless,
            enable_livestream=self.cfg.enable_livestream,
        )
        
        logger.info(f"Environment created with {env.num_envs} parallel environments")
        logger.info(f"Action space: {env.action_spec}")
        logger.info(f"Observation space: {env.observation_spec}")
        
        return env
        
    def _setup_logging(self):
        """Setup logging and monitoring."""
        if self.cfg.wandb_log:
            # Initialize Weights & Biases
            wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
            wandb.init(
                project="mpc-mappo-omnidrones",
                config=wandb_config,
                name=f"mpc_mappo_{self.cfg.task.task}_{int(time.time())}",
            )
            
        # Create model directory
        self.model_dir = Path(self.cfg.algorithm.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
            
        logger.info(f"Model directory: {self.model_dir}")
        
    def collect_rollouts(self, num_steps: int) -> TensorDict:
        """Collect rollouts from the environment."""
        logger.debug(f"Collecting {num_steps} steps of experience...")
        
        rollouts = []
        current_obs = self.env.reset()
        
        for step in range(num_steps):
            # Convert observation to TensorDict format expected by algorithm
            obs_td = TensorDict({
                "agents": {
                    "observation": current_obs["agents"]["observation"],
                }
            }, batch_size=[])
            
            # Get actions from algorithm
            with torch.no_grad():
                action_td = self.algorithm.actor(obs_td)
                value_td = self.algorithm.critic(obs_td)
                
            # Step environment
            actions = action_td["agents"]["action"]
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # Store transition
            transition = TensorDict({
                "agents": {
                    "observation": current_obs["agents"]["observation"],
                    "action": actions,
                    "action_log_probs": action_td["agents"]["action_log_probs"],
                    "reward": rewards["agents"],
                    "done": dones["agents"],
                    "value": value_td["agents"]["state_value"],
                }
            }, batch_size=[])
            
            rollouts.append(transition)
            
            # Update observations
            current_obs = next_obs
            
            # Track episode completion
            if torch.any(dones["agents"]):
                self.episode_count += torch.sum(dones["agents"]).item()
                
        # Stack rollouts
        rollout_td = torch.stack(rollouts, dim=1)  # [batch, time, ...]
        
        # Compute advantages using GAE
        rollout_td = self._compute_advantages(rollout_td)
        
        return rollout_td
        
    def _compute_advantages(self, rollout_td: TensorDict) -> TensorDict:
        """Compute advantages using Generalized Advantage Estimation."""
        from omni_drones.learning.utils.gae import compute_gae
        
        rewards = rollout_td["agents"]["reward"]
        values = rollout_td["agents"]["value"]
        dones = rollout_td["agents"]["done"]
        
        # Compute last value for bootstrap
        last_obs = TensorDict({
            "agents": {"observation": rollout_td["agents"]["observation"][:, -1]}
        }, batch_size=[])
        
        with torch.no_grad():
            last_value = self.algorithm.critic(last_obs)["agents"]["state_value"]
            
        # Compute GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=last_value,
            gamma=self.cfg.algorithm.mpc_mappo.gamma,
            gae_lambda=self.cfg.algorithm.mpc_mappo.gae_lambda,
        )
        
        # Add to tensordict
        rollout_td["agents"]["advantage"] = advantages
        rollout_td["agents"]["return"] = returns
        
        return rollout_td
        
    def train_step(self, rollout_td: TensorDict) -> Dict[str, float]:
        """Perform one training step."""
        logger.debug("Performing training step...")
        
        # Flatten batch dimension for training
        batch_size = rollout_td.shape[0] * rollout_td.shape[1]
        flat_rollout = rollout_td.flatten(0, 1)
        
        # Training metrics
        metrics = {}
        
        # Multiple epochs of training
        for epoch in range(self.cfg.algorithm.mpc_mappo.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(batch_size)
            mini_batch_size = batch_size // self.cfg.algorithm.mpc_mappo.num_mini_batches
            
            for i in range(self.cfg.algorithm.mpc_mappo.num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size
                mini_batch_indices = indices[start_idx:end_idx]
                
                mini_batch = flat_rollout[mini_batch_indices]
                
                # Compute loss
                loss_td, batch_metrics = self.algorithm.loss_fn(mini_batch)
                
                # Backward pass
                self.algorithm.actor_optimizer.zero_grad()
                self.algorithm.critic_optimizer.zero_grad()
                
                loss_td["loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.algorithm.actor.parameters(),
                    self.cfg.algorithm.mpc_mappo.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.algorithm.critic.parameters(),
                    self.cfg.algorithm.mpc_mappo.max_grad_norm
                )
                
                # Optimizer step
                self.algorithm.actor_optimizer.step()
                self.algorithm.critic_optimizer.step()
                
                # Update metrics
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
                    
        # Average metrics across mini-batches
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
            
        return metrics
        
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy."""
        logger.info(f"Evaluating policy for {num_episodes} episodes...")
        
        eval_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'formation_success': [],
            'gate_completion': [],
        }
        
        # Set algorithm to evaluation mode
        self.algorithm.actor.eval()
        self.algorithm.critic.eval()
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # Get action
                    obs_td = TensorDict({
                        "agents": {"observation": obs["agents"]["observation"]}
                    }, batch_size=[])
                    
                    action_td = self.algorithm.actor(obs_td)
                    actions = action_td["agents"]["action"]
                    
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions)
                    
                    episode_reward += torch.mean(rewards["agents"]).item()
                    episode_length += 1
                    done = torch.any(dones["agents"]).item()
                    
                eval_metrics['episode_rewards'].append(episode_reward)
                eval_metrics['episode_lengths'].append(episode_length)
                
                # Extract success metrics from info if available
                if 'formation_success' in infos:
                    eval_metrics['formation_success'].append(infos['formation_success'])
                if 'gate_completion' in infos:
                    eval_metrics['gate_completion'].append(infos['gate_completion'])
                    
        # Set back to training mode
        self.algorithm.actor.train()
        self.algorithm.critic.train()
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in eval_metrics.items():
            if values:
                avg_metrics[f'eval_{key}'] = np.mean(values)
                
        logger.info(f"Evaluation complete. Average reward: {avg_metrics.get('eval_episode_rewards', 0):.2f}")
        
        return avg_metrics
        
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'actor_state_dict': self.algorithm.actor.state_dict(),
            'critic_state_dict': self.algorithm.critic.state_dict(),
            'actor_optimizer_state_dict': self.algorithm.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.algorithm.critic_optimizer.state_dict(),
            'metrics': metrics,
            'config': OmegaConf.to_container(self.cfg),
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.model_dir / "latest_checkpoint.pt")
        
        # Save best model if improved
        current_reward = metrics.get('eval_episode_rewards', float('-inf'))
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            torch.save(checkpoint, self.model_dir / "best_model.pt")
            logger.info(f"New best model saved with reward: {current_reward:.2f}")
            
    def train(self):
        """Main training loop."""
        logger.info("Starting MPC-MAPPO training...")
        logger.info(f"Training configuration: {OmegaConf.to_yaml(self.cfg.algorithm.mpc_mappo)}")
        
        # Training parameters
        rollout_steps = self.cfg.episode_len
        total_frames = self.cfg.total_frames
        eval_interval = self.cfg.eval_interval
        save_interval = self.cfg.save_interval
        
        start_time = time.time()
        
        while self.global_step < total_frames:
            # Collect rollouts
            rollout_start = time.time()
            rollout_td = self.collect_rollouts(rollout_steps)
            rollout_time = time.time() - rollout_start
            
            # Training step
            train_start = time.time()
            train_metrics = self.train_step(rollout_td)
            train_time = time.time() - train_start
            
            # Update global step
            self.global_step += rollout_steps * self.env.num_envs
            
            # Log metrics
            if self.global_step % self.cfg.log_interval == 0:
                # Compute additional metrics
                episode_rewards = torch.mean(rollout_td["agents"]["reward"], dim=1).mean().item()
                
                log_metrics = {
                    'global_step': self.global_step,
                    'episode_count': self.episode_count,
                    'episode_reward': episode_rewards,
                    'rollout_time': rollout_time,
                    'train_time': train_time,
                    **train_metrics,
                }
                
                # Log to console
                logger.info(
                    f"Step {self.global_step}: "
                    f"Reward={episode_rewards:.3f}, "
                    f"PolicyLoss={train_metrics.get('policy_loss', 0):.4f}, "
                    f"ValueLoss={train_metrics.get('value_loss', 0):.4f}"
                )
                
                # Log to wandb
                if self.cfg.wandb_log:
                    wandb.log(log_metrics, step=self.global_step)
                    
            # Evaluation
            if self.global_step % eval_interval == 0:
                eval_metrics = self.evaluate()
                
                if self.cfg.wandb_log:
                    wandb.log(eval_metrics, step=self.global_step)
                    
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    all_metrics = {**train_metrics, **eval_metrics}
                    self.save_checkpoint(all_metrics)
                    
        # Final evaluation and save
        logger.info("Training completed. Performing final evaluation...")
        final_metrics = self.evaluate(num_episodes=20)
        self.save_checkpoint(final_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final average reward: {final_metrics.get('eval_episode_rewards', 0):.2f}")


@hydra.main(version_base=None, config_path="../configs", config_name="train_mpc_mappo")
def main(cfg: DictConfig):
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create trainer and start training
    trainer = MPCMAPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
