#!/usr/bin/env python3

"""
MPC-MAPPO Training Script for OmniDrones
========================================

This script trains the MPC-enhanced MAPPO algorithm for multi-drone formation control
and gate traversal tasks. It integrates Model Predictive Control (MPC) guidance
with Multi-Agent Proximal Policy Optimization (MAPPO).

Usage:
    python train_mpc_mappo.py
    python train_mpc_mappo.py task=MPCFormationGateTraversal algo=mpc_mappo
    
Features:
- MPC trajectory guidance for improved sample efficiency
- Multi-agent formation control with collision avoidance
- Gate traversal in formation
- Real-time performance monitoring
- Tensorboard and WandB logging support
"""

import os
import sys
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tensordict import TensorDict

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from omni_drones.envs import IsaacEnv
from omni_drones.utils.torchrl.env import AgentSpec
from omni_drones.utils.wandb import init_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MPCMAPPOTrainingStats:
    """Statistics tracking for MPC-MAPPO training."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.formation_success_rates = []
        self.gate_completion_rates = []
        self.mpc_solve_times = []
        self.policy_losses = []
        self.value_losses = []
        self.total_steps = 0
        self.total_episodes = 0
        
    def update(self, metrics: Dict[str, Any]):
        """Update statistics with new metrics."""
        if 'episode_reward' in metrics:
            self.episode_rewards.append(metrics['episode_reward'])
        if 'episode_length' in metrics:
            self.episode_lengths.append(metrics['episode_length'])
        if 'formation_success_rate' in metrics:
            self.formation_success_rates.append(metrics['formation_success_rate'])
        if 'gate_completion_rate' in metrics:
            self.gate_completion_rates.append(metrics['gate_completion_rate'])
        if 'mpc_solve_time' in metrics:
            self.mpc_solve_times.append(metrics['mpc_solve_time'])
        if 'policy_loss' in metrics:
            self.policy_losses.append(metrics['policy_loss'])
        if 'value_loss' in metrics:
            self.value_losses.append(metrics['value_loss'])
            
        self.total_steps += metrics.get('steps', 0)
        self.total_episodes += metrics.get('episodes', 0)
        
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {}
        
        if self.episode_rewards:
            summary['mean_episode_reward'] = np.mean(self.episode_rewards[-100:])
            summary['std_episode_reward'] = np.std(self.episode_rewards[-100:])
            
        if self.episode_lengths:
            summary['mean_episode_length'] = np.mean(self.episode_lengths[-100:])
            
        if self.formation_success_rates:
            summary['mean_formation_success'] = np.mean(self.formation_success_rates[-100:])
            
        if self.gate_completion_rates:
            summary['mean_gate_completion'] = np.mean(self.gate_completion_rates[-100:])
            
        if self.mpc_solve_times:
            summary['mean_mpc_solve_time'] = np.mean(self.mpc_solve_times[-100:])
            
        if self.policy_losses:
            summary['mean_policy_loss'] = np.mean(self.policy_losses[-50:])
            
        if self.value_losses:
            summary['mean_value_loss'] = np.mean(self.value_losses[-50:])
            
        summary['total_steps'] = self.total_steps
        summary['total_episodes'] = self.total_episodes
        
        return summary


def create_mpc_mappo_algorithm(cfg: DictConfig, agent_spec: AgentSpec, device: str):
    """Create MPC-enhanced MAPPO algorithm instance."""
    try:
        # Try to import the MPC-MAPPO implementation
        from omni_drones.learning.mpc_mappo import MPCMAPPO
        
        logger.info("Creating MPC-MAPPO algorithm...")
        algorithm = MPCMAPPO(
            cfg=cfg.algo,
            agent_spec=agent_spec,
            device=device,
        )
        logger.info("MPC-MAPPO algorithm created successfully")
        return algorithm
        
    except ImportError:
        logger.warning("MPC-MAPPO implementation not found, falling back to standard MAPPO")
        from omni_drones.learning.mappo import MAPPO
        
        # Create standard MAPPO with MPC configuration adapted
        mappo_cfg = OmegaConf.create({
            "name": "mappo",
            "train_every": cfg.algo.train_every,
            "num_minibatches": cfg.algo.num_minibatches,
            "ppo_epochs": cfg.algo.ppo_epochs,
            "clip_param": cfg.algo.clip_param,
            "entropy_coef": cfg.algo.entropy_coef,
            "gae_lambda": cfg.algo.gae_lambda,
            "gamma": cfg.algo.gamma,
            "max_grad_norm": cfg.algo.max_grad_norm,
            "normalize_advantages": cfg.algo.normalize_advantages,
            "reward_weights": cfg.algo.reward_weights,
            "share_actor": cfg.algo.share_actor,
            "critic_input": cfg.algo.critic_input,
            "actor": cfg.algo.actor,
            "critic": cfg.algo.critic,
        })
        
        algorithm = MAPPO(
            cfg=mappo_cfg,
            agent_spec=agent_spec,
            device=device,
        )
        logger.info("Standard MAPPO algorithm created with MPC-compatible configuration")
        return algorithm


def train_mpc_mappo(cfg: DictConfig):
    """Main training function for MPC-MAPPO."""
    
    logger.info("Starting MPC-MAPPO training...")
    logger.info(f"Configuration: \\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create environment
    logger.info("Creating training environment...")
    env_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    
    from omni_drones.envs import Isaac
    env = Isaac(cfg.task)
    
    logger.info(f"Environment created: {cfg.task.name}")
    logger.info(f"Number of environments: {env.num_envs}")
    logger.info(f"Number of agents per environment: {env.n_agents}")
    
    # Get agent specification
    agent_spec = AgentSpec(
        name="quadrotor",
        action_spec=env.action_spec["agents"],
        observation_spec=env.observation_spec["agents"],
        reward_spec=env.reward_spec["agents"],
    )
    
    # Create algorithm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    algorithm = create_mpc_mappo_algorithm(cfg, agent_spec, device)
    
    # Initialize logging
    stats = MPCMAPPOTrainingStats()
    
    # Initialize WandB if enabled
    if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
        logger.info("Initializing WandB logging...")
        init_wandb(cfg, algorithm=algorithm)
    
    # Training loop
    logger.info("Starting training loop...")
    tensordict = env.reset()
    
    steps = 0
    episodes = 0
    start_time = time.time()
    
    try:
        while steps < cfg.total_frames:
            # Collect rollouts
            with torch.no_grad():
                tensordict = algorithm.collect(env, tensordict)
            
            # Training step
            algorithm.update()
            
            steps += env.num_envs * env.n_agents
            
            # Check for episode completion
            if tensordict["done"].any():
                episodes += tensordict["done"].sum().item()
                
                # Extract metrics for completed episodes
                if tensordict["done"].any():
                    episode_metrics = extract_episode_metrics(tensordict)
                    stats.update(episode_metrics)
            
            # Logging
            if steps % cfg.get('performance', {}).get('log_frequency', 1000) == 0:
                elapsed_time = time.time() - start_time
                fps = steps / elapsed_time if elapsed_time > 0 else 0
                
                summary = stats.get_summary()
                summary.update({
                    'steps': steps,
                    'episodes': episodes,
                    'fps': fps,
                    'elapsed_time': elapsed_time,
                })
                
                # Log to console
                logger.info(
                    f"Step: {steps:,} | Episodes: {episodes:,} | "
                    f"FPS: {fps:.1f} | "
                    f"Reward: {summary.get('mean_episode_reward', 0):.3f} | "
                    f"Success: {summary.get('mean_formation_success', 0):.3f}"
                )
                
                # Log to WandB if enabled
                if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
                    import wandb
                    wandb.log(summary, step=steps)
            
            # Evaluation
            if (cfg.eval_interval > 0 and 
                steps % cfg.eval_interval == 0 and 
                steps > 0):
                logger.info(f"Running evaluation at step {steps}")
                eval_metrics = run_evaluation(env, algorithm, cfg)
                logger.info(f"Evaluation results: {eval_metrics}")
                
                if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
                    import wandb
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=steps)
            
            # Save checkpoint
            if (cfg.save_interval > 0 and 
                steps % cfg.save_interval == 0 and 
                steps > 0):
                checkpoint_path = save_checkpoint(algorithm, steps, cfg)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        env.close()
        logger.info("Training completed")


def extract_episode_metrics(tensordict: TensorDict) -> Dict[str, Any]:
    """Extract metrics from completed episodes."""
    metrics = {}
    
    if "episode_reward" in tensordict:
        metrics['episode_reward'] = tensordict["episode_reward"][tensordict["done"]].mean().item()
    
    if "episode_length" in tensordict:
        metrics['episode_length'] = tensordict["episode_length"][tensordict["done"]].mean().item()
        
    # Add more episode-specific metrics here
    metrics['episodes'] = tensordict["done"].sum().item()
    
    return metrics


def run_evaluation(env, algorithm, cfg: DictConfig) -> Dict[str, float]:
    """Run evaluation episodes."""
    eval_episodes = cfg.get('mpc_training', {}).get('eval_episodes', 10)
    
    eval_rewards = []
    eval_success_rates = []
    
    for episode in range(eval_episodes):
        tensordict = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while episode_length < cfg.env.max_episode_length:
            with torch.no_grad():
                tensordict = algorithm.eval_step(tensordict)
                env.step(tensordict)
            
            episode_reward += tensordict.get("reward", 0).sum().item()
            episode_length += 1
            
            if tensordict.get("done", False).any():
                break
        
        eval_rewards.append(episode_reward)
        # Add success rate calculation based on task completion
        # eval_success_rates.append(calculate_success_rate(tensordict))
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'min_reward': np.min(eval_rewards),
    }


def save_checkpoint(algorithm, step: int, cfg: DictConfig) -> str:
    """Save training checkpoint."""
    checkpoint_dir = Path(cfg.get('checkpoints_dir', './checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"mpc_mappo_step_{step}.pt"
    
    torch.save({
        'step': step,
        'algorithm_state_dict': algorithm.state_dict(),
        'config': OmegaConf.to_container(cfg),
    }, checkpoint_path)
    
    return str(checkpoint_path)


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="train_mpc_mappo"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for MPC-MAPPO training."""
    
    # Print configuration
    print("=" * 80)
    print("MPC-MAPPO Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Start training
    train_mpc_mappo(cfg)


if __name__ == "__main__":
    main()
