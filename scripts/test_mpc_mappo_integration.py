#!/usr/bin/env python3

"""
Test script for MPC-MAPPO integration in OmniDrones.
This script performs basic functionality tests to ensure the integration works correctly.
"""

import os
import sys
from pathlib import Path
import traceback

# Add OmniDrones to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

def test_mpc_controller():
    """Test the standalone MPC controller."""
    print("Testing MPC Controller...")
    
    try:
        from omni_drones.learning.mpc_mappo import QuadrotorMPC
        
        # Create MPC controller
        mpc = QuadrotorMPC(
            horizon=5,
            dt=0.02,
            device=torch.device("cpu")
        )
        
        # Test state and target
        batch_size = 2
        current_state = torch.zeros(batch_size, 13)
        current_state[:, :3] = torch.tensor([[0, 0, 1], [1, 1, 2]])  # positions
        current_state[:, 3] = 1.0  # quaternion w component
        
        target_state = torch.zeros(batch_size, 13) 
        target_state[:, :3] = torch.tensor([[2, 0, 1], [3, 1, 2]])  # target positions
        target_state[:, 3] = 1.0  # quaternion w component
        
        # Solve MPC
        optimal_control, trajectory = mpc.solve(current_state, target_state, max_iters=5)
        
        print(f"✓ MPC Controller test passed")
        print(f"  - Input state shape: {current_state.shape}")
        print(f"  - Output control shape: {optimal_control.shape}")
        print(f"  - Trajectory shape: {trajectory.shape}")
        print(f"  - Control values: {optimal_control[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPC Controller test failed: {e}")
        traceback.print_exc()
        return False


def test_mpc_mappo_import():
    """Test importing the MPC-MAPPO algorithm."""
    print("\\nTesting MPC-MAPPO Import...")
    
    try:
        from omni_drones.learning.mpc_mappo import MPCMAPPO, MPCActorHead
        from omni_drones.learning import ALGOS
        
        # Check if MPC-MAPPO is registered
        assert "mpc_mappo" in ALGOS, "MPC-MAPPO not found in ALGOS registry"
        assert ALGOS["mpc_mappo"] == MPCMAPPO, "MPC-MAPPO class mismatch in registry"
        
        print("✓ MPC-MAPPO import test passed")
        print(f"  - MPCMAPPO class imported successfully")
        print(f"  - MPCActorHead class imported successfully")  
        print(f"  - Algorithm registered in ALGOS: {ALGOS['mpc_mappo']}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPC-MAPPO import test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test loading MPC-MAPPO configuration files."""
    print("\\nTesting Configuration Loading...")
    
    try:
        # Test algorithm config
        algo_config_path = Path(__file__).parent.parent / "cfg" / "algo" / "mpc_mappo.yaml"
        if algo_config_path.exists():
            algo_cfg = OmegaConf.load(algo_config_path)
            assert "name" in algo_cfg, "Algorithm config missing 'name' field"
            assert algo_cfg.name == "mpc_mappo", f"Expected name 'mpc_mappo', got '{algo_cfg.name}'"
            assert "mpc_config" in algo_cfg, "Algorithm config missing 'mpc_config' section"
            print("✓ Algorithm configuration loaded successfully")
        else:
            print("⚠ Algorithm configuration file not found, skipping")
            
        # Test task config
        task_config_path = Path(__file__).parent.parent / "cfg" / "task" / "MPCFormationGateTraversal.yaml"
        if task_config_path.exists():
            task_cfg = OmegaConf.load(task_config_path)
            assert "name" in task_cfg, "Task config missing 'name' field"
            assert task_cfg.name == "MPCFormationGateTraversal", "Task name mismatch"
            print("✓ Task configuration loaded successfully")
        else:
            print("⚠ Task configuration file not found, skipping")
            
        # Test training config
        train_config_path = Path(__file__).parent.parent / "cfg" / "train_mpc_mappo.yaml"
        if train_config_path.exists():
            train_cfg = OmegaConf.load(train_config_path)
            print("✓ Training configuration loaded successfully")
        else:
            print("⚠ Training configuration file not found, skipping")
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        traceback.print_exc()
        return False


def test_algorithm_instantiation():
    """Test creating an instance of MPC-MAPPO algorithm."""
    print("\\nTesting Algorithm Instantiation...")
    
    try:
        from omni_drones.learning.mpc_mappo import MPCMAPPO
        from omni_drones.utils.torchrl.env import AgentSpec
        from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
        
        # Create mock agent spec
        action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(4,),  # thrust + 3 torques
            dtype=torch.float32
        )
        
        observation_spec = UnboundedContinuousTensorSpec(
            shape=(20,),  # Example observation dimension
            dtype=torch.float32
        )
        
        reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32
        )
        
        agent_spec = AgentSpec(
            name="quadrotor",
            action_spec=action_spec,
            observation_spec=observation_spec,
            reward_spec=reward_spec,
        )
        
        # Create algorithm configuration
        cfg = OmegaConf.create({
            "name": "mpc_mappo",
            "train_every": 64,
            "num_minibatches": 4,
            "ppo_epochs": 4,
            "clip_param": 0.1,
            "entropy_coef": 0.001,
            "gae_lambda": 0.95,
            "gamma": 0.995,
            "max_grad_norm": 10.0,
            "normalize_advantages": True,
            "share_actor": False,
            "critic_input": "obs",
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": True,
            "actor": {
                "lr": 0.0005,
                "hidden_units": [64, 64],
                "layer_norm": True,
                "weight_decay": 0.0,
                "gain": 0.01,
                "use_orthogonal": True
            },
            "critic": {
                "num_critics": 1,
                "lr": 0.0005,
                "hidden_units": [64, 64],
                "layer_norm": True,
                "weight_decay": 0.0,
                "gain": 0.01,
                "use_huber_loss": True,
                "huber_delta": 10,
                "use_orthogonal": True
            },
            "mpc_config": {
                "horizon": 5,
                "dt": 0.02,
                "mass": 1.0,
                "Q_pos": 1.0,
                "Q_vel": 0.1,
                "Q_quat": 0.5,
                "Q_omega": 0.1,
                "R_thrust": 0.01,
                "R_torque": 0.01,
                "max_thrust": 20.0,
                "max_torque": 1.0,
            }
        })
        
        # Create MPC-MAPPO instance
        algorithm = MPCMAPPO(
            cfg=cfg,
            agent_spec=agent_spec,
            device="cpu",
            mpc_config=cfg.mpc_config
        )
        
        print("✓ Algorithm instantiation test passed")
        print(f"  - Algorithm created: {type(algorithm).__name__}")
        print(f"  - Device: {algorithm.device}")
        print(f"  - MPC config: {algorithm.mpc_config}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm instantiation test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MPC-MAPPO Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_mpc_controller,
        test_mpc_mappo_import, 
        test_configuration_loading,
        test_algorithm_instantiation,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MPC-MAPPO integration is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
