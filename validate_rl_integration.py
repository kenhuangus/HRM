#!/usr/bin/env python3
"""
Quick validation of RL Integration in HRM Pipeline
Tests the PPO trainer integration without full training
"""

import torch
import os
from hrl_red_teaming_rl import PPOTrainer, create_red_teaming_environment, tokenize_text

def validate_rl_integration():
    """Quick test of RL integration functionality"""
    print("🔍 VALIDATING RL INTEGRATION")
    print("=" * 40)

    # Check if base model exists
    base_model_path = 'cybersecurity_hrm_model.pt'
    if not os.path.exists(base_model_path):
        print("❌ No base model found. Creating synthetic model for testing...")

        # Create synthetic model config for testing
        from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1
        config = {
            'seq_len': 512,
            'hidden_size': 256,  # Smaller for testing
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'vocab_size': 5000
        }

        model = LanguageAdaptedHRM_ACTV1(config)
        synthetic_checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': 5000,
            'phase': 'synthetic_for_testing'
        }
        torch.save(synthetic_checkpoint, 'synthetic_test_model.pt')
        base_model_path = 'synthetic_test_model.pt'

    print("📥 Loading model checkpoint...")
    checkpoint = torch.load(base_model_path)
    print(f"   Vocab size: {checkpoint['vocab_size']}")
    print(f"   Model phase: {checkpoint.get('phase', 'unknown')}")

    # Create environment
    print("\n🎮 Creating red teaming environment...")
    tasks = create_red_teaming_environment()
    print(f"   Generated {len(tasks)} red teaming scenarios")

    # Initialize PPO trainer
    print("\n🤖 Initializing PPO Trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = PPOTrainer(None, checkpoint['vocab_size'], checkpoint['config'])

    print(f"   Device: {device}")
    print(f"   Clipping epsilon: {trainer.clip_epsilon}")
    print(f"   PPO epochs: {trainer.ppo_epochs}")

    # Test environment interaction
    print("\n🧪 Testing environment interaction...")
    task = tasks[0]
    print(f"   Scenario: {task['description']}")

    # Generate a short trajectory
    rollout = trainer.rollout_trajectory(task, max_steps=10)  # Very short for testing
    trajectory = rollout['trajectory']

    print(f"   Generated trajectory with {len(trajectory['states'])} steps")
    print(f"   Average reward: {sum(trajectory['rewards'])/len(trajectory['rewards']):.4f}"
    # Test RL integration method
    print("\n🔗 Testing pipeline integration method...")
    try:
        enhanced_checkpoint, enhanced_path = trainer.integrate_with_pipeline(
            checkpoint['model_state_dict'],
            [task],  # Use single task for quick test
            rl_epochs=2  # Very few epochs for testing
        )
        print(f"   ✅ Integration successful!")
        print(f"   Enhanced model saved: {enhanced_path}")

        # Load and inspect the enhanced model
        enhanced_data = torch.load(enhanced_path)
        print(f"   RL epochs: {enhanced_data.get('rl_config', {}).get('epochs', 'N/A')}")
        print(f"   Training time: {enhanced_data.get('training_time', 'N/A')}")
        print(f"   Final reward: {enhanced_data['rl_stats']['rewards'][-1]:.4f}")

    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        return False

    print("\n✅ RL INTEGRATION VALIDATION COMPLETE!")
    print("🎯 HRM Pipeline RL Integration is working correctly")
    return True

if __name__ == "__main__":
    success = validate_rl_integration()
    if not success:
        exit(1)

    print("\n🎉 Ready to run full pipeline with: python3 hrm_full_pipeline_training.py")
