#!/usr/bin/env python3

"""
Test HRM model instantiation and basic functionality
"""

import torch
import sys

def test_hrm():
    print("Testing HRM on CPU...")

    # Import HRM modules
    try:
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        print("✅ HRM module imported")
    except ImportError as e:
        print(f"❌ HRM import failed: {e}")
        return False

    # Create basic config
    config = {
        "batch_size": 1,
        "seq_len": 81,  # Sudoku size
        "puzzle_emb_ndim": 128,
        "num_puzzle_identifiers": 10,
        "vocab_size": 11,  # PAD + 0-9

        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 4,
        "L_layers": 4,
        "hidden_size": 256,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1
    }

    try:
        model = HierarchicalReasoningModel_ACTV1(config)
        print("✅ HRM model created")
    except Exception as e:
        print(f"❌ HRM model creation failed: {e}")
        return False

    # Test forward pass
    try:
        # Create dummy data
        batch_size = 1
        seq_len = 81
        inputs = torch.randint(0, 10, (batch_size, seq_len))
        labels = torch.randint(0, 10, (batch_size, seq_len))
        puzzle_identifiers = torch.zeros(batch_size, dtype=torch.long)
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers
        }

        # Initial carry
        carry = model.initial_carry(batch)

        # Forward pass (first step)
        carry, outputs = model(carry, batch)

        print("✅ HRM forward pass (first step)")
        print(f"   Output logits shape: {outputs['logits'].shape}")
        print(f"   Halt logits: {outputs['q_halt_logits'].item():.4f}")
        print(f"   Continue logits: {outputs['q_continue_logits'].item():.4f}")

    except Exception as e:
        print(f"❌ HRM forward pass failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_hrm()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
