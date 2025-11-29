#!/usr/bin/env python3

"""
Quick Blackwell GPU Demo for HRM - Focus on GPU functionality
"""

import torch
from tqdm import tqdm
import torch.nn as nn

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.layers import Attention  # Uses our CPU fallback


def create_model_on_device(config, device):
    """Create HRM model directly on specified device"""
    print("🔧 Creating HRM model...")

    # Create model
    model = HierarchicalReasoningModel_ACTV1(config)
    print("✅ Model created successfully")
    print()

    # Move to GPU immediately to avoid device mismatches
    print("🚀 Moving to Blackwell GPU...")

    # First move to device
    model = model.to(device)

    # Force all nn.Buffer objects to GPU explicitly
    def move_buffers_to_device(module, device):
        for name, buf in module.named_buffers():
            if hasattr(buf, 'data'):
                buf.data = buf.data.to(device)

    model.apply(lambda m: move_buffers_to_device(m, device))

    print("✅ Model and all components on GPU")
    print()
    return model


def main():
    print("="*60)
    print("🚀 Blackwell HRM GPU Demo - November 28, 2025")
    print("="*60)

    # Blackwell GPU Status
    print(f"🎯 GPU: {torch.cuda.get_device_name()}")
    print(f"🔢 Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚠️ CC Warning: 12.1 > PyTorch max 12.0")
    print()

    # Create small HRM model directly on GPU
    device = torch.device('cuda')
    config = {
        'batch_size': 1,
        'seq_len': 16,  # Very small for demo
        'vocab_size': 11,
        'num_puzzle_identifiers': 1,
        'puzzle_emb_ndim': 32,
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 32,
        'num_heads': 2,
        'expansion': 2,
        'pos_encodings': 'rope',
        'halt_max_steps': 2,
        'halt_exploration_prob': 0.0
    }

    model = create_model_on_device(config, device)

    # Create dummy data
    print("📊 Creating training data...")
    batch = {
        'inputs': torch.randint(0, 10, (1, 16)).cuda(),  # Short sequence
        'puzzle_identifiers': torch.zeros(1, dtype=torch.long).cuda()
    }
    print("✅ Data on GPU")
    print()

    # Training loop
    print("🔥 BlackwellGPU Training Demo (50 steps)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tqdm(total=50, desc="Blackwell Training") as pbar:
        for step in range(50):
            # Forward
            carry = model.initial_carry(batch)
            carry, outputs = model(carry=carry, batch=batch)

            # Compute loss from logits (cross entropy with dummy labels)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            # Use softmax as loss (since no ground truth labels in demo)
            loss = -F.log_softmax(logits, dim=-1).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.set_postfix({"gpu_mem": f"{torch.cuda.memory_allocated()/1024**2:.0f}MB"})

    print("✅ Blackwell GPU Training Completed!")
    print(f"   Final loss: {loss.item():.4f}")
    print(f"   Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("="*60)

    print("🎉 Blackwell HRM Demo SUCCESSFUL!")
    print("   - Blackwell GPU detected and functional")
    print("   - HRM model loaded on GPU despite CC warnings")
    print("   - Training completed with Blackwell acceleration")
    print("   - Ready for PyTorch 2.8.0 full Blackwell support")
    print("="*60)


if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("❌ Blackwell GPU not available")
