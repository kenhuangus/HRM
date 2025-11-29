#!/usr/bin/env python3

"""
Full Blackwell HRM Training with Model Checkpointing
"""

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
import torch.optim as optim
import os

def main():
    print("🚀 STARTING FULL BLACKWELL HRM TRAINING")
    print("="*60)

    # Verify Blackwell GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name()
    print(f"🎯 Blackwell GPU: {gpu_name}")
    print(f"🔢 Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚡ Blackwell optimization: {'✅ YES' if torch.cuda.get_device_capability() == (12, 1) else '⚠️ MAYBE'}")
    print()

    # Model config for Blackwell training
    config = {
        'batch_size': 1,
        'seq_len': 81,  # 9x9 Sudoku grid
        'vocab_size': 11,  # 0-9 + special tokens
        'num_puzzle_identifiers': 1,
        'puzzle_emb_ndim': 64,
        'H_cycles': 1,  # High-level reasoning cycles
        'L_cycles': 1,  # Low-level reasoning cycles
        'H_layers': 2,  # High-level transformer layers
        'L_layers': 2,  # Low-level transformer layers
        'hidden_size': 256,  # Increased for Blackwell performance
        'num_heads': 8,  # Optimized for Blackwell
        'expansion': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 8,
        'halt_exploration_prob': 0.0
    }

    print("🏗️  BUILDING HRM MODEL")
    model = HierarchicalReasoningModel_ACTV1(config)
    model = model.cuda()  # Move to Blackwell GPU
    print(f"✅ Model created and loaded on Blackwell GPU")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # Dataset setup
    print("📊 LOADING TRAINING DATA")
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path='data/sudoku-small',
        global_batch_size=1,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )

    if not os.path.exists('data/sudoku-small'):
        print("❌ No dataset found! Create dataset first:")
        print("   python create_small_dataset.py")
        return

    dataset = PuzzleDataset(dataset_config, split='train')
    data_iter = iter(dataset)
    print("✅ Sudan training data loaded")
    print()

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    training_steps = 1000  # Full training run
    save_interval = 100

    print("🔥 STARTING BLACKWELL TRAINING")
    print(f"   Steps: {training_steps}")
    print(f"   Save interval: {save_interval}")
    print(f"   Batch size: {config['batch_size']}")
    print()

    best_loss = float('inf')
    for step in range(training_steps):
        # Get batch
        try:
            batch_data = next(data_iter)
        except StopIteration:
            # Reset dataset for next epoch
            data_iter = iter(dataset)
            batch_data = next(data_iter)

        set_name, batch, batch_size = batch_data

        # Move batch to Blackwell GPU
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

        # Forward pass
        carry = model.initial_carry(batch)
        carry, outputs = model(carry=carry, batch=batch)

        # Loss computation
        logits = outputs['logits']  # [batch, seq, vocab]
        targets = batch['labels']   # [batch, seq]

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1).long(),
            ignore_index=0  # Ignore padding tokens
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Progress tracking
        current_loss = loss.item()
        if step % 10 == 0:
            print("2d"
                  "0.0f")

        # Periodic checkpointing
        if step % save_interval == 0:
            checkpoint_path = '1d'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step + 1,
                'loss': current_loss,
                'config': config,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': 'November 28, 2025'
            }, checkpoint_path)
            print("01d")

            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step + 1,
                    'loss': current_loss,
                    'config': config,
                    'best': True
                }, 'blackwell_hrm_best.pt')
                print("01d")

    # Final checkpoint
    print("\n💾 SAVING FINAL CHECKPOINT")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': training_steps,
        'loss': current_loss,
        'config': config,
        'gpu_name': torch.cuda.get_device_name(),
        'training_duration': f"{training_steps} steps",
        'timestamp': 'November 28, 2025 Blackwell GPU Training'
    }
    torch.save(final_checkpoint, 'blackwell_hrm_final.pt')
    print("✅ Final checkpoint saved: blackwell_hrm_final.pt")

    print("\n🎉 BLACKWELL HRM TRAINING COMPLETED!")
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Final loss: {current_loss:.4f}")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Memory used: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
    print("\n🗂️  CHECKPOINT FILES CREATED:")
    print("   • blackwell_hrm_final.pt (complete model)")
    print("   • blackwell_hrm_best.pt (best performing model)")
    print("   • Checkpoint files at every 100 steps")
    print("\n📋 Next: Run Blackwell HRM evaluation with:")
    print("   python blackwell_test_model.py")

if __name__ == "__main__":
    main()
