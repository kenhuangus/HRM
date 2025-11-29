#!/usr/bin/env python3
"""
Optimized HRM Red Teaming Training - Phase 1 Quick Wins
Target: Loss < 1.2 with improved hyperparameters
"""

import torch
import torch.nn as nn
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def main():
    print("🚀 PHASE 1: OPTIMIZED HRM RED TEAMING TRAINING")
    print("=" * 55)

    # Verify Blackwell GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name()
    print("🎯 Blackwell GPU: {}".format(gpu_name))
    print("🔢 Compute Capability: {}".format(torch.cuda.get_device_capability()))

    print("\n📊 OPTIMIZATION STRATEGY - PHASE 1 QUICK WINS:")
    print("   • Learning Rate: 1e-3 (increased from 3e-4)")
    print("   • Batch Size: 4 (increased from 1)")
    print("   • Gradient Clipping: 1.0 (added)")
    print("   • Cosine LR Scheduling (added)")
    print("   • Model Size: Reduced from 512→384 hidden")

    # Optimized configuration for red teaming language task
    config = {
        'batch_size': 4,           # Increased for stable gradients
        'seq_len': 256,
        'vocab_size': 5000,
        'num_puzzle_identifiers': 1,
        'puzzle_emb_ndim': 256,
        'H_cycles': 2,
        'L_cycles': 3,
        'H_layers': 3,
        'L_layers': 3,
        'hidden_size': 384,        # Reduced for better convergence
        'num_heads': 12,
        'expansion': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 20,
        'halt_exploration_prob': 0.1
    }

    print("\n🏗️ BUILDING OPTIMIZED HRM MODEL")
    model = HierarchicalReasoningModel_ACTV1(config)
    model.cuda()
    print("✅ Optimized HRM instantiated: {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("   📝 Configuration: Batch={}, Hidden={}, LR=1e-3".format(config['batch_size'], config['hidden_size']))

    # Load red teaming dataset
    print("\n📊 LOADING RED TEAMING DATASET")
    inputs = torch.load('data/red_teaming/train/all_inputs.pt')
    labels = torch.load('data/red_teaming/train/all_labels.pt')
    print("   Dataset loaded: {} examples".format(inputs.shape[0]))

    # Optimized training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # AdamW with weight decay
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)    # Cosine annealing

    training_steps = 1000  # Focused training run
    save_interval = 100
    grad_clip = 1.0  # Gradient clipping

    print("\n🎯 OPTIMIZED TRAINING PLAN")
    print("   Steps: {}".format(training_steps))
    print("   Target Loss: < 1.2")
    print("   Optimizer: AdamW + Cosine LR")
    print("   Gradient Clipping: {}".format(grad_clip))
    print("   Batch Size: {}".format(config['batch_size']))

    best_loss = float('inf')

    # Optimized training loop
    print("\n🚀 STARTING OPTIMIZED TRAINING...")
    for step in range(training_steps):
        # Sample multiple examples for batch
        batch_indices = torch.randint(0, len(inputs), (config['batch_size'],))
        batch_inputs = inputs[batch_indices]
        batch_labels = labels[batch_indices]

        # Convert to HRM format
        batch_data = {
            'inputs': batch_inputs.cuda(),
            'puzzle_identifiers': torch.zeros(config['batch_size'], dtype=torch.long, device='cuda')
        }

        # Forward pass
        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        # Loss computation
        logits = outputs['logits']
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels.reshape(-1).cuda(),
            ignore_index=0
        )

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # LR scheduling

        current_loss = loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        if step % 20 == 0:
            print("Step {:3d}: Loss {:.4f}, LR {:.1e}, Best {:.4f}".format(
                step, current_loss, current_lr, best_loss))

        # Track best loss
        if current_loss < best_loss:
            best_loss = current_loss
            if best_loss < 1.2:
                print("🎉 TARGET ACHIEVED! Loss {:.4f} < 1.2".format(best_loss))

        # Checkpointing
        if step % save_interval == 0 and step > 0:
            checkpoint_path = "red_teaming_optimized_{}.pt".format(step)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step + 1,
                'loss': current_loss,
                'best_loss': best_loss,
                'config': config,
                'phase': 'optimization_phase_1'
            }, checkpoint_path)
            print("   💾 Checkpoint saved: {}".format(checkpoint_path))

    # Final results
    print("\n" + "="*55)
    print("🏆 PHASE 1 OPTIMIZATION RESULTS")
    print("="*55)
    print("Total Steps: {}".format(training_steps))
    print("Final Loss: {:.4f}".format(current_loss))
    print("Best Loss: {:.4f}".format(best_loss))
    print("Target Achieved (loss < 1.2): {}".format("✅ YES" if best_loss < 1.2 else "❌ NO"))

    if best_loss >= 1.2:
        loss_to_go = best_loss - 1.2
        percent_to_target = (1 - best_loss/9.31) * 100  # vs initial random loss
        print("Loss to Target: {:.4f}".format(loss_to_go))
        print("Progress vs Random: {:.1f}% reduction".format(percent_to_go))

    print("\nOptimization Techniques Applied:")
    print("✅ Higher Learning Rate (1e-3)")
    print("✅ Larger Batch Size (4)")
    print("✅ Gradient Clipping (1.0)")
    print("✅ Cosine LR Scheduling")
    print("✅ AdamW Optimizer with Weight Decay")
    print("✅ Reduced Model Complexity (384 hidden)")

    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': training_steps,
        'loss': current_loss,
        'best_loss_achieved': best_loss,
        'target_achieved': best_loss < 1.2,
        'phase': 'optimization_phase_1_complete',
        'config': config,
        'timestamp': 'November 28, 2025 Optimized Training Complete'
    }
    torch.save(final_checkpoint, 'red_teaming_phase1_final.pt')
    print("✅ Final model saved: red_teaming_phase1_final.pt")

if __name__ == "__main__":
    main()
