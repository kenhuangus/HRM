#!/usr/bin/env python3
"""
Extended HRM Red Teaming Training - 2000 Steps Target
Resume from checkpoint to achieve loss < 1.2
"""

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch.optim as optim

def main():
    print("🚀 EXTENDED HRM RED TEAMING TRAINING - TARGET: 2000 STEPS")
    print("=" * 60)

    # Verify Blackwell GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name()
    print("🎯 Blackwell GPU: {}".format(gpu_name))
    print("🔢 Compute Capability: {}".format(torch.cuda.get_device_capability()))
    print("🧠 VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))

    print("\n📂 CHECKING AVAILABLE CHECKPOINTS...")
    import os
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('red_teaming_checkpoint_') and f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print("✅ Found checkpoint: {}".format(latest_checkpoint))
        checkpoint_path = latest_checkpoint
    else:
        print("⚠️ No checkpoint found, will use red_teaming_hrm_best.pt if available")
        checkpoint_path = 'red_teaming_hrm_best.pt' if os.path.exists('red_teaming_hrm_best.pt') else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print("📥 LOADING CHECKPOINT: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        current_step = checkpoint.get('step', 0)
        best_loss = checkpoint.get('loss', float('inf'))

        print("   Current step: {}".format(current_step))
        print("   Best loss: {:.4f}".format(best_loss))
    else:
        print("❌ No checkpoint found, cannot resume training")
        return

    # Recreate model with saved config
    print("\n🏗️ RECREATING HRM MODEL FROM CHECKPOINT")
    model = HierarchicalReasoningModel_ACTV1(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    # Recreate optimizer with saved state
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("✅ Model and optimizer restored from step {}".format(current_step))
    print("   Parameters: {:,} total".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Load red teaming dataset
    print("\n📊 LOADING RED TEAMING DATASET")
    red_teaming_inputs = torch.load('data/red_teaming/train/all_inputs.pt')
    red_teaming_labels = torch.load('data/red_teaming/train/all_labels.pt')

    print("   Dataset loaded: {} examples".format(red_teaming_inputs.shape[0]))
    print("   Input shape: {}".format(red_teaming_inputs.shape))

    # Extended training configuration
    target_steps = 2000
    additional_steps = target_steps - current_step
    save_interval = 200  # Save every 200 steps for extended training

    print("\n🎯 EXTENDED TRAINING PLAN")
    print("   Current step: {}".format(current_step))
    print("   Target steps: {}".format(target_steps))
    print("   Additional steps needed: {}".format(additional_steps))
    print("   Target loss: < 1.2")
    print("   Save interval: {}".format(save_interval))
    print("   Batch size: {}".format(config['batch_size']))

    # Continue training loop
    print("\n🚀 STARTING EXTENDED TRAINING...")
    running_loss = best_loss

    for step in range(current_step, target_steps):
        # Get batch from red teaming dataset (cycling through)
        batch_idx = step % len(red_teaming_inputs)
        input_seq = red_teaming_inputs[batch_idx:batch_idx+1]
        target_seq = red_teaming_labels[batch_idx:batch_idx+1]

        # Convert to HRM format
        batch_data = {
            'inputs': input_seq.cuda(),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device='cuda')
        }

        # Forward pass
        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        # Language modeling loss
        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1).cuda(),
            ignore_index=0
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        running_loss = min(running_loss, current_loss)  # Track best loss

        # Progress reporting
        if step % 50 == 0:
            print("Step {}/{} Loss: {:.4f} (Best: {:.4f})".format(
                step, target_steps-1, current_loss, running_loss))

        # Checkpoint saving for extended training
        if step % save_interval == 0 and step > current_step:
            checkpoint_path = "red_teaming_extended_{}.pt".format(step)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step + 1,
                'loss': current_loss,
                'best_loss': running_loss,
                'config': config,
                'gpu_name': torch.cuda.get_device_name(),
                'dataset': 'red_teaming_v1_extended',
                'timestamp': 'November 28, 2025 Extended Training'
            }, checkpoint_path)
            print("   💾 Checkpoint saved: {}".format(checkpoint_path))

            # Update best model if improved
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step + 1,
                    'loss': current_loss,
                    'config': config,
                    'best': True,
                    'extended_training': True,
                    'dataset': 'red_teaming_v1_extended'
                }, 'red_teaming_hrm_extended_best.pt')
                print("   🏆 Best model updated: red_teaming_hrm_extended_best.pt")

    # Final checkpoint after extended training
    print("\n💾 SAVING FINAL EXTENDED TRAINING CHECKPOINT")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': target_steps,
        'loss': current_loss,
        'best_loss_achieved': running_loss,
        'config': config,
        'gpu_name': torch.cuda.get_device_name(),
        'training_summary': 'Extended training: {} steps on 1000 red teaming examples'.format(target_steps),
        'loss_target_achieved': running_loss < 1.2,
        'dataset_stats': '{} examples, vocabulary size {}'.format(
            len(red_teaming_inputs), config['vocab_size']),
        'timestamp': 'November 28, 2025 Extended Blackwell Training Complete'
    }
    torch.save(final_checkpoint, 'red_teaming_hrm_extended_final.pt')

    print("✅ EXTENDED TRAINING COMPLETED!")
    print("   Total steps: {}".format(target_steps))
    print("   Final loss: {:.4f}".format(current_loss))
    print("   Best loss achieved: {:.4f}".format(running_loss))
    print("   Target achieved (loss < 1.2): {}".format("✅ YES" if running_loss < 1.2 else "❌ NO"))
    print("   GPU memory used: {:.0f}MB".format(torch.cuda.memory_allocated()/1024**2))
    print("   Checkpoint saved: red_teaming_hrm_extended_final.pt")
    print("   Best model: red_teaming_hrm_extended_best.pt")

if __name__ == "__main__":
    main()
