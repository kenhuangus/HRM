#!/usr/bin/env python3
"""
HRM Training on Red Teaming Dataset
Adapt HRM for conversational red teaming instruction prediction
"""

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch.optim as optim

def main():
    print("🚨 HRM RED TEAMING TRAINING")
    print("=" * 50)

    # Verify Blackwell GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name()
    print("🎯 Blackwell GPU: {}".format(gpu_name))
    print("🔢 Compute Capability: {}".format(torch.cuda.get_device_capability()))
    print("🧠 VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))

    # Configuration adapted for red teaming language task
    config = {
        'batch_size': 1,
        'seq_len': 256,  # Longer sequences for conversations vs 81 for Sudoku
        'vocab_size': 5000,  # Larger vocab for natural language vs 11 for Sudoku
        'num_puzzle_identifiers': 1,  # Still single task type (red teaming)
        'puzzle_emb_ndim': 256,  # Larger embeddings for richer language tasks
        'H_cycles': 3,  # More reasoning cycles for complex language understanding
        'L_cycles': 4,  # More low-level processing for language patterns
        'H_layers': 4,  # Deeper high-level reasoning for adversarial strategies
        'L_layers': 4,  # Deeper low-level for tactical understanding
        'hidden_size': 512,  # Larger model for language vs 256 for puzzles
        'num_heads': 16,  # More attention heads for language complexity
        'expansion': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 24,  # Allow longer reasoning chains for strategic planning
        'halt_exploration_prob': 0.15  # Higher exploration for diverse red teaming approaches
    }

    print("🏗️ BUILDING HRM MODEL FOR RED TEAMING TASKS")
    model = HierarchicalReasoningModel_ACTV1(config)
    model.cuda()
    print("✅ Adapted HRM instantiated: {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("   📝 Adapted parameters:")
    print("      - seq_len: 256 (vs 81 for Sudoku)")
    print("      - vocab_size: 5000 (vs 11 for Sudoku)")
    print("      - hidden_size: 512 (vs 256 for puzzles)")
    print("      - More layers and cycles for language complexity")

    # Load red teaming dataset (custom loading since HRM expects different format)
    print("\n📊 LOADING RED TEAMING DATASET")

    red_teaming_inputs = torch.load('data/red_teaming/train/all_inputs.pt')
    red_teaming_labels = torch.load('data/red_teaming/train/all_labels.pt')

    print("✅ Red teaming dataset loaded: {} examples".format(red_teaming_inputs.shape[0]))
    print("   Input shape: {}".format(red_teaming_inputs.shape))
    print("   Label shape: {}".format(red_teaming_labels.shape))

    # Training setup adapted for language modeling task
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    training_steps = 200  # Shorter training for initial adaptation testing
    save_interval = 50

    print("\n🔥 STARTING RED TEAMING HRM TRAINING")
    print("   Steps: {}".format(training_steps))
    print("   Save interval: {}".format(save_interval))
    print("   Batch size: {}".format(config['batch_size']))
    print("   📝 Task: Language modeling on red teaming conversations")
    best_loss = float('inf')

    # Manual training loop (adapted from HRM's puzzle-solving loop)
    for step in range(training_steps):
        # Get batch from red teaming dataset
        batch_idx = step % len(red_teaming_inputs)
        input_seq = red_teaming_inputs[batch_idx:batch_idx+1]
        target_seq = red_teaming_labels[batch_idx:batch_idx+1]

        # Convert to HRM's expected batch format
        batch_data = {
            'inputs': input_seq.cuda(),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device='cuda')
        }

        # Forward pass through adapted HRM
        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        # Language modeling loss (next-token prediction)
        logits = outputs['logits']  # [batch, seq, vocab]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1).cuda(),
            ignore_index=0
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Progress tracking
        current_loss = loss.item()
        if step % 10 == 0:
            print("Step {:2d} Loss: {:.4f}".format(step, loss.item()))

        # Periodic checkpointing
        if step % save_interval == 0:
            checkpoint_path = "red_teaming_checkpoint_{}.pt".format(step)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step + 1,
                'loss': current_loss,
                'config': config,
                'gpu_name': torch.cuda.get_device_name(),
                'dataset': 'red_teaming_v1',
                'timestamp': 'November 28, 2025'
            }, checkpoint_path)
            print("   Checkpoint saved: {}".format(checkpoint_path))

            # Save best model
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step + 1,
                    'loss': current_loss,
                    'config': config,
                    'best': True,
                    'dataset': 'red_teaming_v1'
                }, 'red_teaming_hrm_best.pt')
                print("   Best model saved: red_teaming_hrm_best.pt")

    # Final checkpoint
    print("\n💾 SAVING FINAL RED TEAMING CHECKPOINT")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': training_steps,
        'loss': current_loss,
        'config': config,
        'gpu_name': torch.cuda.get_device_name(),
        'training_duration': "{} steps on red teaming dataset".format(training_steps),
        'dataset_stats': "{} examples, 5000 vocab".format(len(red_teaming_inputs)),
        'timestamp': 'November 28, 2025 Blackwell GPU Red Teaming Training'
    }, 'red_teaming_hrm_final.pt')
    print("✅ Red teaming final checkpoint saved: red_teaming_hrm_final.pt")

    print("\n🎯 RED TEAMING HRM TRAINING COMPLETED!")
    print("   GPU: {}".format(torch.cuda.get_device_name()))
    print("   Final loss: {:.4f}".format(current_loss))
    print("   Best loss: {:.4f}".format(best_loss))
    print("   Memory used: {:.0f}MB".format(torch.cuda.memory_allocated()/1024**2))
    print("   Dataset adapted: Red teaming conversations (1000 examples)")
    print("\n🗂️  CHECKPOINT FILES CREATED:")
    print("   • red_teaming_hrm_final.pt (final adapted model)")
    print("   • red_teaming_hrm_best.pt (best performing model)")
    print("   • red_teaming_checkpoint_*.pt files")
    print("   • Next: Evaluate red teaming performance")

if __name__ == "__main__":
    main()
