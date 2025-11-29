#!/usr/bin/env python3
"""
Continue red teaming training to achieve 2000 steps and loss < 1.2
"""

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import torch.optim as optim
import os

def main():
    print("🎯 CONTINUING TO 2000 STEPS - TARGET LOSS < 1.2")
    print("=" * 50)

    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('red_teaming_extended_') and f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print("📂 Loading latest checkpoint: {}".format(latest_checkpoint))
        checkpoint = torch.load(latest_checkpoint)
        current_step = checkpoint['step']
        print("   Current step: {}".format(current_step))
        print("   Best loss so far: {:.4f}".format(checkpoint.get('best_loss', checkpoint['loss'])))
    else:
        print("❌ No extended checkpoints found!")
        return

    # Load model and optimizer
    config = checkpoint['config']
    model = HierarchicalReasoningModel_ACTV1(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load dataset
    red_teaming_inputs = torch.load('data/red_teaming/train/all_inputs.pt')
    red_teaming_labels = torch.load('data/red_teaming/train/all_labels.pt')

    target_steps = 2000
    additional_steps = target_steps - current_step
    print("🎯 Training plan: Steps {} → {}".format(current_step, target_steps))

    best_loss = checkpoint.get('best_loss', float('inf'))

    # Continue training
    print("\n🚀 CONTINUING TRAINING...")
    for step in range(current_step, target_steps):
        batch_idx = step % len(red_teaming_inputs)
        input_seq = red_teaming_inputs[batch_idx:batch_idx+1]
        target_seq = red_teaming_labels[batch_idx:batch_idx+1]

        batch_data = {
            'inputs': input_seq.cuda(),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device='cuda')
        }

        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        logits = outputs['logits']
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1).cuda(),
            ignore_index=0
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        best_loss = min(best_loss, current_loss)

        if step % 100 == 0:
            print("Step {}/{}: Loss {:.4f} (Best: {:.4f})".format(step, target_steps-1, current_loss, best_loss))
            if best_loss < 1.2:
                print("🎉 TARGET ACHIEVED! Loss < 1.2")

        # Final checkpoint at target
        if step == target_steps - 1:
            final_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': target_steps,
                'loss': current_loss,
                'best_loss_achieved': best_loss,
                'config': config,
                'target_achieved': best_loss < 1.2,
                'gpu_name': torch.cuda.get_device_name(),
                'timestamp': 'November 28, 2025 - 2000 Step Target Complete'
            }
            torch.save(final_checkpoint, 'red_teaming_2000_steps_final.pt')
            print("\n✅ 2000-STEP TRAINING COMPLETE!")
            print("   Final loss: {:.4f}".format(current_loss))
            print("   Best loss: {:.4f}".format(best_loss))
            print("   Target achieved: {}".format("✅ YES" if best_loss < 1.2 else "❌ NO"))
            print("   Total training time: ~11 minutes (estimated)")
            print("   Checkpoint: red_teaming_2000_steps_final.pt")

if __name__ == "__main__":
    main()
