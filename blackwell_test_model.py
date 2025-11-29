#!/usr/bin/env python3

"""
Blackwell HRM Model Testing and Evaluation
"""

import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
import os

def test_blackwell_model():
    print("🧪 BLACKWELL HRM MODEL TESTING")
    print("="*60)

    # Check for trained model
    if not os.path.exists('blackwell_hrm_final.pt'):
        print("❌ blackwell_hrm_final.pt not found!")
        print("Run training first: python blackwell_full_training.py")
        return

    # Load checkpoint
    print("📁 Loading Blackwell trained model...")
    checkpoint = torch.load('blackwell_hrm_final.pt')
    config = checkpoint['config']

    # Recreate model
    model = HierarchicalReasoningModel_ACTV1(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    print(f"✅ Model loaded on Blackwell GPU: {torch.cuda.get_device_name()}")
    print(f"   Trained for {checkpoint['step']} steps")
    print(f"   Final loss: {checkpoint['loss']:.4f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # Test inference on sample puzzle
    print("🔍 TESTING INFERENCE")
    print("-" * 40)

    # Create a simple test puzzle
    test_input = torch.zeros(1, 81, dtype=torch.long)  # Empty 9x9 Sudoku

    # Add some initial numbers (make it partially solved)
    test_input[0, [0, 1, 2, 9, 10, 11, 18, 19]] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # First few rows

    batch = {
        'inputs': test_input.cuda(),
        'puzzle_identifiers': torch.zeros(1, dtype=torch.long).cuda()
    }

    with torch.no_grad():
        carry = model.initial_carry(batch)
        carry, outputs = model(carry=carry, batch=batch)

        # Get predictions
        predictions = outputs['logits'].argmax(dim=-1).cpu()

    print("Test Puzzle (Input):")
    display_sudoku(test_input[0])
    print("\nBlackwell HRM Predictions:")
    display_sudoku(predictions[0])
    print()

    # Run validation on test dataset
    if os.path.exists('data/sudoku-small'):
        print("📊 VALIDATION ACCURACY TEST")
        print("-" * 40)
        test_accuracy(model, config)

    print("\n🎯 TEST SUMMARY:")
    print("   ✅ Blackwell model loaded successfully")
    print("   ✅ Inference completed on GPU")
    print("   ✅ Predictions generated")
    print(f"   📍 Loaded from: blackwell_hrm_final.pt")
    print(f"   🎯 GPU: {torch.cuda.get_device_name()}")

def test_accuracy(model, config):
    # Load test dataset
    val_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path='data/sudoku-small',
        global_batch_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )

    try:
        val_dataset = PuzzleDataset(val_config, split='test')
        val_iter = iter(val_dataset)
    except Exception as e:
        print(f"❌ Could not load validation data: {e}")
        return

    total_correct = 0
    total_tokens = 0
    num_batches = min(50, len(val_dataset))  # Test first 50 puzzles

    print(f"Testing on {num_batches} validation puzzles...")

    with torch.no_grad():
        for i, batch_data in enumerate(val_iter):
            if i >= num_batches:
                break

            set_name, batch, batch_size = batch_data
            batch = {k: v.cuda() for k, v in batch.items()}

            # Run inference
            carry = model.initial_carry(batch)
            carry, outputs = model(carry=carry, batch=batch)

            # Calculate accuracy (excluding padding tokens)
            preds = outputs['logits'].argmax(dim=-1)
            labels = batch['labels']

            # Only count non-padding tokens (label != 0)
            mask = labels != 0
            correct = ((preds == labels) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0

    print(f"   Accuracy: {accuracy:.2f}% ({total_correct}/{total_tokens} tokens)")
    print(f"   Blackwell GPU Performance: {'Excellent' if accuracy > 60 else 'Good' if accuracy > 30 else 'Needs More Training'}")

    # Save results
    results = {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_tokens': total_tokens,
        'test_size': num_batches,
        'gpu_name': torch.cuda.get_device_name(),
        'model_checkpoint': 'blackwell_hrm_final.pt'
    }
    torch.save(results, 'blackwell_test_results.pt')
    print(f"   Results saved: blackwell_test_results.pt")

def display_sudoku(grid_tensor):
    """Display Sudoku grid in readable format"""
    grid = grid_tensor.numpy()

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            row_str += str(grid[i*9 + j]) + " "
        print(row_str.strip())

if __name__ == "__main__":
    test_blackwell_model()
