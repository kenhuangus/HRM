#!/usr/bin/env python3

"""
GPU Training for HRM on Blackwell GPU (with known CC compatibility warnings)
"""

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
# from adam_atan2 import AdamATan2  # Problematic dependency, using AdamW instead

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class


def create_gpu_train_config():
    """GPU training configuration optimized for Blackwell"""
    config = DictConfig({
        # Data
        'data_path': 'data/sudoku-small',

        # Model arch
        'arch': {
            'name': 'hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
            'loss': {
                'name': 'losses@ACTLossHead',
                'loss_type': 'stablemax_cross_entropy'
            }
        },

        # GPU training params
        'global_batch_size': 1,  # Small batch size for Blackwell testing
        'max_steps': 100,        # More steps to see training progress
        'lr': 1e-3,
        'lr_warmup_steps': 5,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,

        # Puzzle embedding
        'puzzle_emb_lr': 1e-2,
        'puzzle_emb_weight_decay': 0.1,

        # Smaller model for initial Blackwell testing
        'H_cycles': 1,
        'L_cycles': 1,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 64,    # Smaller for GPU memory
        'num_heads': 2,
        'expansion': 4,
        'pos_encodings': 'rope',
        'halt_exploration_prob': 0.1,
        'halt_max_steps': 4,

        'loss_type': 'stablemax_cross_entropy'
    })
    return config


def create_dataloader(config, split="train"):
    """GPU-aware dataloader with Blackwell optimizations"""
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path=config.data_path,
        global_batch_size=config.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=10,  # Cycle through data more frequently
        rank=0,
        num_replicas=1
    )
    dataset = PuzzleDataset(dataset_config, split=split)

    # GPU-aware loader
    data_iter = iter(dataset)
    return lambda: next(data_iter, None), dataset.metadata


def create_model_gpu(config, train_metadata):
    """Create model with GPU device placement"""
    model_cfg = dict(
        batch_size=config.global_batch_size,
        seq_len=train_metadata.seq_len,
        vocab_size=train_metadata.vocab_size,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,

        puzzle_emb_ndim=64,  # Match hidden_size

        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        H_layers=config.H_layers,
        L_layers=config.L_layers,

        hidden_size=config.hidden_size,
        expansion=config.expansion,
        num_heads=config.num_heads,
        pos_encodings=config.pos_encodings,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob
    )

    # Load model classes
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    model = loss_head_cls(model, loss_type=config.arch.loss.loss_type)

    # Move model to Blackwell GPU
    print(f"🚀 Moving model to Blackwell GPU...")
    model.cuda()
    print(f"✅ Model on GPU: {torch.cuda.get_device_name()}")

    # Optimizers
    optimizers = [
        # Puzzle embedding optimizer
        torch.optim.AdamW(
            list(model.model.puzzle_emb.buffers())[:1],  # Take first buffer(s)
            lr=config.puzzle_emb_lr,
            weight_decay=config.weight_decay,
        ),
        # Main optimizer with GPU tensors
        torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]

    return model, optimizers


def gpu_train():
    """GPU training on Blackwell with known CC compatibility warnings"""
    config = create_gpu_train_config()

    print("="*50)
    print("🚀 HRM Blackwell GPU Training Session")
    print(f"📅 Date: November 28, 2025")
    print(f"🎯 GPU: {torch.cuda.get_device_name()}")
    print(f"🔢 CC: {torch.cuda.get_device_capability()}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"📈 Steps: {config.max_steps}")
    print("="*50)

    # Data loaders
    train_loader, train_metadata = create_dataloader(config, "train")
    eval_loader, eval_metadata = create_dataloader(config, "test")

    print(f"📊 Dataset: {train_metadata.total_groups} training puzzles")
    print(f"📝 Sequence length: {train_metadata.seq_len}")

    # Model on GPU
    model, optimizers = create_model_gpu(config, train_metadata)
    model.train()

    step = 0
    best_loss = float('inf')

    print("\n🔥 Starting Blackwell GPU training...")

    with tqdm(total=config.max_steps, desc="Training") as pbar:
        while step < config.max_steps:
            batch_data = train_loader()
            if batch_data is None:
                # Reset data loader
                train_loader, _ = create_dataloader(config, "train")
                batch_data = train_loader()

            set_name, batch, global_batch_size = batch_data

            # Move batch to Blackwell GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            # Initialize carry on GPU
            carry = model.initial_carry(batch)

            try:
                # Forward pass with loss
                carry, loss, metrics, preds, finished = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[]
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  Warning: Invalid loss at step {step}: {loss.item()}")
                    continue

                # Backward and optimize
                for optim in optimizers:
                    optim.zero_grad()

                loss.backward()

                # Check gradients
                total_grad_norm = 0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.norm().item()
                        total_grad_norm += param_grad_norm ** 2
                        param_count += 1

                total_grad_norm = total_grad_norm ** 0.5
                if total_grad_norm > 10.0:
                    print(f"\n⚠️  Gradient explosion at step {step}: {total_grad_norm:.3f}")
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                for optim in optimizers:
                    optim.step()

                # Update progress
                step += 1
                current_loss = loss.item()
                best_loss = min(best_loss, current_loss)

                # Update progress bar
                pbar.update(1)
                pbar.set_description(f"Loss: {current_loss:.4f} | Best: {best_loss:.4f}")

                # GPU memory status
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    "gpu_mem": ".1f",
                    "grad_norm": f"{total_grad_norm:.2f}"
                })

                # Periodic evaluation
                if step % 25 == 0:
                    eval_accuracy = evaluate_model(model, eval_loader, step)
                    print(f"\n📊 Step {step}: Eval accuracy = {eval_accuracy:.3f}")

            except Exception as e:
                print(f"\n❌ Training error at step {step}: {e}")
                print("Continuing training...")
                continue

    print("\n" + "="*50)
    print("🎉 Blackwell GPU Training Completed!")
    print(f"   Final Loss: {current_loss:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Total Steps: {step}")
    print(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("="*50)

    # Save final model
    print("💾 Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': step,
        'loss': current_loss,
        'config': dict(config)
    }, 'blackwell_hrm_checkpoint.pt')
    print("✅ Model saved to blackwell_hrm_checkpoint.pt")

    return model


def evaluate_model(model, eval_loader, step):
    """Evaluate model accuracy on test set"""
    model.eval()
    total_correct = 0
    total_tokens = 0

    try:
        while True:
            batch_data = eval_loader()
            if batch_data is None:
                break

            set_name, batch, global_batch_size = batch_data
            batch = {k: v.cuda() for k, v in batch.items()}

            carry = model.initial_carry(batch)

            # Run until finished (with timeout)
            final_preds = None
            for _ in range(10):  # Max steps
                carry, loss, metrics, preds, finished = model(
                    carry=carry,
                    batch=batch,
                    return_keys=['logits']
                )
                if finished:
                    final_preds = preds
                    break

            if final_preds and 'logits' in final_preds:
                pred_tokens = final_preds['logits'].argmax(-1)
                correct = (pred_tokens == batch['labels']).sum()
                total_correct += correct.item()
                total_tokens += batch['labels'].numel()

    except Exception as e:
        print(f"Eval error: {e}")

    model.train()

    if total_tokens > 0:
        return total_correct / total_tokens
    return 0.0


if __name__ == "__main__":
    # Verify Blackwell GPU before starting
    print("🔍 Blackwell GPU Detection...")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Compiled: {torch.version.cuda}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    print(f"   GPU Name: {torch.cuda.get_device_name()}")
    print(f"   GPU CC: {torch.cuda.get_device_capability()}")
    print()

    if not torch.cuda.is_available():
        print("❌ Blackwell GPU not available")
        exit(1)

    print("⚠️  Note: Blackwell CC 12.1 > PyTorch max 12.0 - proceeding despite warnings")
    print()

    trained_model = gpu_train()
