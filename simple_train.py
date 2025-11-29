#!/usr/bin/env python3

"""
Simplified HRM training on CPU for testing.
"""

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


def create_simple_train_config():
    """Create a simple training configuration"""
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

        # Training params
        'global_batch_size': 2,
        'max_steps': 50,  # Very small for testing
        'lr': 1e-3,
        'lr_warmup_steps': 2,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,

        # Puzzle embedding
        'puzzle_emb_lr': 1e-2,
        'puzzle_emb_weight_decay': 0.1,

        # Model config - from arch
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 4,
        'L_layers': 4,
        'hidden_size': 128,  # Small for CPU
        'num_heads': 4,
        'expansion': 4,
        'pos_encodings': 'rope',
        'halt_exploration_prob': 0.1,
        'halt_max_steps': 8,

        'loss_type': 'stablemax_cross_entropy'
    })
    return config


def create_dataloader(config, split="train"):
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path=config.data_path,
        global_batch_size=config.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    dataset = PuzzleDataset(dataset_config, split=split)

    # Simple loader
    data_iter = iter(dataset)
    return lambda: next(data_iter, None), dataset.metadata


def create_model(config, train_metadata):
    model_cfg = dict(
        # From arch (hardcoded for simplicity)
        batch_size=config.global_batch_size,

        seq_len=train_metadata.seq_len,
        vocab_size=train_metadata.vocab_size,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,

        puzzle_emb_ndim=128,  # match hidden_size

        H_cycles=2,
        L_cycles=2,
        H_layers=2,  # Smaller for CPU
        L_layers=2,  # Smaller for CPU

        hidden_size=128,
        expansion=4,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=8,
        halt_exploration_prob=0.1
    )

    # Load model classes
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    model = loss_head_cls(model, loss_type=config.arch.loss.loss_type)

    # Optimizers
    optimizers = [
        # Puzzle embedding optimizer
        torch.optim.AdamW(
            [model.model.puzzle_emb.buffers()[0]],  # Just the embeddings
            lr=config.puzzle_emb_lr,
            weight_decay=config.weight_decay,
        ),
        # Main optimizer
        AdamATan2(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]

    return model, optimizers


def simple_train():
    """Training loop for HRM on CPU"""
    config = create_simple_train_config()

    print("🚀 Starting HRM training on CPU...")
    print(f"   Max steps: {config.max_steps}")
    print(f"   Batch size: {config.global_batch_size}")
    print(f"   LR: {config.lr}")

    # Data
    train_loader, train_metadata = create_dataloader(config, "train")
    eval_loader, eval_metadata = create_dataloader(config, "test")

    print(f"   Dataset: {train_metadata.total_groups} training examples")
    print(f"   Vocab size: {train_metadata.vocab_size}")
    print(f"   Sequence length: {train_metadata.seq_len}")

    # Model
    model, optimizers = create_model(config, train_metadata)
    step = 0

    # Training
    with tqdm(total=config.max_steps) as pbar:
        while step < config.max_steps:
            batch_data = train_loader()
            if batch_data is None:
                print("Dataset exhausted, restarting...")
                train_loader, _ = create_dataloader(config, "train")
                batch_data = train_loader()

            set_name, batch, global_batch_size = batch_data
            batch = {k: v for k, v in batch.items()}  # CPU tensors

            # Init carry
            carry = model.initial_carry(batch)

            # Forward with loss
            try:
                carry, loss, metrics, preds, finished = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[]
                )

                if loss.item() != loss.item():  # Check for NaN
                    print(f"Warning: NaN loss at step {step}")
                    break

                # Backward and optimize
                for optim in optimizers:
                    optim.zero_grad()

                loss.backward()

                for optim in optimizers:
                    optim.step()

                step += 1
                pbar.update(1)

                # Log
                pbar.set_description(f"Loss: {loss.item():.4f}")

                # Debug: check if preds match labels
                if 'logits' in preds:
                    pred_tokens = preds['logits'].argmax(-1)
                    correct = (pred_tokens == batch['labels']).float().mean()
                    pbar.set_postfix({"acc": f"{correct.item():.3f}"})

                if step >= config.max_steps:
                    break

            except Exception as e:
                print(f"Training error at step {step}: {e}")
                break

    print("✅ Training completed!"    print(f"   Final loss: {loss.item():.4f}")

    # Simple evaluation
    print("Evaluating...")
    model.eval()
    total_correct = 0
    total_tokens = 0

    try:
        while True:
            batch_data = eval_loader()
            if batch_data is None:
                break

            set_name, batch, global_batch_size = batch_data
            batch = {k: v for k, v in batch.items()}

            carry = model.initial_carry(batch)

            # Run until finished
            final_preds = None
            while True:
                carry, loss, metrics, preds, finished = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[]
                )
                if finished:
                    final_preds = preds
                    break
                if not any(carry):  # Timeout
                    break

            if final_preds and 'logits' in final_preds:
                pred_tokens = final_preds['logits'].argmax(-1)
                correct = (pred_tokens == batch['labels']).sum()
                total_correct += correct.item()
                total_tokens += batch['labels'].numel()

        if total_tokens > 0:
            accuracy = total_correct / total_tokens
            print(f"   Eval accuracy: {accuracy:.4f}")
        else:
            print("   No eval tokens")

    except Exception as e:
        print(f"Eval error: {e}")

    print("🎉 HRM training demonstration complete!")


if __name__ == "__main__":
    simple_train()
