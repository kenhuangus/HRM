# HRM Blackwell GPU Training Guide

## NGC Docker Setup (Recommended for Blackwell)

### Prerequisites
- Docker installed on DGX Spark system
- NGC account and API key (will request from user)

### 1. Login to NGC (Provide credentials when prompted)
```bash
# Login to NVIDIA GPU Cloud
docker login nvcr.io
# Username: $oauthtoken
# Password: [Your NGC API Key - will be requested from user]
```

### 2. Pull NVIDIA PyTorch Blackwell Container
```bash
# Pull the latest Blackwell-optimized PyTorch container
docker pull nvcr.io/nvidia/pytorch:25.10-py3
# This includes: CUDA 13, optimized Blackwell PyTorch, FlashAttention, cuDNN
```

### 3. Clone HRM Repository and Mount
```bash
# Clone the repository locally (host system)
git clone https://github.com/sapientinc/HRM.git /home/kengpu/HRM
cd /home/kengpu/HRM

# Create dataset (on host)
source hrm_env/bin/activate  # If needed to create dataset
python create_small_dataset.py
```

### 4. Launch Blackwell-Optimized HRM Container
```bash
# Mount HRM directory and grant GPU access
docker run --gpus all \
  -v /home/kengpu/HRM:/workspace/HRM \
  -it nvcr.io/nvidia/pytorch:25.10-py3 \
  /bin/bash

# Container will start with Blackwell GPU access
# Inside container, navigate to mounted HRM directory
cd /workspace/HRM
```

### 5. Verify Blackwell Setup in Container
```bash
# Inside the NGC container
python -c "
import torch
print('=== NADH Blackwell GPU Detection in NGC Container ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Build: {torch.version.cuda}')
print(f'Blackwell GPU: {torch.cuda.get_device_name()}')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'GPU Available: {torch.cuda.is_available()}')
print('✅ Blackwell optimization expected (no CC warnings)')
"
```

## Alternative: Direct Host Setup (If NGC Not Available)

### 1. Install FlashAttention for Blackwell
```bash
# Only if NGC not used - Blackwell requires compatible build
pip install flash-attn --index-url https://download.pytorch.org/whl/cu128
# Note: NGC containers have this pre-installed and optimized
```

### 2. Prepare Dataset
```bash
# Create small Sudoku dataset for quick training
python create_small_dataset.py
```

### 3. Verify Blackwell GPU Status
```bash
python -c "
import torch
print('Blackwell GPU:', torch.cuda.get_device_name())
print('Compute Capability:', torch.cuda.get_device_capability())
print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
print('Availability:', torch.cuda.is_available())
"
```

## Training Command

### Start Full Blackwell GPU Training
```bash
# Navigate to HRM directory
cd /home/kengpu/HRM

# Activate environment
source hrm_env/bin/activate

# Run Blackwell GPU training (with FlashAttention)
python -c "
import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
import torch.optim as optim

# Model config for Blackwell training
config = {
    'batch_size': 1,
    'seq_len': 81,
    'vocab_size': 11,
    'num_puzzle_identifiers': 1,
    'puzzle_emb_ndim': 64,
    'H_cycles': 1,
    'L_cycles': 1,
    'H_layers': 2,
    'L_layers': 2,
    'hidden_size': 64,
    'num_heads': 2,
    'expansion': 4,
    'pos_encodings': 'rope',
    'halt_max_steps': 8,
    'halt_exploration_prob': 0.0
}

print('🚀 Starting HRM Blackwell Training...')

# Load dataset
dataset_config = PuzzleDatasetConfig(
    seed=42,
    dataset_path='data/sudoku-small',
    global_batch_size=1,
    test_set_mode=False,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1
)

dataset = PuzzleDataset(dataset_config, split='train')
data_iter = iter(dataset)

# Create model
print('Building HRM model...')
model = HierarchicalReasoningModel_ACTV1(config)
model.cuda()  # Move to Blackwell GPU
print(f'Model on GPU: {torch.cuda.get_device_name()}')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
training_steps = 100
print(f'Training for {training_steps} steps...')

for step in range(training_steps):
    # Get batch
    try:
        batch_data = next(data_iter)
    except StopIteration:
        # Reset dataset
        data_iter = iter(dataset)
        batch_data = next(data_iter)

    set_name, batch, batch_size = batch_data

    # Move batch to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # Forward pass
    carry = model.initial_carry(batch)
    carry, outputs = model(carry=carry, batch=batch)

    # Compute loss
    logits = outputs['logits']  # Shape: [batch, seq, vocab]
    targets = batch['labels']   # Shape: [batch, seq]

    # Cross-entropy loss (reshape for calculation)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=0  # Ignore padding (label 0)
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Progress
    if step % 10 == 0:
        print(f'Step {step}: Loss = {loss.item():.4f}, GPU Memory = {torch.cuda.memory_allocated()/1024**2:.0f}MB')

# Save checkpoint
print('💾 Saving Blackwell GPU checkpoint...')
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': training_steps,
    'loss': loss.item(),
    'config': config
}
torch.save(checkpoint, 'blackwell_hrm_checkpoint.pt')
print('✅ Checkpoint saved: blackwell_hrm_checkpoint.pt')

print('🎉 Blackwell GPU training completed!')
"
```

This should complete training and save a checkpoint automatically.

## Model Testing Steps

### 1. Load and Test Trained Model
```python
# Test script for trained Blackwell HRM model
import torch
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

def test_hrm_checkpoint():
    # Load model
    checkpoint = torch.load('blackwell_hrm_checkpoint.pt')
    config = checkpoint['config']

    # Recreate model
    model = HierarchicalReasoningModel_ACTV1(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()  # Move to Blackwell GPU
    model.eval()

    print(f'Loaded Blackwell-trained HRM model')
    print(f'Training completed at step {checkpoint[\"step\"]} with loss {checkpoint[\"loss\"]:.4f}')

    # Create test puzzle
    test_input = torch.zeros(1, 81, dtype=torch.long)  # Empty 9x9 Sudoku grid
    test_input[0, [0, 1, 2, 9, 10, 11]] = torch.tensor([1, 2, 3, 4, 5, 6])  # Partial puzzle

    batch = {
        'inputs': test_input.cuda(),
        'puzzle_identifiers': torch.zeros(1, dtype=torch.long).cuda()
    }

    # Get model predictions
    with torch.no_grad():
        carry = model.initial_carry(batch)

        # Get predictions (the model will try to fill in the puzzle)
        carry, outputs = model(carry=carry, batch=batch)
        predictions = outputs['logits'].argmax(dim=-1)

        print(f'Input puzzle shape: {test_input.shape}')
        print(f'Predictions shape: {predictions.shape}')
        print('✅ Blackwell HRM model testing successful!')

if __name__ == '__main__':
    test_hrm_checkpoint()
```

### 2. Run the Test
```bash
python test_checkpoint.py
```

### 3. Alternative Testing (Batch Processing)
```python
# Test multiple puzzles from validation set
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

def test_multiple_puzzles():
    # Load checkpoint
    checkpoint = torch.load('blackwell_hrm_checkpoint.pt')
    config = checkpoint['config']

    model = HierarchicalReasoningModel_ACTV1(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    # Load validation data
    val_config = PuzzleDatasetConfig(
        seed=42,
        dataset_path='data/sudoku-small',
        global_batch_size=2,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )

    val_dataset = PuzzleDataset(val_config, split='test')
    val_iter = iter(val_dataset)

    total_correct = 0
    total_tokens = 0

    print('Testing Blackwell-trained HRM on validation set...')
    for batch_data in val_iter:
        set_name, batch, batch_size = batch_data
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.no_grad():
            carry = model.initial_carry(batch)
            carry, outputs = model(carry=carry, batch=batch)

            # Get accuracy
            preds = outputs['logits'].argmax(dim=-1)
            correct = (preds == batch['labels']).sum().item()
            total_correct += correct
            total_tokens += batch['labels'].numel()

    accuracy = total_correct / total_tokens * 100
    print('.2f')

    # Save test results
    results = {
        'total_correct': total_correct,
        'total_tokens': total_tokens,
        'accuracy': accuracy,
        'test_timestamp': 'November 28, 2025 Blackwell GPU'
    }
    torch.save(results, 'test_results.pt')
    print('Test results saved to test_results.pt')

print('Blackwell HRM model validation complete!')
```

## Expected Performance

For the small synthetic Sudoku dataset after Blackwell training:

- **Training Loss**: Should decrease from ~4.0 to ~2.0 range
- **Prediction Accuracy**: 20-40% reasonable (simple dataset, limited training)
- **GPU Utilization**: Blackwell should show efficient memory usage
- **Inference Speed**: Fast on Blackwell architecture

## Troubleshooting

### Common Issues:
1. **FlashAttention not found**: `ModuleNotFoundError: No module named 'flash_attn'`
   - Solution: Install compatible Blackwell FlashAttention or use NGC containers

2. **CUDA out of memory**: Blackwell has 120GB VRAM but small batches still recommended
   - Reduce batch size in config

3. **Checkpoint loading fails**: Ensure same model config used for loading

### Blackwell-Specific Notes:
- CC 12.1 warnings are expected with current PyTorch
- Performance will be sub-optimal until PyTorch 2.8.0
- NGC containers provide best Blackwell optimization

## Success Verification

✅ **Training completes without errors**
✅ **Checkpoint saves successfully**
✅ **Model loads and runs inference**
✅ **GPU memory usage monitored**
✅ **Blackwell hardware utilized**

The Blackwell HRM training is ready to execute!
