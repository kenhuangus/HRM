#!/usr/bin/env python3
"""
Minimal HRM Cybersecurity Training Script
"""

print("🚀 Starting HRM Cybersecurity Training")

import torch
import torch.nn as nn
import os
from datetime import datetime

print("📚 Generating cybersecurity vocabulary...")

vocab = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<system>': 3, '<user>': 4, '<assistant>': 5}
terms = ['tcp', 'udp', 'http', 'https', 'nmap', 'firewall', 'encryption', 'malware']
max_id = 6

for term in terms:
    vocab[term] = max_id
    max_id += 1

while len(vocab) < 5000:
    vocab[f'term_{len(vocab)}'] = len(vocab)

os.makedirs('data/red_teaming/enhanced', exist_ok=True)
torch.save(vocab, 'data/red_teaming/enhanced/vocab_cybersecurity.pt')
print(f"✅ Vocabulary created with {len(vocab)} tokens")

print("📝 Creating training data...")

conversations = []
for i in range(100):
    conv = [
        {'role': 'system', 'content': 'You are a cybersecurity expert.'},
        {'role': 'user', 'content': f'How does {list(vocab.keys())[i%10]} work?'},
        {'role': 'assistant', 'content': f'{list(vocab.keys())[i%10]} is a security tool.'}
    ]
    conversations.append({'conversation': conv})

all_inputs = []
all_labels = []

for conv in conversations:
    tokens = []
    for turn in conv['conversation']:
        role_id = vocab[f'<{turn["role"]}>']
        tokens.append(role_id)

        for word in turn['content'].lower().split():
            word = word.strip('.,!?')
            token_id = vocab.get(word, vocab.get('<user>', 4))
            tokens.append(token_id)

        tokens.append(vocab['<eos>'])

    input_seq = tokens[:-1][:256]
    label_seq = tokens[1:][:256]

    while len(input_seq) < 256:
        input_seq.append(0)
        label_seq.append(0)

    all_inputs.append(input_seq)
    all_labels.append(label_seq)

inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)

os.makedirs('data/red_teaming/large_vocab', exist_ok=True)
torch.save(inputs_tensor, 'data/red_teaming/large_vocab/inputs.pt')
torch.save(labels_tensor, 'data/red_teaming/large_vocab/labels.pt')

print("✅ Training data created")
print(f"🚀 Training HRM model ({len(vocab)} tokens)")

from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1

print("Checking GPU...")
if not torch.cuda.is_available():
    print("❌ No GPU available!")
    exit(1)

gpu_name = torch.cuda.get_device_name()
print(f"✅ GPU: {gpu_name}")

config = {
    'batch_size': 2, 'seq_len': 256, 'vocab_size': len(vocab),
    'hidden_size': 128, 'expansion': 4, 'num_heads': 4,
    'H_layers': 1, 'L_layers': 1, 'H_cycles': 1, 'L_cycles': 1
}

model = LanguageAdaptedHRM_ACTV1(config)
model.cuda()
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

inputs = torch.load('data/red_teaming/large_vocab/inputs.pt')
labels = torch.load('data/red_teaming/large_vocab/labels.pt')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

print("Step | Loss | Time")
start_time = datetime.now()
best_loss = float('inf')

for step in range(50):
    batch_idx = torch.randint(0, len(inputs), (config['batch_size'],))

    with torch.cuda.amp.autocast():
        batch_data = {
            'inputs': inputs[batch_idx].cuda(),
            'puzzle_identifiers': torch.zeros(config['batch_size'], dtype=torch.long, device='cuda')
        }

        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        loss = nn.functional.cross_entropy(
            outputs['logits'].reshape(-1, outputs['logits'].size(-1)),
            labels[batch_idx].cuda().reshape(-1),
            ignore_index=0
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss

    elapsed = datetime.now() - start_time
    print(f"{step+1:2d} | {current_loss:.4f} | {elapsed.seconds}s")

final_loss = current_loss
torch.save({
    'model': model.state_dict(),
    'vocab_size': len(vocab),
    'config': config,
    'loss': final_loss
}, 'cybersecurity_hrm_model.pt')

print(f"\n✅ Training completed!")
print(f"   Vocabulary size: {len(vocab)}")
print(f"   Final loss: {final_loss:.4f}")
print("   Model saved: cybersecurity_hrm_model.pt")
print("🔒 HRM Cybersecurity Training Complete!")
