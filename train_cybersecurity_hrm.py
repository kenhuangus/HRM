#!/usr/bin/env python3
"""
Simple HRM Red Teaming Training with Cybersecurity Vocabulary
"""

import torch
import torch.nn as nn
import os
from datetime import datetime

def generate_vocab():
    """Generate comprehensive cybersecurity vocabulary"""
    print("🔐 GENERATING CYBERSECURITY VOCABULARY")

    vocab = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<system>': 3, '<user>': 4, '<assistant>': 5}
    max_id = 6

    # Cybersecurity terms
    terms = [
        'tcp', 'udp', 'http', 'https', 'ssh', 'dns', 'ip', 'firewall', 'encryption',
        'nmap', 'metasploit', 'wireshark', 'burpsuite', 'sqlmap', 'gobuster', 'john_ripper',
        'hashcat', 'aircrack', 'ettercap', 'tcpdump', 'snort', 'suricata', 'iptables',
        'cobalt_strike', 'empire', 'mimikatz', 'bloodhound', 'impacket', 'crackmapexec',
        'sql_injection', 'xss', 'csrf', 'command_injection', 'remote_code_execution',
        'buffer_overflow', 'man_in_the_middle', 'dns_poisoning', 'session_hijacking',
        'phishing', 'spear_phishing', 'malware', 'ransomware', 'trojan', 'rootkit',
        'brute_force', 'social_engineering', 'zero_day', 'apt', 'red_team', 'blue_team',
        'penetration_testing', 'reconnaissance', 'scanning', 'gaining_access', 'maintaining_access',
        'covering_tracks', 'exploit_development', 'metasploit_framework',
        'gdpr', 'hipaa', 'pci_dss', 'iso_27001', 'nist'
    ]

    for term in terms:
        if term not in vocab:
            vocab[term] = max_id
            max_id += 1

    # Fill to 5000+ tokens
    while len(vocab) < 5000:
        for i in range(10):
            if len(vocab) < 5000:
                vocab[f'term_{len(vocab)}'] = len(vocab)

    vocab_path = 'data/red_teaming/enhanced/vocab_cybersecurity.pt'
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    torch.save(vocab, vocab_path)

    print(f"✅ Generated vocabulary with {len(vocab)} tokens")
    return vocab_path, vocab

def create_data(vocab_path):
    """Create training data"""
    print("📝 CREATING TRAINING DATA")

    vocab = torch.load(vocab_path)

    conversations = []
    for i in range(500):  # Fewer examples for faster testing
        conv = [
            {'role': 'system', 'content': 'You are a cybersecurity expert.'},
            {'role': 'user', 'content': f'How does {list(vocab.keys())[i%10]} work?'},
            {'role': 'assistant', 'content': f'{list(vocab.keys())[i%10]} is an important cybersecurity tool.'}
        ]
        conversations.append({'conversation': conv})

    # Tokenize
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

        input_seq = tokens[:-1][:256]  # Shorter for speed
        label_seq = tokens[1:][:256]

        # Pad
        while len(input_seq) < 256:
            input_seq.append(0)
            label_seq.append(0)

        all_inputs.append(input_seq)
        all_labels.append(label_seq)

    inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    torch.save(inputs_tensor, 'data/red_teaming/large_vocab/inputs.pt')
    torch.save(labels_tensor, 'data/red_teaming/large_vocab/labels.pt')

    print("✅ Created training data")
    return len(conversations), len(vocab)

def train_model(vocab_size, num_examples):
    """Train HRM model"""
    print(f"🚀 TRAINING HRM MODEL ({vocab_size} tokens)")

    from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1

    print("Checking GPU...")
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")

    config = {
        'batch_size': 4, 'seq_len': 256, 'vocab_size': vocab_size,
        'hidden_size': 256, 'expansion': 4, 'num_heads': 8,
        'H_layers': 2, 'L_layers': 2, 'H_cycles': 1, 'L_cycles': 1
    }

    model = LanguageAdaptedHRM_ACTV1(config)
    model.cuda()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load data
    inputs = torch.load('data/red_teaming/large_vocab/inputs.pt')
    labels = torch.load('data/red_teaming/large_vocab/labels.pt')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    print("Step | Loss    | Best Loss | Time")

    best_loss = float('inf')
    start_time = datetime.now()

    for step in range(100):  # Quick test training
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
        if step % 10 == 0:
            print(f"{step+1:4d} | {current_loss:.6f} | {best_loss:.6f} | {elapsed.seconds}s")
    print(f"\nTraining complete. Final loss: {current_loss:.4f}")

    # Save model
    torch.save({
        'model': model.state_dict(),
        'vocab_size': vocab_size,
        'config': config,
        'loss': current_loss
    }, 'cybersecurity_hrm_model.pt')

    print("✅ Model saved: cybersecurity_hrm_model.pt")
    return current_loss

def main():
    print("🔒 HRM CYBERSECURITY TRAINING")

    # Generate vocab
    vocab_path, vocab = generate_vocab()
    vocab_size = len(vocab)

    # Create data
    num_examples, final_vocab_size = create_data(vocab_path)

    # Train
    final_loss = train_model(final_vocab_size, num_examples)

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"   Vocabulary size: {final_vocab_size}")
    print(f"   Final loss: {final_loss:.4f}")
    print("   Ready for red teaming applications")
if __name__ == "__main__":
    main()
