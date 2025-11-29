#!/usr/bin/env python3
"""
HRM Red Teaming Training with Large Cybersecurity Vocabulary (5000+ tokens)
NGC Docker Training with Blackwell GPU
"""

import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json

def generate_large_vocab():
    """Generate comprehensive 5000+ token cybersecurity vocabulary"""
    print("🔐 GENERATING COMPREHENSIVE CYBERSECURITY VOCABULARY")

    # Base vocab
    vocab = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<system>': 3, '<user>': 4, '<assistant>': 5}
    max_id = 6

    # Comprehensive cybersecurity terms
    cybersecurity_terms = [
        # Networking
        'tcp', 'udp', 'icmp', 'ip', 'ipv4', 'ipv6', 'arp', 'dns', 'dhcp', 'ntp', 'snmp',
        'http', 'https', 'ftp', 'sftp', 'ssh', 'telnet', 'smtp', 'imap', 'pop3',
        'ldap', 'kerberos', 'radius', 'tacacs', 'diameter', 'bgp', 'ospf', 'rip', 'mpls',
        'vxlan', 'gre', 'ipsec', 'ike', 'ssl', 'tls', 'dtls', 'pptp', 'l2tp', 'openvpn', 'wireguard',

        # Tools
        'nmap', 'metasploit', 'wireshark', 'burpsuite', 'sqlmap', 'gobuster', 'john_ripper',
        'hashcat', 'aircrack', 'ettercap', 'tcpdump', 'snort', 'suricata', 'iptables',
        'fail2ban', 'cobalt_strike', 'empire', 'mimikatz', 'bloodhound', 'impacket', 'crackmapexec',

        # Attacks
        'sql_injection', 'sqli', 'cross_site_scripting', 'xss', 'stored_xss', 'reflected_xss',
        'cross_site_request_forgery', 'csrf', 'directory_traversal', 'lfi', 'rfi',
        'command_injection', 'remote_code_execution', 'rce', 'buffer_overflow', 'heap_overflow',
        'stack_overflow', 'integer_overflow', 'use_after_free', 'double_free', 'race_condition',
        'man_in_the_middle', 'mitm', 'arp_poisoning', 'dns_poisoning', 'session_hijacking',
        'phishing', 'spear_phishing', 'vishing', 'smishing', 'malware', 'ransomware', 'trojan',
        'virus', 'worm', 'rootkit', 'backdoor', 'spyware', 'adware', 'cryptojacking',
        'zero_day', 'brute_force', 'dictionary_attack', 'password_cracking', 'social_engineering',
        'pretexting', 'baiting', 'tailgating', 'shoulder_surfing', 'dumpster_diving', 'honeypot',

        # Defense
        'firewall', 'next_generation_firewall', 'intrusion_detection_system', 'ids',
        'intrusion_prevention_system', 'ips', 'security_information_and_event_management', 'siem',
        'endpoint_detection_and_response', 'edr', 'data_loss_prevention', 'dlp',
        'multi_factor_authentication', 'mfa', 'oauth', 'saml', 'zero_trust', 'network_segmentation',
        'defense_in_depth', 'threat_hunting', 'vulnerability_management', 'patch_management',
        'gdpr', 'hipaa', 'pci_dss', 'iso_27001', 'nist', 'cis_benchmarks', 'mitre_att&ck',

        # Cryptography
        'encryption', 'decryption', 'symmetric_encryption', 'asymmetric_encryption', 'public_key',
        'private_key', 'digital_signature', 'certificate_authority', 'hash_function', 'md5', 'sha1',
        'sha256', 'sha512', 'bcrypt', 'scrypt', 'argon2', 'aes', 'des', 'blowfish', 'rsa',
        'ecdsa', 'diffie_hellman', 'elliptic_curve_cryptography', 'hmac', 'cmac', 'salt',
        'nonce', 'initialization_vector', 'key_derivation_function', 'perfect_forward_secrecy',

        # Web Security
        'owasp_top10', 'broken_access_control', 'cryptographic_failures', 'injection',
        'insecure_design', 'security_misconfiguration', 'vulnerable_components',
        'identification_and_authentication_failures', 'server_side_request_forgery', 'ssrf',
        'xml_external_entity', 'xxe', 'cross_origin_resource_sharing', 'cors',
        'content_security_policy', 'csp', 'helmet', 'subresource_integrity',

        # Cloud Security
        'aws_security', 'azure_security', 'gcp_security', 'iam', 'vpc', 'subnet', 'security_group',
        'kubernetes_security', 'docker_security', 'rbac', 'secret_management', 'vault',
        'key_management_service', 'kms', 'cloud_access_security_broker', 'casb',

        # Red Teaming
        'red_team', 'blue_team', 'purple_team', 'penetration_testing', 'pentest',
        'reconnaissance', 'footprinting', 'enumeration', 'scanning', 'gaining_access',
        'maintaining_access', 'covering_tracks', 'exploit_development', 'metasploit_framework',
        'exploit_db', 'social_engineer_toolkit', 'browser_exploitation_framework',

        # APT Groups
        'apt', 'advanced_persistent_threat', 'apt28', 'fancy_bear', 'apt29', 'cozy_bear',
        'sandworm', 'apt41', 'barium', 'lazarus_group', 'hidden_cobra', 'comment_crew',

        # IoT & Embedded
        'internet_of_things', 'iot', 'embedded_systems', 'zigbee', 'zwave', 'bluetooth',
        'industrial_control_systems', 'ics', 'scada', 'plc', 'modbus', 'profinet', 'opc_ua',
        'industrial_internet_of_things', 'iiot', 'smart_grid',

        # Mobile
        'android_security', 'ios_security', 'mobile_application_security', 'owasp_mobile_top10',
        'jadx', 'apktool', 'frida', 'drozer', 'wi_fi_security', 'wpa2', 'wpa3', 'wep',

        # AI/ML Security
        'ai_security', 'ml_security', 'adversarial_machine_learning', 'data_poisoning',
        'model_poisoning', 'membership_inference', 'deepfake_detection', 'federated_learning_security'
    ]

    # Add all terms
    for term in cybersecurity_terms:
        if term not in vocab:
            vocab[term] = max_id
            max_id += 1

    # Fill to 5000 tokens with combined terms
    while len(vocab) < 5000:
        base_terms = list(set([term.split('_')[0] for term in vocab.keys() if '_' in term]))
        for base in base_terms[:100]:  # Limit to avoid infinite loop
            new_term = f"{base}_security"
            if new_term not in vocab and len(vocab) < 5000:
                vocab[new_term] = max_id
                max_id += 1

    vocab_path = 'data/red_teaming/enhanced/vocab_cybersecurity_5000.pt'
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    torch.save(vocab, vocab_path)

    print(f"✅ Generated vocabulary with {len(vocab)} tokens")
    print(f"💾 Saved to: {vocab_path}")
    return vocab_path, vocab

def create_training_data(vocab_path):
    """Create training data with new vocabulary"""
    print("\n📝 CREATING TRAINING DATA WITH NEW VOCABULARY")

    vocab = torch.load(vocab_path)
    print(f"Loaded vocab with {len(vocab)} tokens")

    # Create synthetic conversations covering cybersecurity topics
    conversations = []

    templates = [
        {
            'system': 'You are a cybersecurity expert specializing in {topic}.',
            'user': 'How can I perform {attack} attack?',
            'assistant': 'First, identify the target system vulnerabilities. Use {tool} for reconnaissance.'
        },
        {
            'system': 'You are conducting red team operations.',
            'user': 'What tool should I use for {task}?',
            'assistant': '{tool} is excellent for {task}. Remember to maintain operational security.'
        },
        {
            'system': 'You are a penetration tester.',
            'user': 'Guide me through {phase} phase.',
            'assistant': 'In {phase}, focus on {focus}. Use {technique} for best results.'
        }
    ]

    topics = ['network_security', 'web_application_security', 'cryptography', 'social_engineering']
    attacks = ['sql_injection', 'xss', 'mitm', 'phishing']
    tools = ['nmap', 'metasploit', 'wireshark', 'burpsuite']
    tasks = ['scanning', 'exploitation', 'post_exploitation']
    phases = ['reconnaissance', 'enumeration', 'gaining_access']
    techniques = ['passive_reconnaissance', 'active_scanning', 'social_engineering']

    for i in range(1000):  # Create 1000 conversations
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        attack = attacks[i % len(attacks)]
        tool = tools[i % len(tools)]
        task = tasks[i % len(tasks)]
        phase = phases[i % len(phases)]
        technique = techniques[i % len(techniques)]

        conversation = [
            {'role': 'system', 'content': template['system'].format(topic=topic)},
            {'role': 'user', 'content': template['user'].format(attack=attack, task=task)},
            {'role': 'assistant', 'content': template['assistant'].format(tool=tool, task=task, phase=phase, focus=topic, technique=technique)}
        ]

        conversations.append({
            'id': i,
            'conversation': conversation,
            'metadata': {'topic': topic, 'tool': tool, 'technique': technique}
        })

    # Save conversations
    os.makedirs('data/red_teaming/large_vocab', exist_ok=True)
    with open('data/red_teaming/large_vocab/train.jsonl', 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

    # Tokenize and create tensors
    all_inputs = []
    all_labels = []

    for conv in conversations:
        tokens = []
        for turn in conv['conversation']:
            role_token = f"<{turn['role']}>"
            tokens.append(vocab.get(role_token, vocab.get('<sos>', 2)))

            for word in turn['content'].lower().split():
                word = word.strip('.,!?')
                if word:
                    tokens.append(vocab.get(word, vocab.get('<user>', 4)))

            tokens.append(vocab.get('<eos>', 1))

        input_seq = tokens[:-1]
        label_seq = tokens[1:]

        max_length = 512
        if len(input_seq) < max_length:
            input_seq.extend([0] * (max_length - len(input_seq)))
            label_seq.extend([0] * (max_length - len(label_seq)))

        input_seq = input_seq[:max_length]
        label_seq = label_seq[:max_length]

        all_inputs.append(input_seq)
        all_labels.append(label_seq)

    inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    torch.save(inputs_tensor, 'data/red_teaming/large_vocab/all_inputs.pt')
    torch.save(labels_tensor, 'data/red_teaming/large_vocab/all_labels.pt')
    torch.save(vocab, 'data/red_teaming/large_vocab/vocab.pt')

    print(f"✅ Created {len(conversations)} training examples")
    print("✅ Training tensors saved to data/red_teaming/large_vocab/")

    return len(conversations)

def train_model(vocab_size, num_examples):
    """Train HRM model with large vocabulary"""
    print(f"\n🚀 TRAINING HRM MODEL WITH {vocab_size} TOKENS")

    from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1

    config = {
        'batch_size': 8,
        'seq_len': 512,
        'vocab_size': vocab_size,
        'num_puzzle_identifiers': 1,
        'hidden_size': 512,
        'expansion': 4,
        'num_heads': 8,
        'H_layers': 4,
        'L_layers': 4,
        'H_cycles': 2,
        'L_cycles': 2
    }

    print("🏗️ Creating HRM model with large vocabulary...")
    model = LanguageAdaptedHRM_ACTV1(config)
    model.cuda()

    # Calculate warmup steps based on dataset size
    total_steps = min(2000, num_examples * 4)  # Adaptive training length
    warmup_steps = min(100, total_steps // 10)

    print(f"⚙️ Training configuration:")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Dataset examples: {num_examples}")
    print(f"   Training steps: {total_steps}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load data
    inputs = torch.load('data/red_teaming/large_vocab/all_inputs.pt')
    labels = torch.load('data/red_teaming/large_vocab/all_labels.pt')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    best_loss = float('inf')

    print("\n🚀 STARTING TRAINING...")

    for step in range(total_steps):
        batch_indices = torch.randint(0, len(inputs), (config['batch_size'],))
        batch_inputs = inputs[batch_indices].cuda()
        batch_labels = labels[batch_indices].cuda()

        batch_data = {
            'inputs': batch_inputs,
            'puzzle_identifiers': torch.zeros(config['batch_size'], dtype=torch.long, device='cuda')
        }

        carry = model.initial_carry(batch_data)
        carry, outputs = model(carry=carry, batch=batch_data)

        logits = outputs['logits']
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch_labels.reshape(-1),
            ignore_index=0
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step < warmup_steps:
            scheduler.step()

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss

        if step % 50 == 0:
            print("Step {:4d}/{:4d}: Loss {:.4f}, Best {:.4f}".format(
                step, total_steps, current_loss, best_loss))

        # Save checkpoints
        if step % 500 == 0 and step > 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': current_loss,
                'best_loss': best_loss,
                'config': config,
                'vocab_size': vocab_size
            }
            torch.save(checkpoint, f'red_teaming_large_vocab_{step}.pt')
            print(f"💾 Checkpoint saved: red_teaming_large_vocab_{step}.pt")

    print("\n🏆 TRAINING COMPLETE!")
    print(f"Final Loss: {current_loss:.4f}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model trained on {vocab_size} cybersecurity tokens")

    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': total_steps,
        'loss': current_loss,
        'best_loss': best_loss,
        'config': config,
        'vocab_size': vocab_size,
        'training_data_examples': num_examples,
        'timestamp': 'November 29, 2025 - Large Vocab Cybersecurity Training'
    }
    torch.save(final_checkpoint, 'red_teaming_large_vocab_final.pt')
    print("✅ Final model saved: red_teaming_large_vocab_final.pt")

def main():
    print("🔒 HRM RED TEAMING - LARGE VOCABULARY CYBERSECURITY TRAINING")
    print("="*70)

    # Step 1: Generate large vocabulary
    vocab_path, vocab = generate_large_vocab()
    vocab_size = len(vocab)
    print(f"📈 Vocabulary size: {vocab_size} tokens")

    # Step 2: Create training data
    num_examples = create_training_data(vocab_path)

    # Step 3: Train model
    train_model(vocab_size, num_examples)

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"✅ Generated {vocab_size}-token cybersecurity vocabulary")
    print(f"✅ Created {num_examples} training examples")
    print("✅ Trained HRM model with large vocabulary")
    print("🚀 Ready for red teaming applications with comprehensive cyber terms")

if __name__ == "__main__":
    main()
