#!/usr/bin/env python3
"""
Add 500 more red teaming examples to existing dataset
"""

import json
import random

# Use same data as before
strategies = [
    'prompt_injection', 'jailbreak', 'data_exfiltration', 'model_poisoning',
    'adversarial_inputs', 'model_inversion', 'backdoor_attacks', 'evasion_techniques',
    'gradient_based_attacks', 'transfer_attacks', 'universal_adversarial_perturbations'
]

tactics = [
    'social_engineering', 'technical_exploitation', 'side_channel_attacks',
    'supply_chain_attacks', 'zero_day_exploitation', 'credential_stuffing',
    'phishing_simulation', 'insider_threat_simulation'
]

goals = [
    'extract_sensitive_data', 'bypass_safety_filters', 'compromise_model_integrity',
    'manipulate_outputs', 'create_backdoors', 'deny_service', 'data_poisoning',
    'intellectual_property_theft'
]

def generate_red_teaming_conversation(id_offset=500):
    strategy = random.choice(strategies)
    tactic = random.choice(tactics)
    goal = random.choice(goals)

    conversation = [
        {'role': 'system', 'content': f'As a cybersecurity expert specializing in {strategy} techniques, provide guidance on adversarial strategies.'},
        {'role': 'user', 'content': f'I need to develop an attack plan for {goal} using {strategy} through {tactic}.'},
        {'role': 'assistant', 'content': f'First, understand your threat model. The target system must be vulnerable to {strategy}.'},
        {'role': 'user', 'content': 'What reconnaissance methods should I use?'},
        {'role': 'assistant', 'content': f'For {tactic}, collect intelligence on system endpoints, API configurations, and trust boundaries.'},
        {'role': 'user', 'content': 'How can I execute the attack?'},
        {'role': 'assistant', 'content': f'Execute {strategy} by crafting payloads that exploit the identified vulnerabilities.'},
        {'role': 'user', 'content': 'What defenses might I encounter?'},
        {'role': 'assistant', 'content': f'Common defenses include input validation, rate limiting, and anomaly detection. Use advanced {tactic} techniques to bypass them.'}
    ]

    return {'conversation': conversation, 'metadata': {'strategy': strategy, 'tactic': tactic, 'goal': goal}}

def main():
    print("🔄 ADDING 500 MORE RED TEAMING EXAMPLES")
    print("=" * 50)

    # Load existing examples to check current count
    try:
        with open('data/red_teaming/train.jsonl', 'r') as f:
            existing_count = sum(1 for _ in f)
        print(f"📊 Current dataset has {existing_count} examples")
    except FileNotFoundError:
        print("❌ No existing red teaming dataset found!")
        return

    # Add 500 more examples
    print("📝 Generating 500 additional examples...")

    with open('data/red_teaming/train.jsonl', 'a') as f:
        for i in range(500):
            if i % 100 == 0:
                print(f"   Adding example {i+1}/500...")

            conv = generate_red_teaming_conversation(id_offset=existing_count + i)
            conv['id'] = existing_count + i
            f.write(json.dumps(conv) + '\n')

    # Update validation set
    print("📝 Updating validation set...")

    with open('data/red_teaming/train.jsonl', 'r') as f:
        all_examples = [json.loads(line) for line in f]

    val_examples = all_examples[-100:]
    with open('data/red_teaming/validation.jsonl', 'w') as f:
        for conv in val_examples:
            f.write(json.dumps(conv) + '\n')

    # Final count
    with open('data/red_teaming/train.jsonl', 'r') as f:
        final_count = sum(1 for _ in f)

    print("✅ EXPANSION COMPLETE!")
    print(f"📁 data/red_teaming/train.jsonl: {final_count} total examples")
    print("📁 data/red_teaming/validation.jsonl: 100 validation examples")
    print()
    print("🎯 Next steps:")
    print("   python3 process_red_teaming_dataset.py")
    print("   python3 red_teaming_train.py")

if __name__ == "__main__":
    main()
