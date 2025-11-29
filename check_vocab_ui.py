#!/usr/bin/env python3
"""
Quick script to inspect vocabulary issues in the UI
"""

import torch
import os

print("🔍 VOCAB INSPECTION FOR UI DEBUGGING")
print("=" * 50)

# Check trained model
try:
    checkpoint = torch.load('cybersecurity_hrm_model.pt')
    print(f"✅ Model vocab size: {checkpoint.get('vocab_size')}")
    print(f"   Config vocab size: {checkpoint.get('config', {}).get('vocab_size')}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Check available vocabs
vocab_paths = [
    'data/red_teaming/enhanced/vocab_cybersecurity.pt',
    'data/red_teaming/enhanced/vocab_enhanced.pt',
    'data/red_teaming/train/vocab.pt'
]

for path in vocab_paths:
    try:
        if os.path.exists(path):
            vocab = torch.load(path)
            vocab_size = len(vocab)
            print(f"\n📚 Vocab at {path}: {vocab_size} tokens")
            print(f"   Sample tokens: {list(vocab.keys())[:5]}")

            # Check common words
            common_words = ['how', 'can', 'i', 'hack', 'a', 'website', 'the', 'is', 'and']
            found = [w for w in common_words if w in vocab]
            print(f"   Common words found: {len(found)}/{len(common_words)} -> {found[:3]}...")

            # Check special tokens
            special_tokens = ['<sos>', '<eos>', '<user>', '<assistant>', '<pad>']
            found_special = [t for t in special_tokens if t in vocab]
            print(f"   Special tokens: {len(found_special)}/{len(special_tokens)} -> {found_special}")

            if vocab_size == checkpoint.get('vocab_size'):
                print("   ✅ VOCAB SIZE MATCHES MODEL!")
            else:
                print(f"   ⚠️ VOCAB SIZE MISMATCH: model expects {checkpoint.get('vocab_size')}")

        else:
            print(f"\n❌ {path} not found")
    except Exception as e:
        print(f"\n❌ Failed to load {path}: {e}")

print("\n🔧 RECOMMENDED FIXES:")
print("1. Use a vocab that includes basic English words")
print("2. Implement proper BPE/subword tokenization instead of word-level")
print("3. Or retrain model with a standard tokenizer (like GPT-style)")

# Test tokenization with current logic
print("\n🧪 TESTING CURRENT TOKENIZATION:")
test_input = "how can i hack a website"
try:
    vocab = torch.load('data/red_teaming/enhanced/vocab_cybersecurity.pt')
    id_to_token = {v: k for k, v in vocab.items()}

    # Current UI tokenization
    tokens = [vocab.get('<user>', 4)]
    for word in test_input.lower().split():
        word = word.strip('.,!?')
        if word:
            token_id = vocab.get(word, vocab.get('<sos>', 2))
            tokens.append(token_id)
    tokens.append(vocab.get('<eos>', 1))

    print(f"Input: '{test_input}'")
    print(f"Tokens: {tokens}")
    print(f"Detokenized: {[id_to_token.get(t, f'<{t}>') for t in tokens]}")

except Exception as e:
    print(f"❌ Tokenization test failed: {e}")
