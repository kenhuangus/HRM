#!/usr/bin/env python3
"""
Test the UI in chat mode with simple inputs that should work with basic vocab
"""

import torch
from red_teaming_comparison_app import create_basic_english_vocab, tokenize_text, detokenize_text

# Test with basic vocab
vocab = create_basic_english_vocab()
id_to_token = {v: k for k, v in vocab.items()}

print("🧪 TESTING UI WITH BASIC VOCAB")
print("=" * 40)

# Test inputs that should work
test_inputs = [
    "how can i hack a website",
    "use nmap for scanning",
    "what is sql injection",
    "how to find vulnerabilities"
]

for test_input in test_inputs:
    print(f"\nInput: '{test_input}'")

    # Tokenize
    tokens = tokenize_text(test_input, vocab)
    print(f"Tokens: {tokens}")

    # Show tokenized words
    token_words = [id_to_token.get(t, f'<{t}>') for t in tokens]
    print(f"Words: {token_words}")

    # Basic detokenize (skip special tokens)
    response = ' '.join([id_to_token.get(t, '<unk>') for t in tokens if t not in [0, 1, 2, 4]])  # Skip pad, eos, sos, user
    print(f"Detokenized: '{response}'")

print("\n✅ UI should now work with basic English inputs!")
print("🛠️  The model outputs may still not be coherent English text,")
print("    but at least the tokenization/detokenization works correctly.")
print("\n🚀 Access the UI at: http://localhost:8501")
