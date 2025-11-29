#!/usr/bin/env python3
import torch
import sys

# Load the enhanced vocabulary
vocab = torch.load('data/red_teaming/enhanced/vocab_enhanced.pt')
print('Vocab size:', len(vocab))
print('First 10 vocab items:', list(vocab.items())[:10])
print('Vocab type examples:', [type(k) for k, v in list(vocab.items())[:10]])

# Test tokenization of a sample input
input_text = "how can i hack a website"
tokens = [vocab.get('<user>', 4)]  # Start with user role

# Basic word-level tokenization
words = input_text.lower().split()
for word in words:
    word = word.strip('.,!?')
    if word:
        token_id = vocab.get(word, vocab.get('<sos>', 2))  # Use <sos> for unknown
        tokens.append(token_id)

tokens.append(vocab.get('<eos>', 1))  # End with EOS
print('Input text:', input_text)
print('Tokenized:', tokens)
print('Token length:', len(tokens))

# Create reverse vocab for detokenization
id_to_token = {v: k for k, v in vocab.items()}
print('Reverse vocab sample:', list(id_to_token.items())[:10])

# Test detokenization of generated tokens
generated_tokens = [5, 10, 15, 20]  # Sample generated tokens
detokenized = []
for token_id in generated_tokens:
    token = id_to_token.get(token_id, '<unk>')
    if token.startswith('<') and token.endswith('>'):
        continue  # Skip special tokens
    else:
        detokenized.append(token)

print('Generated tokens:', generated_tokens)
print('Detokenized:', ' '.join(detokenized))
