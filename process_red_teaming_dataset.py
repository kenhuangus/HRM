#!/usr/bin/env python3
"""
Process Red Teaming Dataset for HRM Training
Convert JSON conversations to PyTorch tensors
"""

import json
import torch
import os
from collections import defaultdict

class RedTeamingDataset:
    def __init__(self, data_path, max_length=256):
        self.data_path = data_path
        self.max_length = max_length
        self.vocab = self.build_vocab()

    def build_vocab(self):
        vocab = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<system>': 3, '<user>': 4, '<assistant>': 5}
        idx = 6

        with open(self.data_path, 'r') as f:
            for line in f:
                conv = json.loads(line)
                for turn in conv['conversation']:
                    content = turn['content']
                    for word in content.split():
                        word = word.lower().strip('.,!?')
                        if word and word not in vocab:
                            vocab[word] = idx
                            idx += 1

        return vocab

    def tokenize_conversation(self, conversation):
        tokens = []
        for turn in conversation:
            role = turn['role']
            content = turn['content']

            # Add role token
            role_token = f'<{role}>'
            tokens.append(self.vocab.get(role_token, 2))  # Default to <sos> if not found

            # Add content tokens
            for word in content.split():
                word = word.lower().strip('.,!?')
                if word:
                    tokens.append(self.vocab.get(word, 2))  # Use <sos> for unknown words

            # Add turn separator
            tokens.append(1)  # <eos>

        return tokens[:self.max_length]

def create_red_teaming_tensors(data_path, output_dir):
    print(f"🎯 Processing {data_path} for HRM format...")
    dataset = RedTeamingDataset(data_path)
    os.makedirs(output_dir, exist_ok=True)

    all_inputs = []
    all_labels = []

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 50 == 0:
                print(f"   Processing example {line_num}...")

            conv = json.loads(line)
            tokens = dataset.tokenize_conversation(conv['conversation'])

            if len(tokens) > 1:
                input_seq = tokens[:-1]  # All tokens except last as input
                label_seq = tokens[1:]   # Shifted by 1 for prediction

                # Pad/truncate to fixed length
                max_seq_len = 256
                if len(input_seq) < max_seq_len:
                    input_seq.extend([0] * (max_seq_len - len(input_seq)))
                    label_seq.extend([0] * (max_seq_len - len(label_seq)))

                input_seq = input_seq[:max_seq_len]
                label_seq = label_seq[:max_seq_len]

                all_inputs.append(input_seq)
                all_labels.append(label_seq)

    inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    torch.save(inputs_tensor, f'{output_dir}/all_inputs.pt')
    torch.save(labels_tensor, f'{output_dir}/all_labels.pt')
    torch.save(dataset.vocab, f'{output_dir}/vocab.pt')

    print("✅ Dataset conversion complete!")
    print(f"   Examples: {len(all_inputs)}")
    print(f"   Vocabulary size: {len(dataset.vocab)}")
    print(f"   Sequence length: {max_seq_len}")
    print(f"   Saved to: {output_dir}/")

    # Show vocabulary sample
    print("   📝 Vocabulary sample:")
    vocab_items = list(dataset.vocab.items())[:10]
    for token, idx in vocab_items:
        print(f"      {token}: {idx}")
    if len(dataset.vocab) > 10:
        print(f"      ... (+{len(dataset.vocab) - 10} more tokens)")
    print()

def main():
    print("🔄 PROCESSING RED TEAMING DATASET FOR HRM")
    print("=" * 50)

    # Process both train and validation sets
    print("📊 Processing training set...")
    create_red_teaming_tensors('data/red_teaming/train.jsonl', 'data/red_teaming/train')

    print("📊 Processing validation set...")
    create_red_teaming_tensors('data/red_teaming/validation.jsonl', 'data/red_teaming/validation')

    print("✅ ALL DATASET PROCESSING COMPLETE!")
    print()
    print("📂 Generated files:")
    print("   data/red_teaming/train/all_inputs.pt     (training inputs)")
    print("   data/red_teaming/train/all_labels.pt     (training labels)")
    print("   data/red_teaming/train/vocab.pt          (vocabulary)")
    print("   data/red_teaming/validation/all_inputs.pt(validation inputs)")
    print("   data/red_teaming/validation/all_labels.pt(validation labels)")
    print("   data/red_teaming/validation/vocab.pt     (vocabulary)")
    print()
    print("🎯 Ready for HRM red teaming training!")
    print("   Next: Run red_teaming_train.py in NGC container")

if __name__ == "__main__":
    main()
