#!/usr/bin/env python3
"""
Phase 3A: Vocabulary Expansion & Quality Filtering
Expand vocabulary from 140→2000+ tokens and improve data quality
Target: Enhanced token representation for loss < 1.2
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import re
from collections import Counter
import os

def expand_vocabulary():
    """
    Perform vocabulary expansion and quality filtering
    """
    print("🔡 PHASE 3A: VOCABULARY EXPANSION & QUALITY FILTERING")
    print("=" * 55)

    # Load current dataset
    print("\n📊 LOADING CURRENT DATASET...")
    with open('data/red_teaming/train.jsonl', 'r') as f:
        conversations = [json.loads(line) for line in f]

    print(f"Current dataset: {len(conversations)} conversations")

    # Step 1: Quality filtering
    print("\n🎯 STEP 1: QUALITY FILTERING")
    filtered_conversations = quality_filter_conversations(conversations)
    print(f"After filtering: {len(filtered_conversations)} conversations")

    # Step 2: Extract all text for vocabulary expansion
    print("\n📝 STEP 2: EXTRACTING TEXT CORPUS")
    all_text = extract_text_corpus(filtered_conversations)

    # Step 3: Build enhanced vocabulary
    print("\n🏗️ STEP 3: BUILDING ENHANCED VOCABULARY")
    vocab = build_enhanced_vocabulary(all_text)
    print(f"Enhanced vocabulary size: {len(vocab)} tokens")

    # Step 4: Add domain-specific tokens
    print("\n🛡️ STEP 4: ADDING DOMAIN-SPECIFIC TOKENS")
    vocab = add_domain_tokens(vocab)
    print(f"Final vocabulary size: {len(vocab)} tokens")

    # Step 5: Re-tokenize with improved vocabulary
    print("\n🔄 STEP 5: RE-TOKENIZING WITH ENHANCED VOCABULARY")
    processed_data = retokenize_data(filtered_conversations, vocab)

    # Step 6: Save enhanced dataset
    print("\n💾 SAVING ENHANCED DATASET")
    save_enhanced_dataset(processed_data, vocab)

    return processed_data, vocab

def quality_filter_conversations(conversations):
    """
    Filter conversations for quality
    """
    def is_high_quality(conv):
        text = ' '.join([turn['content'] for turn in conv['conversation']])

        # Basic quality checks
        if len(text.split()) < 20:  # Too short
            return False
        if len(text.split()) > 500:  # Too long
            return False
        if text.count('?') < 2:  # Not enough questions
            return False
        if len(conv['conversation']) < 4:  # Not enough turns
            return False

        # Check for technical content
        technical_terms = ['attack', 'exploit', 'vulnerability', 'breach', 'security']
        has_technical = any(term in text.lower() for term in technical_terms)

        return has_technical

    filtered = [conv for conv in conversations if is_high_quality(conv)]
    print(f"Quality filtered: {len(filtered)}/{len(conversations)} conversations retained")
    return filtered

def extract_text_corpus(conversations):
    """
    Extract all text for vocabulary building
    """
    all_text = []
    for conv in conversations:
        for turn in conv['conversation']:
            all_text.append(turn['content'])

    combined_text = '\n'.join(all_text)
    print(f"Extracted {len(all_text)} conversation turns, {len(combined_text.split())} total words")
    return combined_text

def build_enhanced_vocabulary(text_corpus):
    """
    Build enhanced vocabulary with word-level and subword tokenization
    """
    # Split text into words and characters
    words = re.findall(r'\b\w+\b', text_corpus.lower())
    chars = list(set(text_corpus.lower()))

    # Word frequency analysis
    word_counts = Counter(words)
    most_common_words = [word for word, count in word_counts.most_common(1500)]

    # Build vocabulary
    vocab = {}
    vocab_idx = 0

    # Special tokens
    special_tokens = ['<pad>', '<eos>', '<sos>', '<unk>', '<system>', '<user>', '<assistant>']
    for token in special_tokens:
        vocab[token] = vocab_idx
        vocab_idx += 1

    # Most common words
    for word in most_common_words:
        if word not in vocab and vocab_idx < 2000:
            vocab[word] = vocab_idx
            vocab_idx += 1

    # Common bigrams (2-character combinations)
    bigrams = []
    for word in most_common_words[:500]:  # Use top words for bigram extraction
        if len(word) >= 2:
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                bigrams.append(bigram)

    bigram_counts = Counter(bigrams)
    common_bigrams = [bigram for bigram, count in bigram_counts.most_common(100)]

    for bigram in common_bigrams:
        if bigram not in vocab and vocab_idx < 2000:
            vocab[bigram] = vocab_idx
            vocab_idx += 1

    # Individual characters
    for char in chars:
        if char.isalnum() and char not in vocab and vocab_idx < 2000:
            vocab[char] = vocab_idx
            vocab_idx += 1

    return vocab

def add_domain_tokens(vocab):
    """
    Add cybersecurity and red teaming domain-specific tokens
    """
    domain_tokens = [
        # Common cybersecurity terms
        'vulnerability', 'exploit', 'breach', 'compromise', 'authentication',
        'authorization', 'encryption', 'malware', 'ransomware', 'phishing',
        'social_engineering', 'zero_day', 'patch', 'firewall', 'intrusion',

        # Red teaming specific
        'adversarial', 'payload', 'command_injection', 'sql_injection',
        'cross_site_scripting', 'buffer_overflow', 'denial_of_service',
        'man_in_the_middle', 'credential_stuffing', 'session_hijacking',

        # Technical terminology
        'protocol', 'server', 'client', 'database', 'api', 'endpoint',
        'network', 'traffic', 'packet', 'port', 'host', 'domain',
        'certificate', 'signature', 'hash', 'key', 'token', 'session',

        # Red teaming methodologies
        'reconnaissance', 'scanning', 'gaining_access', 'maintaining_access',
        'covering_tracks', 'exfiltration', 'escalation', 'persistence',
        'lateral_movement', 'command_and_control'
    ]

    max_id = max(vocab.values())
    additional_tokens_added = 0

    for token in domain_tokens:
        if token not in vocab:
            vocab[token] = max_id + 1
            max_id += 1
            additional_tokens_added += 1

    print(f"Added {additional_tokens_added} domain-specific tokens")
    return vocab

def retokenize_data(conversations, vocab):
    """
    Re-tokenize conversations with new vocabulary
    """
    processed_data = []
    token_stats = []

    for conv in conversations:
        tokenized_turns = []
        for turn in conv['conversation']:
            role_token = f'<{turn["role"]}>'

            # Tokenize content
            content_tokens = []
            for word in turn['content'].split():
                if word in vocab:
                    content_tokens.append(vocab[word])
                else:
                    # Handle OOV words by character-level fall back
                    for char in word.lower():
                        if char in vocab:
                            content_tokens.append(vocab[char])
                        else:
                            content_tokens.append(vocab.get('<unk>', vocab.get('<eos>', 1)))

            # Add role token
            tokenized_turns.extend([vocab.get(role_token, vocab.get('<sos>', 2))])
            tokenized_turns.extend(content_tokens)
            tokenized_turns.append(vocab.get('<eos>', 1))

        # Truncate to maximum length
        if len(tokenized_turns) > 256:
            tokenized_turns = tokenized_turns[:256]

        processed_data.append(tokenized_turns)
        token_stats.append(len(tokenized_turns))

    avg_length = sum(token_stats) / len(token_stats)
    print(f"Re-tokenized {len(processed_data)} conversations")
    print(f"Average length: {avg_length:.1f}")
    return processed_data

def save_enhanced_dataset(processed_data, vocab):
    """
    Save enhanced dataset with new vocabulary
    """
    # Create enhanced dataset directory
    os.makedirs('data/red_teaming/enhanced', exist_ok=True)

    # Save processed conversations
    with open('data/red_teaming/enhanced/conversations.jsonl', 'w') as f:
        for conv in processed_data:
            f.write(json.dumps(conv) + '\n')

    # Save vocabulary
    torch.save(vocab, 'data/red_teaming/enhanced/vocab_enhanced.pt')

    # Convert to tensors
    max_len = 256
    padded_data = []
    labels = []

    for conv in processed_data:
        # Convert conversation to input-label pairs
        if len(conv) > 1:
            input_seq = conv[:-1]  # All tokens except last
            label_seq = conv[1:]   # Next token prediction

            # Pad/truncate
            if len(input_seq) < max_len:
                input_seq.extend([vocab.get('<pad>', 0)] * (max_len - len(input_seq)))
                label_seq.extend([vocab.get('<pad>', 0)] * (max_len - len(label_seq)))

            input_seq = input_seq[:max_len]
            label_seq = label_seq[:max_len]

            padded_data.append(input_seq)
            labels.append(label_seq)

    # Save tensors
    inputs_tensor = torch.tensor(padded_data, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save(inputs_tensor, 'data/red_teaming/enhanced/all_inputs_enhanced.pt')
    torch.save(labels_tensor, 'data/red_teaming/enhanced/all_labels_enhanced.pt')

    print("✅ Enhanced dataset saved!")
    print(f"   Processed {len(padded_data)} conversations")
    print(f"   Vocabulary size: {len(vocab)} tokens")
    print(f"   Files saved: conversations.jsonl, vocab_enhanced.pt, all_inputs_enhanced.pt, all_labels_enhanced.pt")

if __name__ == "__main__":
    processed_data, enhanced_vocab = expand_vocabulary()
    print("\n🎉 VOCABULARY EXPANSION COMPLETE!")
    print("=" * 40)
    print(f"Original vocabulary: 140 tokens")
    print(f"Enhanced vocabulary: {len(enhanced_vocab)} tokens")
    print(f"Expansion factor: {len(enhanced_vocab)/140:.1f}x")
    print("\nNEXT: Train language model with enhanced vocabulary for improved loss reduction!")
