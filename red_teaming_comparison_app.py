#!/usr/bin/env python3

"""
Red Teaming HRM Comparison UI - Final Version
Compare best trained model vs initial checkpoint (no red teaming knowledge)
Professional Streamlit interface for testing red teaming capabilities

Requires NGC Docker container for Blackwell GPU support:
docker run --gpus all -p 8501:8501 -v /home/kengpu/HRM:/workspace/HRM --rm nvcr.io/nvidia/pytorch:25.10-py3 /bin/bash -c "
cd /workspace/HRM &&
source hrm_env/bin/activate &&
streamlit run red_teaming_comparison_app.py --server.port 8501 --server.address 0.0.0.0
"
"""

import streamlit as st
import torch
import torch.nn as nn
import json
from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import os

# Page configuration
st.set_page_config(
    page_title="HRM Red Teaming Comparison",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E3440;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        border: 2px solid #E5E9F0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #F8F9FA;
    }
    .loss-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }
    .best-loss { background-color: #28A745; }
    .initial-loss { background-color: #DC3545; }
    .response-box {
        background-color: #FFFFFF;
        border-left: 4px solid #007ACC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models_and_vocab():
    """Load both models and vocabulary"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.write(f"🖥️ Using device: {device}")

    # Load vocabulary - match to model training vocabularies
    vocab_path = None
    vocab = None
    vocab_name = "Unknown"

    try:
        # For best model (phase3_enhanced_final.pt), use enhanced vocabulary (303 tokens)
        vocab_path = 'data/red_teaming/enhanced/vocab_enhanced.pt'
        vocab = torch.load(vocab_path)
        vocab_name = f"Enhanced vocabulary loaded ({len(vocab)} tokens)"
        vocab_size = len(vocab)
        st.sidebar.success(f"✅ Enhanced vocab loaded ({vocab_size} tokens) - matches phase3 model training")
    except Exception as e:
        st.sidebar.warning(f"Enhanced vocab failed, falling back: {e}")
        try:
            vocab_path = 'data/red_teaming/train/vocab.pt'
            vocab = torch.load(vocab_path)
            vocab_name = f"Training vocabulary loaded ({len(vocab)} tokens)"
            vocab_size = len(vocab)
        except Exception as e2:
            st.sidebar.warning(f"Training vocab failed: {e2}")
            try:
                vocab_path = 'data/red_teaming/enhanced/vocab_cybersecurity.pt'
                vocab = torch.load(vocab_path)
                vocab_name = f"Fallback cybersecurity vocabulary loaded ({len(vocab)} tokens)"
                vocab_size = len(vocab)
            except Exception as e3:
                st.error(f"❌ All vocabularies failed: {e3}")
                return None, None, None, None, None, None, None

    if vocab is not None:
        vocab_size = len(vocab)
        st.sidebar.success(f"✅ Vocabulary loaded ({vocab_size} tokens)")
        cyber_terms = ['hack', 'exploit', 'backdoor', 'payload', 'vulnerability', 'red_team', 'penetration']
        if vocab_size < 400 and any(cyber_term in [k.lower() for k in vocab.keys() if isinstance(k, str)] for cyber_term in cyber_terms):
            vocab_name = f"Cybersecurity vocabulary ({vocab_size} tokens)"
        elif vocab_size > 1000:
            vocab_name = "Large training vocabulary"
        elif vocab_size == 303:
            vocab_name = "Basic enhanced vocabulary (no cyber terms)"
        else:
            vocab_name = f"Custom vocabulary ({vocab_size} tokens)"
    else:
        vocab_name = "No vocabulary loaded"
        vocab_size = 0

    # Debug: Show some vocab info
    if vocab:
        sample_tokens = list(vocab.keys())[:5]
        st.sidebar.write(f"**Debug Sample:** {sample_tokens}")
        # Check for key cyber terms
        cyber_terms = ['hack', 'exploit', 'backdoor', 'payload', 'vulnerability']
        present_terms = [term for term in cyber_terms if term in vocab]
        st.sidebar.write(f"**Cyber terms present:** {len(present_terms)}/{len(cyber_terms)}")

    # Store vocab info for later use
    vocab_info = {
        'vocab': vocab,
        'size': vocab_size,
        'name': vocab_name,
        'path': vocab_path
    }

    # Try to create a basic English vocab as fallback if the loaded vocab is inadequate
    basic_english_vocab = create_basic_english_vocab()
    if vocab and len([k for k in vocab.keys() if isinstance(k, str) and len(k) > 1 and not k.startswith('<')]) < 20:
        st.sidebar.warning("⚠️ Loaded vocab appears incomplete. Using basic English fallback for demo.")
        vocab = basic_english_vocab
        vocab_size = len(vocab)
        id_to_token = {v: k for k, v in vocab.items()}

    # Load best model (Language-Adapted HRM from Phase 3)
    best_checkpoint_path = 'phase3_enhanced_final.pt'
    try:
        checkpoint_best = torch.load(best_checkpoint_path, map_location=device)

        # Use config from checkpoint if available, otherwise use defaults
        config = checkpoint_best.get('config', {
            'batch_size': 1,
            'seq_len': 256,
            'vocab_size': len(vocab) if vocab else 303,
            'num_puzzle_identifiers': 1,
            'puzzle_emb_ndim': 128,
            'H_cycles': 1,
            'L_cycles': 4,
            'H_layers': 2,
            'L_layers': 4,
            'hidden_size': 384,
            'num_heads': 12,
            'expansion': 4
        })

        model_best = LanguageAdaptedHRM_ACTV1(config)
        model_best.load_state_dict(checkpoint_best['model_state_dict'])
        model_best.to(device)
        model_best.eval()
        model_best.config = config  # Store config for inference

        best_loss = checkpoint_best.get('best_loss_achieved', checkpoint_best.get('loss', 'N/A'))
        st.sidebar.success(f"✅ Best model loaded (Loss: {best_loss})")
    except Exception as e:
        st.error(f"❌ Error loading best model: {e}")
        model_best = None
        best_loss = 'N/A'

    # Load initial model (Original HRM from early red teaming training)
    initial_checkpoint_path = 'red_teaming_checkpoint_0.pt'
    try:
        checkpoint_initial = torch.load(initial_checkpoint_path, map_location=device)

        # Use config from checkpoint if available
        config = checkpoint_initial.get('config', {
            'batch_size': 1,
            'seq_len': 256,
            'vocab_size': 5000,  # Original vocab size from red_teaming_train.py
            'num_puzzle_identifiers': 1,
            'puzzle_emb_ndim': 256,
            'H_cycles': 3,
            'L_cycles': 4,
            'H_layers': 4,
            'L_layers': 4,
            'hidden_size': 512,
            'num_heads': 16,
            'expansion': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 24,
            'halt_exploration_prob': 0.15
        })

        model_initial = HierarchicalReasoningModel_ACTV1(config)
        model_initial.load_state_dict(checkpoint_initial['model_state_dict'])
        model_initial.to(device)
        model_initial.eval()
        model_initial.config = config  # Store config for inference

        initial_loss = checkpoint_initial.get('loss', 'N/A')
        st.sidebar.success(f"✅ Initial model loaded (Loss: {initial_loss})")
    except Exception as e:
        st.error(f"❌ Error loading initial model: {e}")
        model_initial = None
        initial_loss = 'N/A'

    # Reverse vocab for detokenization
    id_to_token = {v: k for k, v in vocab.items()} if vocab else {}

    return model_best, model_initial, vocab, id_to_token, device, best_loss, initial_loss

def create_basic_english_vocab():
    """Create a basic English vocabulary for demo purposes"""
    basic_words = [
        '<pad>', '<eos>', '<sos>', '<unk>', '<user>', '<assistant>',
        'how', 'can', 'i', 'hack', 'a', 'website', 'what', 'is', 'the',
        'best', 'way', 'to', 'find', 'vulnerabilities', 'in', 'web', 'applications',
        'use', 'nmap', 'for', 'scanning', 'ports', 'and', 'services', 'then',
        'try', 'sql', 'injection', 'attacks', 'with', 'burpsuite', 'or', 'sqlmap',
        'wireless', 'network', 'security', 'aircrack', 'wireshark', 'ettercap',
        'social', 'engineering', 'phishing', 'pretexting', 'maltego',
        'password', 'cracking', 'john', 'ripper', 'hashcat', 'brute', 'force',
        'exploit', 'metasploit', 'framework', 'meterpreter', 'shell',
        'defense', 'firewall', 'ids', 'ips', 'honeypot', 'siem',
        'gdpr', 'hipaa', 'pci', 'dss', 'compliance', 'ethical', 'hacking',
        'red', 'team', 'blue', 'pentest', 'authorized', 'permission'
    ]

    vocab = {word: idx for idx, word in enumerate(basic_words)}
    return vocab

def tokenize_text(text, vocab):
    """Simple tokenization matching dataset processing"""
    tokens = [vocab.get('<user>', 4)]  # Start with user role

    # Basic word-level tokenization
    words = text.lower().split()
    for word in words:
        word = word.strip('.,!?')
        if word:
            token_id = vocab.get(word, vocab.get('<unk>', 3))  # Use <unk> for unknown
            tokens.append(token_id)

    tokens.append(vocab.get('<eos>', 1))  # End with EOS
    return tokens

def detokenize_text(tokens, id_to_token):
    """Convert token IDs back to text"""
    text = []
    skip_role = True  # Skip initial role token for display

    for token_id in tokens:
        token = id_to_token.get(token_id, f'token_{token_id}')  # Fallback to token_ID for missing tokens
        if token.startswith('<') and token.endswith('>'):
            if skip_role:
                skip_role = False
                continue
            # Replace role tokens with readable labels
            if token == '<user>':
                continue  # Skip user token
            elif token == '<assistant>':
                text.append("Assistant:")
                continue
        elif token in ['<pad>', '<eos>', '<sos>']:
            continue
        else:
            text.append(token)

    return ' '.join(text).strip()

# Simple config object to handle dict access like object attributes
class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def get_config_value(config, key):
    """Get config value from either dict or object"""
    if isinstance(config, dict):
        return config.get(key)
    else:
        return getattr(config, key, None)

def generate_response(model, input_tokens, config, device, max_new_tokens=100, temperature=1.0):
    """Generate response using the model"""
    model.eval()

    # Fix config access for original HRM model
    if isinstance(model.config, dict):
        model.config = ConfigObject(model.config)

    with torch.no_grad():
        # Truncate/pad input to seq_len
        seq_len = get_config_value(config, 'seq_len')
        if seq_len is None:
            seq_len = 256  # fallback

        if len(input_tokens) > seq_len:
            input_tokens = input_tokens[:seq_len]
        else:
            input_tokens.extend([0] * (seq_len - len(input_tokens)))  # Pad with <pad>

        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

        generated = input_tokens.copy()

        for _ in range(max_new_tokens):
            # Current input for model
            current_input = generated[-seq_len:] if len(generated) > seq_len else generated
            if len(current_input) < seq_len:
                current_input = [0] * (seq_len - len(current_input)) + current_input

            batch = {
                'inputs': torch.tensor([current_input], dtype=torch.long, device=device),
                'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=device)
            }

            carry = model.initial_carry(batch)
            carry, outputs = model(carry=carry, batch=batch)

            logits = outputs['logits'][0, -1, :]  # Last token logits

            # Clamp to available vocabulary size
            vocab_size = get_config_value(config, 'vocab_size')
            if vocab_size is not None and logits.numel() > vocab_size:
                logits = logits[:vocab_size]
            elif vocab_size is not None and logits.numel() < vocab_size:
                # Pad logits if needed
                padding = torch.full((vocab_size - logits.numel(),), float('-inf'), device=logits.device)
                logits = torch.cat([logits, padding])

            if temperature > 0:
                logits = logits / temperature
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.argmax().item()

            generated.append(next_token)

            # Stop if EOS token
            if next_token == 1:  # <eos>
                break

        return generated[len(input_tokens):]  # Return only generated tokens

# Main UI
def main():
    st.markdown('<div class="main-header">🛡️ HRM Red Teaming Comparison Tool</div>', unsafe_allow_html=True)

    st.markdown("""
    Compare the performance of HRM models trained on red teaming data:

    - **Best Model**: After extensive training with 63% loss reduction (final loss ≈3.45)
    - **Initial Model**: Before red teaming training, no red teaming knowledge

    This tool demonstrates the impact of cross-domain adaptation from puzzle-solving to conversational AI.

    **⚠️ Important**: Models require NGC Docker environment for proper Blackwell GPU support.
    """)

    # Load models and data
    with st.spinner("Loading models and vocabulary..."):
        model_best, model_initial, vocab, id_to_token, device, best_loss, initial_loss = load_models_and_vocab()

        if model_best is None or model_initial is None or vocab is None:
            st.error("""
            **❌ Model Loading Failed**

            This app requires NGC Docker container for Blackwell GPU model execution:

            ```bash
            docker run --gpus all -p 8501:8501 -v /home/kengpu/HRM:/workspace/HRM --rm nvcr.io/nvidia/pytorch:25.10-py3 /bin/bash -c "
            cd /workspace/HRM &&
            source hrm_env/bin/activate &&
            streamlit run red_teaming_comparison_app.py --server.port 8501 --server.address 0.0.0.0
            "
            ```

            Local execution may fail due to:
            - Blackwell GPU CC 12.1 compatibility
            - Missing NGC optimized PyTorch
            - FlashAttention Blackwell binaries
            """)
            return

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Information:**")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown('<span class="loss-badge best-loss">Best Model</span>', unsafe_allow_html=True)
        st.write(f"Loss: {best_loss:.4f}")

    with col2:
        st.markdown('<span class="loss-badge initial-loss">Initial Model</span>', unsafe_allow_html=True)
        st.write(f"Loss: {initial_loss}")

    st.sidebar.markdown("**Training Details:**")
    vocab_size = len(vocab) if vocab else "Unknown"
    st.sidebar.write(f"• Vocabulary: {vocab_size} tokens")
    st.sidebar.write("• Quality-filtered dataset: 810 conversations")
    st.sidebar.write("• Language-adapted HRM architecture")

    if vocab_size == 303:
        st.sidebar.warning("⚠️ Small vocabulary (303 tokens) - many input words may be unknown, leading to gibberish outputs")

    if vocab_size == 5000:
        st.sidebar.info("✅ Large vocabulary (5000 tokens) - should handle more diverse input")

    # Input section
    st.markdown("---")
    st.markdown("### 🔍 Enter Your Red Teaming Question")

    user_input = st.text_area(
        "Ask a red teaming question (e.g., 'How can I hack a website?')",
        height=100,
        placeholder="Enter your question here..."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        max_tokens = st.slider("Max response length", 50, 200, 100)
    with col2:
        temperature = st.slider("Creativity (temperature)", 0.1, 2.0, 0.8)
    with col3:
        generate_button = st.button("🚀 Generate Responses", use_container_width=True)

    if generate_button and user_input.strip():
        # Debug: Show tokenization
        input_tokens = tokenize_text(user_input, vocab)
        st.write(f"**Debug - Input tokens:** {input_tokens[:10]}... (length: {len(input_tokens)})")

        with st.spinner("Generating responses from both models..."):
            # Generate from both models
            response_best_tokens = generate_response(model_best, input_tokens, model_best.config, device,
                                                    max_new_tokens=max_tokens, temperature=temperature)
            response_initial_tokens = generate_response(model_initial, input_tokens, model_initial.config, device,
                                                       max_new_tokens=max_tokens, temperature=temperature)

            # Debug: Show generated tokens
            st.write(f"**Debug - Best model tokens:** {response_best_tokens[:10]}...")
            st.write(f"**Debug - Initial model tokens:** {response_initial_tokens[:10]}...")

            # Detokenize
            response_best = detokenize_text(response_best_tokens, id_to_token)
            response_initial = detokenize_text(response_initial_tokens, id_to_token)

            # Debug: Show final text
            st.write(f"**Debug - Best response:** '{response_best[:100]}...'")
            st.write(f"**Debug - Initial response:** '{response_initial[:100]}...'")

            if not response_best:
                response_best = "[No response generated]"
            if not response_initial:
                response_initial = "[No response generated]"

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("#### 🏆 Best Model (Red Teaming Trained)")
            st.markdown(f'<span class="loss-badge best-loss">Loss: {best_loss:.4f}</span>', unsafe_allow_html=True)
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            if len(response_best) > 500:
                st.write(response_best[:500] + "...")
            else:
                st.write(response_best)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("#### 📚 Initial Model (No Red Teaming Knowledge)")
            st.markdown(f'<span class="loss-badge initial-loss">Loss: {initial_loss}</span>', unsafe_allow_html=True)
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            if len(response_initial) > 500:
                st.write(response_initial[:500] + "...")
            else:
                st.write(response_initial)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Analysis
        st.markdown("---")
        st.markdown("### 🔬 Analysis")
        st.info("""
        **Expected Behavior:**
        - **Best Model**: Should provide more coherent, red-teaming-appropriate responses
        - **Initial Model**: May generate confused or irrelevant responses since it lacks red teaming training

        The performance difference demonstrates HRM's ability to adapt from spatial reasoning (puzzles)
        to sequential reasoning (conversations) through specialized training.
        """)

    elif generate_button and not user_input.strip():
        st.warning("Please enter a question first!")

if __name__ == "__main__":
    main()
