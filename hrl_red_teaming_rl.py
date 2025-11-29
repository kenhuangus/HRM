#!/usr/bin/env python3
"""
HRM Red Teaming RL Integration - Reinforcement Learning Training
Advanced reward functions for cybersecurity red teaming
PPO-based policy optimization for ethical hacking proficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any

class RedTeamingRewardFunction:
    """Comprehensive reward function for red teaming scenarios"""

    def __init__(self, vocab_path='data/red_teaming/enhanced/vocab_cybersecurity.pt'):
        """Initialize reward evaluator"""
        self.vocab = torch.load(vocab_path)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Cybersecurity expertise keywords
        self.security_tools = {
            'nmap', 'metasploit', 'wireshark', 'burpsuite', 'sqlmap',
            'john_ripper', 'hashcat', 'aircrack', 'ettercap', 'maltego'
        }

        self.attack_vectors = {
            'sql_injection', 'xss', 'csrf', 'buffer_overflow', 'mitm',
            'dns_poisoning', 'social_engineering', 'zero_day', 'apt'
        }

        self.defense_mechanisms = {
            'firewall', 'ids', 'ips', 'honeypot', 'siem', 'endpoint_protection',
            'gdpr', 'hipaa', 'pci_dss', 'iso_27001'
        }

        # Ethical boundaries
        self.illegal_activities = {
            'malware_deployment', 'ransomware', 'data_theft', 'ddos',
            'unauthorized_access', 'data_destruction'
        }

    def compute_reward(self, response_tokens: List[int], task_context: Dict) -> Dict[str, float]:
        """
        Compute multi-dimensional reward for red teaming response

        Args:
            response_tokens: Generated token sequence
            task_context: Task information (query, expected_tools, legal_constraints)

        Returns:
            Dict with reward components and total
        """
        response_text = self._tokens_to_text(response_tokens)

        rewards = {}

        # 1. Cybersecurity Expertise (40%)
        rewards['expertise'] = self._evaluate_expertise(response_text, task_context)
        rewards['expertise'] *= 0.4

        # 2. Tool Proficiency (25%)
        rewards['tools'] = self._evaluate_tools(response_text, task_context)
        rewards['tools'] *= 0.25

        # 3. Ethical Compliance (20%)
        rewards['ethics'] = self._evaluate_ethics(response_text, task_context)
        rewards['ethics'] *= 0.2

        # 4. Creativity & Methodology (10%)
        rewards['creativity'] = self._evaluate_creativity(response_text, task_context)
        rewards['creativity'] *= 0.1

        # 5. Task Completion (5%)
        rewards['completion'] = self._evaluate_completion(response_text, task_context)
        rewards['completion'] *= 0.05

        # Bonus for multi-step reasoning
        rewards['reasoning'] = self._evaluate_reasoning(response_text) * 0.1

        # Ethical penalty (overrides positive rewards)
        if rewards['ethics'] < -0.5:
            rewards['total'] = -1.0  # Strong negative for policy violation
        else:
            rewards['total'] = sum(rewards.values())

        return rewards

    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        words = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, '<unk>')
            if token not in ['<pad>', '<eos>', '<sos>', '<user>', '<assistant>']:
                words.append(token)
        return ' '.join(words)

    def _evaluate_expertise(self, response: str, context: Dict) -> float:
        """Score cybersecurity knowledge accuracy"""
        score = 0.0
        response_lower = response.lower()

        # Expected tools/attacks for this scenario
        expected_tools = context.get('expected_tools', [])
        expected_attacks = context.get('expected_attacks', [])

        # Tool mentions
        mentioned_tools = sum(1 for tool in self.security_tools
                            if tool.replace('_', ' ') in response_lower)
        if expected_tools:
            expected_found = sum(1 for tool in expected_tools
                               if tool.replace('_', ' ') in response_lower)
            score += (expected_found / len(expected_tools)) * 0.6

        # Attack vector accuracy
        attack_mentions = sum(1 for attack in self.attack_vectors
                            if attack.replace('_', ' ') in response_lower)
        score += min(attack_mentions * 0.1, 0.3)  # Bonus for correct attack knowledge

        # Defense mechanism awareness
        defense_mentions = sum(1 for defense in self.defense_mechanisms
                             if defense.replace('_', ' ') in response_lower)
        score += min(defense_mentions * 0.05, 0.1)  # Awareness bonus

        return min(score, 1.0)

    def _evaluate_tools(self, response: str, context: Dict) -> float:
        """Score tool recommendations appropriateness"""
        score = 0.0
        response_lower = response.lower()

        # Primary tool recommendations
        primary_tools = context.get('primary_tools', [])
        if primary_tools:
            primary_found = sum(1 for tool in primary_tools
                              if tool.replace('_', ' ') in response_lower)
            score += (primary_found / len(primary_tools)) * 0.5

        # Secondary tools (bonus)
        secondary_tools = context.get('secondary_tools', [])
        if secondary_tools:
            secondary_found = sum(1 for tool in secondary_tools
                                if tool.replace('_', ' ') in response_lower)
            score += (secondary_found / len(secondary_tools)) * 0.3

        # Tool usage instructions
        if any(indicator in response_lower for indicator in
               ['step', 'first', 'then', 'next', 'command', 'run', 'execute']):
            score += 0.2

        return min(score, 1.0)

    def _evaluate_ethics(self, response: str, context: Dict) -> float:
        """Score ethical and legal compliance"""
        score = 0.0
        response_lower = response.lower()

        # Explicit mention of legal/ethical considerations
        ethical_keywords = [
            'legal', 'authorized', 'permission', 'consent', 'ethical',
            'responsible', 'professional', 'pentest', 'red team',
            'bug bounty', 'compliance', 'gdpr', 'safe'
        ]

        ethical_mentions = sum(1 for keyword in ethical_keywords
                             if keyword in response_lower)
        score += min(ethical_mentions * 0.2, 0.4)

        # Warnings about unauthorized access
        warning_keywords = [
            'unauthorized', 'illegal', 'warning', 'caution', 'risk',
            'only with permission', 'ethical hacker', 'white hat'
        ]

        warnings_given = sum(1 for warning in warning_keywords
                           if warning in response_lower)
        score += min(warnings_given * 0.3, 0.4)

        # Strong penalty for illegal activity promotion
        illegal_promoted = sum(1 for illegal in self.illegal_activities
                             if illegal.replace('_', ' ') in response_lower
                             and 'avoid' not in response_lower)
        if illegal_promoted > 0:
            score -= 0.8  # Heavy penalty

        return max(-1.0, min(score, 1.0))

    def _evaluate_creativity(self, response: str, context: Dict) -> float:
        """Score creative problem-solving approaches"""
        score = 0.0
        response_lower = response.lower()

        # Creative attack chaining
        if 'and then' in response_lower or 'followed by' in response_lower:
            score += 0.3

        # Alternative approach suggestions
        if 'or alternatively' in response_lower or 'another way' in response_lower:
            score += 0.3

        # Advanced technique mentions
        advanced_techs = [
            'zero-day', 'advanced persistent threat', 'supply chain',
            'social engineering', 'zero trust', 'defense evasion'
        ]

        advanced_count = sum(1 for tech in advanced_techs
                           if tech in response_lower)
        score += min(advanced_count * 0.2, 0.4)

        # Multi-phase approach
        if 'phase' in response_lower and 'reconnaissance' in response_lower:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_completion(self, response: str, context: Dict) -> float:
        """Score task completion thoroughness"""
        score = 0.0

        # Length reward (encourage detailed responses)
        response_length = len(response.split())
        if 20 <= response_length <= 100:
            score += 0.4
        elif response_length > 100:
            score += 0.6

        # Structured response (numbered lists, clear steps)
        if '1.' in response or 'step 1' in response:
            score += 0.3

        # Practical guidance
        if any(indicator in response.lower() for indicator in
               ['command:', 'run:', 'execute:', 'use:', 'tool:']):
            score += 0.3

        return min(score, 1.0)

    def _evaluate_reasoning(self, response: str) -> float:
        """Score multi-step reasoning and logical flow"""
        score = 0.0
        response_lower = response.lower()

        # Logical connectors
        logical_indicators = [
            'because', 'therefore', 'however', 'although', 'thus',
            'consequently', 'furthermore', 'moreover', 'additionally'
        ]

        logical_count = sum(1 for indicator in logical_indicators
                          if indicator in response_lower)
        score += min(logical_count * 0.2, 0.5)

        # Step-by-step reasoning
        step_indicators = [
            'first', 'second', 'then', 'next', 'finally', 'lastly',
            'initially', 'subsequently'
        ]

        step_count = sum(1 for step in step_indicators
                        if step in response_lower)
        score += min(step_count * 0.1, 0.3)

        # Caution/reasoned approach
        if 'consider' in response_lower or 'however' in response_lower:
            score += 0.2

        return min(score, 1.0)


class PPOTrainer:
    """Proximal Policy Optimization for HRM red teaming fine-tuning"""

    def __init__(self, model, vocab_size, config):
        self.model = model
        self.vocab_size = vocab_size
        self.config = config

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.ppo_epochs = 4
        self.batch_size = 64
        self.num_mini_batches = 8

        # PPO critic (value network)
        self.critic = nn.Linear(config['hidden_size'], 1)
        self.critic.to(model.device)

        # Optimizer for PPO (policy + value)
        self.optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': 1e-5},  # Lower learning rate for RL
            {'params': self.critic.parameters(), 'lr': 1e-4}
        ])

        # Reward function
        self.reward_fn = RedTeamingRewardFunction()

        # Experience replay buffer
        self.buffer = []

        # Statistics
        self.stats = {
            'episodes': 0,
            'rewards': [],
            'losses': [],
            'kl_divs': []
        }

    def rollout_trajectory(self, task_context: Dict, max_steps: int = 100) -> Dict:
        """Generate rollout trajectory using current policy"""
        self.model.eval()

        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

        # Initial state (question tokens)
        current_tokens = task_context['input_tokens'].copy()
        trajectory['states'].append(current_tokens.copy())

        for step in range(max_steps):
            with torch.no_grad():
                # Get action logits and value
                state_input = self._prepare_model_input([current_tokens])
                logits, value = self._get_policy_and_value(state_input)

                # Sample action (next token)
                action_dist = Categorical(logits=logits[0])
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Execute action (add token)
                current_tokens.append(action.item())

                # Compute immediate reward (sliding window)
                if step >= 3:  # Wait for some tokens accumulation
                    sliding_tokens = current_tokens[-50:]  # Recent tokens
                    rewards = self.reward_fn.compute_reward(sliding_tokens, task_context)
                    immediate_reward = rewards['total']
                else:
                    immediate_reward = 0.0

                # Store transition
                trajectory['actions'].append(action.item())
                trajectory['log_probs'].append(log_prob.item())
                trajectory['values'].append(value[0, -1].item())
                trajectory['rewards'].append(immediate_reward)

                # Check completion
                done = self._is_trajectory_complete(current_tokens, task_context)
                trajectory['dones'].append(done)

                if done:
                    break

        # Compute returns and advantages
        returns, advantages = self._compute_returns_and_advantages(trajectory)

        return {
            'trajectory': trajectory,
            'returns': returns,
            'advantages': advantages,
            'final_tokens': current_tokens
        }

    def _prepare_model_input(self, token_sequences):
        """Prepare input tensors for HRM model"""
        max_len = self.config['seq_len']
        batch_inputs = []

        for tokens in token_sequences:
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]
            else:
                tokens = [self.vocab.get('<pad>', 0)] * (max_len - len(tokens)) + tokens

            batch_inputs.append(tokens)

        return torch.tensor(batch_inputs, dtype=torch.long).to(self.model.device)

    def _get_policy_and_value(self, input_tensor):
        """Get policy (logits) and value from current model"""
        with torch.no_grad():
            batch = {
                'inputs': input_tensor,
                'puzzle_identifiers': torch.zeros(input_tensor.size(0), dtype=torch.long, device=input_tensor.device)
            }

            carry = self.model.initial_carry(batch)
            carry, outputs = self.model(carry=carry, batch=batch)

            logits = outputs['logits']  # [batch, seq, vocab]
            values = self.critic(outputs['logits'].mean(dim=-1))  # Value estimation

            return logits, values

    def _is_trajectory_complete(self, tokens: List[int], context: Dict) -> bool:
        """Check if trajectory should terminate"""
        response_length = len(tokens) - len(context['input_tokens'])

        # Maximum length constraint
        if response_length >= context.get('max_length', 100):
            return True

        # EOS token generation
        if tokens[-1] == self.vocab.get('<eos>', 1):
            return True

        return False

    def _compute_returns_and_advantages(self, trajectory):
        """Compute discounted returns and GAE advantages"""
        rewards = np.array(trajectory['rewards'])
        values = np.array(trajectory['values'])
        dones = np.array(trajectory['dones'])

        gamma = 0.99
        lambda_gae = 0.95

        # Bootstrap value for incomplete trajectories
        if not dones[-1]:
            _, next_values = self._get_policy_and_value(
                self._prepare_model_input([trajectory['states'][-1]])
            )
            bootstrap_value = next_values[0, -1].item()
        else:
            bootstrap_value = 0.0

        # Compute returns
        returns = np.zeros_like(rewards)
        adv_buffer = []

        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = bootstrap_value if not dones[step] else 0.0
            else:
                next_value = values[step + 1] if step + 1 < len(values) else bootstrap_value

            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * lambda_gae * (1 - dones[step]) * gae
            adv_buffer.insert(0, gae)

        returns = values + np.array(adv_buffer)
        advantages = np.array(adv_buffer)

        return returns, advantages

    def ppo_update(self, trajectories: List[Dict]):
        """Perform PPO policy update"""
        self.model.train()
        total_loss = 0

        # Flatten all trajectories into training batches
        all_states, all_actions, all_log_probs_old, all_returns, all_advantages = [], [], [], [], []

        for traj_data in trajectories:
            traj = traj_data['trajectory']
            returns = traj_data['returns']
            advantages = traj_data['advantages']

            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_log_probs_old.extend(traj['log_probs'])
            all_returns.extend(returns)
            all_advantages.extend(advantages)

        # Convert to tensors
        states = self._prepare_model_input(all_states)
        actions = torch.tensor(all_actions, dtype=torch.long).to(self.model.device)
        old_log_probs = torch.tensor(all_log_probs_old, dtype=torch.float).to(self.model.device)
        returns = torch.tensor(all_returns, dtype=torch.float).to(self.model.device)
        advantages = torch.tensor(all_advantages, dtype=torch.float).to(self.model.device)

        # PPO update epochs
        for _ in range(self.ppo_epochs):
            # Shuffle for mini-batch updates
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.batch_size // self.num_mini_batches):
                end_idx = min(start_idx + self.batch_size // self.num_mini_batches, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Get current policy and values
                logits, values = self._get_policy_and_value(batch_states)

                # Policy loss (PPO clipped objective)
                action_dist = Categorical(logits=logits[torch.arange(len(batch_states)), -1])
                new_log_probs = action_dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus (exploration)
                entropy_loss = -action_dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                total_loss += loss.item()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        return total_loss / (self.ppo_epochs * len(trajectories))

    def train_rl_epoch(self, tasks: List[Dict], num_rollouts: int = 32) -> Dict:
        """Train one RL epoch: generate rollouts + PPO update"""
        trajectories = []

        print(f"⬜ Generating {num_rollouts} rollouts...")
        for i in range(num_rollouts):
            task = np.random.choice(tasks)
            rollout = self.rollout_trajectory(task)
            trajectories.append(rollout)

        print("📈 Computing advantages and updating policy...")
        loss = self.ppo_update(trajectories)

        # Update statistics
        episode_rewards = []
        for traj in trajectories:
            total_reward = sum(traj['trajectory']['rewards'])
            episode_rewards.append(total_reward)

        avg_reward = np.mean(episode_rewards)
        self.stats['rewards'].append(avg_reward)
        self.stats['losses'].append(loss)
        self.stats['episodes'] += len(trajectories)

        return {
            'avg_reward': avg_reward,
            'loss': loss,
            'num_trajectories': len(trajectories)
        }

    def integrate_with_pipeline(self, model_path: str, tasks: List[Dict], rl_epochs: int = 10):
        """Convenience method to integrate RL into training pipeline"""
        print("🔗 INTEGRATING RL INTO HRM TRAINING PIPELINE")

        # Load base model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Base model not found: {model_path}")

        checkpoint = torch.load(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1
        self.model = LanguageAdaptedHRM_ACTV1(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

        # Initialize critic and optimizer
        self.critic = nn.Linear(checkpoint['config']['hidden_size'], 1).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-5},
            {'params': self.critic.parameters(), 'lr': 1e-4}
        ])

        # Reset stats
        self.stats = {'episodes': 0, 'rewards': [], 'losses': [], 'kl_divs': []}

        # Train RL epochs
        start_time = datetime.now()
        for epoch in range(rl_epochs):
            metrics = self.train_rl_epoch(tasks, num_rollouts=16)
            print(f"🎯 RL Epoch {epoch+1}/{rl_epochs} | Reward: {metrics['avg_reward']:+.4f} | Loss: {metrics['loss']:.4f}")

        training_time = datetime.now() - start_time

        # Save RL-enhanced model
        enhanced_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'rl_config': {'epochs': rl_epochs, 'vocab_size': checkpoint['vocab_size']},
            'rl_stats': self.stats,
            'training_time': str(training_time),
            'base_model_info': f"Inherited from {model_path}",
            **checkpoint  # Include original info
        }

        enhanced_path = model_path.replace('.pt', '_rl_enhanced.pt')
        torch.save(enhanced_checkpoint, enhanced_path)
        print(f"🚀 RL-Enhanced Model Saved: {enhanced_path}")

        return enhanced_checkpoint, enhanced_path


def create_red_teaming_environment():
    """Create challenging red teaming training scenarios"""

    scenarios = [
        {
            'description': 'Web application penetration testing',
            'input_tokens': tokenize_text('How would you perform web app penetration testing?', vocab),
            'expected_tools': ['burpsuite', 'sqlmap', 'nmap'],
            'secondary_tools': ['wireshark', 'metasploit'],
            'expected_attacks': ['sql_injection', 'xss', 'csrf'],
            'legal_constraints': ['authorized', 'bug_bounty', 'pentest'],
            'max_length': 150
        },
        {
            'description': 'Network reconnaissance and mapping',
            'input_tokens': tokenize_text('What tools and techniques for network reconnaissance?', vocab),
            'expected_tools': ['nmap', 'wireshark', 'maltego'],
            'secondary_tools': ['metasploit', 'burpsuite'],
            'expected_attacks': ['network_scanning', 'service_enumeration'],
            'legal_constraints': ['authorized', 'pentest'],
            'max_length': 120
        },
        {
            'description': 'Social engineering assessment',
            'input_tokens': tokenize_text('How to conduct social engineering for security testing?', vocab),
            'expected_tools': ['setoolkit', 'social_engineer_toolkit'],
            'secondary_tools': ['burpsuite', 'metasploit'],
            'expected_attacks': ['phishing', 'pretexting', 'baiting'],
            'legal_constraints': ['authorized', 'training_context', 'ethical'],
            'max_length': 140
        },
        {
            'description': 'Wireless network security',
            'input_tokens': tokenize_text('Wireless network pentesting methodology?', vocab),
            'expected_tools': ['aircrack', 'airodump', 'airmon'],
            'secondary_tools': ['ettercap', 'wireshark'],
            'expected_attacks': ['wep_cracking', 'wpa_handshake', 'deauth'],
            'legal_constraints': ['your_network', 'authorized'],
            'max_length': 130
        },
        {
            'description': 'Database security assessment',
            'input_tokens': tokenize_text('Database penetration testing guide?', vocab),
            'expected_tools': ['sqlmap', 'burpsuite', 'metasploit'],
            'secondary_tools': ['nmap', 'wireshark'],
            'expected_attacks': ['sql_injection', 'privilege_escalation'],
            'legal_constraints': ['authorized', 'security_testing'],
            'max_length': 135
        }
    ]

    return scenarios


# Global vocab reference for helper functions
vocab = torch.load('data/red_teaming/enhanced/vocab_cybersecurity.pt')

def tokenize_text(text: str, vocab: Dict) -> List[int]:
    """Tokenize text for HRM input"""
    tokens = [vocab.get('<user>', 4)]  # Start with user token

    for word in text.lower().split():
        word = word.strip('.,!?')
        if word:
            token_id = vocab.get(word, vocab.get('<sos>', 2))
            tokens.append(token_id)

    tokens.append(vocab.get('<eos>', 1))
    return tokens


if __name__ == "__main__":
    print("🎯 HRM RED TEAMING RL-FINE-TUNING")
    print("=" * 60)

    # Load model and vocabulary
    base_model_path = 'cybersecurity_hrm_model.pt'
    if os.path.exists(base_model_path):
        checkpoint = torch.load(base_model_path)
        model_config = checkpoint['config']
        print(f"📥 Loaded trained HRM model with {checkpoint['vocab_size']} tokens")
    else:
        print("❌ No trained model found. Run training first!")
        exit(1)

    # Create reward function and RL environment
    reward_fn = RedTeamingRewardFunction()
    training_scenarios = create_red_teaming_environment()

    print(f"🎮 Created {len(training_scenarios)} red teaming scenarios")

    # Initialize PPO trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from models.hrm.hrm_language_v1 import LanguageAdaptedHRM_ACTV1

    model = LanguageAdaptedHRM_ACTV1(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    ppo_trainer = PPOTrainer(model, checkpoint['vocab_size'], model_config)

    print("\n🚀 STARTING RL FINE-TUNING")
    print("Episode | Avg Reward | Loss     | Time")
    print("--------|------------|----------|-----")

    # Training loop
    start_time = datetime.now()
    for episode in range(20):  # Train for 20 episodes
        metrics = ppo_trainer.train_rl_epoch(training_scenarios, num_rollouts=16)
        elapsed = datetime.now() - start_time

        print("5d")

        # Save checkpoint every 5 episodes
        if episode % 5 == 0:
            rl_checkpoint = {
                'model_state_dict': model.state_dict(),
                'critic_state_dict': ppo_trainer.critic.state_dict(),
                'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                'rl_stats': ppo_trainer.stats,
                'episode': episode,
                'base_config': model_config,
                'vocab_size': checkpoint['vocab_size']
            }
            torch.save(rl_checkpoint, f'red_teaming_rl_checkpoint_{episode}.pt')
            print(f"💾 Saved RL checkpoint: red_teaming_rl_checkpoint_{episode}.pt")

    # Final model save
    final_rl_model = {
        'model_state_dict': model.state_dict(),
        'critic_state_dict': ppo_trainer.critic.state_dict(),
        'rl_stats': ppo_trainer.stats,
        'reward_function': 'RedTeamingRewardFunction',
        'training_scenarios': len(training_scenarios),
        'final_episode': len(ppo_trainer.stats['rewards']) - 1,
        'training_duration': str(datetime.now() - start_time)
    }
    torch.save(final_rl_model, 'red_teaming_rl_final_model.pt')

    print("\n✅ RL FINE-TUNING COMPLETED!")
    print(".4f")
    print(f"   Training Episodes: {ppo_trainer.stats['episodes']}")
    print("   Final Model Saved: red_teaming_rl_final_model.pt")
    print("\n🎉 HRM Model Now RL-Optimized for Expert Red Teaming!")
