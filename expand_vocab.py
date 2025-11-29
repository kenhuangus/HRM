#!/usr/bin/env python3
import torch
from collections import defaultdict

# Load existing enhanced vocabulary
vocab = torch.load('data/red_teaming/enhanced/vocab_enhanced.pt')
print(f"Original vocab size: {len(vocab)}")

# Extensively expand with cybersecurity and red teaming terms
cybersec_terms = [
    # Attack methods
    'hack', 'hacking', 'hacker', 'hackable', 'hacked', 'hacks',
    'exploit', 'exploits', 'exploiting', 'exploitable', 'exploited', 'exploitation',
    'vulnerability', 'vulnerabilities', 'vulnerable', 'vulnerabilistic',
    'attack', 'attacks', 'attacking', 'attacker', 'attackers', 'attacked',
    'injection', 'injections', 'sql_injection', 'command_injection', 'ldap_injection',
    'cross_site_scripting', 'xss', 'stored_xss', 'reflected_xss', 'dom_xss',
    'cross_site_request_forgery', 'csrf', 'xsrf',
    'denial_of_service', 'dos', 'ddos', 'distributed_denial_of_service',
    'buffer_overflow', 'heap_overflow', 'stack_overflow',
    'logic_bombs', 'trojan_horse', 'trojans', 'backdoors', 'backdoor',
    'keylogger', 'keyloggers', 'spyware', 'rootkit', 'rootkits',
    'malware', 'malicious_software', 'virus', 'viruses', 'worm', 'worms',
    'ransomware', 'crypto-ransomware', 'locker_ransomware',
    'phishing', 'spear_phishing', 'phishing_scams', 'smishing',
    'man-in-the-middle', 'mitm', 'mitm_attack',
    'session_hijacking', 'cookie_theft', 'manipulation',
    'zero-day', 'zero_days', 'unknown_vulnerability',
    'privilege_escalation', 'horizontal_escalation', 'vertical_escalation',
    'code_execution', 'remote_code_execution', 'arbitrary_code_execution',
    'data_tampering', 'tampering',
    'brute_force', 'brute_force_attack', 'dictionary_attack',
    'social_engineering', 'pretexting', 'baiting', 'quid_pro_quo',

    # Defense methods
    'firewall', 'firewalls', 'access_control_lists', 'intrusion_detection',
    'intrusion_prevention_system', 'encryption', 'multi_factor_authentication',
    'zero_trust', 'segmentation', 'network_segmentation',
    'incident_response', 'forensic_analysis', 'penetration_testing',
    'vulnerability_scanning', 'patch_management',
    'security_information_and_event_management', 'siem',

    # Red teaming specific
    'red_team', 'red_teaming', 'blue_team', 'purple_team',
    'adversarial_simulation', 'ethical_hacking', 'pentest',
    'exploit_development', 'proof_of_concept', 'security_assessment',
    'threat_modeling', 'risk_assessment',
    'command_and_control', 'c2', 'c_c',
    'payload', 'payloads', 'shellcode',
    'shell_access', 'reverse_shell', 'bind_shell',
    'persistence', 'persistent_malware', 'persistence_mechanisms',

    # Common security terms
    'authentication', 'authorization', 'confidentiality', 'integrity', 'availability',
    'threat', 'threats', 'threat_actor', 'attribution',
    'cybersecurity', 'cyber_security', 'infosec', 'it_security',
    'compliance', 'hipaa', 'gdpr', 'pci_dss', 'sox',
    'encryption_algorithms', 'aes', 'rsa', 'ecdsa', 'hashing',
    'biometrics', 'facial_recognition', 'retina_scanning'
]

# Find the highest existing ID
max_id = max(vocab.values()) + 1 if vocab else 100

# Add new terms to vocabulary
added_count = 0
for term in cybersec_terms:
    if term not in vocab:
        vocab[term] = max_id
        max_id += 1
        added_count += 1

# Save expanded vocabulary
torch.save(vocab, 'data/red_teaming/enhanced/vocab_expanded_cybersecurity.pt')
print(f"Expanded vocab saved with {len(vocab)} total tokens (added {added_count} cybersecurity terms)")

# Verify the expansion
loaded_vocab = torch.load('data/red_teaming/enhanced/vocab_expanded_cybersecurity.pt')
print(f"Verification: Loaded vocab size: {len(loaded_vocab)}")

# Test some key terms
test_terms = ['hack', 'exploit', 'backdoor', 'payload', 'red_team']
print("Verification of key terms:")
for term in test_terms:
    if term in loaded_vocab:
        print(f"✓ '{term}' -> ID {loaded_vocab[term]}")

print("Cybersecurity vocabulary expansion complete!")
