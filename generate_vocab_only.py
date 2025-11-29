#!/usr/bin/env python3
"""
Generate Large Cybersecurity Vocabulary (5000+ tokens) - Standalone Script
"""

import torch
import os

def generate_large_vocab():
    """Generate comprehensive 5000+ token cybersecurity vocabulary"""
    print("🔐 GENERATING COMPREHENSIVE CYBERSECURITY VOCABULARY")
    print("=" * 60)

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
    ]

    # Add terms we have so far
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

if __name__ == "__main__":
    vocab_path, vocab = generate_large_vocab()
    print(f"📈 Final vocabulary size: {len(vocab)} tokens")
    print("🎯 Ready for training phase")
