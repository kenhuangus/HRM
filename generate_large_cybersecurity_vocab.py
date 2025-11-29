#!/usr/bin/env python3
"""
Generate Large Cybersecurity Vocabulary with 5000+ tokens
For HRM Red Teaming Model Expansion
"""

import torch
from collections import defaultdict
import json

def main():
    print("🔐 GENERATING COMPREHENSIVE CYBERSECURITY VOCABULARY")
    print("=" * 60)

    # Start with base vocab from enhanced version
    try:
        base_vocab = torch.load('data/red_teaming/enhanced/vocab_enhanced.pt')
        print("✅ Loaded base vocabulary: {} tokens".format(len(base_vocab)))
    except:
        print("⚠️ Base vocab not found, starting fresh")
        base_vocab = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<system>': 3, '<user>': 4, '<assistant>': 5}
        print("✅ Created minimal base vocabulary: {} tokens".format(len(base_vocab)))

    # Get the next available ID
    max_id = max(base_vocab.values()) + 1

    print("📚 Building comprehensive cybersecurity vocabulary...")

    # Extensive cybersecurity terms organized by categories
    cybersecurity_categories = {

        # Networking Fundamentals
        'networking_protocols': [
            'tcp', 'udp', 'icmp', 'ip', 'ipv4', 'ipv6', 'arp', 'dns', 'dhcp', 'ntp', 'snmp',
            'http', 'https', 'ftp', 'ftps', 'sftp', 'ssh', 'telnet', 'smtp', 'imap', 'pop3',
            'ldap', 'ldaps', 'kerberos', 'ntlm', 'radius', 'tacacs', 'diameter', 'bgp', 'ospf',
            'rip', 'eigrp', 'isis', 'mpls', 'vxlan', 'gre', 'ipsec', 'ike', 'ssl', 'tls',
            'tls1', 'tls1_1', 'tls1_2', 'tls1_3', 'dtls', 'srtp', 'zrtp', 'pptp', 'l2tp',
            'openvpn', 'wireguard', 'socks', 'socks4', 'socks5', 'proxy', 'reverse_proxy',
            'load_balancer', 'cdn', 'wan', 'lan', 'man', 'vlan', 'subnet', 'netmask', 'cidr',
            'mac_address', 'ipv6_address', 'domain_name', 'fqdn', 'hostname', 'port', 'socket'
        ],

        # Security Tools & Software
        'security_tools': [
            'nmap', 'metasploit', 'wireshark', 'burpsuite', 'sqlmap', 'dirbuster', 'gobuster',
            'john_ripper', 'hashcat', 'aircrack', 'airodump', 'reaver', 'pixie_dust', 'ettercap',
            'dsniff', 'tcpdump', 'tshark', 'snort', 'suricata', 'fail2ban', 'iptables',
            'firewalld', 'pf', 'cisco_asa', 'palo_alto', 'checkpoint', 'fortigate', 'sophos',
            'clamav', 'ossec', 'tripwire', 'aide', 'rkhunter', 'chkrootkit', 'lynis', 'openvas',
            'nessus', 'qualys', 'rapid7', 'acunetix', 'owasp_zap', 'nikto', 'dirb', 'wfuzz',
            'cewl', 'hydra', 'medusa', 'patator', 'ncrack', 'thc_hydra', 'cobalt_strike',
            'empire', 'powersploit', 'mimikatz', 'bloodhound', 'responder', 'evil_winrm',
            'crackmapexec', 'impacket', 'bloodhound', 'responder', 'evil_winrm', 'crackmapexec',
            'zerologon', 'printnightmare', 'nomad', 'petitpotam', 'ressource_based_constraint_delegation'
        ],

        # Attack Techniques & Vectors
        'attack_techniques': [
            'sql_injection', 'sqli', 'blind_sqli', 'time_based_sqli', 'error_based_sqli',
            'cross_site_scripting', 'xss', 'stored_xss', 'reflected_xss', 'dom_xss',
            'cross_site_request_forgery', 'csrf', 'xsrf', 'clickjacking', 'ui_redressing',
            'directory_traversal', 'path_traversal', 'lfi', 'rfi', 'file_inclusion',
            'command_injection', 'remote_command_execution', 'rce', 'code_execution',
            'buffer_overflow', 'heap_overflow', 'stack_overflow', 'integer_overflow',
            'use_after_free', 'double_free', 'format_string', 'race_condition', 'time_of_check_time_of_use',
            'man_in_the_middle', 'mitm', 'arp_poisoning', 'dns_poisoning', 'dhcp_spoofing',
            'session_hijacking', 'cookie_hijacking', 'tcp_hijacking', 'ssl_stripping',
            'phishing', 'spear_phishing', 'whaling', 'vishing', 'smishing', 'pharming',
            'malware', 'ransomware', 'trojan', 'virus', 'worm', 'rootkit', 'backdoor',
            'spyware', 'adware', 'scareware', 'cryptojacking', 'fileless_malware',
            'zero_day', 'zero_day_attack', 'day_zero', 'unknown_vulnerability',
            'brute_force', 'dictionary_attack', 'rainbow_table', 'password_cracking',
            'credential_stuffing', 'password_spray', 'pass_the_hash', 'golden_ticket',
            'pass_the_ticket', 'kerberoasting', 'asreproasting', 'dcsync', 'dcshadow',
            'social_engineering', 'pretexting', 'baiting', 'tailgating', 'shoulder_surfing',
            'dumpster_diving', 'reverse_social_engineering', 'honeypot', 'honeynet'
        ],

        # Defense Mechanisms
        'defense_mechanisms': [
            'firewall', 'next_generation_firewall', 'web_application_firewall', 'waf',
            'intrusion_detection_system', 'ids', 'intrusion_prevention_system', 'ips',
            'security_information_and_event_management', 'siem', 'log_analysis',
            'endpoint_detection_and_response', 'edr', 'extended_detection_and_response', 'xdr',
            'network_access_control', 'nac', 'data_loss_prevention', 'dlp', 'encryption',
            'multi_factor_authentication', 'mfa', 'two_factor_authentication', '2fa',
            'biometric_authentication', 'oauth', 'openid', 'saml', 'kerberos_authentication',
            'zero_trust', 'least_privilege', 'principle_least_privilege', 'defense_in_depth',
            'network_segmentation', 'micro_segmentation', 'air_gapping', 'sandboxing',
            'virtualization_security', 'container_security', 'cloud_security', 'devsecops',
            'threat_hunting', 'threat_intelligence', 'vulnerability_management', 'patch_management',
            'configuration_management', 'compliance_monitoring', 'gdpr', 'hipaa', 'pci_dss',
            'sox', 'iso_27001', 'nist', 'cmmc', 'cis_benchmarks', 'nist_csf', 'mitre_att&ck',
            'kill_chain', 'diamond_model', 'cyber_kill_chain', 'lockheed_martin', 'mandiant'
        ],

        # Cryptography & Encryption
        'cryptography': [
            'symmetric_encryption', 'asymmetric_encryption', 'public_key', 'private_key',
            'digital_signature', 'certificate_authority', 'ca', 'certificate', 'ssl_certificate',
            'hash_function', 'md5', 'sha1', 'sha256', 'sha512', 'bcrypt', 'scrypt', 'argon2',
            'aes', 'des', '3des', 'blowfish', 'twofish', 'camellia', 'serpent', 'idea',
            'rsa', 'dsa', 'ecdsa', 'ed25519', 'diffie_hellman', 'elliptic_curve_cryptography',
            'ecc', 'quantum_resistant', 'homomorphic_encryption', 'zkp', 'zero_knowledge_proof',
            'block_cipher', 'stream_cipher', 'one_time_pad', 'perfect_secrecy', 'salt',
            'nonce', 'initialization_vector', 'iv', 'key_derivation_function', 'kdf',
            'hmac', 'cmac', 'digital_envelope', 'key_exchange', 'perfect_forward_secrecy'
        ],

        # Web Security & Application Security
        'web_security': [
            'owasp_top10', 'broken_access_control', 'cryptographic_failures', 'injection',
            'insecure_design', 'security_misconfiguration', 'vulnerable_components',
            'identification_and_authentication_failures', 'software_integrity_failures',
            'security_logging_and_monitoring_failures', 'server_side_request_forgery', 'ssrf',
            'xml_external_entity', 'xxe', 'cors', 'cross_origin_resource_sharing',
            'content_security_policy', 'csp', 'x_frame_options', 'xfo', 'hsts',
            'http_strict_transport_security', 'same_origin_policy', 'sop', 'subresource_integrity',
            'sri', 'helmet', 'express_helmet', 'django_security', 'spring_security'
        ],

        # Cloud & Infrastructure Security
        'cloud_infrastructure_security': [
            'aws_security', 'azure_security', 'gcp_security', 'cloudtrail', 'cloudwatch',
            'vpc', 'subnet', 'security_group', 'nacl', 'iam', 'role', 'policy', 's3_bucket_policy',
            'cloudformation', 'terraform', 'kubernetes_security', 'docker_security', 'pod_security',
            'network_policy', 'rbac', 'service_account', 'secret_management', 'vault',
            'secret_management', 'key_management_service', 'kms', 'hsms', 'cloud_hsm',
            'load_balancer_security', 'cdn_security', 'waf', 'ddos_protection',
            'cloud_access_security_broker', 'casb', 'data_classification', 'data_sovereignty'
        ],

        # Red Teaming & Penetration Testing
        'red_teaming_pentesting': [
            'red_team', 'blue_team', 'purple_team', 'white_hat', 'ethical_hacker',
            'penetration_testing', 'pentest', 'black_box_testing', 'white_box_testing',
            'gray_box_testing', 'reconnaissance', 'footprinting', 'enumeration', 'scanning',
            'gaining_access', 'maintaining_access', 'covering_tracks', 'reporting',
            'exploit_development', 'weaponization', 'delivery', 'exploitation', 'installation',
            'command_and_control', 'actions_on_objectives', 'metasploit_framework',
            'exploit_db', 'securityfocus', 'packetstorm', 'exploit_pack', 'canvas',
            'immunity_canvas', 'core_impact', 'social_engineer_toolkit', 'set',
            'beef', 'browser_exploitation_framework', 'king_phisher', 'gophish', 'cobalt_strike',
            'empire', 'posh_c2', 'sliver', 'merlin', 'brute_ratel', 'havoc', 'mythic'
        ],

        # Threat Actors & Groups
        'threat_actors': [
            'apt', 'advanced_persistent_threat', 'nation_state', 'russia', 'china', 'iran', 'north_korea',
            'apt28', 'fancy_bear', 'apt29', 'cozy_bear', 'sandworm', 'apt41', 'barium', 'ke3chang',
            'apt38', 'lazarus_group', 'apt33', 'magnium', 'chaos', 'labyrinth_chollima',
            'apt1', 'comment_crew', 'shanghai_group', 'apt3', 'gothic_panda', 'ups_team',
            'apt5', 'scanbox', 'chinese_government', 'apt10', 'menu_pass', 'red_aprils',
            'apt12', 'numbered_panda', 'ixespionage', 'apt15', 'vixen_panda', 'keen_malware',
            'apt16', 'tg_0416', 'apt17', 'hidden_cobra', 'volgmer', 'apt18', 'dynamite_panda',
            'threat_group_0416', 'apt19', 'codoso', 'sunshop_group', 'apt20', 'drooping_duck',
            'apt21', 'apt22', 'apt23', 'apt24', 'apt25', 'apt26', 'apt27', 'apt28', 'apt29', 'apt30'
        ],

        # IoT & Embedded Security
        'iot_embedded_security': [
            'internet_of_things', 'iot', 'embedded_systems', 'firmware', 'field_programmable_gate_array',
            'fpga', 'microcontroller', 'arduino', 'raspberry_pi', 'esp32', 'embedded_linux',
            'rtos', 'real_time_operating_system', 'zigbee', 'zwave', 'bluetooth', 'bluetooth_low_energy',
            'ble', 'wifi', 'lora', 'sigfox', 'nb_iot', 'lte_m', 'gsm', '5g_security',
            'industrial_control_systems', 'ics', 'scada', 'plc', 'dcs', 'siemens',
            'modbus', 'profibus', 'profinet', 'opc_ua', 'bacnet', 'dnp3', 'industrial_internet_of_things',
            'iiot', 'smart_grid', 'medical_devices', 'iot_medical', 'pacemaker_hacking'
        ],

        # Mobile & Wireless Security
        'mobile_wireless_security': [
            'android_security', 'ios_security', 'mobile_application_security',
            'reverse_engineering', 'jadx', 'apktool', 'frida', 'drozer', 'mobsf',
            'owasp_mobile_top10', 'weak_server_side_controls', 'insecure_data_storage',
            'insufficient_transport_layer_protection', 'unintended_data_leakage',
            'poor_authorization_and_authentication', 'broken_cryptography',
            'client_side_injection', 'security_decisions_via_untrusted_inputs',
            'improper_session_handling', 'lack_of_binary_protections', 'wi_fi_security',
            'wpa2', 'wpa3', 'wep', 'evil_twin', 'rogue_access_point', 'deauthentication_attack',
            'krack', 'fragattack', 'dragonblood', 'bluesnarfing', 'bluebugging', 'bluejacking'
        ],

        # Privacy & Data Protection
        'privacy_data_protection': [
            'privacy_by_design', 'data_minimization', 'purpose_limitation', 'storage_limitation',
            'data_accuracy', 'accountability', 'transparency', 'consent', 'right_to_access',
            'right_to_rectification', 'right_to_erasure', 'right_to_data_portability',
            'right_to_object', 'automated_decision_making', 'profiling', 'data_breach',
            'breach_notification', 'data_protection_officer', 'dpo', 'privacy_impact_assessment',
            'pia', 'cookie_consent', 'tracking_pixel', 'web_beacon', 'fingerprinting',
            'canvas_fingerprinting', 'device_fingerprinting', 'behavioral_tracking',
            'cross_device_tracking', 'retention_policy', 'anonymization', 'pseudonymization'
        ],

        # Artificial Intelligence & Machine Learning Security
        'ai_ml_security': [
            'ai_security', 'ml_security', 'adversarial_machine_learning', 'adversarial_examples',
            'data_poisoning', 'model_poisoning', 'model_evasion', 'model_inversion',
            'membership_inference', 'attribute_inference', 'deepfake_detection', 'ai_ethics',
            'algorithmic_bias', 'explainable_ai', 'federated_learning_security', 'homomorphic_ml',
            'differential_privacy', 'secure_multi_party_computation', 'trusted_execution_environment',
            'tee', 'sgx', 'arm_trustzone', 'tee_security', 'secure_boot', 'measured_boot',
            'remote_attestation', 'confidential_computing', 'enclave', 'secure_enclave'
        ]
    }

    # Flatten all categories into one list
    all_cyber_terms = []
    for category, terms in cybersecurity_categories.items():
        all_cyber_terms.extend(terms)

    print(f"📊 Collected {len(all_cyber_terms)} cybersecurity terms from {len(cybersecurity_categories)} categories")

    # Add terms to vocabulary, avoiding duplicates
    added_count = 0
    for term in all_cyber_terms:
        if term not in base_vocab:
            base_vocab[term] = max_id
            max_id += 1
            added_count += 1

    print(f"✅ Added {added_count} new cybersecurity terms to vocabulary")

    # If still not enough, add more technical variations
    if len(base_vocab) < 5000:
        additional_terms = [
            # Technical variations and combinations
            'cyber_attack', 'cyber_defense', 'cyber_threat', 'cyber_intelligence',
            'digital_forensics', 'incident_response', 'threat_hunting', 'vulnerability_assessment',
            'risk_assessment', 'compliance_audit', 'security_policy', 'access_control',
            'authentication_system', 'authorization_system', 'accounting_system',
            'confidentiality_integrity_availability', 'cia_triad', 'park', 'non_repudiation',
            'availability', 'reliability', 'resilience', 'robustness', 'fault_tolerance',
            'penetration_test_report', 'vulnerability_scan', 'configuration_scan',
            'source_code_review', 'binary_analysis', 'reverse_engineering_tools',
            'exploit_code', 'proof_of_concept', 'poc', 'zero_day_vulnerability',
            'public_disclosure', 'responsible_disclosure', 'bug_bounty', 'vulnerability_reward',
            'certified_ethical_hacker', 'ceh', 'oscp', 'offensive_security', 'sans_institute',
            'cresta_certification', 'iso_certification', 'nist_framework', 'cve_database',
            'common_vulnerabilities_exposures', 'cve', 'cvss_score', 'severity_rating',
            'exploitability_index', 'impact_score', 'base_score', 'temporal_score', 'environmental_score'
        ]

        for term in additional_terms:
            if term not in base_vocab:
                base_vocab[term] = max_id
                max_id += 1
                added_count += 1

    final_vocab_size = len(base_vocab)
    print(f"📈 Final vocabulary size: {final_vocab_size} tokens")

    if final_vocab_size < 5000:
        print(f"⚠️ Target is 5000 tokens, currently have {final_vocab_size}. Adding generic technical terms...")

        # Add generic technical terms to reach 5000
        generic_terms = []
        common_prefixes = ['cyber', 'digital', 'network', 'security', 'system', 'data', 'access', 'threat', 'risk', 'vulnerable']
        suffixes = ['_analysis', '_control', '_monitoring', '_protection', '_detection', '_prevention', '_response', '_assessment', '_management']

        for prefix in common_prefixes:
            for suffix in suffixes:
                term = prefix + suffix
                if term not in base_vocab:
                    generic_terms.append(term)

        # Add enough to reach 5000
        needed = 5000 - len(base_vocab)
        for i in range(min(needed, len(generic_terms))):
            term = generic_terms[i]
            base_vocab[term] = max_id
            max_id += 1
            added_count += 1

        final_vocab_size = len(base_vocab)
        print(f"📈 Final vocabulary size after generic terms: {final_vocab_size} tokens")

    # Save the expanded vocabulary
    vocab_path = 'data/red_teaming/enhanced/vocab_cybersecurity_large.pt'
    torch.save(base_vocab, vocab_path)
    print(f"💾 Saved large cybersecurity vocabulary to: {vocab_path}")

    # Verify and show statistics
    print("\n📊 VOCABULARY STATISTICS:")
    print(f"Total tokens: {len(base_vocab)}")
    print(f"Base tokens (original): {len(base_vocab) - added_count}")
    print(f"Added cybersecurity tokens: {added_count}")
    print("")

    # Show some sample additions
    sample_terms = ['nmap', 'metasploit', 'sql_injection', 'xss', 'firewall', 'encryption', 'threat_hunting', 'zero_trust']
    print("🔍 SAMPLE CYBERSECURITY TERMS:")
    for term in sample_terms:
        if term in base_vocab:
            print(f"   '{term}' → ID {base_vocab[term]}")

    print("\n✅ LARGE CYBERSECURITY VOCABULARY GENERATION COMPLETE!")
    print("   Ready for HRM model training with comprehensive cyber terms")

    return vocab_path

if __name__ == "__main__":
    main()
