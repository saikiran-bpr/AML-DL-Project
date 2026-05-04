"""Shared configuration for Phase 3 hybrid experiments."""
import os
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'Phase 1', 'data'))
RESULTS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'results'))
MODELS_DIR = os.path.normpath(os.path.join(_HERE, '..', 'models'))
FIGURES_DIR = os.path.normpath(os.path.join(_HERE, '..', 'figures'))
PHASE1_RESULTS = os.path.normpath(os.path.join(_HERE, '..', '..', 'Phase 1', 'results'))
PHASE2_RESULTS = os.path.normpath(os.path.join(_HERE, '..', '..', 'Phase 2', 'results'))

for d in (RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Verify the chosen accelerator actually works on this machine.
try:
    _t = torch.randn(2, 2, device=DEVICE)
    _ = _t + _t
except Exception:
    DEVICE = torch.device('cpu')

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

ATTACK_MAPPING = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'snmpgetattack': 'R2L', 'named': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'sendmail': 'R2L', 'httptunnel': 'R2L', 'worm': 'R2L', 'snmpguess': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'xterm': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
}

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
DROP_COLS = ['label', 'difficulty_level', 'attack_cat', 'binary_label']

# Multi-class label order used everywhere a confusion matrix or report is printed.
ATTACK_CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
ATTACK_TO_IDX = {c: i for i, c in enumerate(ATTACK_CLASSES)}
