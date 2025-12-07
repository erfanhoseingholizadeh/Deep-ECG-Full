# ==============================================================================
# Copyright (c) 2023 Erfan Hoseingholizadeh. All Rights Reserved.
# 
# This software is distributed under the MIT License.
# You are free to use, modify, and distribute this software, provided that
# the above copyright notice and this permission notice appear in all copies.
# 
# This project utilizes the MIT-BIH Arrhythmia Database (PhysioNet).
# See README.md for full data attribution and license details.
# ==============================================================================

import torch

# --- REPRODUCIBILITY ---
SEED = 100

# --- PATHS ---
DATA_DIR = 'mit_db_raw'
MODEL_SAVE_PATH = 'hybrid_ecg_model.pth'

# --- SIGNAL PROCESSING ---
FS = 360
WINDOW_SIZE = 280  # ~0.77 seconds

# --- AAMI STANDARD MAPPING ---
# 0:N, 1:S, 2:V, 3:F, 4:Q
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

# --- HYPERPARAMETERS ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'