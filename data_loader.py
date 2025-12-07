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

import wfdb
import numpy as np
import torch
import os
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import config

class MITBIHDataset(Dataset):
    def __init__(self, mode='train'):
        self.samples = []
        self.labels = []
        self.rhythm_features = []
        
        # Standard MIT-BIH Inter-Patient Split
        if mode == 'train':
            self.records = [
                '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
                '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', 
                '223', '230'
            ]
        else:
            self.records = [
                '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', 
                '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', 
                '233', '234'
            ]
            
        self._process_data()

    def _process_data(self):
        if not os.path.exists(config.DATA_DIR):
            print("Downloading MIT-BIH Database...")
            os.makedirs(config.DATA_DIR, exist_ok=True)
            wfdb.dl_database('mitdb', config.DATA_DIR)

        print(f"Processing {len(self.records)} records...")
        
        for record_name in tqdm(self.records):
            path = os.path.join(config.DATA_DIR, record_name)
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
            
            signal = record.p_signal[:, 0]
            signal = self._bandpass_filter(signal)
            
            for i in range(1, len(ann.sample) - 1):
                symbol = ann.symbol[i]
                if symbol not in config.AAMI_MAPPING:
                    continue
                
                peak = ann.sample[i]
                
                # A. Morphology
                left = peak - config.WINDOW_SIZE // 2
                right = peak + config.WINDOW_SIZE // 2
                
                if left < 0 or right > len(signal):
                    continue
                    
                wave = signal[left:right]
                # Z-Score Normalization
                wave = (wave - np.mean(wave)) / (np.std(wave) + 1e-6)
                
                # B. Rhythm Features
                prev_peak = ann.sample[i-1]
                next_peak = ann.sample[i+1]
                
                pre_rr = (peak - prev_peak) / config.FS
                post_rr = (next_peak - peak) / config.FS
                
                # --- [CONFIGURATION SWITCH] ---
                # MODE 1: DIAGNOSTIC (Default)
                # Uses future context (post_rr). Higher accuracy, 1-beat latency.
                self.rhythm_features.append([pre_rr, post_rr])
                
                # MODE 2: REAL-TIME (Strictly Causal)
                # Uncomment lines below for instant detection. (Must also update model.py!)
                # self.rhythm_features.append([pre_rr]) 
                # ------------------------------

                self.samples.append(wave)
                self.labels.append(config.AAMI_MAPPING[symbol])

    def _bandpass_filter(self, data):
        nyquist = 0.5 * config.FS
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.rhythm_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def get_loaders():
    train_ds = MITBIHDataset(mode='train')
    test_ds = MITBIHDataset(mode='test')
    
    # Calculate Class Weights (for info only, handled by Focal Loss dynamically)
    labels = train_ds.labels
    counts = np.bincount(labels)
    weights = 1. / (counts + 1e-6)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)
    
    print(f"Class Distribution: {counts}")
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, class_weights