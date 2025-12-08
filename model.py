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
import torch.nn as nn
import config

class HybridECGNet(nn.Module):
    def __init__(self):
        super(HybridECGNet, self).__init__()
        
        # BRANCH 1: MORPHOLOGY (CNN)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Forces output to (Batch, 128, 1) regardless of input length
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        
        self.cnn_out_dim = 128
        
        # BRANCH 2: RHYTHM (Dense Net)
        # Dynamically set input dimension based on config
        rhythm_input_dim = 2 if config.DIAGNOSTIC_MODE else 1
        
        self.rhythm_net = nn.Sequential(
            nn.Linear(rhythm_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # FUSION HEAD
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_out_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # 5 AAMI Classes
        )

    def forward(self, wave, rhythm):
        x1 = self.cnn(wave)
        x2 = self.rhythm_net(rhythm)
        combined = torch.cat((x1, x2), dim=1)
        out = self.fusion(combined)
        return out