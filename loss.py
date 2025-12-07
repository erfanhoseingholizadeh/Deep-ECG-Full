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
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for Imbalanced Classification
        Formula: Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        
        Args:
            alpha: Tensor of class weights (to handle absolute count imbalance).
            gamma: Focusing parameter. Higher gamma = Ignore easy examples more.
                   Gamma=0 is just standard Cross Entropy.
                   Gamma=2 is standard for Focal Loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculate Standard Cross Entropy (Log probabilities)
        # We use reduction='none' because we need to modify the loss per-sample first
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 2. Get the probability of the true class (p_t)
        # log_p_t = -ce_loss  ->  p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Calculate the Modulating Factor (1 - p_t)^gamma
        # This becomes 0 if the model is confident (p_t -> 1)
        focal_term = (1 - pt) ** self.gamma
        
        # 4. Combine
        loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss