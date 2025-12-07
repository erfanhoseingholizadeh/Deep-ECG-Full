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
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
from data_loader import get_loaders
from loss import FocalLoss

# --- 1. REPRODUCIBILITY ---
def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. THE TRANSFORMER ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HeartTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        
        # Branch 1: Transformer for Waveform
        self.embedding = nn.Linear(1, self.d_model) 
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Branch 2: Rhythm Features (Dense Net)
        # ============================================================
        # [CONFIGURATION SWITCH] - Must match data_loader.py
        # ============================================================
        
        # MODE 1: DIAGNOSTIC (Default - 2 Features)
        self.rhythm_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # MODE 2: REAL-TIME (Strictly Causal - 1 Feature)
        # self.rhythm_net = nn.Sequential(
        #     nn.Linear(1, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.ReLU()
        # )
        # ============================================================
        
        # Fusion Head
        self.fc = nn.Linear(self.d_model + 32, 5) 

    def forward(self, wave, rhythm):
        # Wave Shape: [Batch, 1, 280] -> Permute to [Batch, 280, 1] for Transformer
        x_wave = wave.permute(0, 2, 1)
        
        # Transformer Stream
        x_wave = self.embedding(x_wave)
        x_wave = self.pos_encoder(x_wave)
        x_wave = self.transformer(x_wave)
        x_wave = x_wave.mean(dim=1) 
        
        # Rhythm Stream
        x_rhythm = self.rhythm_net(rhythm)
        
        # Concatenate
        combined = torch.cat((x_wave, x_rhythm), dim=1)
        return self.fc(combined)

# --- 3. HELPER FUNCTIONS ---
def get_all_predictions(model, loader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for waves, rhythms, labels in loader:
            waves = waves.to(config.DEVICE)
            rhythms = rhythms.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(waves, rhythms)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def plot_training_results(history, cm, classes, roc_data):
    plt.figure(figsize=(16, 12))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='navy')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title('Learning Dynamics (Transformer)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='purple')
    plt.title('Clinical Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')

    # ROC
    plt.subplot(2, 2, 4)
    plt.plot(roc_data['train_fpr'], roc_data['train_tpr'], color='blue', lw=2, label=f'Train AUC={roc_data["train_auc"]:.2f}')
    plt.plot(roc_data['test_fpr'], roc_data['test_tpr'], color='darkorange', lw=2, linestyle='--', label=f'Test AUC={roc_data["test_auc"]:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

# --- 4. TRAINING ENGINE ---
def train_engine():
    set_deterministic(config.SEED)
    print(f"--- STARTING TRANSFORMER EXPERIMENT (SEED={config.SEED}) ---")
    
    train_loader, test_loader, class_weights = get_loaders()
    model = HeartTransformer().to(config.DEVICE)
    print(f"Model initialized on {config.DEVICE}")
    
    # Use Focal Loss (Relaxed Gamma=1.0)
    criterion = FocalLoss(alpha=None, gamma=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Progress Bar for CPU sanity
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)
        
        for waves, rhythms, labels in pbar:
            waves, rhythms, labels = waves.to(config.DEVICE), rhythms.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(waves, rhythms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for waves, rhythms, labels in test_loader:
                waves, rhythms, labels = waves.to(config.DEVICE), rhythms.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(waves, rhythms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'transformer_model.pth')

    # Final Metrics
    print("\n--- TRANSFORMER RESULTS ---")
    model.load_state_dict(torch.load('transformer_model.pth'))
    y_train, y_train_probs = get_all_predictions(model, train_loader)
    y_test, y_test_probs = get_all_predictions(model, test_loader)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    
    class_names = ['N', 'S', 'V', 'F', 'Q']
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # ROC Stats
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    fpr_train, tpr_train, _ = roc_curve(y_train_bin.ravel(), y_train_probs.ravel())
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test_bin.ravel(), y_test_probs.ravel())
    roc_auc_test = auc(fpr_test, tpr_test)
    
    roc_data = {'train_fpr': fpr_train, 'train_tpr': tpr_train, 'train_auc': roc_auc_train,
                'test_fpr': fpr_test, 'test_tpr': tpr_test, 'test_auc': roc_auc_test}

    # COMPARISON METRIC: QUANTIZATION
    print("\n--- BENCHMARK: SIZE COMPARISON ---")
    model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), 'transformer_quantized.pth')
    
    size_fp32 = os.path.getsize('transformer_model.pth')
    size_int8 = os.path.getsize('transformer_quantized.pth')
    print(f"Transformer Original:  {size_fp32/1024:.2f} KB")
    print(f"Transformer Quantized: {size_int8/1024:.2f} KB")

    # Visualization
    print("\nGenerating Experiment Dashboard...")
    cm = confusion_matrix(y_test, y_test_pred)
    plot_training_results(history, cm, class_names, roc_data)

if __name__ == "__main__":
    train_engine()