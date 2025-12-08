# ==============================================================================
# Copyright (c) 2023 Erfan Hoseingholizadeh. All Rights Reserved.
# 
# This software is distributed under the MIT License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import config
from data_loader import get_loaders
from model import HybridECGNet
from loss import FocalLoss

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # 1. Learning Curve
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='navy')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title('Learning Dynamics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Accuracy Curve
    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.title('Clinical Accuracy Improvement')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Diagnostic Confusion Matrix')
    plt.ylabel('True Pathology')
    plt.xlabel('AI Prediction')

    # 4. ROC Curves
    plt.subplot(2, 2, 4)
    plt.plot(roc_data['train_fpr'], roc_data['train_tpr'], color='blue', lw=2, label=f'Train ROC (AUC = {roc_data["train_auc"]:.2f})')
    plt.plot(roc_data['test_fpr'], roc_data['test_tpr'], color='darkorange', lw=2, linestyle='--', label=f'Test ROC (AUC = {roc_data["test_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Generalization Check (ROC-AUC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/training_dashboard.png') 
    print("✅ Dashboard saved to images/training_dashboard.png")

def visualize_samples(model, loader, classes):
    model.eval()
    waves, rhythms, labels = next(iter(loader))
    indices = random.sample(range(len(labels)), 5)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        wave = waves[idx].cpu().numpy().flatten()
        label = labels[idx].item()
        with torch.no_grad():
            w_t = waves[idx].unsqueeze(0).to(config.DEVICE)
            r_t = rhythms[idx].unsqueeze(0).to(config.DEVICE)
            output = model(w_t, r_t)
            pred = torch.argmax(output, dim=1).item()
        
        color = 'green' if pred == label else 'red'
        plt.subplot(1, 5, i+1)
        plt.plot(wave, color='black', linewidth=1)
        plt.title(f"True: {classes[label]}\nPred: {classes[pred]}", color=color, fontweight='bold')
        plt.axis('off')
    plt.suptitle("Live Model Predictions (Green=Correct, Red=Error)")
    plt.savefig('images/sample_predictions.png')

def train_engine():
    set_deterministic(config.SEED)
    print(f"--- STARTING HYBRID ECG TRAINING (SEED={config.SEED}) ---")
    
    # 1. Load Data
    train_loader, test_loader, class_weights = get_loaders()
    model = HybridECGNet().to(config.DEVICE)
    print(f"Model initialized on {config.DEVICE}")
    
    # --- CORRECTION: REMOVE EXPLICIT WEIGHTS ---
    # We rely on Gamma=2.0 to handle hard examples dynamically.
    # Adding alpha=class_weights caused the gradient explosion.
    criterion = FocalLoss(alpha=None, gamma=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Keep the Scheduler (It helps convergence)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    os.makedirs('images', exist_ok=True)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for waves, rhythms, labels in train_loader:
            waves, rhythms, labels = waves.to(config.DEVICE), rhythms.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(waves, rhythms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
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
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"   └── Saved Best Model (Loss: {best_val_loss:.4f})")

    print("\n--- CALCULATING FINAL METRICS ---")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    y_train, y_train_probs = get_all_predictions(model, train_loader)
    y_test, y_test_probs = get_all_predictions(model, test_loader)
    
    y_test_pred = np.argmax(y_test_probs, axis=1)
    class_names = ['N', 'S', 'V', 'F', 'Q']
    # zero_division=0 handles the warning gracefully without crashing or spamming logs
    print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
    
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    fpr_train, tpr_train, _ = roc_curve(y_train_bin.ravel(), y_train_probs.ravel())
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test_bin.ravel(), y_test_probs.ravel())
    roc_auc_test = auc(fpr_test, tpr_test)
    
    roc_data = {'train_fpr': fpr_train, 'train_tpr': tpr_train, 'train_auc': roc_auc_train,
                'test_fpr': fpr_test, 'test_tpr': tpr_test, 'test_auc': roc_auc_test}

    print("\n--- OPTIMIZING FOR EDGE DEPLOYMENT ---")
    model.to('cpu')
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized_path = config.MODEL_SAVE_PATH.replace('.pth', '_quantized.pth')
    torch.save(quantized_model.state_dict(), quantized_path)
    size_fp32 = os.path.getsize(config.MODEL_SAVE_PATH)
    size_int8 = os.path.getsize(quantized_path)
    print(f"Original: {size_fp32/1024:.2f} KB | Quantized: {size_int8/1024:.2f} KB | Ratio: {size_fp32/size_int8:.2f}x")

    print("\nGenerating Dashboard...")
    cm = confusion_matrix(y_test, y_test_pred)
    plot_training_results(history, cm, class_names, roc_data)
    visualize_samples(model, test_loader, class_names)

if __name__ == "__main__":
    train_engine()