# Deep-ECG-Full: Clinical-Grade Arrhythmia Detection 🫀⚡

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 📋 Executive Summary

This repository hosts an end-to-end Deep Learning pipeline designed to detect cardiac arrhythmias (specifically Premature Ventricular Contractions) from raw single-lead ECG signals. Unlike standard tutorials, this project solves specific **Real-World Engineering Challenges**: class imbalance, signal noise, inter-patient variability, and edge hardware constraints.

The system features a **Hybrid Dual-Stream Architecture** combining morphological (waveform) and temporal (rhythm) analysis, achieving **93% Clinical Accuracy** on the MIT-BIH Arrhythmia Database. It is containerized via Docker and optimized for embedded deployment (Raspberry Pi/ESP32) using Post-Training Quantization.

---

## 🏗️ Technical Architecture

To emulate the diagnostic process of a cardiologist, we designed a **Hybrid Neural Network** that processes two distinct data streams simultaneously:

### 1. Morphology Stream (1D-CNN)
* **Input:** 0.77s raw signal window (280 samples centered on R-peak).
* **Architecture:** 3-Layer **1D Convolutional Neural Network (CNN)** with Batch Normalization and Max Pooling.
* **Purpose:** Extracts high-level spatial features (e.g., QRS width, P-wave inversion).

### 2. Rhythm Stream (Dense Network)
* **Input:** RR-Intervals (Timing between beats).
* **Architecture:** Multi-Layer Perceptron (MLP).
* **Purpose:** Extracts temporal context (e.g., Tachycardia, Compensatory Pauses).

### 3. Fusion Head
* Features from both streams are concatenated and passed through a final Fully Connected classification head with **Dropout (0.3)** to prevent overfitting.
* **Decision Logic:** The model outputs logits for 5 AAMI classes. We use `torch.max` (Argmax) to determine the final prediction.

---

## 🔬 Engineering Decisions & Methodology

### 1. Data Engineering & DSP
**Dataset:** MIT-BIH Arrhythmia Database (PhysioNet).
* **Challenge:** Raw medical data is noisy and unstructured.
* **Solution:** We implemented a hybrid ingestion pipeline that supports both Online (automatic download via `wfdb`) and Offline (local disk) loading.
* **Signal Processing:**
    * **Filtering:** Applied a **Zero-Phase Butterworth Bandpass Filter (0.5Hz - 50Hz)**. This removes baseline wander (breathing) and powerline interference without phase-shifting the R-peak.
    * **Normalization:** Instance-level Z-Score normalization per window.

### 2. Handling Class Imbalance (Focal Loss)
**The Problem:** Normal beats (45,000) vastly outnumber Arrhythmias (900). Standard Cross-Entropy training causes "Model Collapse" (predicting only Normal).
**The Solution:** We implemented **Focal Loss** ($\gamma=1.0$).
* **Mechanism:** This custom loss function mathematically "down-weights" the easy examples (Normal beats) and forces the model to focus on hard, misclassified examples (Arrhythmias).

### 3. Deployment Modes: Diagnostic vs. Real-Time
The architecture supports two deployment configurations via toggle switches in `data_loader.py` and `model.py`, allowing a trade-off between **Latency** and **Accuracy**.

* **Mode 1: Diagnostic Mode (Default)**
    * **Logic:** Uses **Pre-RR** (past interval) AND **Post-RR** (future interval).
    * **Latency:** 1-beat lag (~0.8 seconds).
    * **Performance:** Higher Accuracy on PVCs due to detection of the "Compensatory Pause."
* **Mode 2: Real-Time Mode (Strictly Causal)**
    * **Logic:** Uses **Pre-RR** only. No future knowledge.
    * **Latency:** **0 ms (Instant)**.
    * **Performance:** Slightly lower sensitivity to ventricular ectopy, but suitable for wearable alerts.

### 4. Edge Optimization (Quantization)
To enable deployment on microcontrollers:
* **Technique:** Post-Training Dynamic Quantization.
* **Result:** Converted FP32 weights to **Int8**.
* **Compression:** Reduced model size from **4.7 MB** to **1.3 MB** (**3.64x reduction**) with <0.1% loss in accuracy.

---

## 📊 Visual Analysis & Results

### 1. Signal Processing Pipeline
Before training, raw ECG signals are processed to remove noise and baseline wander.
![Signal Processing](images/Live%20Model%20Prediction.png)
*Figure 1: Comparison of raw noisy signal (red) vs. cleaned signal (blue) after bandpass filtering.*

### 2. Model Performance (Hybrid CNN - Diagnostic Mode)
Our primary model achieves robust classification across major arrhythmia classes.

**Training Dynamics:**
![Learning Dynamics](images/Learning%20Dynamics%20(Focal%20Loss).png)
*Figure 2: Stable loss convergence using Focal Loss.*

**Clinical Accuracy:**
![Accuracy](images/Clinical%20Accuracy%20Improvement.png)
*Figure 3: Validation accuracy stabilizes >90% within 10 epochs.*

**Diagnostic Confusion Matrix:**
![Confusion Matrix](images/Diagnostic%20Confusion%20Matrix.png)
*Figure 4: Confusion Matrix confirms high sensitivity to Ventricular Ectopy (Class V).*

**Generalization Check:**
![ROC-AUC](images/Generalization%20(ROC-AUC).png)
*Figure 5: ROC-AUC of 0.98 indicates excellent class separation.*

### 3. Real-Time Predictions
The model provides instant classification on live data streams.
![Live Predictions](images/Live%20Model%20Prediction.png)
*Figure 6: Sample predictions on test data showing accurate detection of Normal beats.*

### 4. Comparative Analysis: Transformer Architecture
We benchmarked a Transformer-based architecture against our Hybrid CNN.
![Transformer Results](images/Screenshot%202025-12-07%20184525.png)
*Figure 7: Generalization check (ROC-AUC) for the Transformer model.*

**Benchmark Verdict:**
| Architecture | Accuracy | PVC Recall | Model Size (Int8) | Training Speed | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid CNN (Selected)** | **93%** | **89%** | **15 KB** | Fast (CPU optimized) | ✅ **Deployed** |
| Transformer (Self-Attention) | 93% | 82% | ~200 KB | Slow (GPU required) | ❌ Too Heavy |

---

## 🛠️ Usage

### Option A: Using Docker (Recommended)
This runs the entire pipeline in a sealed, reproducible container.

```bash
# 1. Build the container
docker build -t deep-ecg .

# 2. Run the pipeline
docker run deep-ecg

### Option 2: Local Python
To run the code directly on your machine (Linux/WSL/Mac):

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python main.py

Deep-ECG-Full/
│
├── main.py                     # Master pipeline (Train -> Eval -> Quantize)
├── config.py                   # Global hyperparameters & settings
├── data_loader.py              # Hybrid data ingestion & feature extraction
├── model.py                    # PyTorch Architecture (CNN + MLP)
├── loss.py                     # Custom Focal Loss implementation
├── experimental_transformer.py # Transformer Benchmark script
├── dsp_filters.py              # Signal processing utilities
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── images/                     # Visualization assets

License & Attribution
Software License
Copyright (c) 2023 Erfan Hoseingholizadeh. This software is distributed under the MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy of this software to deal in the Software without restriction.

Dataset Attribution
This project utilizes the MIT-BIH Arrhythmia Database provided by PhysioNet (ODC Attribution License).

Required Citations:

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation 101(23):e215-e220 (2000).