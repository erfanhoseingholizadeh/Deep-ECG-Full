# Deep-ECG: A Resilient Hybrid Architecture for Clinical-Grade Arrhythmia Detection

**Author:** Erfan Hoseingholizadeh
**Date:** December 2025
**Repository:** [github.com/erfanhoseingholizadeh/Deep-ECG-Full](https://github.com/erfanhoseingholizadeh/Deep-ECG-Full)

---

## 1. Abstract
Cardiac arrhythmia detection from single-lead ECG signals presents a dual challenge: extreme class imbalance (normal beats vastly outnumber pathological ones) and significant signal noise. Standard Deep Learning approaches often succumb to the "Accuracy Paradox," achieving high accuracy by ignoring minority classes entirely. This project introduces **Deep-ECG**, a deployment-ready system utilizing a **Hybrid Dual-Stream Neural Network**. By integrating morphological analysis (1D-CNN) with rhythm context (Dense Network) and employing a **Weighted Focal Loss** objective, the system achieves **95.1% accuracy** and, crucially, **90.0% recall** for Ventricular Ectopy on the MIT-BIH Arrhythmia Database. Furthermore, the model was optimized via Post-Training Dynamic Quantization, reducing the memory footprint by **600%** (to 213 KB), simulating feasibility for edge hardware (ESP32/Cortex-M).

---

## 2. Introduction
Cardiovascular diseases remain the leading cause of death globally. While 12-lead ECGs in clinical settings are the gold standard for diagnosis, the rise of wearable technology (smartwatches, Holter monitors) offers the potential for continuous monitoring. However, automating diagnosis on these devices is non-trivial.

The primary engineering bottleneck is **Data Imbalance**. In the standard MIT-BIH dataset, Normal (N) beats outnumber Ventricular (V) beats by a factor of 15:1. A naive model can achieve ~90% accuracy simply by predicting "Normal" for every heartbeat, rendering it clinically useless.

This project aims to engineer a solution that is:
1.  **Sensitive:** Prioritizing the detection of rare, dangerous arrhythmias.
2.  **Robust:** Capable of handling signal noise and variable input lengths.
3.  **Efficient:** Small enough to run on embedded processors without a GPU.

---

## 3. Methodology

### 3.1 Data Engineering & Signal Processing
Raw ECG signals are inherently noisy, containing baseline wander and powerline interference. Before entering the neural network, the signal passes through a rigorous Digital Signal Processing (DSP) pipeline:

* **Bandpass Filtering:** A Zero-Phase Butterworth filter (0.5Hz – 50Hz) isolates the cardiac frequencies, removing low-frequency wander and high-frequency noise without altering the phase (timing) of the R-peak.
* **Z-Score Normalization:** Each heartbeat window is normalized to have a mean of 0 and a standard deviation of 1. This ensures the neural network receives inputs on a consistent scale, independent of the recording device's gain.
* **Inter-Patient Splitting:** To prevent "data leakage," training and test sets are strictly separated by Patient ID.

### 3.2 The Hybrid Architecture
Cardiologists diagnose arrhythmia using two cues: the **shape** of the beat (Morphology) and the **timing** between beats (Rhythm). Deep-ECG mimics this via a dual-stream architecture:

1.  **Morphology Stream (1D-CNN):** A 3-layer Convolutional Neural Network extracts spatial features from the raw waveform (e.g., QRS width, T-wave inversion). Crucially, this stream utilizes **Adaptive Average Pooling (GAP)**. Unlike standard flattening, GAP forces the output to a fixed dimension regardless of input length, making the model robust to variable sampling rates.
2.  **Rhythm Stream (MLP):** A Dense Network processes the RR-intervals (time elapsed since the previous beat). This captures temporal anomalies like "Compensatory Pauses" that are characteristic of Premature Ventricular Contractions (PVCs).
3.  **Fusion Head:** The feature vectors from both streams are concatenated and passed through a final classification layer with Dropout (0.3).

### 3.3 The Mathematical Fix: Weighted Focal Loss
Standard Cross-Entropy Loss treats all errors equally. To counter the 15:1 imbalance, we implemented **Weighted Focal Loss**:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

* **Focusing Parameter ($\gamma=2.0$):** This term reduces the loss contribution from "easy" examples (confident Normal predictions) to zero, forcing the model to focus entirely on "hard," misclassified examples (Arrhythmias).
* **Result:** This mathematical intervention prevented model collapse and raised the recall for the minority Class V from <10% (baseline) to 90%.

---

## 4. Experimental Results

### 4.1 Diagnostic Performance
The model was evaluated on the unseen test set of the MIT-BIH database.

| Class | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- |
| **N (Normal)** | 0.96 | 0.99 | 0.97 |
| **S (Supraventricular)** | 0.60 | 0.16 | 0.26 |
| **V (Ventricular)** | **0.95** | **0.90** | **0.92** |

**Analysis:** The system excels at detecting **Ventricular Ectopy (Class V)**, identifying 90% of these dangerous beats with high precision.

### 4.2 Edge Optimization (Quantization)
To verify deployment feasibility, the trained FP32 model was compressed using Dynamic Quantization to Int8 (8-bit integer).

| Metric | Original Model (FP32) | Quantized Model (Int8) | Improvement |
| :--- | :--- | :--- | :--- |
| **File Size** | 1.31 MB | **0.21 MB** | **6.1x Smaller** |
| **Accuracy Loss** | - | < 0.2% | Negligible |

---

## 5. System Implementation

Beyond the core model, the project was engineered as a complete microservice architecture to ensure reproducibility and scalability.

### 5.1 The Tech Stack
* **REST API:** Developed using **FastAPI**, serving predictions via JSON with strict Pydantic data validation.
* **Containerization:** The entire pipeline (Training $\to$ Inference $\to$ API) is encapsulated in **Docker**, guaranteeing identical performance across development (Windows WSL2) and production environments.
* **CI/CD:** Automated testing suite (`pytest`) verifies model integrity and API health before deployment.

---

## 6. Conclusion
Deep-ECG successfully bridges the gap between theoretical deep learning and practical medical engineering. By addressing the specific challenges of data imbalance and hardware constraints, we delivered a system that is not only accurate (95.1%) but also clinically sensitive (90% Recall on Class V) and computationally efficient (213 KB).