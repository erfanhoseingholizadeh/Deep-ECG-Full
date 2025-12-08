# Deep-ECG-Full: Clinical-Grade Arrhythmia Detection System 🫀⚡

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

## 📋 Executive Summary

**Deep-ECG-Full** is an end-to-end Machine Learning Engineering pipeline designed to detect cardiac arrhythmias (specifically Ventricular Ectopy) from raw single-lead ECG signals.

Unlike standard data science projects, this repository implements a complete **Production Lifecycle**:
1.  **Robust Training:** Solves extreme class imbalance (1:200) using Weighted Focal Loss.
2.  **Deployment Ready:** Features a quantized inference engine (200KB model size).
3.  **Full-Stack Interface:** Includes a **REST API** (FastAPI) for integration and a **Live Dashboard** (Streamlit) for visualization.

---

## 🏗️ System Architecture

### 1. The Model (Hybrid CNN)
A dual-stream neural network processing **Morphology** (Waveform shape) and **Rhythm** (Inter-beat timing).
* **Robustness:** Implements **Adaptive Average Pooling** to handle variable signal lengths.
* **Optimization:** Dynamic Quantization reduces model size by **6x** compared to standard FP32 implementations.

### 2. Configuration Modes
Controlled via `config.py` to support different clinical use cases:
* **Diagnostic Mode:** Uses future context (1-beat lag) for maximum accuracy.
* **Real-Time Mode:** Strictly causal (past data only) for instant wearable alerts.

---

## 🛠️ Usage Guide

### 1. Installation
```bash
git clone [https://github.com/erfanhoseingholizadeh/Deep-ECG-Full.git](https://github.com/erfanhoseingholizadeh/Deep-ECG-Full.git)
cd Deep-ECG-Full
pip install -r requirements.txt

Training the Brain
Downloads MIT-BIH data, processes signals, and trains the model.
python3 main.py
Output: Saves hybrid_ecg_model.pth and generates performance charts in images/.

utomated Testing
Verifies the API, Model, and Config logic before deployment.
pytest

Running the Application
Option A: The API (Backend)
Start the REST API to serve predictions via JSON.
python api.py
Docs: Open http://localhost:8000/docs to test the API interactively.

Option B: The Dashboard (Frontend)
Launch the interactive web interface.
streamlit run dashboard.py

Features: Upload CSV files, generate synthetic hearts, and visualize predictions in real-time.

To run the entire suite in an isolated container:
# 1. Build the image
docker build -t deep-ecg .

# 2. Run the container (Exposing API & Streamlit ports)
docker run -p 8000:8000 -p 8501:8501 deep-ecg

Metric,Result,Notes
Accuracy,95.1%,Weighted average across all classes
PVC Recall,90.0%,Sensitivity to Ventricular Ectopy (Class V)
Inference Time,<15ms,On standard CPU (Quantized)
Model Size,213 KB,Fits on embedded edge devices

License & Attribution
MIT License - Copyright (c) 2023 Erfan Hoseingholizadeh.

Data Source: MIT-BIH Arrhythmia Database (PhysioNet).

### 3. Quick Sanity Check for `Dockerfile`
Since you are about to build the container, you need to update your `Dockerfile` as well. Your old one only ran `main.py`. A production Dockerfile needs to expose ports for your API and Dashboard.

**Action:** Overwrite `Dockerfile` with this:

```dockerfile
# 1. Base Image
FROM python:3.9-slim

# 2. Environment Setup
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY . /app

# 5. Expose Ports (8000 for API, 8501 for Streamlit)
EXPOSE 8000
EXPOSE 8501

# 6. Default Command: Run the API Server
# (You can override this to run 'python main.py' or 'streamlit run dashboard.py')
CMD ["python", "api.py"]
```


## ⚖️ Data License & Attribution

The **MIT-BIH Arrhythmia Database** is provided by PhysioNet and is available under the **ODC Attribution License (ODC-By)**.

**Required Citations:**
If you use this software in research, please cite the original data source:

1.  **Goldberger, A., et al.** "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation* 101.23 (2000): e215-e220.
2.  **Moody, G. B., & Mark, R. G.** "The impact of the MIT-BIH Arrhythmia Database." *IEEE Engineering in Medicine and Biology Magazine* 20.3 (2001): 45-50.

**Link to Original Data:** https://physionet.org/content/mitdb/