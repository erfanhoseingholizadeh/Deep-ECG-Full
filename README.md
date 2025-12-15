# 🫀 Deep-ECG: Clinical-Grade Arrhythmia Detection System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat\&logo=fastapi\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat\&logo=streamlit\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat\&logo=docker\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**Deep-ECG** is an **end-to-end, clinical-grade machine learning system** for detecting dangerous cardiac arrhythmias—specifically **Ventricular Ectopy (V)**—from **single-lead ECG signals**.

Unlike standard ML demos, this project is designed to **simulate a real production environment**, featuring:

* A deployment-ready **REST API**
* A **live interactive dashboard**
* **Edge-optimized inference** suitable for low-resource devices

---

## 📂 Project Structure

```text
Deep-ECG-Full/
├── api.py                 # FastAPI backend (REST API)
├── config.py              # Central configuration (hyperparameters, paths)
├── dashboard.py           # Streamlit frontend (interactive demo)
├── data_loader.py         # MIT-BIH downloader & signal preprocessing
├── Dockerfile             # Production container setup
├── inference.py           # Inference engine (preprocessing + prediction)
├── LICENSE                # MIT license
├── loss.py                # Custom weighted focal loss
├── main.py                # Training loop & validation engine
├── model.py               # Hybrid dual-stream NN (CNN + MLP)
├── requirements.txt       # Python dependencies
├── technical_report.md    # Detailed engineering documentation
└── tests/                 # Automated unit tests
```

---

## 🚀 Key Features

### 🧠 Hybrid Architecture

* **1D-CNN stream** for ECG morphology
* **Dense (MLP) stream** for rhythm-level features
* Feature fusion for holistic arrhythmia detection

### ⚖️ Class Imbalance Handling

* Custom **Weighted Focal Loss**
* Effectively addresses severe **1:15 class imbalance**

### ⚡ Edge Optimization

* **Post-training INT8 quantization**
* Final model size: **213 KB**

### 🏗️ Production-Ready Stack

* **FastAPI** backend for inference
* **Streamlit** frontend for real-time visualization
* Fully containerized with **Docker**

---

## 📊 Performance

**Evaluation Dataset:** MIT-BIH Arrhythmia Database (held-out test set)

|                Class | Precision |  Recall |
| -------------------: | :-------: | :-----: |
|           Normal (N) |    96%    |   99%   |
|      Ventricular (V) |    95%    | **90%** |
| Supraventricular (S) |    60%    |   16%   |

> **Clinical Note**
> The high **recall (90%)** for **Ventricular ectopy** demonstrates strong suitability for **screening dangerous ventricular events**, where missed detections are critical.

---

## 🛠️ Installation & Usage

### 1️⃣ Local Setup (Linux / macOS / WSL)

> ⚠️ Due to **PEP 668** on modern Linux distributions (e.g., Ubuntu 24.04+), a virtual environment is required.

```bash
# Clone the repository
git clone https://github.com/erfanhoseingholizadeh/Deep-ECG-Full.git
cd Deep-ECG-Full

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Training the Model

This step automatically downloads the **MIT-BIH dataset (~50 MB)** and starts training.

```bash
python main.py
```

---

### 3️⃣ Running the Demo

Launch the interactive dashboard to visualize predictions on real or synthetic ECG signals.

```bash
streamlit run dashboard.py
```

---

## 🐳 Docker Deployment

Run the entire system in an isolated container for **guaranteed reproducibility**.

```bash
# Build the image
docker build -t deep-ecg .

# Run the container
docker run -p 8000:8000 -p 8501:8501 deep-ecg
```

* **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Dashboard:** [http://localhost:8501](http://localhost:8501)

---

## 📜 License & Attribution

### 🧾 License

This project is licensed under the **MIT License**.

### 📚 Data Attribution

This project uses the **MIT-BIH Arrhythmia Database** provided by **PhysioNet**:

* Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol, 20(3):45–50 (2001).
* Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23):e215–e220 (2000).

---

⭐ If you find this project useful, consider giving it a star!
