# ==============================================================================
# Copyright (c) 2023 Erfan Hoseingholizadeh. All Rights Reserved.
# Deep-ECG-Full Container Configuration
# ==============================================================================

# 1. Base Image
FROM python:3.12-slim

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Source Code
COPY . /app

# 5. Expose Ports (Crucial for Documentation & Cloud Deployment)
# Port 8000 = FastAPI Backend
# Port 8501 = Streamlit Frontend
EXPOSE 8000
EXPOSE 8501

# 6. Default Command: Run the API Server
# (Override this with "streamlit run dashboard.py" to run the frontend)
CMD ["python", "api.py"]