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

# 1. Base Image
FROM python:3.9-slim

# 2. Environment Variables
# Prevents Python from buffering stdout so you see logs immediately
ENV PYTHONUNBUFFERED=1

# 3. Workspace
WORKDIR /app

# 4. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code
COPY . /app

# 6. Run the Training Engine
CMD ["python", "main.py"]