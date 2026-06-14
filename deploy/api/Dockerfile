FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Evita que Python escriba archivos .pyc y fuerza a que la salida de consola se vea en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos dependencias del sistema necesarias para OpenCV y conversión de video
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt