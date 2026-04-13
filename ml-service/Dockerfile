# ml-service/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installa le dipendenze di sistema per OpenCV e healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installa PyTorch ottimizzato per CUDA (versione 12.1 come standard attuale)
# Separare questo layer velocizza le build successive se modifichi solo le altre dipendenze
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installa le dipendenze per l'API e YOLO
RUN pip install --no-cache-dir \
    "numpy<2" \
    fastapi \
    uvicorn \
    ultralytics \
    pillow \
    python-multipart

# Copia il codice sorgente
COPY main2.py .

# Esponi la porta interna
EXPOSE 8001

# Avvia l'applicazione (punta a main2.py)
CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8001"]
