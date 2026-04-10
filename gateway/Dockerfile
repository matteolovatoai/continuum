# gateway/Dockerfile
FROM python:3.11-slim

# Evita la scrittura di file .pyc e forza l'output dei log senza buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installa solo le librerie strettamente necessarie per il gateway
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    websockets \
    python-multipart

# Copia il codice sorgente
COPY main.py .

# Esponi la porta (documentazione interna del container)
EXPOSE 8000

# Comando di avvio ottimizzato per la produzione
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]