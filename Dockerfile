FROM python:3.11-slim

WORKDIR /app

# build-essential por si alguna dependencia no trae wheel precompilado para la imagen slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage/datasets storage/models

EXPOSE 8000

# Railway (y la mayoría de los PaaS) asigna el puerto en la variable PORT; en local cae a 8000.
# Forma shell (sh -c) para que ${PORT} se expanda; exec para que uvicorn reciba las señales.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
