FROM python:3.11-slim

WORKDIR /app

# build-essential por si alguna dependencia no trae wheel precompilado para la imagen slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# La app corre sin privilegios de root; solo necesita escribir en storage/.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p storage/datasets storage/models \
    && chown -R appuser:appuser storage
USER appuser

EXPOSE 8000

# Railway (y la mayoría de los PaaS) asigna el puerto en la variable PORT; en local cae a 8000.
# Forma shell (sh -c) para que ${PORT} se expanda; exec para que uvicorn reciba las señales.
# La IP real detrás de un proxy se resuelve con TRUSTED_PROXY_HOPS (ver rate_limit.py). Por eso
# se apaga el manejo de proxies de uvicorn, que le cree al cliente lo que escriba en X-Forwarded-For.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-proxy-headers"]
