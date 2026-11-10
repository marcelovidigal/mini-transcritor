# Mini Transcritor — modo servidor (web). O Ollama roda em outro contêiner (ver docker-compose.yml).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home app && mkdir -p /data && chown -R app /data
USER app

# /data: upload/ e download/ (temporários, recriados vazios a cada início) e config/ (settings).
ENV SERVER_DATA_DIR=/data HOST=0.0.0.0 PORT=8000 TRUST_PROXY=1 MINITRANSCRITOR_HOME=/data/config
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["python", "server.py"]
