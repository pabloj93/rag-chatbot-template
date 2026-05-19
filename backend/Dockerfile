FROM python:3.11-slim

WORKDIR /app

# System deps required by some Python wheels (e.g. lxml).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only BEFORE requirements.txt.
# Why: sentence-transformers depends on torch. On Linux, pip defaults to the
# full CUDA build (~1.5 GB). The CPU-only wheel is ~200 MB and sufficient
# because we run embeddings on CPU (see embeddings.py, device="cpu").
RUN pip install --no-cache-dir \
    torch==2.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps. torch is already satisfied above, so pip skips it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last — most-frequently-changing layer goes at the end.
COPY app ./app

# .env is NOT copied here. In docker-compose, env vars arrive via the
# `env_file` directive and land in os.environ. pydantic-settings reads
# os.environ before the file, so the path resolution in config.py is
# only a fallback for local dev (uvicorn run outside Docker).
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
