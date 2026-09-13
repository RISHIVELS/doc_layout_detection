# CPU inference by default - a forward pass doesn't need a GPU, and this
# way the image runs anywhere without CUDA drivers.
FROM python:3.11-slim

WORKDIR /app

# libgl/libglib for Pillow + matplotlib's font rendering (used for
# legible label renders in scripts/_render_utils.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# requirements first so this layer only rebuilds when deps actually change
COPY requirements.txt .

# CPU-only torch wheel - the default PyPI package pulls the full CUDA
# toolkit (multiple GB) for an image that never touches a GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/_render_utils.py scripts/__init__.py scripts/

# Weights aren't baked into the image (too large for git, see README for
# the download link) - mounted or fetched at container start instead
ENV MODEL_PATH=/app/weights/best.pt
ENV MAX_UPLOAD_MB=10

RUN useradd --create-home appuser && mkdir -p /app/weights && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
