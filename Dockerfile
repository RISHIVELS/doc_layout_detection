# CPU inference by default. The trained weights only need to run a forward
# pass to serve requests - no GPU required for that, and it means this image
# runs anywhere (a reviewer's laptop, a free-tier host) without needing CUDA
# drivers or a GPU-enabled base image, which is a real deployment obstacle
# a lot of ML demos skip past and then can't actually be run by anyone else.
FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow's image codecs and matplotlib's font rendering
# (scripts/_render_utils.py leans on matplotlib's bundled DejaVu font for
# legible labels - libgl/libglib cover headless image/font operations that
# a minimal slim image does not ship with by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Installing requirements before copying the rest of the source means this
# layer only gets invalidated when dependencies actually change, not on
# every code edit - meaningfully faster rebuilds while iterating.
COPY requirements.txt .

# CPU-only torch wheel. The default PyPI torch package pulls the full CUDA
# toolkit (multiple GB) even when nothing in this image ever touches a GPU -
# pointless bloat for a service that only does CPU inference.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/_render_utils.py scripts/__init__.py scripts/

# Weights are not baked into the image - see README for exactly why (they are
# too large for git, and rebuilding the image every time a checkpoint changes
# is wasteful). MODEL_PATH is mounted or downloaded at container start instead.
ENV MODEL_PATH=/app/weights/best.pt
ENV MAX_UPLOAD_MB=10

# Runs as a non-root user - no functional need for root here, and no reason
# to grant it when there is none.
RUN useradd --create-home appuser && mkdir -p /app/weights && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
