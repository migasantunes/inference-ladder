# Container image for the inference-ladder HTTP service (rung 06).
#
# Build:   docker build -t inference-ladder .
# Run:     docker run --cpus=2 --memory=2g -p 8000:8000 \
#            -v ladder-models:/models inference-ladder
#
# By default the model is downloaded from Hugging Face on first start into
# /models (HF_HOME). Mount a volume there so the download happens only once.
# To bake the model into the image instead (no network needed at run time):
#   docker build --build-arg BAKE_MODEL=HuggingFaceTB/SmolLM2-135M -t inference-ladder .
#
# Choose the rung and model at run time with -e LADDER_RUNG=... -e LADDER_MODEL=...
# and optionally -e LADDER_THREADS=2.

FROM python:3.12-slim

# No .pyc files, unbuffered logs (so `docker logs` shows output immediately),
# and no pip cache kept in the image layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as an unprivileged user, never as root.
RUN useradd --create-home --uid 1000 ladder

WORKDIR /app

# Install CPU-only PyTorch first, from the PyTorch CPU wheel index. The default
# PyPI torch wheel bundles CUDA libraries and is several times larger. Because
# torch is then already installed, the next pip install does not fetch it again.
# This layer is cached as long as the index URL does not change.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.2"

# Copy only what the package needs to install. README.md is read by pyproject.toml.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# Hugging Face cache location. Downloaded models live here; mount a volume on /models
# to keep them across container restarts.
ENV HF_HOME=/models/hf
RUN mkdir -p /models && chown ladder:ladder /models

USER ladder

# Optional: download the model at build time so it ships inside the image.
# Empty by default, which skips this step. Note that a *bind* mount on /models at
# run time hides the baked copy; a new named volume is filled from it instead.
ARG BAKE_MODEL=""
RUN if [ -n "${BAKE_MODEL}" ]; then \
        python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "${BAKE_MODEL}"; \
    fi

# Defaults, overridable with -e at run time.
ENV LADDER_RUNG=01_torch_fp32 \
    LADDER_MODEL=HuggingFaceTB/SmolLM2-135M

EXPOSE 8000

# slim images have no curl, so the check uses Python. urlopen raises on a non-2xx
# status (the 503 before the model is loaded), which makes the check fail.
# start-period gives the first start time to download and load the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" || exit 1

# One uvicorn process: several workers would each load their own copy of the
# model and compete for the same CPUs. --factory calls create_app() at startup.
CMD ["uvicorn", "ladder.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
