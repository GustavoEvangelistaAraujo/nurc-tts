ARG BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${BASE}

# ---- Build/runtime environment -------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10
ENV PYTHONNOUSERSITE=1
# ^ Keep user-site OFF and install everything at build time as root.
#   (This is what made `pip install --user` fail at runtime.)

# Coqui/Trainer runtime settings
ENV COQUI_TOS_AGREED=1
ENV TRAINER_TELEMETRY=0
ENV TTS_TELEMETRY=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Hugging Face + cache settings (baked so `docker run` stays clean) ----
# Persistent HF cache on the mounted volume -> dataset downloads ONCE.
# 60s timeout survives the CDN read-timeout stalls.
ENV HF_HUB_DOWNLOAD_TIMEOUT=60
ENV HF_HOME=/workspace/data/hf-cache
# Writable caches so the non-root runtime UID doesn't crash on import
# (numba via librosa, matplotlib via the trainer's plotting).
ENV NUMBA_CACHE_DIR=/workspace/data/numba-cache
ENV MPLCONFIGDIR=/workspace/data/mpl-cache

# ---- System dependencies --------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-wheel \
        espeak-ng \
        libsndfile1-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Binary dependency used by the TTS/numba stack
RUN pip3 install --no-cache-dir llvmlite --prefer-binary

# ---- PyTorch (pinned; do NOT let later installs upgrade it) ---------------
# CUDA 12.1 wheels run fine on the newer 12.x host driver. Newer torch can
# break the old Coqui sampler classes, so 2.1.2 is intentional.
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        pip3 install --no-cache-dir \
            torch==2.1.2 \
            torchaudio==2.1.2 \
            --index-url https://download.pytorch.org/whl/cu121 \
            --prefer-binary && break; \
        echo "PyTorch install failed. Retry ${attempt}/5..."; \
        [ "$attempt" -eq 5 ] && { echo "Giving up after 5 attempts."; exit 1; }; \
        sleep 60; \
    done

# Verify the pinned build survived resolution (fail fast if not).
RUN python3 - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)
assert torch.__version__.startswith("2.1.2"), f"Expected torch 2.1.2, got {torch.__version__}"
assert torch.version.cuda == "12.1", f"Expected CUDA 12.1 build, got {torch.version.cuda}"
PY

# Freeze torch so downstream resolution can't bump it.
RUN pip3 freeze | grep -E '^(torch|torchaudio)==' > /tmp/torch-constraints.txt && \
    cat /tmp/torch-constraints.txt

# ---- Project source -------------------------------------------------------
WORKDIR /opt/TTS
COPY . /opt/TTS
RUN chmod -R a+rX /opt/TTS

# ---- Dataset / runtime Python deps ----------------------------------------
# Baked here so the launch script needs NO runtime pip step.
RUN pip3 install --no-cache-dir \
        datasets==2.19.2 \
        dill==0.3.8 \
        multiprocess==0.70.16 \
        pyarrow \
        pyarrow-hotfix \
        xxhash \
        soundfile \
        tqdm==4.64.1 \
        --constraint /tmp/torch-constraints.txt \
        --prefer-binary

# Install the local TTS package (pulls in `trainer`) without upgrading torch.
RUN pip3 install --no-cache-dir . \
        --constraint /tmp/torch-constraints.txt \
        --prefer-binary

# ---- Final safety check ---------------------------------------------------
# Pre-create baked cache dirs so importing TTS (-> librosa -> numba) has a
# writable target during THIS build step, before any volume is mounted.
RUN mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME"
RUN python3 - <<'PY'
import sys
from pathlib import Path
import torch, TTS

print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| available:", torch.cuda.is_available())
print("TTS:", Path(TTS.__file__).resolve())
assert torch.__version__.startswith("2.1.2"), f"Expected torch 2.1.2, got {torch.__version__}"
assert torch.version.cuda == "12.1", f"Expected torch CUDA 12.1 build, got {torch.version.cuda}"
PY

WORKDIR /workspace
ENTRYPOINT ["tts"]
CMD ["--help"]