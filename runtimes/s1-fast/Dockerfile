# REAX s1-fast runtime sidecar. Same base digest and pins as the reference fast engine;
# serves the reference batch-1 CUDA-graph readout on loopback.
# Build from the repository root:  docker build -f runtimes/s1-fast/Dockerfile -t reax-s1-fast-runtime .
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -q --target /tmp/pip26 "pip==26.2.1" \
 && PYTHONPATH=/tmp/pip26 python -m pip install -q --uploaded-prior-to=2026-09-25T04:21:31Z \
      "jevk5[fast] @ https://github.com/allebee/jevk5/archive/85238d7be5527370c43206fe54cd752eb3134c1b.tar.gz#sha256=1afc1b863dc7535e173e434b64ed71ca1b29acd664794403f187e720321a467f" \
      "transformers==5.17.0" "accelerate==1.15.0" "huggingface-hub==1.33.0" \
      "flash-linear-attention==0.5.2" "einops==0.8.2" \
 && rm -rf /tmp/pip26
# Same upload-date bound as above, so transitive dependencies resolve the same way on every rebuild
# (the transitive resolution is date-bounded but is not hash-locked; see docs/THREAT-MODEL.md).
RUN python -m pip install -q --target /tmp/pip26 "pip==26.2.1" \
 && PYTHONPATH=/tmp/pip26 python -m pip install --no-cache-dir --uploaded-prior-to=2026-09-25T04:21:31Z \
      "fastapi==0.115.14" "uvicorn==0.29.0" \
 && rm -rf /tmp/pip26
RUN mkdir -p /opt/reax
COPY src/reax_subnet/manifests/s1-fast-v1.json /opt/reax/s1-fast-v1.json
COPY runtimes/s1-fast/plumb_server.py /opt/reax/plumb_server.py
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
COPY LICENSES /licenses/LICENSES/
RUN useradd --uid 10002 --create-home s1m
USER s1m
# Writable caches for any uid (operators run the sidecar with their own uid to read a 0600 token file).
ENV PLUMB_MODEL_DIR=/model RUNTIME_HOST=127.0.0.1 RUNTIME_PORT=8101 HOME=/tmp TRITON_CACHE_DIR=/tmp/triton \
    XDG_CACHE_HOME=/tmp/.cache
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD curl -sf "http://127.0.0.1:${RUNTIME_PORT}/health" || exit 1
ENTRYPOINT ["python", "/opt/reax/plumb_server.py"]
