# REAX s1-lite / s1-cpu runtime sidecar, CPU build (linux/amd64 and linux/arm64 from the same file).
# Build from the repository root:
#   docker build -f runtimes/s1-gguf/Dockerfile --build-arg MODEL=s1-cpu -t reax-s1-cpu-runtime .
#   docker build --platform linux/arm64 -f runtimes/s1-gguf/Dockerfile --build-arg MODEL=s1-cpu -t reax-s1-cpu-runtime:arm64 .
# Unreleased (REAX_FEATURE_TIERED_POOLS, development networks only). Base: multi-arch index digest (amd64 + arm64).
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update -qq && apt-get install -y -qq build-essential cmake git >/dev/null && rm -rf /var/lib/apt/lists/*
# Portable CPU code: no -march=native, so the same image runs on every x86-64-v3 (AVX2) or ARMv8.2 machine and
# gives the same numbers on all of them (see docs/DESIGN-POOLS.md section 6).
ARG CMAKE_ARGS_AMD64="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
ARG CMAKE_ARGS_ARM64="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
RUN set -e; case "$(uname -m)" in x86_64) A="$CMAKE_ARGS_AMD64";; aarch64) A="$CMAKE_ARGS_ARM64";; *) exit 1;; esac; \
    CMAKE_ARGS="$A" CMAKE_BUILD_PARALLEL_LEVEL=8 pip wheel --no-deps -w /wheels "llama-cpp-python==0.3.36"
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update -qq && apt-get install -y -qq curl libgomp1 >/dev/null && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install -q /wheels/*.whl "numpy==2.3.3" "fastapi==0.115.14" "uvicorn==0.29.0" && rm -rf /wheels
ARG MODEL=s1-cpu
RUN mkdir -p /opt/reax
COPY src/reax_subnet/manifests/${MODEL}-v1.json /opt/reax/manifest.json
COPY runtimes/s1-gguf/gguf_server.py runtimes/s1-gguf/readouts.py /opt/reax/
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
RUN useradd --uid 10002 --create-home s1m
USER s1m
ENV REAX_MANIFEST=/opt/reax/manifest.json GGUF_MODEL_DIR=/model GGUF_N_GPU_LAYERS=0 RUNTIME_HOST=127.0.0.1 \
    RUNTIME_PORT=8103 HOME=/tmp
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD curl -sf "http://127.0.0.1:${RUNTIME_PORT}/health" || exit 1
ENTRYPOINT ["python", "/opt/reax/gguf_server.py"]
