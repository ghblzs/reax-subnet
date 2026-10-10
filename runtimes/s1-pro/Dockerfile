# REAX s1-pro runtime sidecar. Same recipe as the reference Rune image (rune-vllm@sha256:b2b2ee1f...):
# vLLM 0.30.0 by digest + the reference decision adapter (unchanged) + chat template + warm-up.
# Build from the repository root:  docker build -f runtimes/s1-pro/Dockerfile -t reax-s1-pro-runtime .
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
RUN mkdir -p /opt/rune /opt/reax
COPY runtimes/s1-pro/rune_adapter.py runtimes/s1-pro/chat_template_nospace.jinja runtimes/s1-pro/start.sh runtimes/s1-pro/warmup.py /opt/rune/
COPY src/reax_subnet/manifests/s1-pro-v1.json /opt/reax/s1-pro-v1.json
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
COPY LICENSES /licenses/LICENSES/
RUN chmod +x /opt/rune/start.sh
ENV RUNE_DETERMINISTIC=1 RUNE_HOST=127.0.0.1 RUNE_PORT=8102 RUNE_MODEL_DIR=/model
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 CMD curl -sf "http://127.0.0.1:${RUNE_PORT}/health" || exit 1
ENTRYPOINT ["/opt/rune/start.sh"]
