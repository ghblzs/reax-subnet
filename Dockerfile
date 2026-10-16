# REAX miner/validator image. Build: docker build -t reax:1.2.0rc6 .   (default target = miner, the last stage)
# Targets: `--target miner` (entrypoint reax-miner), `--target validator` (entrypoint reax-validator) and
# `--target competition-worker` (reax-competition-worker + docker CLI, used by deploy/validator-1.2 and
# deploy/competition-worker) and `--target validator-competitions` (validator + docker CLI, REAX_COMPETITIONS_MODE=local).
# Both are the same code; the plain image also runs the validator with --entrypoint reax-validator.
# CI builds both on every push (.github/workflows/docker.yml); see docs/MINER.md and docs/VALIDATOR.md.
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app
# torch 2.14 runs some CUDA ops (e.g. Qwen3's rotary embedding) through Triton, which compiles a small
# driver module with a C compiler on first use. Without gcc the GPU path crashes (found on an RTX PRO 6000,
# 3 Oct 2026; CPU runs are unaffected).
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/lock.txt requirements/lock.txt
# Every dependency is pinned by hash.
RUN pip install --require-hashes -r requirements/lock.txt
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
COPY LICENSES /licenses/LICENSES/
COPY src ./src
RUN pip install --no-deps --no-build-isolation . \
    && useradd --uid 10001 --create-home reax \
    && mkdir -p /models /state \
    && chmod 1777 /models /state

# Runs as uid 10001 by default. Bittensor key files are mode 0600 and owned by your
# host user, so start the container with --user "$(id -u):$(id -g)" (see docs/MINER.md).
USER reax
# Wallet (hotkey only) is mounted read-only at /wallets; never bake keys into the image.
ENV REAX_WALLET_PATH=/wallets \
    REAX_STATE_PATH=/state/validator-state.json \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton XDG_CACHE_HOME=/tmp/.cache
# HOME and the caches point at /tmp so the GPU path (Triton compiles on first use) also works for the
# arbitrary --user "$(id -u):$(id -g)" the docs use; with no passwd entry Docker would set HOME=/ (not writable).

# Slim s1 miner for any architecture (linux/amd64 and linux/arm64, e.g. Jetson or ARM servers): mechanism s1 runs
# the model in its runtime sidecar, so this image needs no torch/CUDA. Hash-locked for all platforms
# (requirements/lock-s1.txt, same versions as lock.txt). Build: docker buildx build --platform linux/arm64
# --target miner-s1 -t reax-miner-s1 .   (tiered pools, unreleased: REAX_FEATURE_TIERED_POOLS on dev networks)
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016 AS miner-s1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements/lock-s1.txt requirements/lock-s1.txt
RUN pip install --require-hashes --only-binary=:all: -r requirements/lock-s1.txt
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY LICENSE THIRD_PARTY_NOTICES.md /licenses/
COPY LICENSES /licenses/LICENSES/
COPY src ./src
RUN pip install --no-deps --no-build-isolation . && useradd --uid 10001 --create-home reax && mkdir -p /state \
    && chmod 1777 /state
USER reax
ENV REAX_WALLET_PATH=/wallets HOME=/tmp
EXPOSE 8091
HEALTHCHECK --interval=60s --timeout=5s --start-period=120s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('REAX_MINER_PORT','8091'), timeout=4)" || exit 1
ENTRYPOINT ["reax-miner"]

FROM base AS validator
# The validator opens no port; watch its logs instead.
HEALTHCHECK NONE
ENTRYPOINT ["reax-validator"]

# The competition worker: runs ONLY the competition cycles (engine measurements and specialist evaluations start sibling
# containers through the host's docker socket). No wallet and no token is ever mounted into it. This is the only image
# that needs the docker CLI; deploy/validator-1.2/README.md explains the security implication.
# TODO(release): pin the docker CLI image by digest like the other base images (docker buildx imagetools inspect docker:27-cli).
FROM base AS competition-worker
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
HEALTHCHECK NONE
ENTRYPOINT ["reax-competition-worker"]

# Validator with the docker CLI, for REAX_COMPETITIONS_MODE=local (competitions inside the validator process, which then
# needs the docker socket and the hotkey in one place: not recommended on a host that serves customers).
FROM validator AS validator-competitions
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

FROM base AS miner
EXPOSE 8091
# Miner liveness.
HEALTHCHECK --interval=60s --timeout=5s --start-period=600s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('REAX_MINER_PORT','8091'), timeout=4)" || exit 1
ENTRYPOINT ["reax-miner"]

