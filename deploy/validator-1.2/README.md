# Validator 1.2 on one 96 GB GPU

Release 1.2 turns the **s1-pro pool on for every network**, so a validator runs the reference runtime of both
pools: s1-fast (Plumb-4B, about 12 GB of VRAM in use) and s1-pro (Surogate Rune 26B-A4B, deterministic vLLM 0.30.0).
`compose.yml` here starts both next to each other on **one** 96 GB card (RTX PRO 6000) plus the validator. Both
runtimes listen on loopback only (`127.0.0.1:8101`, `127.0.0.1:8102`); never publish those ports.

| Service | Listens on | VRAM | Notes |
| --- | --- | --- | --- |
| `s1-fast-runtime` | 127.0.0.1:8101 | about 12 GB | reference fast runtime, as in 1.1; starts first |
| `s1-pro-runtime` | 127.0.0.1:8102 | 70 % of the card = about 67 GB | `RUNE_GPU_MEM_UTIL` (default 0.70), deterministic build, 52 GB bf16 weights + KV cache |
| `validator` | none | none | `reax-validator`, hotkey-only wallets mounted read-only |

From 16 October 2026, 12:00 UTC the validator also runs the **competitions** (engine champion and the two use-case
tracks, 50 % of the weights; see [docs/COMPETITIONS.md](../../docs/COMPETITIONS.md)). The competition measurements run
**one at a time** on the same GPU, as a background task: serving rounds go on at their normal interval. A validator
that is not set up for them (below) logs one warning and serves only, which means it sets 100 % inference weights and
therefore differs from validators that run the competitions.

About 15 GB stay free for driver overhead and a short engine-competition measurement of about 12 GB. If the
measurement needs more, lower `REAX_S1_PRO_GPU_MEM_UTIL` (for example 0.65); the numerics do not depend on it
(batch-invariant kernels), only the KV-cache size does. Run `reax-selfcheck --model s1-pro` after any change.

## Setup (from the repository root)

```bash
docker build -t reax:1.2.0rc1 .
docker compose -f deploy/validator-1.2/compose.yml build            # builds both runtimes
cp deploy/validator.env.example deploy/validator.env                # netuid, wallet names; keep REAX_SET_WEIGHTS=0 for the first hour
mkdir -p deploy/secrets && openssl rand -hex 32 > deploy/secrets/runtime-token && chmod 0600 deploy/secrets/runtime-token
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b
hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune   # gated: your own HF account
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b REAX_S1_PRO_WEIGHTS=/srv/reax/rune \
  docker compose -f deploy/validator-1.2/compose.yml up -d
```

### Competitions: extra setup

Needs three more things on the host. All paths are host paths; the compose file mounts them at the same path inside the
validator container.

| Variable | What | Size |
| --- | --- | --- |
| `REAX_S1_FAST_WEIGHTS` | the Plumb-4B directory from above; the engine competition measures with these weights (mounted read-only, also used as `REAX_ENGINE_WEIGHTS_DIR`) | 8 GB |
| `REAX_SPEC_SEALED_DIR` | the sealed item pools of the use-case tracks (`<track>/sealed/sealed.clean.jsonl`), **read-only**, obtained from the subnet owner | small |
| `REAX_COMPETITION_DIR` | writable: `engines/` (per-measurement sockets and logs) and `spec-work/` (downloaded miner models, up to ~20 GB each, kept only for champions and pending confirmations) | 200 GB free is plenty |
| `REAX_HF_CACHE` | writable Hugging Face metadata cache (`HF_HOME`) | 5 GB |
| `REAX_DOCKER_GID` | group id that may use the docker socket: `getent group docker \| cut -d: -f3` | |

```bash
docker build -t reax:1.2.0rc1 --target validator-competitions .      # compose builds it too
export REAX_SPEC_SEALED_DIR=/srv/reax/sealed REAX_COMPETITION_DIR=/srv/reax/competitions REAX_HF_CACHE=/srv/reax/hf
export REAX_DOCKER_GID=$(getent group docker | cut -d: -f3)
mkdir -p "$REAX_COMPETITION_DIR" "$REAX_HF_CACHE" && chown "$(id -u):$(id -g)" "$REAX_COMPETITION_DIR" "$REAX_HF_CACHE"
```

State lives in the `state` volume (`engine-competition.json`, `spec-state.json`); keep it across restarts (it holds the
champions and the pending confirmations). The reference engine image is a **release constant**; do not set
`REAX_ENGINE_REFERENCE` (finney refuses it). `REAX_SPEC_TRACKS_FILE` is ignored on finney: the track schedule is fixed.

**GPU memory.** An engine measurement needs two s1-fast engines (2 x 14 GB) free; a specialist evaluation needs up to
30 GB. Each cycle checks `nvidia-smi` first and, if the memory is short, skips cleanly (logged as "cycle skipped"),
without recording anything against any submission. With s1-pro at the default 70 % only ~15 GB are free, so on a
single card the measurements will mostly skip until `REAX_S1_PRO_GPU_MEM_UTIL` is lowered (about 0.45 leaves
~50 GB). Choose the value on your card with `nvidia-smi` and confirm with `reax-selfcheck --model s1-pro`.

**Missing setup.** If the release reference image is not published in this build, or `REAX_ENGINE_WEIGHTS_DIR` or
`REAX_SPEC_SEALED_DIR` is missing, the validator starts and serves only (one warning in the log at start).

### Security: what the docker socket means

The engine and the specialist evaluator run code and models that **miners chose**, in short-lived sibling containers
(`--network none`, read-only root, all capabilities dropped, `no-new-privileges`, one GPU, memory/pids limits,
weights mounted read-only). To start them, the validator talks to the host's docker daemon. **Access to
`/var/run/docker.sock` is equivalent to root on the host.** The validator code itself is not miner-controlled, but a
bug in it, or in the way it handles a miner's repository metadata, would then be a host compromise. Mitigations:

* Run this validator on a machine that holds **hotkey-only** wallets and nothing else of value (the compose file mounts
  no coldkey; never copy one there).
* Prefer **rootless docker** (or a dedicated user with its own daemon) for the validator host; then the socket is the
  socket of that user, not root. NVIDIA GPU support works with rootless docker.
* The sealed pools are mounted read-only into the validator container only; they are never mounted into a sibling
  container. Miner models and engines never see them.
* Without the socket and the three directories the validator still serves and sets weights, only without the
  competition share.

Always use `--local-dir` and mount that directory (a HF cache snapshot is made of relative symlinks that do not
exist inside the container). The s1-pro runtime verifies all 18 weight files against the release manifest at start
(a few minutes) and refuses to run on a mismatch; its healthcheck allows up to 15 minutes for start-up.

## Check

```bash
docker compose -f deploy/validator-1.2/compose.yml ps                 # both runtimes "healthy" before the validator starts
nvidia-smi --query-gpu=memory.used,memory.total --format=csv          # expect about 80 GB used of 96 GB
docker compose -f deploy/validator-1.2/compose.yml logs validator | grep "reference runtime for"   # s1-fast and s1-pro ready
# selfcheck against the running sidecars (needs the reax image or an install of this release)
REAX_RUNTIME_TOKEN_FILE=deploy/secrets/runtime-token REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 \
  REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 reax-selfcheck --model s1-fast
REAX_RUNTIME_TOKEN_FILE=deploy/secrets/runtime-token REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 \
  reax-selfcheck --model s1-pro
```

`RUNE_DETERMINISTIC=1` is required on a validator: stock vLLM is not reproducible and would score honest
miners wrongly. The validator refuses to start if either runtime does not answer an authenticated test decision.
