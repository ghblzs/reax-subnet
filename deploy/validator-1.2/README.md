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
