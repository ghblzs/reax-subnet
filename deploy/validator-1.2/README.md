# Validator 1.2 on one 96 GB GPU

Release 1.2 turns the **s1-pro pool on for every network**, so a validator runs the reference runtime of both
pools: s1-fast (Plumb-4B, 12.3 GB of VRAM measured) and s1-pro (Surogate Rune 26B-A4B, deterministic vLLM 0.30.0).
`compose.yml` here starts both next to each other on **one** 96 GB card (RTX PRO 6000) plus the validator. Both
runtimes listen on loopback only (`127.0.0.1:8101`, `127.0.0.1:8102`); never publish those ports.

| Service | Listens on | VRAM | Notes |
| --- | --- | --- | --- |
| `s1-fast-runtime` | 127.0.0.1:8101 | 12.3 GB | reference fast runtime, as in 1.1; starts first |
| `s1-pro-runtime` | 127.0.0.1:8102 | 58.7 GB (`REAX_S1_PRO_GPU_MEM_UTIL` 0.60, `REAX_S1_PRO_MAX_MODEL_LEN` 8192) | deterministic build, 52 GB bf16 weights + KV cache; bit-identical to the calibration runs (871/871 items, concurrency 1 and 16) |
| `s1-cpu-runtime` | 127.0.0.1:8104 | none (CPU) | s1-cpu reference: Gutsy 0.8B Q8_0 GGUF on llama.cpp CPU, `GGUF_THREADS=8` (3.7 decisions/s measured on 8 threads) |
| `validator` | none | none | `reax-validator`, hotkey-only wallets mounted read-only; **no docker socket**, competitions in remote mode |
| `competition-worker` | none | up to 25 GB while a cycle runs | `reax-competition-worker`: the docker socket, the sealed pools and the work directories live here; **no wallets, no tokens** |

From 16 October 2026, 12:00 UTC the weights include the **competitions** (engine champion and the two use-case
tracks, 50 % of the weights; see [docs/COMPETITIONS.md](../../docs/COMPETITIONS.md)). They are **not run by the validator
process**: the `competition-worker` service runs the measurements **one at a time** on the same GPU, as its own container,
and writes `engine-competition.json` and `spec-state.json`; the validator (`REAX_COMPETITIONS_MODE=remote`) only reads
them each round. A validator without a fresh state file (no worker, worker down for more than 26 hours) logs one warning
and pays that competition's share to inference, which means it differs from validators that run the competitions.

Measured on an RTX PRO 6000 Blackwell Server Edition (97,887 MiB): 12.3 GB + 58.7 GB = about 71 GB, so about 27 GB stay
free for one competition measurement at a time (engine: 2 x 12.5 GB; specialist: up to 25 GB; 12.3 + 58.7 + 25 = ~96).
Do not go lower than 0.60 with the 8k context: 0.58 with a 32k context fails (6.9 GiB KV cache needed). The numerics do
not depend on the share (batch-invariant kernels), only the KV-cache size does. Run `reax-selfcheck --model s1-pro`
after any change.

### Confidential-compute GPUs

On confidential-compute GPUs (e.g. some cloud RTX PRO 6000 / H100 CC instances) vLLM returns garbage unless UVA is
disabled (a zero-copy GPU view of pinned host memory does not see later host writes). Set `REAX_VLLM_DISABLE_UVA=1`
for the s1-pro runtime: `runtimes/vllm-ccfix/sitecustomize.py` (mounted read-only at `/opt/reax-ccfix`, on
`PYTHONPATH`) switches vLLM to its explicit-copy buffers. Numerics are unchanged (measured bit-identical). The
worker copies the same file into the run directory of every specialist vLLM container (`REAX_VLLM_DISABLE_UVA=1` is
passed to the worker).

```bash
export REAX_VLLM_DISABLE_UVA=1      # 0 (default) on normal GPUs
```

## Setup (from the repository root)

```bash
docker build -t reax:1.2.0rc3 .
docker compose -f deploy/validator-1.2/compose.yml build            # builds both runtimes
cp deploy/validator.env.example deploy/validator.env                # netuid, wallet names; keep REAX_SET_WEIGHTS=0 for the first hour
mkdir -p deploy/secrets && openssl rand -hex 32 > deploy/secrets/runtime-token && chmod 0600 deploy/secrets/runtime-token
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b
hf download kouhxp/gutsy --revision f9995fe26a337f1a4ea3f1a5ed38c9799e5a8f63 --local-dir /srv/reax/gutsy   # s1-cpu reference (812 MB GGUF)
hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune   # gated: your own HF account
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b REAX_S1_PRO_WEIGHTS=/srv/reax/rune REAX_S1_CPU_WEIGHTS=/srv/reax/gutsy \
  docker compose -f deploy/validator-1.2/compose.yml up -d
```

### Competitions: extra setup

Needs a few more things on the host. All paths are host paths; the compose file mounts them at the same path inside the
worker container (docker resolves the `-v` paths of the sibling containers on the host).

| Variable | What | Size |
| --- | --- | --- |
| `REAX_S1_FAST_WEIGHTS` | the Plumb-4B directory from above; the engine competition measures with these weights (mounted read-only into the worker as `REAX_ENGINE_WEIGHTS_DIR`) | 8 GB |
| `REAX_SPEC_SEALED_DIR` | the sealed item pools of the use-case tracks (`<track>/sealed/sealed.clean.jsonl`), **read-only into the worker only**, obtained from the subnet owner | small |
| `REAX_COMPETITION_DIR` | writable: `engines/` (per-measurement sockets and logs), `spec-work/` (downloaded miner models, up to ~20 GB each, kept only for champions and pending confirmations; `spec-work/sock/` holds the vLLM sockets), `spec-bases/` (the base models miners declare, fetched by the worker and kept, ~8-18 GB each) and `state/` (the two JSON files; the validator mounts only this subdirectory, read-only). **Its filesystem needs >= 60 GB free at all times** (the worker skips a specialist cycle below that) and should not be the one that holds `/` | 200 GB free is plenty |
| `REAX_HF_CACHE` | writable Hugging Face metadata cache (`HF_HOME`) | 5 GB |
| `REAX_DOCKER_GID` | group id that may use the docker socket (worker only): `getent group docker \| cut -d: -f3` | |
| `REAX_CCFIX_DIR` | absolute host path of `runtimes/vllm-ccfix` in your checkout, for the **s1-pro runtime** only (read-only; used when `REAX_VLLM_DISABLE_UVA=1`) | tiny |

```bash
docker build -t reax:1.2.0rc3 --target competition-worker .      # compose builds it too
export REAX_SPEC_SEALED_DIR=/srv/reax/sealed REAX_COMPETITION_DIR=/srv/reax/competitions REAX_HF_CACHE=/srv/reax/hf
export REAX_DOCKER_GID=$(getent group docker | cut -d: -f3)
export REAX_CCFIX_DIR=$(pwd)/runtimes/vllm-ccfix   # absolute path; needed by compose on every host (used only with REAX_VLLM_DISABLE_UVA=1)
mkdir -p "$REAX_COMPETITION_DIR/state" "$REAX_HF_CACHE" && chown "$(id -u):$(id -g)" "$REAX_COMPETITION_DIR" "$REAX_COMPETITION_DIR/state" "$REAX_HF_CACHE"
```

The champions and the pending confirmations live in `$REAX_COMPETITION_DIR/state` (`engine-competition.json`,
`spec-state.json`); keep it across restarts. The reference engine image is a **release constant**; do not set
`REAX_ENGINE_REFERENCE` (finney refuses it). `REAX_SPEC_TRACKS_FILE` is ignored on finney: the track schedule is fixed.

**GPU memory.** An engine measurement needs two s1-fast engines (2 x 12,500 MiB) free; a specialist evaluation needs up
to 25 GB. Each cycle checks `nvidia-smi` first and, if the memory is short, skips cleanly (logged as "cycle skipped"),
without recording anything against any submission. With the defaults (s1-pro 0.60, 8k context) about 27 GB stay free on
a 96 GB card, enough for either. On a smaller card the measurements skip.

**Timing noise.** Serving goes on during a measurement, so the GPU is shared with the s1-fast and s1-pro runtimes. Every
engine session first measures the reference alone and treats any later reference run more than 12 % away from that as
inconclusive (retried in a later cycle, never counted against the candidate until three tries fail).

**Missing setup.** A validator in remote mode with no state directory content, or only a stale one, pays 100 % inference
for the competitions (one warning). A worker that lacks the sealed pools or the weights directory refuses to start.

**Moving the worker to another box.** Delete the `competition-worker` service from this compose file and run
[deploy/competition-worker](../competition-worker/README.md) on a keyless GPU box that copies the two state files to
`$REAX_COMPETITION_DIR/state` on this host. Nothing else changes for the validator.

### Security: what the docker socket means

The engine and the specialist evaluator run code and models that **miners chose**, in short-lived sibling containers
(`--network none`, read-only root, all capabilities dropped, `no-new-privileges`, one GPU, CPU/memory/swap/pids/ulimit/log
limits, dedicated uids 10002 (engines) and 10003 (vLLM) that differ from every host user, weights mounted read-only, the
miner's own chat template never rendered). To start them, the worker talks to the host's docker daemon. **Access to
`/var/run/docker.sock` is equivalent to root on the host**, and an image-only escape of the container runtime or the NVIDIA
toolkit is host root too. That is why, since 1.2.0rc2:

* **The socket exists only in the `competition-worker` container.** The validator container has no docker CLI, no socket and
  no sealed data. A bug in the validator's parsing of chain data, HF metadata or miner responses no longer reaches the
  docker daemon.
* **The worker is still root on the host while it parses miner input.** It reads Hugging Face metadata, safetensors headers,
  registry manifests and engine/vLLM responses with the docker socket in hand: a Python-level code-execution bug there is
  host root (hotkey, router, WireGuard), no container escape needed. The target is the worker on a separate keyless box
  (option E below / `deploy/competition-worker`); until then keep the parsers' surface small and the host patched.
* **The worker's heartbeat shows that it is alive, not that its cycles succeed.** A worker whose cycles keep failing keeps its
  last champion paid (decaying) until you notice; watch its log for "competition cycle failed".
* **The worker holds nothing to steal**: no wallets, no runtime token, no router token, a bridge network (not the host
  loopback with the runtimes), read-only root, no capabilities. This is *not* a sandbox against a host escape: the worker still
  runs as a process of the same machine, and a root escape on this host still reaches the hotkey files, the router and the
  WireGuard keys. To remove that, run the worker on a separate keyless box ([deploy/competition-worker](../competition-worker/README.md)).
* Patch the escape chain on the host before enabling the worker: NVIDIA container toolkit >= 1.17.8 (latest is better)
  **and** `features.disable-cuda-compat-lib-hook = true` in `/etc/nvidia-container-toolkit/config.toml`; runc >= 1.2.8 / 1.3.3;
  current docker/containerd, kernel and NVIDIA driver; recheck the NVIDIA and runc security bulletins published after June 2026.
* The pinned vLLM image must ship `jinja2 >= 3.1.6` (sandbox-escape fixes): `scripts/check-vllm-image.sh` prints the versions.
* Hotkey-only wallets and nothing else of value on the validator host (the compose file mounts no coldkey; never copy one
  there; `find /w/wallets -name coldkey` must print nothing).
* Prefer a **separate rootless docker daemon** (or gVisor) for the candidates; NVIDIA GPU support works with both.
* The sealed pools are mounted read-only into the worker only; they are never mounted into a sibling container. Miner models
  and engines never see them.

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

## Tiered pools: s1-lite and s1-cpu

Release 1.2 also turns the **tiered pools** on for every network (like s1-pro): `s1-lite` (small GPUs incl. Jetson Orin)
and `s1-cpu` (CPU-only machines incl. ARM), with capacity scoring inside every pool. Inside the inference part the tier
budgets are GPU 80 % (s1-fast 35 / s1-pro 65 inside it), small GPU 15 %, CPU 5 %; a pool or tier without a miner passes
its share pro rata to the others (never burned). Details: [docs/DESIGN-POOLS.md](../../docs/DESIGN-POOLS.md).
`REAX_FEATURE_TIERED_POOLS` is a fixed release variable: operators cannot set it on finney.

The validator needs a reference for each pool:

| Pool | Reference | Where | Cost per round |
| --- | --- | --- | --- |
| `s1-lite` | the **s1-fast runtime** (same Plumb-4B bf16 weights and contract; the pool's tolerance `tau_mean` 0.30 is set for this reference) | `REAX_S1_LITE_RUNTIME_URL`, default `http://127.0.0.1:8101` | at most 240 decisions, about 7.5 s on that runtime; no extra GPU memory |
| `s1-cpu` | the CPU reference service `s1-cpu-runtime` of this compose file | `REAX_S1_CPU_RUNTIME_URL`, default `http://127.0.0.1:8104` | at most 160 decisions, at most 43 s on 8 threads |

Validator-local caps (not scoring constants): `REAX_REFERENCE_CAP_S1_LITE` (default 240) and `REAX_REFERENCE_CAP_S1_CPU`
(default 160) bound the reference decisions per round; beyond that, miners are verified in fair rotation. If you want a
CPU reference for s1-lite instead (`runtimes/s1-gguf`, `MODEL=s1-lite`, 12 threads = 0.6 decisions/s measured), set
`REAX_REFERENCE_CAP_S1_LITE=50`.

**A validator without a reference for a tiered pool does not crash.** If `REAX_S1_CPU_RUNTIME_URL` is unset or the
runtime does not answer at start, it logs a warning (`tiered pool s1-cpu is NOT scored on this validator`) and leaves
that pool unscored: its share then goes to the other pools in that validator's weights, so such a validator's weights
differ from those of validators that run the reference. Run the reference.

Check: `docker compose -f deploy/validator-1.2/compose.yml logs validator | grep "reference runtime for"` shows s1-fast,
s1-pro, s1-lite and s1-cpu ready.
