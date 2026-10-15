# Configuration (environment variables)

## Common

| Variable | Default | Meaning |
|---|---|---|
| `REAX_NETWORK` | `finney` | `finney`, `test`, or a `ws://`/`wss://` endpoint (logged without credentials) |
| `REAX_NETUID` | `92` on `finney`; required otherwise | subnet id. Testnet and local endpoints must set their own netuid. |
| `REAX_WALLET_NAME` / `REAX_WALLET_HOTKEY` | `default` / `default` | wallet and hotkey names |
| `REAX_WALLET_PATH` | `~/.bittensor/wallets` | wallet directory (hotkey only is enough) |
| `REAX_MECHANISM` | `s1` | `s1` (release 1.1: miners serve the REAX reference models, protocol `reax-s1/1`) or `mpm1` (unchanged rc3 mechanism, protocol `reax-mpm/1`) |
| `REAX_MODEL` | `s1-fast` with mechanism s1 / `qwen3-4b-instruct-2507` with mpm1 | with `REAX_MECHANISM=s1`: `s1-fast` or `s1-pro` (the REAX reference model this process serves or references); with mpm1: `qwen3-4b-instruct-2507` |
| `REAX_S1_FAST_RUNTIME_URL` / `REAX_S1_PRO_RUNTIME_URL` | none (required for each model used with mechanism s1) | loopback http URL of that model's runtime sidecar (must start with `http://127.`, `http://localhost` or `http://[::1]`; a non-loopback URL aborts startup). `hash` selects the hash test runtime and needs `REAX_ALLOW_DEV_MODELS=1` |
| `REAX_RUNTIME_TOKEN_FILE` | none (required with mechanism s1) | path of the shared internal token file; a config error stops startup unless it names a readable, non-empty file, and an authenticated test decision against the runtime follows at start. The sidecar must hold the same secret (sidecar: `S1M_INTERNAL_TOKEN` env or `S1M_INTERNAL_TOKEN_FILE`) |
| `REAX_DEVICE` | `cuda` | mpm1 only (the s1 model runs in its own sidecar): `cuda`, `cuda:<n>` or `cpu`. Startup fails if CUDA is requested but unavailable |
| `REAX_DTYPE` | `bfloat16` | mpm1 only: `bfloat16`, `float16` or `float32`; production uses bfloat16 everywhere |
| `REAX_ALLOW_DEV_MODELS` | `0` | development networks only: allows test models and scoring overrides (refused on finney) |
| `REAX_LOG_LEVEL` / `REAX_LOG_JSON` | `INFO` / `0` | logging |
| `HF_HOME` | `~/.cache/huggingface` (`/models` in Docker) | mpm1 model download cache; the s1 sidecar gets its weights mounted read-only at `/model` |

## Miner

| Variable | Default | Meaning |
|---|---|---|
| `REAX_MINER_HOST` / `REAX_MINER_PORT` | `0.0.0.0` / `8091` | listen address |
| `REAX_EXTERNAL_IP` / `REAX_EXTERNAL_PORT` | none / miner port | endpoint published on chain (an IP address, not a host name) |
| `REAX_SERVE_AXON` | `1` | publish the endpoint on start if it changed |
| `REAX_MIN_VALIDATOR_STAKE` | `1000` | minimum `total_stake` of a validator hotkey before the miner answers it |
| `REAX_METAGRAPH_REFRESH_S` | `300` | how often the validator list is refreshed |
| `REAX_MAX_CONCURRENT` | `8` (1.2.0rc2: 4) | requests passed to the runtime at once; more wait in a queue of 4x this size, served round-robin per validator (one validator's burst cannot starve another's capacity probe) |
| `REAX_PER_CALLER_CONCURRENCY` | `16` (1.2.0rc2: 2) | parallel requests per validator. Capacity probes offer up to 16; a lower value caps how fast this hotkey can be measured |
| `REAX_PER_CALLER_PER_MINUTE` | `240` (1.2.0rc2: 20) | requests per validator per minute (a capacity probe at the pool cap sends about 220) |
| `REAX_RUNTIME_INFLIGHT` | `16` | decisions in flight to the runtime sidecar at once; raise it for a batched or multi-GPU runtime behind one hotkey |
| `REAX_BODY_TIMEOUT_S` | `5` | deadline for receiving a request body |
| `REAX_MAX_PENDING_PER_IP` | `32` (1.2.0rc2: 8) | unfinished request bodies per remote IP |
| `REAX_MAX_PENDING_TOTAL` | `4096` | unfinished request bodies across all remote addresses (a global cap so the per-IP cap cannot be side-stepped with many addresses) |
| `REAX_PUBLISH_COMMITMENT` | `1` | mechanism s1: publish the on-chain commitment `reax-s1/1:<model>:<manifest_id>` on start if it changed (validators only query miners with a commitment matching an enabled pool and this release's manifest; changing it restarts the score from zero) |

## Validator

| Variable | Default | Meaning |
|---|---|---|
| `REAX_ROUND_INTERVAL_S` | `120` | seconds between round starts |
| `REAX_MAX_PARALLEL` | `256` | concurrent miner requests (256 = every UID at once) |
| `REAX_STATE_PATH` | `~/.reax/validator-state.json` | persisted scores |
| `REAX_SET_WEIGHTS` | `1` | `0` = shadow mode |
| `REAX_WEIGHTS_INTERVAL_BLOCKS` | `360` | minimum blocks between weight submissions (the chain rate limit also applies) |
| `REAX_ALLOW_PRIVATE_IPS` | `0` | local test networks only; link-local addresses stay blocked |
| `REAX_BURN_FRACTION` | `0` | share of every weight vector for the owner UID; identical on all validators |
| `REAX_PROBE_CEILING_DPS` | `0` (off) | tiered pools: drop a capacity probe (no capacity update) when its total throughput reaches this many decisions/s, the most this host can measure. Set it from `scripts/bench_probe.py`; it never changes how a kept probe is scored |

**Fixed by the release** (setting them aborts startup unless `REAX_ALLOW_DEV_MODELS=1`, which is refused on
finney): `REAX_TAU` 0.15, `REAX_MIN_ROUND_FIDELITY` 0.8 (mpm1: 10 of the 12 items), `REAX_LATENCY_FAST_S` 2,
`REAX_LATENCY_SLOW_S` 20, `REAX_LATENCY_WEIGHT` 0.2, `REAX_EMA_ALPHA` 0.1, `REAX_MIN_EMA` 0.3,
`REAX_ITEMS_PER_ROUND` 12 (mpm1), `REAX_REQUEST_TIMEOUT_S` (follows `REAX_LATENCY_SLOW_S`, 20),
`REAX_WARMUP_ROUNDS` 6, and `REAX_FEATURE_S1_PRO_POOL` (ON in 1.2 on every network; pool shares are
s1-fast 0.35 / s1-pro 0.65 and every validator needs an 80–96 GB GPU for the s1-pro reference runtime; only
development networks may switch it off).

With mechanism s1 the per-pool scoring constants live with the pools (see
[VALIDATOR.md](VALIDATOR.md)): 8 items per round — s1-fast verifies all 8, s1-pro verifies a secret random 3
of 8 drawn after the answers arrive. Per verified item the validator computes a logit-space distance d against
its own reference (probabilities below a 1e-6 floor are treated as equal), then D = the mean d over the
verified items: D ≤ 0.20 scores `0.8 + 0.2 × speed`, D > 0.20 loses the round (score 0), and D > 0.50 or any
single item d > 2.0 is a strike (moving average reset to 0). Latency bounds (speed term and timeout): s1-fast
0.5–8 s, s1-pro 1–12 s. A round in which the validator's own reference runtime fails on more than 10 % of the
verified items, or whose reference step exceeds 1.5 round intervals, is skipped without scoring anyone. A changed
commitment is applied per UID once two consecutive reads agree; a UID that stays unconfirmed (flipping) decays.
The per-pool tolerances τ and latency bounds are release constants from the GPU calibration — see
CALIBRATION.md.

Disabled features (`REAX_FEATURE_*`): see [FEATURE-FLAGS.md](FEATURE-FLAGS.md).

**Engine competition (**[ENGINE-COMPETITION.md](ENGINE-COMPETITION.md), [COMPETITIONS.md](COMPETITIONS.md)**)**. On finney
(s1 mechanism) these are read always; `REAX_ENGINE_WEIGHTS_DIR` and `REAX_SPEC_SEALED_DIR` are needed to take part (without them,
or while the release reference image is unpublished, the validator logs one warning and serves only); on other
networks only with `REAX_FEATURE_ENGINE_COMPETITION=1`. The start time and the split are release constants, not variables:

| Variable | Default | Meaning |
|---|---|---|
| `REAX_COMPETITIONS_MODE` | `local` (finney: unset = serving-only, `local` only when set explicitly) | `local` = the validator runs the competitions itself; `remote` = it only reads the state files of `reax-competition-worker` (needs only `REAX_COMPETITIONS_STATE_DIR`, no docker socket, no sealed data); `off` = 100 % inference. See [COMPETITIONS.md](COMPETITIONS.md) |
| `REAX_COMPETITIONS_STATE_DIR` | — | absolute directory with `engine-competition.json` and `spec-state.json`: read-only for a `remote` validator, writable for the worker. Files older than 26 h count as missing |
| `REAX_ENGINE_NVML_CHECK` | `0` | `1`: sample `nvidia-smi pmon` for the paused candidate's PIDs during every reference window (worker / local mode) |
| `REAX_ENGINE_REFERENCE` | release constant `config.ENGINE_REFERENCE_IMAGE` | reference engine image, `<image>@sha256:<digest>`; **development networks only** (finney refuses it) |
| `REAX_SPEC_SEALED_DIR` | — | sealed pools of the use-case tracks, read-only (specialist evaluator, [SPECIALIST-TRACKS.md](SPECIALIST-TRACKS.md)) |
| `REAX_ENGINE_WEIGHTS_DIR` | — | pinned model weights, mounted read-only into every engine |
| `REAX_ENGINE_RUNNER` | `docker` | `dev` = local commands from `REAX_ENGINE_DEV_MAP` (needs `REAX_ALLOW_DEV_MODELS=1`) |
| `REAX_ENGINE_GPU` | `0` | GPU index for the engines (serving rounds go on during a cycle; the GPU is shared with the serving runtimes) |
| `REAX_ENGINE_DOCKER_RUNTIME` | — | e.g. `runsc` (gVisor) |
| `REAX_ENGINE_EVAL_INTERVAL_S` | `21600` | seconds between evaluation cycles (minimum 3600; 30 on development networks) |
| `REAX_ENGINE_MAX_EVALS` | `4` | new submissions measured per cycle, oldest commitment first |
| `REAX_ENGINE_WORKDIR` / `REAX_ENGINE_STATE_PATH` | `~/.reax/engines` / `~/.reax/engine-competition.json` | socket directories and logs / champion state |
| `REAX_ENGINE_DEV_PARAMS` | — | JSON overrides of track and reward parameters (development networks only; refused on finney) |
