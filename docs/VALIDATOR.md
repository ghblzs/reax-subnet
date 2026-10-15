# Running a REAX validator

This release ships two mechanisms:

* `REAX_MECHANISM=s1` (default, release 1.1): the validator sends each miner fresh typed decisions over
  protocol `reax-s1/1` (`POST /v1/s1/decide`) and recomputes the verified answers on its own **reference
  runtime** of the same pinned REAX reference model, running in a sidecar on loopback.
* `REAX_MECHANISM=mpm1`: the unchanged rc3 validator (Qwen3-4B reference in-process, `POST /v1/decide`); see
  [the mpm1 section below](#the-mpm1-rc3-validator-fallback) and [MPM.md](MPM.md).

The validator opens no port.
REAX is registered on Finney as netuid 92. Emission start and the TAO emission share are controlled on-chain (subnet owner and root network); check the live chain state. Use a separately configured netuid for testnet or local endpoints.

## Pools, reference runtimes and hardware (s1)

The validator runs the **reference runtime of every enabled pool** as a sidecar on loopback. Default pools:
**s1-fast 0.35 and s1-pro 0.65** (release 1.2: the s1-pro pool is on for every network, finney included;
`REAX_FEATURE_S1_PRO_POOL` is fixed by the release and can only be switched off on development networks with
`REAX_ALLOW_DEV_MODELS=1`). **Every validator needs an 80–96 GB GPU** for the s1-pro reference runtime. One
96 GB card holds both reference runtimes: `deploy/validator-1.2/compose.yml` ([README](../deploy/validator-1.2/README.md)).
A pool without a miner above `REAX_MIN_EMA` does not burn its share: it passes to the other pool pro rata
(s1-pro empty → s1-fast gets the whole miner part; only if no pool has a miner does it go to the owner UID).

| | s1-fast reference | s1-pro reference |
|---|---|---|
| Reference runtime | build from `runtimes/s1-fast/Dockerfile` (reference fast runtime: jevk5 batch-1 CUDA graph) | build from `runtimes/s1-pro/Dockerfile` (reference Rune recipe: deterministic vLLM 0.30.0) |
| Loopback | `127.0.0.1:8101` | `127.0.0.1:8102` (verifies all 18 weight files at start; a few minutes) |
| Weights | `crh225/plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186`, public; manifest `ff6e01b983d48024` | `surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921`, Hugging Face gated (accept the gate on huggingface.co with your own account, `hf auth login`, `hf download … --local-dir`; never mirror); manifest `5b11440bf57b5434` |
| GPU (in the reference sidecar) | 1× NVIDIA with ≥ 24 GB; **measured: RTX PRO 6000 Blackwell, H100 80 GB** (RTX 4090: see CALIBRATION.md) | 1× NVIDIA with 80–96 GB; **measured: RTX PRO 6000 96 GB, H100 80 GB** (bit-identical results) |
| CPU / RAM / disk | 8 cores / 32 GB; disk for the weights (about 9 GB) | disk adds the s1-pro weights (about 52 GB) |
| Network | outbound only (miners' public ports and a subtensor endpoint); the validator opens no port | same |
| Clock | NTP running | same |
| Stake | a validator permit **and** at least `REAX_MIN_VALIDATOR_STAKE` (default 1000, the SDK's `total_stake` of your hotkey on the subnet); miners answer only validators that meet both, otherwise every miner returns 403 | same |

For s1-fast, RTX 4090, RTX 5090, RTX 3090, L40S and RTX A6000 are also measured (CALIBRATION.md). Other cards
(L4, A100, H200 …) should work but are **not yet measured**: run
`scripts/calibration_s1` on your GPU first and stay inside the release tolerances before trusting its scores.

In the s1 mechanism the GPU lives in the reference sidecar; the validator process itself needs no GPU
(`REAX_DEVICE`/`REAX_DTYPE` apply to the mpm1 fallback only).

## Keys

Use bittensor-cli 9.x (`pip install bittensor-cli==9.23.2`, separate virtual environment) for these commands.
Only the **hotkey** goes on the server (plus `coldkeypub.txt`). Register and stake from a separate machine:

```bash
btcli subnets register --netuid 92 --wallet.name reax --wallet.hotkey validator
btcli stake add --netuid 92 --wallet.name reax --wallet.hotkey validator --amount <TAO>
```

Wait one tempo and check that the hotkey has a validator permit (`btcli subnets metagraph --netuid 92`).

## Run with Docker (s1)

The validator runs its **own reference sidecar** of each enabled pool on loopback with its own copy of the
pinned weights and the shared internal token — the same sidecar images, host networking, token file and
read-only weights mount as the miner setup. `../deploy/docker-compose.s1-validator.yml` is a ready-made copy
of the commands below (sidecar + validator, health-gated start); `../deploy/docker-compose.s1.yml` shows the
same pattern for a miner. The validator process itself opens no port and needs no GPU.

```bash
# from the repository root:
docker build -t reax:1.2.0rc3 .
docker build -f runtimes/s1-fast/Dockerfile -t reax-s1-fast-runtime .

# the shared internal token (the sidecar must hold exactly this secret):
mkdir -p deploy/secrets
openssl rand -hex 32 > deploy/secrets/runtime-token
chmod 0600 deploy/secrets/runtime-token      # owned by you; both containers run as your uid below

# the pinned weights (always --local-dir; a HF cache snapshot path is a tree of dangling
# symlinks inside the container and the sidecar refuses to start):
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b

# reference sidecar on loopback, running as YOUR uid so it can read the 0600 token file:
docker run -d --name reax-s1-fast-ref --network host --gpus all --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e S1M_INTERNAL_TOKEN_FILE=/run/s1m-internal-token \
  -v "$PWD/deploy/secrets/runtime-token:/run/s1m-internal-token:ro" \
  -v /srv/reax/plumb-4b:/model:ro \
  reax-s1-fast-runtime

# the validator process itself (opens no port, needs no GPU):
docker run -d --name reax-validator --network host --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" --no-healthcheck \
  --env-file deploy/validator.env --entrypoint reax-validator \
  -e REAX_MECHANISM=s1 -e REAX_MODEL=s1-fast \
  -e REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 \
  -e REAX_RUNTIME_TOKEN_FILE=/run/reax-runtime-token \
  -v $HOME/.bittensor/wallets:/wallets:ro \
  -v "$PWD/deploy/secrets/runtime-token:/run/reax-runtime-token:ro" \
  -v reax-state:/state \
  reax:1.2.0rc3
```

Notes:

* `--user "$(id -u):$(id -g)"` on **both** containers is what makes the 0600 token file readable inside; the
  s1-fast image sets `HOME=/tmp` and `TRITON_CACHE_DIR=/tmp/triton`, so any uid works (no passwd entry needed).
* `REAX_MODEL=s1-fast` is required with `REAX_MECHANISM=s1`: the example `validator.env` pins the mpm1 model
  (`qwen3-4b-instruct-2507`), which mechanism s1 refuses (the command-line values here override the env file).
  The validator's reference runtimes are chosen by the enabled pools, not by `REAX_MODEL`.
* On start the validator refuses to run without a readable, non-empty `REAX_RUNTIME_TOKEN_FILE` and then
  checks each reference sidecar with an **authenticated test decision**; if the probe fails (sidecar down,
  or the two sides hold different tokens) it exits with an error instead of silently scoring nobody.
* The sidecar verifies the pinned weights under `/model` against the release manifest at start.
* Host networking (or one shared network namespace) keeps the sidecar on loopback and unreachable from
  outside; never publish the sidecar port. Every validator also runs the
  s1-pro reference sidecar (`runtimes/s1-pro/Dockerfile`, loopback `127.0.0.1:8102`, verifying all 18 weight
  files at start) and needs an 80–96 GB GPU; its internal vLLM then also listens **unauthenticated** on
  `127.0.0.1:8000` — keep other local users off the machine.

## Run without Docker (reax validator process)

The reference sidecar always runs as a container (it is the pinned reference runtime). The validator process
itself can run from a virtual environment:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements/lock.txt && pip install --no-deps --no-build-isolation .
export REAX_MECHANISM=s1 REAX_NETUID=92 REAX_WALLET_NAME=reax REAX_WALLET_HOTKEY=validator \
  REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 REAX_RUNTIME_TOKEN_FILE=<path to the shared token> \
  REAX_SET_WEIGHTS=0   # shadow mode for the first hour
reax-validator
```

Also run the s1-pro reference sidecar on an 80–96 GB GPU and export `REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102`.

## What it logs

One line per miner per round (`uid … [model] score … items passed/verified mean_d … max_d <largest per-item
logit distance> latency … ema …` plus an optional `STRIKE` marker and `error=…`), one summary per round and
one line per weight submission.
With `REAX_LOG_JSON=1` every line is JSON. Nothing secret is logged (RPC URLs are logged without credentials).

Healthy signs: rounds every ~2 minutes (`REAX_ROUND_INTERVAL_S`, default 120), most committed miners answering,
and "weights set" roughly every `max(weights_rate_limit, REAX_WEIGHTS_INTERVAL_BLOCKS)` blocks (default 360
blocks ≈ 72 minutes).

**Capacity probes (tiered pools).** About every 10 rounds a probe loads every queryable UID of each pool at once for
the pool's window (s1-fast 5 s). It needs one socket per lane, about 16 per s1-fast/s1-pro UID: the compose files set
`ulimits: nofile 65536` for the validator; without Docker run `ulimit -n 65536` first. If the validator's own event loop
lags during the window (p95 above 50 ms), the log says `capacity probe skipped: the validator was the bottleneck` and
nothing is recorded for that probe. Measure the host once with `python scripts/bench_probe.py --uids 120 --procs 8`
(fake miners on 127.0.0.1, no chain, no wallet; stop the validator or use spare cores) and, if its totals stop short of
the miners' speed, set `REAX_PROBE_CEILING_DPS` as the script suggests.

A validator starting with no saved state first scores **6 rounds (about 12 minutes) without submitting
weights**, so honest miners have a non-zero average before its first weights. Restarts with saved state skip
this.

## Check your reference runtime

The validator's own runtime sidecar is the reference every miner is measured against: if it drifts, every honest miner
loses rounds. Run the same check the miners run (docs/MINER.md, "Check your setup before you register") against it once
the sidecar is healthy and before you enable weights (it sends 640 requests to the sidecar), and again after any driver, GPU
or image change; do not set weights from a host that fails:

```bash
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1-validator.yml run --rm --no-deps --entrypoint reax-selfcheck validator
```

## Shadow mode

`REAX_SET_WEIGHTS=0` scores and logs weights but never submits them. Use it for the first hour.

Warm-up counts rounds, not answers. A validator that runs before it holds a permit (or with stake below
`REAX_MIN_VALIDATOR_STAKE`) is refused by every miner (`http 403`), so its warm-up passes with all averages at 0. If
it then submits weights at the first round after the permit arrives, they go entirely to the owner UID and the weights
interval (`REAX_WEIGHTS_INTERVAL_BLOCKS`, one tempo) keeps them for about 72 minutes. Keep `REAX_SET_WEIGHTS=0` until
the log shows miners answering (`items 8/8`) with averages above 0.3 (about 4 rounds), then restart with
`REAX_SET_WEIGHTS=1`; the saved state keeps the averages.

## Pools, commitments and fixed scoring constants

* **Pools and shares.** Default pools: s1-fast 0.35 and s1-pro 0.65 (1.2; see [FEATURE-FLAGS.md](FEATURE-FLAGS.md)).
  Every validator needs an 80–96 GB GPU for the s1-pro reference runtime. `REAX_FEATURE_S1_PRO_POOL` is fixed by
  the release; only development networks (`REAX_ALLOW_DEV_MODELS=1`) may switch the pool off. A pool with no miner
  above `REAX_MIN_EMA` passes its share to the other pool pro rata instead of burning it; with no miner in any pool
  the miner part goes to the owner UID as before. `REAX_BURN_FRACTION` is unchanged.
* **Commitments.** The validator only queries miners whose on-chain commitment
  `reax-s1/1:<model>:<manifest_id>` matches an enabled pool and this release's manifest (`ff6e01b983d48024`
  for s1-fast, `5b11440bf57b5434` for s1-pro). A changed commitment (model or manifest) restarts that UID's
  score from zero.
* **Round scoring (s1).** Per miner and round 8 fresh items; s1-fast verifies all 8, s1-pro verifies a secret
  random 3 of 8 drawn after the answers arrive (the miner cannot tell which). Per verified item the validator
  computes the logit-space distance d between the miner's and its own reference answer (probabilities below a
  1e-6 floor are treated as equal); D = the mean d over the verified items. D ≤ 0.20 → the round scores
  `0.8 + 0.2 × speed` (speed from the validator-measured latency; bounds s1-fast 0.5–8 s, s1-pro 1–12 s, past
  the slow bound the request times out); D > 0.20 → the round scores 0; D > 0.50 or any single item d > 2.0 →
  **strike: the moving average resets to 0**. Tolerances and latency bounds are release constants from the
  GPU calibration — see CALIBRATION.md.
* **Fixed scoring constants.** `REAX_TAU`, `REAX_MIN_ROUND_FIDELITY`, `REAX_LATENCY_FAST_S`,
  `REAX_LATENCY_SLOW_S`, `REAX_LATENCY_WEIGHT`, `REAX_EMA_ALPHA`, `REAX_MIN_EMA`, `REAX_ITEMS_PER_ROUND`,
  `REAX_REQUEST_TIMEOUT_S`, `REAX_WARMUP_ROUNDS` and `REAX_FEATURE_S1_PRO_POOL` are fixed by the release: the
  validator refuses to start if any of them is set without `REAX_ALLOW_DEV_MODELS=1` (refused on finney).
  Validators with different scoring disagree on weights and lose trust. The per-pool tolerances τ and latency
  bounds are release constants from the GPU calibration — see CALIBRATION.md. `REAX_BURN_FRACTION` is the one
  exception; it must be identical on all validators and is set only by agreement among validators, for example as an
  emergency brake (docs/EMISSIONS.md).

## State and restarts

Scores live in `REAX_STATE_PATH` (written atomically after every round). A restart continues where it stopped.
If the file is corrupt or belongs to another subnet/hotkey it is moved aside and scores rebuild. A metagraph
read that suddenly has fewer neurons is treated as a bad read and the round is skipped (accepted again on a
second identical read). A changed commitment is applied per UID only when two consecutive reads agree; until
then that UID is neither queried nor scored, and if it is still unconfirmed on the next read (it keeps flipping)
its score decays like a miner that did not answer. A failure of the validator's own reference
runtime leaves only the affected miners unscored for that round; if more than 10 % of the verified items
fail, or the reference step takes longer than 1.5 round intervals (a hung sidecar whose /health stays green),
the whole round is skipped without scoring anyone and an error is logged (check the sidecar). Scores follow hotkeys across
renumbered UIDs, and each miner's score is keyed to its committed role (`<model>:<manifest_id>`): a changed
commitment restarts that UID's score from zero.

## The mpm1 (rc3) validator (fallback)

`REAX_MECHANISM=mpm1` runs the unchanged rc3 validator: the reference model (`qwen3-4b-instruct-2507`) runs
**in the validator process** on a 24 GB GPU (no sidecar; `REAX_DEVICE`/`REAX_DTYPE` apply), scoring as in
[MPM.md](MPM.md). The validator refuses to start without a usable GPU unless `REAX_DEVICE=cpu` is set (CPU is
for tests only). Keys, stake and permit requirements, shadow mode, logs, warmup and state behave as above.
When an existing s1 validator switches to mpm1 (rollback, CHANGELOG.md), move its state file aside first (or use a
fresh `REAX_STATE_PATH` / volume): the mpm1 loop ignores roles and would otherwise carry s1 scores into its first rounds.

```bash
docker build -t reax:1.2.0rc3 .
cp deploy/validator.env.example deploy/validator.env   # fill in netuid, wallet names (sets REAX_MECHANISM=mpm1)
docker run -d --name reax-validator --gpus all --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" --no-healthcheck \
  --env-file deploy/validator.env -e REAX_MECHANISM=mpm1 --entrypoint reax-validator \
  -v $HOME/.bittensor/wallets:/wallets:ro -v reax-models:/models -v reax-state:/state \
  reax:1.2.0rc3
```
