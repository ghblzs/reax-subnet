# Running a REAX miner

This release ships two mechanisms:

* `REAX_MECHANISM=s1` (default, release 1.1): your miner serves exactly one System1 production model —
  `s1-fast` (Plumb-4B) or `s1-pro` (Surogate Rune 26B-A4B v3) — in a runtime sidecar container on loopback and
  answers signed validator requests over protocol `reax-s1/1` (`POST /v1/s1/decide`).
* `REAX_MECHANISM=mpm1`: the rc3 mechanism (Qwen3-4B, `POST /v1/decide`), unchanged as a fallback. See
  [the mpm1 section below](#the-mpm1-rc3-miner-fallback) and [MPM.md](MPM.md).

Your score depends on how closely your answers match the validators' reference computation (see
[DESIGN-v1.1.md](DESIGN-v1.1.md)). There is nothing to tune: run the pinned model with the code in this
repository, stay online, answer fast.

## How the s1 miner fits together

Two processes on one machine:

1. the **runtime sidecar** runs the model with exactly the reference engine and serves the reference runtime's
   decision contract (`POST /internal/v1/decide`, header `X-S1M-Internal-Token`) on **loopback only**
   (s1-fast `127.0.0.1:8101`, s1-pro `127.0.0.1:8102`). Never publish this port.
2. the **reax miner** (`reax-miner`) talks to the sidecar over loopback with a shared internal token
   (`REAX_S1_FAST_RUNTIME_URL` / `REAX_S1_PRO_RUNTIME_URL`, `REAX_RUNTIME_TOKEN_FILE`) and serves the public
   signed endpoint (btauth-signed both ways).

Sidecar and reax process share a random internal token: the sidecar reads `S1M_INTERNAL_TOKEN` (env) or
`S1M_INTERNAL_TOKEN_FILE`, the reax process reads `REAX_RUNTIME_TOKEN_FILE`. Keep the token file mode 0600 and
run the sidecar with your uid (`--user "$(id -u):$(id -g)"`, as the compose file does). Both processes
must share one loopback namespace: run them with host networking (as in `deploy/docker-compose.s1.yml`) or in
one network namespace.

Before serving, the sidecar verifies its weights against the release manifest (s1-fast also verifies the
pinned runtime source; s1-pro verifies all 18 weight files, which takes a few minutes) and refuses to run on
any mismatch.

On start the miner also publishes its **commitment** on chain: `reax-s1/1:<model>:<manifest_id>`, signed by
your hotkey (`REAX_PUBLISH_COMMITMENT`, default 1). Validators only query miners whose commitment matches an
enabled pool and this release's manifest; without a matching commitment your miner is not queried and earns
nothing. Changing the commitment (model or manifest) restarts that UID's score from zero.

## Models, licences and hardware (s1)

One model per hotkey; your commitment says which one.

| Pool | Model | Weights (pinned revision) | Licence | `manifest_id` (this release) | GPU (sidecar) |
|---|---|---|---|---|---|
| s1-fast | Plumb-4B (System1 s1-fast) | `crh225/plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186` (public) | Apache-2.0 | `ff6e01b983d48024` | one NVIDIA GPU with ≥ 24 GB; **measured: RTX PRO 6000 Blackwell, H100 80 GB, RTX 4090, RTX 5090, L40S, RTX A6000** (CALIBRATION.md) |
| s1-pro | Surogate Rune 26B-A4B v3 (System1 s1-pro) | `surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921` (Hugging Face gated) | Apache-2.0 | `5b11440bf57b5434` | one NVIDIA GPU with 80–96 GB; **measured: RTX PRO 6000 96 GB, H100 80 GB** (bit-identical results) |

For s1-fast, RTX 4090, RTX 5090, L40S and RTX A6000 are also measured (CALIBRATION.md). Other cards
(L4, A100, H200, RTX 3090 …) should work but are **not yet measured**: run
`scripts/calibration_s1` on your GPU first and serve only if it stays inside the release tolerances.

Both models are Apache-2.0. The s1-pro weights are gated on Hugging Face: accept the gate on huggingface.co
with **your own** HF account, log in (`hf auth login`) and download with your own account; never mirror the
weights for anyone else. The s1-pro pool is OFF by default in this release (see
[FEATURE-FLAGS.md](FEATURE-FLAGS.md)); serve `s1-pro` only where the pool is enabled (in 1.1 that means
development networks with `REAX_ALLOW_DEV_MODELS=1`).

Other requirements: a current NVIDIA driver for the pinned runtime images (CUDA 13 for the s1-fast image),
Docker with the NVIDIA Container Toolkit, 4 cores / 32 GB RAM, disk for the model snapshot (s1-fast about 9 GB
of weights, s1-pro about 52 GB), a public IPv4/IPv6 address with one open TCP port for the miner endpoint (the
sidecar port must **never** be published), and NTP (signed requests older than 10 s or more than 2 s in the
future are refused, in both directions).

## Keys

Use **bittensor-cli 9.x** (`pip install bittensor-cli==9.23.2`) for wallet and registration commands, in a
separate virtual environment from the miner: the `btcli` that comes with `bittensor==11.1.0` (pinned by this
package) has a different command syntax.

The miner needs **only the hotkey** on the server. Create and register keys on a separate machine, then copy
`~/.bittensor/wallets/<name>/hotkeys/<hotkey>` and `coldkeypub.txt` to the server. Never copy the coldkey.
REAX is registered on Finney as netuid 92. Emission start and the TAO emission share are controlled on-chain (subnet owner and root network); check the live chain state. Use a separately configured netuid for testnet or local endpoints.

```bash
btcli wallet new_coldkey --wallet.name reax
btcli wallet new_hotkey  --wallet.name reax --wallet.hotkey miner1
btcli subnets register   --netuid 92 --wallet.name reax --wallet.hotkey miner1   # pays the current burn cost
```

## Get the weights (s1)

The sidecar verifies the weights against the release manifest before serving, so download exactly the pinned
revision:

```bash
# s1-fast (public):
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b

# s1-pro (gated: accept the gate on huggingface.co with your own HF account, log in once, then download;
# never mirror the weights for anyone else):
hf auth login
hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune-v3
```

Always download with `--local-dir <dir>` and mount **that directory** read-only at `/model` in the sidecar
container. Do not mount a path from the Hugging Face cache (`~/.cache/huggingface/hub/.../snapshots/<rev>`):
there every file is a relative symlink into the cache's blob store, which does not exist inside the container,
so the sidecar fails its weights check and refuses to start. The sidecar verifies every pinned file against
the release manifest (s1-pro verifies all 18 weight files at start, which takes a few minutes) and refuses to
serve on any mismatch.

## Run with Docker (s1, compose)

`deploy/docker-compose.s1.yml` runs an **s1-fast miner**: the runtime sidecar
(`runtimes/s1-fast/Dockerfile`, loopback `127.0.0.1:8101`) plus the `reax-miner` process, both on host
networking so the sidecar is reachable on loopback and unreachable from outside. (The s1-pro sidecar mirrors
this on loopback port 8102 with its own weights and an 80–96 GB GPU.)

```bash
# from the repository root:
docker build -t reax:1.1.0rc2 .
docker build -f runtimes/s1-fast/Dockerfile -t reax-s1-fast-runtime .
cp deploy/miner-s1.env.example deploy/miner.env   # fill in netuid, wallet names, public IP (mechanism/model are set in the compose file)
mkdir -p deploy/secrets && openssl rand -hex 32 > deploy/secrets/runtime-token
chmod 0600 deploy/secrets/runtime-token        # the compose file runs both containers as your uid
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir /srv/reax/plumb-4b
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml up -d --build
```

Details that matter:

* both services use **host networking**, so the sidecar is reachable on `127.0.0.1:8101` from the miner and
  not reachable from outside; never publish the sidecar port. The miner endpoint (default port 8091) binds on
  the host directly and is what you publish on chain.
* the sidecar runs the model on the GPU; the reax miner process itself needs no GPU.
* the compose file mounts the weights directory from your `hf download --local-dir` (env
  `REAX_S1_FAST_WEIGHTS`; never a HF cache snapshot path — its files are symlinks that dangle inside the
  container) read-only at `/model`, plus the shared token from `deploy/secrets/runtime-token` into both
  containers.
* the miner service waits for the sidecar's health check (`depends_on: service_healthy`; the image's
  HEALTHCHECK allows up to 10 minutes for the weights hash and model load).

## Run without Docker (reax process only)

The sidecar always runs as a container (it is the pinned reference runtime). The reax miner process itself
can run from a virtual environment:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --require-hashes -r requirements/lock.txt && pip install --no-deps --no-build-isolation .
export REAX_MECHANISM=s1 REAX_MODEL=s1-fast \
  REAX_NETUID=92 REAX_WALLET_NAME=reax REAX_WALLET_HOTKEY=miner1 REAX_EXTERNAL_IP=<public ip> \
  REAX_S1_FAST_RUNTIME_URL=http://127.0.0.1:8101 REAX_RUNTIME_TOKEN_FILE=<path to the shared token>
reax-miner
```

On start the miner refuses to run without a readable, non-empty `REAX_RUNTIME_TOKEN_FILE`, checks the
sidecar with an **authenticated test decision** (if it fails — sidecar down, or the two sides hold different
tokens — the miner exits with an error), checks that the hotkey is registered, publishes
`REAX_EXTERNAL_IP:REAX_EXTERNAL_PORT` on chain if it changed, publishes its commitment
`reax-s1/1:<model>:<manifest_id>` if it changed, and starts serving `POST /v1/s1/decide`. A commitment that
did not land (chain rate limits apply) is retried every 10 minutes until the chain shows it; without a
matching commitment validators never query the miner. `GET /health` returns the protocol, version, model
and manifest_id.

## Check your setup before you register (s1)

`reax-selfcheck` sends 640 published reference decisions through your runtime sidecar exactly as the miner does
and scores your answers with the validator's release constants (per-item logit distance, simulated validator
rounds). It needs no wallet and no chain access. Run it after the sidecar is healthy and before you pay for a
registration, and again after any driver, GPU or image change:

```bash
# Compose: start only the sidecar, then run the check in the miner image with the miner's settings
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml up -d s1-fast-runtime
REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b \
  docker compose -f deploy/docker-compose.s1.yml run --rm --no-deps --entrypoint reax-selfcheck miner
# virtual environment: same variables as reax-miner above
reax-selfcheck            # --json for one machine-readable object; exit code 0 = pass, 1 = fail, 2 = cannot run
```

A pass means every item was answered, no simulated round is struck and at least 99 % of simulated rounds
score. It checks fidelity only, not speed: validators also score latency (s1-fast: full speed credit at 0.5 s per
request of 8 decisions, none at 8 s), so compare the printed decisions/s with docs/CALIBRATION.md. On the measured GPUs (docs/CALIBRATION.md) the median distance is about 0.06–0.08 and the maximum stays
below 1.0. A fail means validators would score you 0 or strike you: check that the weights are the pinned
revision, that you run the pinned runtime image (not your own vLLM or an FP8 build) and that the GPU is not
shared with other work. The reference answers ship with the release (`reax_subnet/selfcheck_data/`) and match
its manifest; a release with a new manifest ships new ones.

## What your score looks like (s1)

* Per round the validator sends you 8 fresh typed decisions (a state plus one `choice`, `noul` or `score`
  question, generated fresh per miner and round — there is no answer key to leak) and compares your answers
  with its own reference runtime of the same pinned model in logit space.
* s1-fast: the validator verifies all 8 items. s1-pro: it verifies a secret random 3 of 8, drawn after your
  answers arrive; you cannot tell which items are verified, so you must answer all 8 properly.
* Per verified item the validator computes the per-item distance d: the largest difference between your
  option logits and its own (logit units, recovered from the probabilities, floored at 1e-6 — below that an
  option carries no usable fingerprint). With D = the mean d over the verified items:
  D ≤ 0.20 → the round scores `0.8 + 0.2 × speed`; D > 0.20 → the round is lost (scores 0);
  D > 0.50 or any single item with d > 2.0 → **strike: your moving average resets to 0**.
  The reference runtime on the same GPU type gives D = 0 exactly; across GPU types see CALIBRATION.md.
  Speed comes from the validator's own clock around the signed exchange (bounds s1-fast 0.5–8 s, s1-pro
  1–12 s; beyond the slow bound the request times out). Items are generated fresh per miner and round; exact
  repeats are rare (under 1 %) and a repeat can only replay an answer the real model produced.
* A new miner starts at 0. The moving average of a perfect miner is about 0.34 after 4 rounds (first weight),
  0.64 after 10 rounds, 0.86 after 20 and 0.93 after about 30 rounds (one hour at the default 2-minute
  rounds).
* A wrong or cheaper model, an FP8 build, stock (non-deterministic) vLLM for s1-pro, or a modified prompt: the
  logits drift beyond these limits (measured in CALIBRATION.md) and you lose rounds or get struck. Run the pinned
  runtime images unchanged.
* Slow answers reduce only the 20 % speed term (past the pool's slow-latency bound the request times out and
  the round scores 0).
* Downtime: every missed round counts as zero in the moving average.
* Changing your commitment (model or manifest) restarts that UID's score from zero.

## Network protection

The miner endpoint (default port 8091) is public. Application limits (5 s body deadline, 8 unfinished requests
per IP, per-validator rate limits) stop single-source abuse, but not attackers with many addresses. Recommended
for production: put a reverse proxy in front (nginx/caddy: header and body timeouts of a few seconds,
connection limits per IP, `client_max_body_size 256k`) and raise the open-file limit
(`--ulimit nofile=65536:65536` in Docker). The sidecar port must never be published; it binds to loopback and
requires the shared internal token. Note for an s1-pro sidecar: its internal vLLM listens on
`127.0.0.1:8000` **without authentication** — with host networking every local process can reach it (and it
clashes with anything else on :8000, e.g. a second Rune), so keep other local users off the machine.

## Settings

See [CONFIG.md](CONFIG.md). The ones s1 miners usually touch: `REAX_MECHANISM` (default `s1`), `REAX_MODEL`
(`s1-fast` or `s1-pro`), `REAX_S1_FAST_RUNTIME_URL` / `REAX_S1_PRO_RUNTIME_URL` (loopback sidecar URLs),
`REAX_RUNTIME_TOKEN_FILE`, `REAX_PUBLISH_COMMITMENT` (default 1), `REAX_EXTERNAL_IP`, `REAX_EXTERNAL_PORT`,
`REAX_MINER_PORT`. `REAX_MAX_CONCURRENT` (default 4) is the number of requests admitted at once; inference
runs one batch at a time, so raising it adds queueing, not throughput. With `REAX_MECHANISM=s1` the model runs
in the sidecar; `REAX_DEVICE`/`REAX_DTYPE` and `HF_HOME` only apply to the mpm1 (in-process) model.

## Upgrades

Follow the release announcements. When the protocol version or the manifest changes, upgrade before the
announced switch time; requests with an unknown protocol or manifest are refused, and a new manifest means a
new commitment (the score restarts from zero).

## The mpm1 (rc3) miner (fallback)

`REAX_MECHANISM=mpm1` runs the unchanged rc3 miner (`REAX_MODEL=qwen3-4b-instruct-2507`, protocol
`reax-mpm/1`, `POST /v1/decide`): the approved Qwen3-4B model runs **in the miner process** (no sidecar).
Keys, registration and network protection are the same as above; mechanism, wire protocol and scoring are
documented in [MPM.md](MPM.md).

| Part | Minimum (mpm1) | Notes |
|---|---|---|
| GPU | 1× NVIDIA with 16 GB (e.g. RTX 4080/4090, L4, A10) | the model needs about 8 GB in bfloat16 |
| NVIDIA driver | a driver that supports CUDA 13 (R580 or newer) | the pinned `torch==2.14.1` wheel is a CUDA 13 build. With an older driver the miner refuses to start rather than silently running on CPU |
| CPU / RAM / disk | 4 cores / 32 GB / 30 GB | model weights in `HF_HOME` |
| Platform | Linux x86_64, Python 3.12 for the hash-locked install | other platforms: `pip install .[model]` without the lock |

```bash
docker build -t reax:1.1.0rc2 .
cp deploy/miner.env.example deploy/miner.env   # fill in netuid, wallet names, public IP (sets REAX_MECHANISM=mpm1)
docker run -d --name reax-miner --gpus all --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" --ulimit nofile=65536:65536 \
  --env-file deploy/miner.env -e REAX_MECHANISM=mpm1 -p 8091:8091 \
  -v $HOME/.bittensor/wallets:/wallets:ro -v reax-models:/models \
  reax:1.1.0rc2
```

`--user` is required: key files are readable only by their owner (mode 0600), so the container must run with
your host user id. `GET /health` returns protocol `reax-mpm/1`, version and model. Do not change the inference
code path on the mpm1 mechanism: the validator computes its reference with exactly the same prompt,
temperature, batching and dtype; different batching or a different serving engine shifts bfloat16 results
slightly and uses up part of the tolerance.
