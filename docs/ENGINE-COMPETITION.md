# Engine competition `reax-eng/1` (release 1.2)

Status: runs on finney from `COMPETITIONS_START_UTC` = 2026-10-16 12:00 UTC, fixed by the release (no operator switch;
before that instant the validator sets exactly the 1.1 weights and runs no evaluation). On development networks and the
testnet `REAX_FEATURE_ENGINE_COMPETITION=1` switches it on, active at once. The share of the weights and the other
competitions are described in [COMPETITIONS.md](COMPETITIONS.md). Parameters below are release candidates that still need
calibration (section 8).

Goal: make the REAX engine faster. Competitors submit engine images; validators measure them against the reference
engine on their own GPUs; the best verified improvement receives a share of the miner weights.

## 1. What is being optimised

A **track** = one model + one engine contract + one GPU class. The first track is **`s1-fast`**: the model the subnet
already serves (Plumb-4B, pinned weights `crh225/plumb-4b@55de037`, temperature 2.07). The reference engine is the
release's s1-fast runtime (`runtimes/s1-fast`: jevk5, batch-1 CUDA graphs). This is the right first track because:

- every validator already has this model and a 24 GB+ GPU, so evaluation needs no new hardware;
- a winning engine is a drop-in runtime for every serving miner (same contract), so the subnet gets faster as a whole;
- correctness can be checked exactly as serving is checked today (logit distance against the reference).

A second track for the 27B REAX Engine (`reaxlabs/reax-engine`, Perplexity Decider v1.1) fits the same code but needs
96 GB validator GPUs and its own calibration. It is listed as a later step (section 9).

**In scope:** serving code (kernels, batching, scheduling, CUDA graphs, memory layout, tokenisation, process layout,
quantisation only if it stays inside the correctness gate). **Out of scope:** other weights, fine-tuning, changing
the prompt/readout, anything that changes answers beyond the gate, caching across requests, network access.

## 2. How a miner submits

1. Build a container image that serves the engine contract (below), with the label
   `ai.reax.engine.hotkey=<ss58 of the hotkey that will commit it>`, and push it to an allowed public registry
   (ghcr.io, docker.io, quay.io). The label binds the digest to that hotkey: nobody else can claim the same digest.
2. Commit it on chain with a hotkey registered on subnet 92 (one commitment per hotkey):
   `reax-eng/1:<track>:<image>@sha256:<64 hex>` (at most 128 bytes, so the image name is at most ~37 characters),
   e.g. `reax-eng/1:s1-fast:ghcr.io/alice/plumb-fast@sha256:…`. Tool: `reax-engine-submit <image>@sha256:<digest>`.
3. Before submitting, the miner can run the exact validator measurement locally: `reax-engine-eval --reference <ref>
   --candidate <image> --weights <model dir>`.

A competition hotkey does not serve traffic; an operator who wants both registers two UIDs. The image is pinned by
digest, so nothing can change after the commitment. Submission time = the on-chain block of the commitment.

**Engine contract** (the existing runtime sidecar contract, on a unix socket because there is no network):
`POST /internal/v1/decide` with header `X-S1M-Internal-Token` and `GET /health`, served on `$RUNTIME_UDS`
(mode 0666); token from `$S1M_INTERNAL_TOKEN_FILE`; weights read-only at `$PLUMB_MODEL_DIR`; `/tmp` is the only
writable path. The reference runtime got a 5-line change to also serve on `RUNTIME_UDS`.

## 3. How a validator measures, fairly

Every validator measures every new submission itself, on its own GPU, **relative to the reference engine measured in
the same session on the same GPU** (both engines loaded at once, interleaved). Absolute speed never counts, so an H100
validator and an RTX 4090 validator rank submissions the same way.

Per evaluation (`engrunner.evaluate`):

0. **Before anything is pulled:** the registry manifest is read (`docker manifest inspect -v`) and an image above
   **15 GB compressed** is refused (the candidate fails; an unreadable manifest is inconclusive). Every path that does not
   end in a finished evaluation removes the image again (`docker rmi`): not-owner images, failed starts, aborted pulls.
   The reference engine is measured **alone** once, before the candidate starts; this is the session baseline (below).
1. `docker pull <image>@sha256:<digest>` (allowed registries only); the image label must name the committing
   hotkey. `docker run --network none --read-only --cap-drop ALL --security-opt no-new-privileges`, pids, CPU,
   memory (no swap), file and log caps, `--gpus device=N`, weights mounted read-only, fresh socket directory and
   token per engine; the socket must be a real socket (not a symlink). Optional gVisor (`--runtime runsc`).
   **The idle engine is paused (`docker pause`) during every measurement**, so neither engine can load the GPU or
   CPU to slow the other one down.
2. Warm-up, then **timed runs**: 256 fresh decisions at concurrency 16, reference and candidate on the same items,
   two pairs in ABBA order. Score = median of the per-pair throughput ratios (candidate / reference).
3. **Correctness on every timed answer** against the reference engine's answer for the same item:
   mean logit distance ≤ 0.10, every item ≤ 1.0, top choice agrees on ≥ 99 %. This is stricter than serving
   (0.20 / 2.0) so that the serving tolerance never becomes a speed budget; a winning engine is valid for serving.
4. **Anti-cache probes:** items that repeat an earlier state with a different question, and the same question with a
   different state, sent one at a time after their base items. A cache keyed by part of the input answers from the
   cache and fails.
5. **Determinism:** 32 timed items re-sent at concurrency 1 must reproduce the timed answers within 0.05 logit
   (the reference is bit-identical at concurrency 1; honest batched engines move ~0.01).
6. **Latency gate:** 24 items one at a time; candidate p50 ≤ 1.25 × reference p50, and the same distance gates.
7. Nothing the engine reports is trusted: all times are the validator's clock; all answers are recomputed.
8. Budget: one evaluation may take at most 45 minutes; a candidate that fails its first timed run is stopped
   at once. Pull or start trouble that may be ours (registry rate limit, Docker error) is inconclusive and retried
   up to 3 times; a reference failure while a candidate is loaded is checked against the reference alone.

**Fresh, unpredictable items:** the validator generates items after the engine has started (it is already sealed,
without network) with a secret seed, and adds a secret random record id to every state, so no answer table can cover
them. Items never repeat inside the timed runs, so a legitimate exact-match cache cannot help either.

## 4. Anti-cheat summary

| Cheat | What stops it | Shown in the tests |
|---|---|---|
| Answer table / lookup | fresh items + secret record id per item | (by construction) |
| Ask an outside service ("oracle") | `--network none` | GPU test: `phonehome` engine |
| Cache by state or by question | anti-cache probes | local + GPU: `cache` engine |
| Lossy shortcut (cut input, aggressive quantisation, other model) | distance gate on every timed answer | local: `broken`; GPU: `shortcut` |
| Non-deterministic shortcuts | determinism re-run (+ distance gates) | local: `jitter` |
| Fast only when unobserved / slow when checked | correctness is checked on the timed answers themselves | by construction |
| Copy the champion's image | the image label binds the digest to its committer (dev runner: the earliest on-chain commitment) | local: `c_copy`; unit tests |
| Slow the reference down while it is measured (GPU/CPU load when idle) | the idle engine is paused during each measurement | GPU: `hog` engine, paused vs not paused |
| Keep GPU kernels running while paused (a paused container still finishes kernels it queued) | the reference is measured alone at session start; every later reference run must stay within +-12 % of that baseline, else the evaluation is inconclusive and retried; optionally (`REAX_ENGINE_NVML_CHECK=1`) the paused candidate's PIDs are sampled with `nvidia-smi pmon` during each reference window and any SM activity fails the candidate | unit tests (`tests/test_engine_hardening.py`) |
| Pull from an internal address / huge images | registry allowlist, manifest size cap (15 GB compressed) before the pull, pull timeout, image removed after the evaluation and on every failure path | unit tests |
| Rebuild the champion with a trivial change | must beat the champion by the 3 % margin | local: `c_ref`, `c_tiny` |
| Lucky measurement (winner's curse) | the title changes only after a second evaluation at least 6 h later (also across restarts) also clears the bar; it is held at the lower score | unit tests, local chain |
| Engine that degrades after winning | the champion is re-measured whenever it is challenged; a failed re-measurement vacates the title | unit tests |
| Escape from the container | no network, no capabilities, read-only, pids/memory caps, optional gVisor; the docker socket lives only in the competition worker, which holds no wallet or token (COMPETITIONS.md, "Where the competitions run") | review item (section 8) |

## 5. Who wins

- A submission counts only if all gates pass **and** it beats the reference by ≥ 3 % (min gain).
- **Champion:** the best valid submission. A challenger takes the title only with ≥ 3 % over the champion's score,
  confirmed by a second evaluation in a later cycle (fresh items); ties go to the earlier commitment.
- Each validator keeps its own champion state; Yuma consensus combines them (stake-weighted), as for serving scores.

## 6. Rewards (share of miner emissions)

The split between serving and the competitions is defined once in [COMPETITIONS.md](COMPETITIONS.md)
(`competition_split.py`). For the engine:

- **Active profile `80/20`: 10 %** of the miner part for the engine champion (profile `50/50`: 20 %). A release constant.
- Inside that share, **winner-take-most with decay**: when crowned the champion gets all of it; its part then decays
  linearly to 50 % over 14 days (100,800 blocks), and the decayed part goes back to serving. A new champion starts at
  100 % again, so the prize for beating the champion stays high while a long reign without progress pays less.
- **Runner-ups are off by default** (code supports them: halving by rank).
  Reason (review finding): an engine is an image, not weights, so a rebuilt copy of the champion or of the
  reference has a new digest and could hold a runner-up slot with one lucky measurement. Switch on only together
  with confirmation and similarity checks for runner-ups.
- **No champion, a struck or withdrawn champion, the decayed part = back to serving.** Never burned, never to the owner
  UID. Before the start time the weights are byte-for-byte the 1.1 weights (tested).
- The share is taken from the miner part only; the owner's burn fraction is unchanged. If a validator sets nothing (full
  burn without an owner UID), it still sets nothing.

## 7. What the branch implements (flag `REAX_FEATURE_ENGINE_COMPETITION`)

- `engcomp.py` (pure rules), `engrunner.py` (docker + dev runners, measurement), `engvalidator.py` (cycle, state,
  weights), `engcli.py` (`reax-engine-eval`, `reax-engine-submit`), hooks in `s1validator.py` and `cli.py`,
  config in `config.py`, the share layer in `competition_split.py`. On finney the competition is on by release (a
  validator needs `REAX_ENGINE_WEIGHTS_DIR` and `REAX_SPEC_SEALED_DIR`; the reference engine is the release constant
  `config.ENGINE_REFERENCE_IMAGE`; without them it serves only; the flag is refused so that no validator can
  switch it off); elsewhere the flag decides. A validator on a custom endpoint whose genesis hash is finney's refuses to
  start (use `REAX_NETWORK=finney`), so every mainnet validator runs the same rules and start time.
- The evaluation cycle (every 6 h, at most 4 new submissions, oldest first, plus re-measurements) runs in the validator
  process (`REAX_COMPETITIONS_MODE=local`) or in `reax-competition-worker` (`remote`, `worker.py`; the validator then
  reads `engine-competition.json` through `remote.py`). Serving rounds go on during a cycle (the GPU is shared with
  the serving runtimes), and weights are still submitted between evaluations.
- Tests: `tests/test_engcomp.py`; local-chain proof `scripts/localnet/e2e_engine.py` with the synthetic engines in
  `scripts/localnet/fake_engine.py`.

## 8. Before switching on (calibration and review)

1. **Calibrate on the s1-fast GPU classes** (4090, 5090, H100, PRO 6000): honest engines' distance and determinism
   distributions, and the run-to-run noise of the speed ratio (sets min gain and margin; the GPU test measured it once).
2. **Security review of the runner**: docker flags, socket directory, image pull, resource caps;
   decide whether gVisor is required. Validators run third-party images: this must be opt-in and documented.
3. **Validator cost:** ~5 min of GPU per evaluation; at most 4 new submissions per 6 h = ≤ 1.5 GPU-h/day per validator.
   **GPU memory on a shared 96 GB card:** the serving runtimes (s1-fast ~12 GB, s1-pro) stay resident. A measurement adds
   two s1-fast engines (reference + candidate, ~12 GB each, so 28 GB with headroom); the idle one is paused, which stops
   its compute but keeps its memory. Before each cycle the runner reads the free GPU memory and skips the cycle (retried
   next interval, no candidate is marked failed) if less than 28 GB is free. **To check on the real host:** `nvidia-smi`
   with all runtimes up and one `reax-engine-eval` run.
4. Testnet run with at least two independent validators to confirm they agree on the champion.
5. Release: version bump, release notes, miner docs, and a short public page ("make the REAX Engine faster").
6. Licence condition for submissions (winners' source must be published under Apache-2.0 before a merge into the
   reference runtime); the payout itself does not require a merge.

### Open items from the code review (not fixed on the branch)

- The engine's socket directory is a bind mount on the worker's disk; a hostile engine could fill it while it runs
  (the directory is removed afterwards). Fix before release: a size-limited mount for it (XFS project quota or a small
  loop-mounted filesystem); until then keep `REAX_COMPETITION_DIR` on its own filesystem.
- Independent validators measure on different GPUs; the local chain showed two validators agreeing, but the
  real-GPU agreement between validators needs the testnet run (step 4 above).
- Container escape is mitigated, not excluded: decide whether gVisor (`runsc`) is mandatory for validators.
- Runner-up payouts stay off until there is a similarity check for rebuilt copies.

## 9. Later

- Track `pplx-27b` for the 27B REAX Engine (96 GB GPUs; the reference = current reax-engine FP8 build).
- Adopting a champion as the new reference runtime (new manifest; serving miners switch).
- A signed public scoreboard per validator (helps miners and audits).
