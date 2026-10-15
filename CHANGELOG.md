# Changelog

## 1.2.0rc1 — unreleased

Weights version 1200 (scoring changes: validators on an older release are rejected by the chain once the owner raises
`weights_version`). The wire protocol is unchanged.

**What this release does, in plain English.** From **16 October 2026, 12:00 UTC** the miner weights are split 50/50:
half goes to serving (inference), half to competitions. Before that instant a 1.2 validator sets exactly the 1.1
weights.

* **50 % inference, 50 % competitions (profile `50/50`, active).** Inference 0.50; the engine champion 0.20; two use-case
  tracks 0.15 each (`demand_forecasting` and `risk_assessment`, 16–30 Oct 12:00 UTC; the next pair follows with a later
  release). Inside a track ranks 1–3 get 75 / 17 / 8 %. The engine champion's share halves linearly over 14 days. Any
  competition share nobody earns (no champion, a missing rank, a score at or below a track's entry floor, a struck
  champion, the decayed part) goes back to the inference part: never burned, never paid to the owner UID. The older
  `80/20` profile ships as an inactive profile. New share layer `competition_split.py`, see docs/COMPETITIONS.md.
* **The s1-pro pool is ON by default on every network, finney included.** Inside the inference part the shares stay
  s1-fast 0.35 / s1-pro 0.65. `REAX_FEATURE_S1_PRO_POOL` stays a fixed release variable: operators cannot set it on
  finney; development networks may still switch it off with `REAX_ALLOW_DEV_MODELS=1`. Every validator needs an
  80–96 GB GPU for the s1-pro reference runtime.
* **An unclaimed pool share is no longer burned.** A pool with no miner at or above `REAX_MIN_EMA` passes its share
  to the other enabled pools pro rata (s1-pro empty → s1-fast gets the whole miner part). Only when no pool has a
  miner does the miner part go to the owner UID, as in 1.1. `REAX_BURN_FRACTION` is unchanged.
* **Engine competition `reax-eng/1`** (fixed on for finney from the start time): validators measure submitted engine
  images against the reference engine in the same session (no network, read-only, fresh perturbed items, correctness on
  every timed answer, anti-cache probes, determinism re-run, latency gate, idle engine paused). The reference engine is a
  release constant (`config.ENGINE_REFERENCE_IMAGE`); `REAX_ENGINE_REFERENCE` is a development-network override. New
  tools `reax-engine-eval` and `reax-engine-submit`; the s1-fast runtime can also serve on a unix socket
  (`RUNTIME_UDS`). A validator that uses a custom endpoint of finney refuses to start (set `REAX_NETWORK=finney`).
  See docs/ENGINE-COMPETITION.md.
* **Specialist (use-case) tracks** `reax_subnet.spec`, commitment `reax-spec/1:<track_id>:<repo>@<revision>`: weights-only
  submission rules, a pinned network-less vLLM evaluation on fresh sealed draws, the 0.75 competence + 0.25 calibration
  score, the margin/second-draw confirmation, tensor-level hashes and weight-delta near-duplicate checks, persisted state,
  and the `reax-spec-eval` miner CLI (docs/SPECIALIST-TRACKS.md). The validator runs it next to the engine competition;
  its schedule is the one in `competition_split.py`. A per-track entry floor (`TRACK_ENTRY_FLOOR`, none yet) keeps
  weak submissions from being paid.
* **One GPU, one measurement at a time.** Engine measurements and specialist evaluations share one GPU lock, run as a
  background task (serving rounds are never paused) and each skips its cycle cleanly when the card has too little free
  memory (engine 2 x 14 GB, specialist up to 30 GB).
* **A finney validator without the competition setup does not crash.** If the sealed pools, the engine weights or the
  published reference image are missing, it logs one warning and serves only (100 % inference weights).
* `deploy/validator-1.2/`: compose file and README for both reference runtimes (s1-fast on 127.0.0.1:8101, s1-pro on
  127.0.0.1:8102) on one 96 GB GPU, plus everything the competitions need: docker socket (with an honest security
  note), the engine and sealed-data directories, the spec work/state directories and the Hugging Face cache. New
  Dockerfile target `validator-competitions` (validator + docker CLI).
* `docs/hardware.md`: hardware per pool and an s1-pro miner quickstart. `scripts/make_selfcheck_reference.py` writes
  the `reax-selfcheck` reference answers of a pool from its reference runtime.
* New workflow `.github/workflows/release-images.yml`: builds and pushes the s1-fast runtime, the REAX image and the
  s1-pro runtime to GHCR on a `v*` tag (or by hand) and prints the digests in the job summary.

* Optional `reax-router` service: a client-facing router that forwards signed requests to miners (read-only
  validator score selection, sequential failover, circuit breakers, bounded shadow verification with a separate
  strike ledger). Disabled until an operator configures and enables it; see docs/ROUTER.md.
* CPU-only pinned Plumb prompt/token accounting in the router; token counts are never taken from miner-supplied usage.
* The runtime client retries once on a dropped sidecar connection.
* The repository moved to the `reaxlabs` organisation; source and GHCR references use `reaxlabs`.
* MIT licence, `CONTRIBUTING.md`, issue and pull-request templates.
* Validator robustness: every chain read has a deadline (a stalled RPC skips the round instead of freezing the loop,
  and a hung metagraph read is not started twice); late `set_weights` outcomes are logged; the sidecar client reads
  runtime responses split across TCP segments in full; integers too large for a float are a contract error; a
  committed miner that is not queried (other manifest, disabled pool, malformed commitment) is explained once in the log;
  a recycled UID no longer inherits the previous hotkey's role-change shield cooldown.
* Compose files cap Docker logs (json-file, 5 x 50 MB per service).
* `reax-selfcheck`: checks a miner's runtime sidecar against 640 published reference answers with the release
  scoring constants before registration (docs/MINER.md). No wallet or chain access.
* Calibration addendum: the s1-fast reference path on RTX A6000, L40S and RTX 5090 stays inside the release tolerances.

## 1.1.0rc2 — 2026-10-05

Weights version stays 1100; the wire protocol is unchanged, so rc1 and rc2 validators and miners interoperate.

* **Security fix:** a request carrying the `X-Bittensor-Hotkey` header twice could pass the miner's validator
  allow-list with one value and the signature check with another. Miners now refuse duplicate `X-Bittensor-*`
  headers and serve only when the verified signer is the allow-listed validator. Upgrade miners.
* A UID that keeps changing its commitment no longer keeps a frozen score; weights count only roles on the current manifest.
* The reference step of a validator round is time-bounded, so a hung sidecar skips the round with an error instead
  of stalling the validator.
* Runtime URLs must parse as loopback; `REAX_ALLOW_PRIVATE_IPS` is refused on every mainnet network name.
* Docker: targets `miner` (default) and `validator`; caches live under /tmp so the GPU path works for any `--user`.
  CI builds, smoke-tests and then pushes `reax-miner`, `reax-validator` and `reax-s1-fast-runtime` images to GHCR.
* Docs: rollback uses the 1.1 image with `REAX_MECHANISM=mpm1`; emission-brake caveats; per-UID role confirmation.

## 1.1.0rc1 — 2026-10-03

Weights version 1100.

* **New default mechanism `reax-s1` (`REAX_MECHANISM=s1`).** A miner commits on chain to one role,
  `reax-s1/1:<model>:<manifest_id>`, and serves `s1-fast` (Plumb-4B) or `s1-pro` (Surogate Rune 26B-A4B v3)
  with the reference prompt, readout and typed-decision answer schema (choice / noul / score with probabilities).
* Versioned model manifests pin weights (every file's sha256), runtime image digests, package pins and the readout.
* Runtime sidecars (`runtimes/s1-fast`, `runtimes/s1-pro`) run the reference readout on loopback.
* Verification in logit space: per-item distance, round mean ≤ 0.20 scores `0.8 + 0.2 × speed`, mean > 0.20 loses
  the round, mean > 0.50 or any item > 2.0 is a strike. Thresholds and latency bounds come from GPU calibration
  (docs/CALIBRATION.md).
* Pools with fixed shares; the s1-pro pool is OFF by default (development networks only in this release).
* Role changes take effect only when two consecutive metagraph reads agree for that UID.
* The runtime token is required in s1 mode and verified with an authenticated test decision at startup; the miner
  retries its commitment every 10 minutes until the chain shows it; a metagraph read that changes more than 20 %
  of committed roles is skipped until confirmed; a full burn without an owner UID sets no weights.
* The `mpm1` mechanism stays available unchanged as `REAX_MECHANISM=mpm1`.
* Fixed: the Docker image crashed on GPUs because torch compiles Triton helpers with a C compiler the slim base
  did not have; the image now installs gcc.

### Upgrading from 1.0.0rc3

1. Upgrade validators first (they understand both mechanisms; `REAX_MECHANISM` selects which one runs).
2. Then miners start their runtime sidecar and publish their commitment `reax-s1/1:<model>:<manifest_id>`
   (docs/MINER.md). A miner without a matching commitment is not queried while validators run s1.
3. Then the subnet owner raises `weights_version` to 1100.

Rollback: run the 1.1 images with `REAX_MECHANISM=mpm1` set explicitly, lower `weights_version` to 1000 and move the
validator state file aside (or use a fresh `REAX_STATE_PATH`).

## 1.0.0rc3 — 2026-10-03

Weights version 1000.

* Probabilities are read at protocol temperature T = 4 (`softmax(logits / T)`), which defeats a "copy the top choice" cheat.
* Rounds use 12 items with a 10-of-12 gate.
* Fixed: the metagraph guard no longer stops a validator forever after an owner trim; the global cap on unfinished
  requests is 4096, the per-IP cap stays 8.
* Development mode is refused on finney; a full burn without an owner UID sets no weights.
* `scripts/calibrate.py` reports round-pass probabilities and adds an "oracle top choice" cheat and `--cheat-model`.

## 1.0.0rc2 — 2026-10-03

* Fixed: a miner answering with a huge integer probability made every validator round abort. Parsing now rejects it.
* Fixed: slow-drip request bodies could hold miner connections forever (5 s body deadline, per-IP and total caps).
* Validators score 6 rounds before their first weights.
* Docker: run with `--user`; CPU fallback is refused; bounded `set_weights` wait; directory fsync for state.

## 1.0.0rc1 — 2026-10-03

First release candidate of the Minimal Production Mechanism (`reax-mpm/1`, weights version 1000).

* Miner: aiohttp endpoint `POST /v1/decide`, request verification, validator-permit and stake check, per-validator
  concurrency and rate limits, signed responses.
* Validator: fresh items per miner per round, local reference computation, total-variation scoring with a fidelity
  gate and a speed term, persisted EMA, owner-UID fallback, SSRF-safe outbound requests.
* Features from the long-term design are disabled behind `REAX_FEATURE_*` (docs/FEATURE-FLAGS.md).
* Local subtensor end-to-end harness, adversarial miners and a threat model.
