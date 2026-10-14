# Changelog

## 1.2.0-dev — unreleased

* Engine competition `reax-eng/1` behind `REAX_FEATURE_ENGINE_COMPETITION` (refused on finney; local chains and the
  testnet only with `REAX_ALLOW_UNRELEASED_FEATURES=1`): competitors commit engine images by digest, validators measure
  them against the reference engine in the same session (no network, read-only, fresh perturbed items, correctness on
  every timed answer, anti-cache probes, determinism re-run, latency gate) and give the confirmed champion a share of
  the miner weights (winner-take-most with decay). New tools `reax-engine-eval` and `reax-engine-submit`; the s1-fast
  runtime can also serve on a unix socket (`RUNTIME_UDS`). With the flag off the weights are unchanged.
  See docs/ENGINE-COMPETITION.md.

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
