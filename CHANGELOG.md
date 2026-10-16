# Changelog

## 1.2.0rc5 — engine track held; owner approval before any competition award (weights version 1220; validator and worker)

Miners need no update. Validators and the competition worker must update before the engine track could pay.

* **The engine track is held in code.** `ENGINE_TRACK_START_UTC = None` (`competition_split.py`), separate from the
  use-case tracks, which still start at 2026-10-16 12:00 UTC. While it is held: no engine cycle runs in the validator or
  in `reax-competition-worker`, `reax-engine-submit` refuses on finney, and the validator pays no engine champion, also
  in remote mode when the synced `engine-competition.json` names one. The engine share goes to inference.
* **Owner approval before payment.** A winning entry is paid only after the subnet owner has checked and approved it:
  the engine champion and every paid rank of a use-case track need an Ed25519-signed approval for exactly (track,
  hotkey, image digest or `repo@revision`). Without it the share goes to inference (never burned; lower ranks do not
  move up) and the validator logs `awaiting owner approval`. Approvals are read from `REAX_AWARD_APPROVALS` (validator
  host only, never the worker) and checked against `award.AWARD_APPROVAL_PUBKEY` or `REAX_AWARD_APPROVAL_PUBKEY`. A
  missing key or file, a bad signature or an expired approval pays nothing. Always on for finney.
* New tool `reax-award-approve` (`keygen`, `sign`, `verify`). `pycryptodome` (already installed through bittensor) is now
  a declared dependency.
* With no champion and no approved leader the weights are exactly the rc4 weights.
* **Specialist tracks: payout ranks follow the champion order.** Ranks 1-3 of a track (75 / 17 / 8 %) are the
  champion, then the former champions, most recently crowned first, as SPECIALIST-TRACKS.md describes. 1.2.0rc4
  re-sorted them by their latest scores, which come from different draws: a former champion holding 90 from an old
  draw outranked the champion that beat it with 85 on a newer draw and took 75 % of the track share. The evaluator
  now hands out `competition_split.RankedLeaders` and `track_payout` pays them in that order; entry floors still apply
  (the leaders after a floored one move up). Plain score lists are still ranked by score.
* **Specialist tracks: no award without the near-duplicate check.** When the declared base model is not on the
  validator (an unknown name or revision), or shares no weight with the model, the submission is *pending
  verification*: kept, not evaluated, ranked or paid, and checked again every cycle until the base is there.
  1.2.0rc4 let it pass with only the exact hash check. A champion that was never checked is held the same way. Near
  duplicates are now compared by the tensor hash of the resolved base, not the declared `base_model` text, so a
  mirror or another name of the same base does not hide a copy. Exact copies are refused as before, with or without
  the base. Operators: `REAX_SPEC_BASE_DIR` must hold the bases miners declare (`pending verification` in the log).
* **Specialist tracks: the worker fetches declared base models itself.** A missing base is downloaded into
  `REAX_SPEC_BASE_DIR` (default `${REAX_COMPETITION_DIR}/spec-bases` in both compose files) with the submission download
  rules (pinned revision, public, safetensors only, size caps, allowlisted architecture, no `auto_map`; any licence),
  into a temp directory renamed into place. Gated, private, missing or oversize bases stay pending with the reason and
  are retried every 6 cycles; at most 40 GB of bases per cycle and 50 GB must stay free. `REAX_SPEC_BASE_PRESEED`
  fetches a list in the background at start (SPECIALIST-TRACKS.md, "Base models").
* **Spec state rollback compatibility.** The state file stays at version 1; new fields (`first_incumbent`, `base_id`)
  have defaults and unknown keys are ignored, so an older release loads a newer file after a rollback and the other
  way round.
* **Specialist tracks: confirmation against a champion crowned after the first draw** (contributed by Ihor-52466,
  reaxlabs/reax-subnet#3). A challenger whose first draw was measured against another incumbent, or against a vacant
  track, now has to beat the incumbent it meets on the confirmation draw by the full margin `max(1.5, 2 x SE)`. Before, it needed half of its first-draw delta, and on a
  vacant track that delta is the entry floor itself: a challenger that cleared the floor one cycle after another one
  needed +16.85 (risk_assessment) or +12.4 (demand_forecasting) over it once that one had been crowned. The submission
  state records the incumbent of the first pass (`first_incumbent`); state files written before fall back to the
  first-pass history. Scoring and the wire protocol are unchanged.

* **The candidate's vLLM share leaves room for what vLLM uses outside it.** vLLM 0.30 takes its memory snapshot after
  the CUDA context exists and allocates more after its memory profile (kernel warm-up, first batches), so a model capped
  at the 25 GB budget really took about 26.2 GB. Measured with Qwen3.5-9B text-only bf16 (8.95 B parameters) on an RTX PRO
  6000 Blackwell: 1.22 GiB outside the share, which explains the reported CUDA OOM a few seconds into scoring with
  26,605 MiB free. The cap is now `budget - 1.5 GiB` (`GPU_OVERHEAD_GB`): the same model gets 0.245 instead of 0.261,
  peaks at 24.46 GiB and scores all 513 public dev items. Models below the cap (4 B and smaller) are unchanged.
* **The free-memory gate before a specialist cycle uses `REAX_SPEC_GPU_BUDGET_GB`** instead of the release default, so
  an operator who lowers the budget is not still asked for 25,600 MiB free.

Review fixes (rc5 review F1-F6):

* **Specialist near-duplicate check works across dtypes, with a coverage rule.** The delta sketch compares every float
  tensor whose name and shape match the base, whatever its dtype (both sides are decoded to float). An honest
  fine-tune saved in F32 or F16 of a BF16 base is now verified (rc5 left it *pending verification* forever), and a
  copy that upcasts the parent and keeps one small tensor in the base dtype with noise is now *derived*. The compared
  tensors must hold at least 90 % of the base's float parameters, else the submission is pending with the coverage in
  its reason. FP8 tensors are never compared and say so in the reason. A submission held under the rc5 rule while
  its base is on the host is queued once more for the check (`sketch_rule` in the state, default 0).
* **Former champions that were never delta-checked are not ranked** (`compute_leaders`).
* **The competition worker refuses a custom finney endpoint** (the validator's genesis check), so it can never run in
  development mode against mainnet.
* **The award key:** the compose comment now says that an empty `REAX_AWARD_APPROVAL_PUBKEY` means the release key, and
  the validator logs an ERROR at start on finney if neither key is valid. `AwardGate.approved` never raises (returns
  False).
* **A candidate whose vLLM container stops at start-up because the GPU has too little free memory** ("Free memory on
  device ... is less than desired GPU memory utilization") is a host problem: it does not count toward the model's
  `MAX_ATTEMPTS` and is retried next cycle.

Engine-track rules, prepared for its opening (the track stays held; nothing here pays out):

* **A pending challenger is protected.** It is displaced only by a submission that beats its pooled score by the same
  3 % margin required against the champion. A later near-copy at noise level, or one that misses the champion's bar,
  no longer resets its confirmation (`update_champion` used to clear the challenger whenever the top contender did not
  clear the margin).
* **The champion's score is held at the minimum of its measurements** instead of the latest one, so one lucky
  re-measurement cannot raise the bar.
* **Confirmation is a second fresh session, not a later cycle.** New containers and new secret items, run right after
  the first in the same cycle (`confirm_gap_blocks` 1800 -> 0); the champion is measured again with the challenger.
  Each measurement carries a session id, so the same one is never counted twice. Crowned at the lower of the two;
  the history records the pooled score and the number of sessions.
* **Frozen thresholds in one block** (`engthresholds.py`: margin, min gain, agreement gate, distance budget, items,
  runs, confirmations). Its sha256 is logged at start and written into the state file as `thresholds_sha256`; a state
  written under other thresholds logs a warning. A test pins the digest.
* Acceptance test: 1000 rebuilt copies of the reference with seeded measurement noise are never crowned; a real +10 %
  engine in the same run is.

## 1.2.0rc4 — capacity probe timing (weights version 1220; validator-only, miners need no update)

The wire protocol and the weights are computed as in 1.2.0rc3; only when a UID's probe window opens changes.

* **A UID's probe window opens at the validator's first send to it.** 1.2.0rc3 opened one window for every UID and then
  opened all 16 lanes of each UID in turn (~0.2 ms of validator CPU per lane): at ~175 UIDs the last UIDs' first
  requests left 0.6 s (bench) to 2.5 s (mainnet report) after their window had opened, so fast miners were measured at
  75-90 % of their speed. Now each UID's window opens at its own first send and has the pool's length, and lanes open
  one per UID per loop pass, so every UID's first request leaves within ~30 ms (175 UIDs, bench: measured/true 0.95-0.97
  median, worst 0.89, was 0.82-0.92 median, worst 0.73).
* **Siblings still share one machine.** All UIDs' windows open within the send skew; a probe whose first sends spread
  over more than 250 ms is dropped like a lagging one (`capacity probe skipped: ... first sends spread over`).
* `scripts/bench_probe.py` emulates the internet (`--rtt-ms`, per-machine speed `--dps LO-HI`, `--pro-uids`,
  `--siblings`, `--loop-load`, `--gil-load`) and prints per-lane timing with `--trace` (`REAX_PROBE_TRACE=1`).

## 1.2.0rc3 — unreleased: capacity scoring v2

Weights version **1210** (scoring changes; 1.2.0rc2 sent 1200). The wire protocol is unchanged.

* **Capacity is paid in a straight line with verified speed**. 1.2.0rc2 paid `ema x min(1, capacity /
  target)`: a GPU faster than the s1-fast target (40/s) earned once per hotkey when it was split, because every share
  still measured at least 40/s. Now the factor is `min(max_factor, capacity / target)`:
  * s1-fast and s1-pro: up to 16 units (640/s and 160/s), far above one machine. Several hotkeys on one machine measure
    and earn at most what one hotkey on it does, and a hotkey with more GPUs earns more.
  * s1-lite and s1-cpu: up to 4 units (32/s and 4/s), near the best honest node of the tier. Their small models run
    10–100x faster on a big GPU; a higher cap would let one GPU take most of a cheap pool.
* **The tier-order rule is unchanged**: the average reward per miner in a tier is at most half of the tier above.
  The 80 / 15 / 5 tables hold however fast GPU miners get.
* **The probe can measure fast machines.**
  * All of a pool's queryable UIDs are probed at the same instant, including UIDs the reference-cap rotation rests and
    UIDs that did not answer this round (a UID that is down measures 0).
  * Requests are prepared and serialised before the window, and the lanes' connections are opened before it.
  * Only answers inside the common window count, so a hotkey that stops early cannot free the machine for its
    siblings.
  * 16 lanes per UID for s1-fast and s1-pro (8 s1-lite, 4 s1-cpu), requests for 4x the UID's previous measurement
    (at least the target, at most the cap). Verification stays at most `verified_max` decisions per UID and probe.
  * s1-fast window 3 -> 5 s, s1-pro 5 -> 8 s, s1-cpu 4 decisions per request.
* **The validator's CPU stays out of the measurement.** Inside the window it only sends and records. Signatures and
  answers are checked after the window; one forged or malformed answer sets that UID's probe to 0. If the validator's
  event loop still lags (p95 above 50 ms) or the probe reaches `REAX_PROBE_CEILING_DPS`, the probe is dropped with no
  capacity update. `scripts/bench_probe.py` measures what a validator host can take.
* **Validator deploy:** `ulimits: nofile 65536` for the validator service (a probe holds about 16 sockets per s1-fast /
  s1-pro UID). The reference budget of a probe starts after its requests are prepared.
* **Miners: new defaults, and how to get measured fully.**
  * Update to 1.2.0rc3, or set `REAX_PER_CALLER_CONCURRENCY=16` and `REAX_PER_CALLER_PER_MINUTE=240` now (old
    defaults: 2 and 20).
  * Batched (vLLM) or multi-GPU setups also raise `REAX_RUNTIME_INFLIGHT` (new, default 16: decisions in flight to the
    runtime) and `REAX_MAX_CONCURRENT` (default 4 -> 8).
  * `REAX_MAX_PENDING_PER_IP` 8 -> 32.
  * The miner's queue is served round-robin per validator, so one validator's burst cannot starve another validator's
    probe.
* **Transition.** Miners on the old defaults keep working: extra probe lanes are refused (no penalty) and such a miner
  is measured up to about 55/s. Faster miners are measured in full only after they update or set the two settings.
  Capacity is a moving average (α 0.3) and probes run about every 20 minutes, so a node reaches its new level within
  1–2 hours. Pay inside s1-fast moves from "equal per hotkey" to "in proportion to verified speed".

## 1.2.0rc2 — unreleased

* Entry floors per use-case track: the s1-fast model's score on the sealed pool + 1.5 points (demand_forecasting 24.8, risk_assessment 33.7).
Weights version stays 1200 (no scoring change). The wire protocol is unchanged. Security hardening of the competitions and a
way to run them outside the validator process, from the security review of 15 October 2026.

* **Review fixes (release review of 15 October 2026).** *Engine:* a worker container without `nvidia-smi` no longer aborts the
  cycle (the free-memory preflight and the optional per-PID check warn and skip); the compose worker service now gets
  `nvidia-smi` through a `utility`-only GPU reservation. OCI manifests and image indexes (buildx default) are sized
  (`unknown/unknown` attestations skipped); the speed denominator is `max(reference run, solo baseline)` per sample, the
  drift check is one-sided (only a slower reference is inconclusive) and also covers the latency-gate reference runs; the
  runner removes only images it pulled itself and never the reference or vLLM image. *Specialist:* vLLM starts through
  `sh -c 'umask 000; exec vllm serve "$@"'` so the worker uid can connect to its socket; a failure that is the host's (docker
  refused, socket not connectable, never healthy, every model of a group of several failing) no longer counts toward
  `MAX_ATTEMPTS`; the tokenizer must define the markers of the architecture's fixed chat template (refused before any GPU run
  otherwise), `{{ bos_token }}` renders empty when a tokenizer has none, Gemma 4 left the architecture allowlist. *Remote
  mode:* the engine weights and the spec leaders never raise on a corrupt state file (outage warning, inference only; a
  corrupt spec file no longer keeps paying the last leaders); non-finite or overflowing `saved_at` is rejected; champion
  fields are type-checked. *Config:* on finney `REAX_COMPETITIONS_MODE=local` must be set explicitly (unset = serving-only,
  with an error if rc1-style in-process variables are present). Worker `HOME=/tmp`. Documented: the heartbeat shows
  liveness, not freshness, and the worker parses miner input while holding the docker socket.

* **Competitions out of the validator process.** New `REAX_COMPETITIONS_MODE` = `local` (today's behaviour, the default) |
  `remote` | `off`. In `remote` the validator runs no competition cycle, needs no docker socket and no sealed data, and each
  round reads `engine-competition.json` and `spec-state.json` from `REAX_COMPETITIONS_STATE_DIR` (read-only). It pays
  exactly like local mode (engine champion, 14-day decay, confirmations; track leaders, entry floors, 75/17/8;
  withdrawn or deregistered competitors drop out to inference) and re-checks on every read that the hotkey still holds its uid
  and the commitment still matches. A missing, invalid or stale (> 26 h) file pays that competition's share to inference,
  with one warning. New CLI `reax-competition-worker [--engine] [--spec]`: only the cycles, chain read-only, no wallet, no
  weights, same GPU lock, preflights and anti-cheat, state files written atomically with a heartbeat. It runs as a container
  next to the validator without wallets or tokens, or unchanged on a separate keyless GPU box. `deploy/validator-1.2` now has
  a `validator` service without the docker socket and a `competition-worker` service with it; `deploy/competition-worker`
  is the compose file for the keyless box. New Dockerfile target `competition-worker`; the plain `validator` target is used
  by the compose file (no docker CLI).
* **Specialist tracks (security review F3, F6, F9, F10, F13).** The vLLM socket directory is created under
  `REAX_SPEC_WORK_DIR/sock/` (the same-path mounted, host-visible work directory), not in the container's `/tmp` (every
  specialist evaluation would have failed in the compose deployment). The near-duplicate check (`delta_sketch`, compare,
  save) no longer runs on the event loop. A submission's chat template is never rendered: a fixed template per architecture
  family ships in the package (`spec/templates/`), `*.jinja` is not downloaded, `chat_template.*` and the `chat_template`
  keys are removed from the validator's copy, `tokenizer.model`/`*.tiktoken` are removed when `tokenizer.json` exists;
  Gemma 4 is refused until its template is shipped. The vLLM container runs as `REAX_SPEC_UID` (default 10003:10003, not the
  validator's uid) with `--cpus 8 --memory-swap 48g --ulimit nofile=4096:4096`, local log driver capped at 2 x 10 MB, and
  `REAX_SPEC_DOCKER_RUNTIME`; downloaded model directories are made readable for that uid. vLLM responses are capped at 1 MiB.
  A specialist cycle is skipped (nothing recorded) with less than 60 GB free on the work directory's filesystem
  (`REAX_SPEC_MIN_FREE_GB`). One mechanism for the confidential-compute UVA fix: `REAX_VLLM_DISABLE_UVA=1` with the
  packaged `spec/ccfix/sitecustomize.py` (copied into the container's run directory); `REAX_CCFIX_DIR` is only for the
  s1-pro runtime in the compose file. The track entry floor of the evaluator now also honours
  `competition_split.TRACK_ENTRY_FLOOR` (`None` for now, so nothing changes yet).
* **Engine competition (F4, F5).** An image above 15 GB compressed (registry manifest, read before the pull) is refused;
  every path that does not end in a finished evaluation removes the image again (not-owner images, failed starts, aborted
  pulls). Every session measures the reference alone first; later reference runs more than 12 % away from that baseline
  (`TrackParams.reference_drift`) make the evaluation inconclusive. `REAX_ENGINE_NVML_CHECK=1` additionally fails a
  candidate that keeps using the GPU while paused. A candidate whose socket is replaced or that uses the GPU while paused
  is now a failed evaluation, not an internal error. The GPU free-memory preflight (2 x 12.5 GB) now actually runs for
  the docker runner (it was defined on the runner protocol instead of `DockerRunner`).
* `config.ENGINE_REFERENCE_IMAGE` is the published 1.2.0rc1 s1-fast runtime (`ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:d857a90e…`):
  a finney validator with the host side set up now runs the competitions instead of serving only.
* Docs: COMPETITIONS.md ("Where the competitions run"), ENGINE-COMPETITION.md, SPECIALIST-TRACKS.md, CONFIG.md, and the
  Security section of `deploy/validator-1.2/README.md` (the socket now lives only in the worker container).

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
  to the other enabled pools pro rata (s1-pro empty → s1-fast gets the whole miner part; same for tiered pools and tiers). Only when no pool has a
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
  memory (engine 2 x 12.5 GB, specialist up to 25 GB; measured: s1-fast 12.3 GB + s1-pro 58.7 GB).
* **A finney validator without the competition setup does not crash.** If the sealed pools, the engine weights or the
  published reference image are missing, it logs one warning and serves only (100 % inference weights).
* `deploy/validator-1.2/`: compose file and README for both reference runtimes (s1-fast on 127.0.0.1:8101, s1-pro on
  127.0.0.1:8102) on one 96 GB GPU, plus everything the competitions need: docker socket (with an honest security
  note), the engine and sealed-data directories, the spec work/state directories and the Hugging Face cache. New
  Dockerfile target `validator-competitions` (validator + docker CLI).
* `docs/hardware.md`: hardware per pool and an s1-pro miner quickstart. `scripts/make_selfcheck_reference.py` writes
  the `reax-selfcheck` reference answers of a pool from its reference runtime.
* Confidential-compute GPUs: `REAX_VLLM_DISABLE_UVA=1` loads `runtimes/vllm-ccfix/sitecustomize.py` (vLLM without UVA
  buffers, numerics unchanged); wired into the validator compose file and the specialist evaluator's vLLM containers.
  The s1-pro validator defaults are now `RUNE_GPU_MEM_UTIL=0.60` and `RUNE_MAX_MODEL_LEN=8192` (58.7 GB, bit-identical).

* **Tiered pools and capacity scoring are ON by default on every network, finney included** (like s1-pro;
  docs/DESIGN-POOLS.md). Two new pools next to s1-fast and s1-pro:
  * `s1-lite`: small GPUs incl. NVIDIA Jetson Orin. It serves the s1-fast model (Plumb-4B v5) as the author's Q8_0 GGUF on
    llama.cpp CUDA.
  * `s1-cpu`: CPU-only machines incl. ARM. It serves Gutsy 0.8B v0.4 Q8_0 GGUF on llama.cpp CPU.
  * Tier budgets inside the **inference** part: GPU 80 % (s1-fast 35 / s1-pro 65 inside it), small GPU 15 %, CPU 5 %. A
    tier-order rule keeps the average reward per miner in a tier at most half of the tier above it (the excess moves up).
    The order of the weight computation is: pools and tiers, then the serving vector, then the 50/50 competition layer,
    then owner burn as in 1.1. A pool or tier without an eligible miner passes its share pro rata to the other pools
    (never burned).
  * Inside each pool, weight follows the correctness-gated score times verified throughput, up to a per-pool target.
    Capacity probes load every UID of a pool at the same time, so hotkeys that share one machine share its throughput.
  * Reference decisions per round are capped per pool (`s1-lite` 240, `s1-cpu` 160 decisions, fair rotation; validator
    local `REAX_REFERENCE_CAP_S1_LITE` / `_S1_CPU`). The s1-lite tolerance (`tau_mean` 0.30) is set for a bf16 reference, so
    a validator verifies s1-lite against its s1-fast runtime (`REAX_S1_LITE_RUNTIME_URL`, default `127.0.0.1:8101`, no
    extra GPU memory) and s1-cpu against a CPU reference service (`127.0.0.1:8104`, 8 threads) that
    `deploy/validator-1.2` now starts. A validator without a tiered reference does not crash: it logs a warning and
    leaves that pool unscored.
  * `REAX_FEATURE_TIERED_POOLS` is a fixed release variable like `REAX_FEATURE_S1_PRO_POOL`: operators cannot set it on
    finney (development networks may switch it off). `REAX_ALLOW_DEV_MODELS=1` is now refused on every spelling of
    finney's endpoint.
  * New runtime sidecar `runtimes/s1-gguf` (CPU image for linux/amd64 and linux/arm64, CUDA image, Jetson recipe), slim
    multi-architecture miner image (Dockerfile target `miner-s1`, `requirements/lock-s1.txt`), cross-backend calibration
    scripts (`scripts/calibration_gguf`) and a tiered localnet end-to-end run (`scripts/localnet/e2e_tiered.py`).
* **Reference isolation (all pools):** each pool's references run concurrently with their own time budget and failure
  share; a hung or failing reference runtime leaves only that pool's miners unscored for the round instead of skipping
  the round for everyone. The round is skipped only when every pool's reference fails.
* Specialist evaluator, first run against the real pinned vLLM 0.30.0 image on an RTX 5090: the container's `/tmp` tmpfs
  is now mounted `exec` (Triton could not load its helper modules, vLLM never started), start-up uses `--enforce-eager`,
  the candidate's GPU budget is 25 GB computed from the detected card size, `REAX_VLLM_DISABLE_UVA=1` mounts a
  read-only `sitecustomize.py` that disables vLLM's UVA buffers (confidential-compute GPUs), and `reax-spec-eval`
  reports start-up time, scoring time and items per second.
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
* Docker: CI also builds, smoke-tests and publishes the tiered-pool runtime images `reax-s1-cpu-runtime` (one index,
  linux/amd64 + linux/arm64, built natively per architecture and merged), `reax-s1-lite-runtime` (CUDA, amd64) and
  `reax-s1-lite-runtime-jetson` (Jetson Orin, arm64; build and file/import-metadata check only, no GPU smoke). Same
  rules as the other images: push only on a manual dispatch with `publish_images`, `sha-` tags written once, a `v*` tag
  promotes the `sha-` manifest byte for byte.
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
