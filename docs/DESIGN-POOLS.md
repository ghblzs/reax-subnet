# SN92 tiered miner pools: GPU, small GPU (incl. Jetson) and CPU

Status: design v0.1, 15 Oct 2026 (copy of the job design; calibration raw data stays with the job). **Release 1.2 ships
the tiered pools and capacity scoring on every network, finney included** (like the s1-pro pool): the flag
`REAX_FEATURE_TIERED_POOLS` defaults to on and switching it off is possible only on development networks with
`REAX_ALLOW_DEV_MODELS=1` (refused on finney). In 1.2 an empty pool or tier is **not burned**: its share goes pro rata
to the other pools (section 5), and a validator without the reference of a tiered pool leaves that pool unscored with a
warning instead of crashing. Sections below that describe the older burn behaviour are superseded by this note. All numbers come from the calibration runs in section 6 (15 Oct 2026).

## 0. Summary

| Question | Short answer |
|---|---|
| Is a Jetson too small? | No. A decision costs one forward pass and no generation. The s1-lite model (Plumb-4B, 8-bit GGUF, 4.5 GB) fits an Orin Nano 8 GB, Orin NX and AGX Orin. The Orin Nano 4 GB is too small for it. |
| Can we add a really small model for them? | Yes: pool **s1-lite** = **the same model as the GPU pool** (Plumb-4B v5, the s1-fast weights), quantised to 8 bits and run on llama.cpp. Jetson miners therefore serve the real s1-fast model, not a toy. |
| A CPU-only model like Laya? | Yes: pool **s1-cpu** = Gutsy 0.8B v0.4 (Apache-2.0, JevBench 43.1; Laya typed-decisions scores 44.2). It runs on any x86 or ARM CPU through the same llama.cpp runtime. Laya itself is an encoder with a custom head, so llama.cpp cannot run it; it would need a second runtime for the same score (section 2.3). |
| Will expensive GPUs earn more than cheap machines? | Yes, by two rules. (1) Fixed tier budgets: GPU 80 %, small GPU 15 %, CPU 5 % of the inference part. (2) The tier-order rule: the average reward per miner in a tier is at most the average per miner of the tier above it (since 1.2.0rc7.post4; it was half before), whatever the miner counts are (so faster GPU miners never shrink the Jetson/CPU pay per miner). Inside a pool, a bigger node earns more through verified throughput. |
| Multiple nodes per person? | Allowed explicitly: one hotkey (UID) per node, and every node can be GPU or CPU. Running many hotkeys on **one** machine gains nothing: capacity probes hit all UIDs of a pool at the same moment, so they share that machine's throughput (section 4.1). |
| Jetson ARM working: possible or hard? | Possible. The CPU pool on ARM is easy: it builds and gives the same answers as x86 within 0.035 mean distance, about one sixth of the tolerance. Orin with GPU is moderate work: it needs a separate L4T container (`Dockerfile.jetson`), and we have no hardware to test it. Thor is a later step. Details are in docs/JETSON.md. |

## 1. Pools

| Tier | Pool | Model (pinned) | Runtime | Hardware | Budget (share of the inference part) |
|---|---|---|---|---|---|
| gpu | s1-fast (and s1-pro when enabled) | Plumb-4B v5 bf16 (`crh225/plumb-4b@55de037`) | reference CUDA-graph runtime (unchanged) | NVIDIA ≥ 24 GB | **80 %** (s1-fast 35 / s1-pro 65 inside the tier, as today) |
| small | **s1-lite** | Plumb-4B v5 **GGUF Q8_0** (`crh225/plumb-4b-GGUF@71c1c57`, 4.48 GB, sha256 `a0d2974e…`) | llama.cpp CUDA (`runtimes/s1-gguf`) | NVIDIA ≥ 8 GB incl. Jetson Orin 8/16/32/64 GB | **15 %** |
| cpu | **s1-cpu** | Gutsy 0.8B v0.4 **GGUF Q8_0** (`kouhxp/gutsy@f9995fe`, 812 MB, sha256 `d71d7a3d…`) | llama.cpp CPU, x86-64 (AVX2) or aarch64 | ≥ 4 cores, 4 GB RAM | **5 %** |

A miner commits to exactly one pool per hotkey, with the existing commitment format
`reax-s1/1:<pool>:<manifest_id>`. All pools use the same wire protocol (`reax-s1/1`), the same signed requests,
the same answer contract and the same miner process. Only the runtime sidecar differs.

## 2. Models: why these

### 2.1 s1-lite = the s1-fast model, 8-bit
- `plumb-4b-v5-Q8_0.gguf` was published by the model author one second after the v5 safetensors that s1-fast pins
  (same "Plumb-4B v5" commit pair, 24 Sep 2026). It is the same model, converted with `convert_hf_to_gguf.py --no-mtp`.
- The prompt is byte-identical to s1-fast. llama.cpp's tokenisation of the s1-lite prompt equals the HF tokenizer's
  on 300 of 300 generated items. The temperature is the same (2.07) and the letter readout is the same.
- Distance between Q8_0 and the bf16 reference: mean 0.115, about the size of honest bf16 drift between cards (section 6). Because the pool shares the s1-fast model, a later
  product could let s1-lite miners carry s1-fast traffic (section 7). That needs a product decision.
- JevBench: Plumb-4B composite 65.4 (official open-weights rank #34). A stronger option exists:
  **H2O-Lightning-4B v1.1** (75.0, best open model ≤ 9B, Apache-2.0, official GGUFs). It needs the vendor's 76 KB
  prompt shim (third-party code review), and moving to it would be a model change for s1-fast too. It is
  recommended as the next model for *both* tiers, as one announced change .
- Q4_K_M (2.7 GB) would also fit 4–6 GB cards. A pool must pin one file, because Q4 and Q8 answers differ more than the
  tolerance (mean 0.60 for s1-lite, 0.25 for s1-cpu), so Q4 would have to be a separate pool. Not proposed for now.

### 2.2 s1-cpu = Gutsy 0.8B v0.4
- Qwen3.5-0.8B fine-tune, Apache-2.0, an official GGUF from the author, made for CPU, with calibrated probabilities
  (per-type temperatures from the author's validation set).
- JevBench (v0.3 row): 43.1, close to Laya typed-decisions (44.2). Both rows are outside the latency cap because
  they were measured on CPU.
- The prompt is the author's training prompt, byte for byte. Choice and score questions include the reject slot
  "Z", and the options are renormalised without it (this equals the softmax over the options alone). There is one
  pass; the author's option-rotation ensemble is not used.
- Measured speed: about 1.1 s per decision on 4 cores of a shared Ryzen 5 3600 build host (under load); 0.7–1.0 s on 8 cores of the pod hosts; 0.01 s on a GPU. Section 6 has more.

### 2.3 Why not Laya
- Laya is ModernBERT-large plus a custom decision head: an encoder with no letter logits. llama.cpp cannot run it.
- It would need a second runtime (onnxruntime or PyTorch) in the kit, the validator image and the calibration, all for
  about the same JevBench score as Gutsy (44.2 vs 43.1).
- Its card also lists weaknesses: non-Latin scripts and more than about 20 options.
- If Laya is preferred, the s1-cpu pool can take it as `readout.kind = "laya"` with an onnxruntime sidecar.
  This costs about one extra day.

## 3. Scoring per pool

Unchanged from 1.1 for the **correctness gate**:
- fresh generated items;
- the validator recomputes a secret random subset on its own copy of the pool's pinned model;
- the logit-space distance is compared with tau;
- any item over `tau_item`, or a mean over `tau_strike`, is a strike: the moving average resets to 0.

Correctness is therefore a gate, not a bonus.

| Pool | Items per round | Verified | Latency curve (full → zero, per request) | Capacity unit (target) | Cap per hotkey | Probe window × items/request × lanes |
|---|---|---|---|---|---|---|
| s1-fast | 8 | 8 | 0.5 s → 8 s | 40 decisions/s | 16 units = 640/s (mechanism v2: x2 = 1280/s, x4 = 2560/s) | 5 s × 16 × 16 (v2: 32 lanes at x2, 64 at x4) |
| s1-pro | 8 | 3 | 1 s → 12 s | 10/s | 16 units = 160/s (mechanism v2: x2 = 320/s, x4 = 640/s) | 8 s × 8 × 16 (v2: 32 lanes at x2, 64 at x4) |
| s1-lite | 4 | 4 | 1 s → 12 s | 8/s | 4 units = 32/s | 16 s × 8 × 8 (tau_mean 0.30) |
| s1-cpu | 4 | 4 | 2 s → 30 s | 1/s | 4 units = 4/s | 30 s × 4 × 4 |

The round score stays `(1 − 0.2) + 0.2 × speed`. Inside a pool, weight ∝ `ema × min(cap, capacity / target)`
(capacity scoring v2; 1.2.0rc2 had `min(1, capacity / target)`, which paid a machine faster than the target once per
hotkey when it was split, see 4.1).
- `capacity` is the moving average of the verified throughput from capacity probes. A probe runs about once every
  10 rounds, at a time the miner cannot predict.
- A struck probe sets capacity to 0.
- A miner that has never been probed has capacity 0. There is deliberately no floor, because a floor would pay per
  registered hotkey instead of per machine.
- Exception: if a validator has **no capacity data at all for a pool** (probes not run yet, or failing on the
  validator's side), that pool falls back to the correctness score alone. A validator-side problem can then never
  send a pool's share to the owner. (The localnet run found this: a validator whose probes overran their time budget
  set all weight to the owner.)
- A probe stops waiting one latency bound after its window, so a slow miner cannot stretch it, and it has its own
  time budget on top of the reference budget.
- Measurements are clamped at the pool cap **before** they are averaged, so one burst far above the cap cannot carry
  a miner through later probes.
- Every probe request is prepared and serialised before the window opens (in a worker thread), the lanes' connections
  are opened before it (a `GET /health` each, at most 2 s), and all UIDs start at nearly the same instant. Each UID's
  window opens at the validator's first send to it and has the pool's length; only answers inside it count (divided by
  the whole window). Opening a lane costs the validator ~0.2 ms of CPU, so lanes open one per UID per loop pass: every
  UID's first request leaves within one pass over the targets (~30 ms at 175 UIDs). 1.2.0rc3 opened one common window
  before the first send and all 16 lanes of a UID before the next UID's, so the UIDs reached last lost 0.6-2.5 s of a
  5 s window (fast miners measured at 75-90 %; `scripts/bench_probe.py --rtt-ms 20-200 --trace`).
- **Probe shards (1.2.0rc5).** On mainnet one event loop sat at 100 % of a core for the whole window (~800 requests/s,
  ~1.2-1.6 ms of validator CPU per request), so the best miners were clipped at what was left of that core (s1-fast
  55-70/s, s1-pro 37-45/s). With many UIDs the window now runs in forked shard processes (`REAX_PROBE_SHARDS`,
  `probeshard.py`), one per ~12 UIDs. UIDs of one coldkey share a shard while the group fits a fair share of the load,
  bigger groups are split (1.2.0rc7; declared axon IPs never group, they are attacker-controlled); all shards open their windows at one common `time.monotonic` deadline (system-wide on Linux) and the send-skew guard
  covers the first sends of all shards together, so the sibling bound (skew / window) is unchanged. A shard inherits
  the loaded wallet through `fork` (no key file, no key material over a pipe), signs its own UIDs' requests and sends
  back only (request index, raw answer, headers, timings); signatures, answers and references are checked in the
  validator as before. A shard that fails or times out drops the whole probe (no partial credit). Every probe logs its
  requests/s, the busiest loop's CPU share, lag p95 and skew; a loop above 90 % of a core (`cpu_saturated`) drops the
  probe (1.2.0rc7), because a saturated loop measured ~0 while the lag guard stayed quiet. Probe answers are capped at
  32 kB and never decompressed (1.2.0rc7).
- Probe answers are checked after the slowest pool's window (s1-lite: 16 s), so the allowed age of an answer's
  signature covers the probe's own duration (bounded by the nonce store's retention). 1.2.0rc4 used each pool's own
  bounds, and on a busy host s1-fast answers turned "stale" and the UID measured 0.
- A probe covers **every queryable UID of the pool**: rested by the reference-cap rotation or not, answered this round
  or not (a UID that is down measures 0). Measuring only part of a pool would let one machine's hotkeys be measured
  apart, each at the whole machine. Probe verification is bounded by the pool's reference cap where the pool has one (s1-lite, s1-cpu); for s1-fast and s1-pro it is bounded by the verified sample below.
- Inside the window the validator only sends and records: answers are counted by request size and checked afterwards
  (signature, strict parse). One forged or malformed answer sets that UID's probe to 0. This keeps the validator's
  own CPU out of the measurement; otherwise a busy validator shares its CPU out per lane, which under-measures fast
  miners and pays extra hotkeys.
- Saturation guard: a 10 ms ticker samples the validator's event loop during the window. A lag p95 above 50 ms, first
  sends spread over more than 250 ms (`PROBE_SEND_SKEW_MAX_S`), or a total at the host's measured ceiling
  (`REAX_PROBE_CEILING_DPS`, from `scripts/bench_probe.py`), means the validator was the bottleneck: the probe is logged and dropped, with no capacity update for anyone (the next probe then waits
  for the normal schedule, also for new UIDs).
- The validator prepares requests for 4× a UID's previous measurement (at least one unit, at most the cap), so a probe
  costs what the pool really serves; a new fast node reaches its speed by its second probe.
- A quarter of each UID's probe answers (at least 3, at most 40; 8 for s1-cpu) is verified against the reference.
  Each verified item is drawn after the answers arrive, and any bad item is a strike, so answering part of a probe
  with fast garbage does not pay.
- A UID that has never been measured (new registration or new role) triggers a probe in the next round in which it
  answers, instead of waiting at zero capacity weight.
- A strike sets capacity to 0, and the average then recovers with α = 0.3 over a few probes. This is a deliberate
  extra cost of a strike.

The target is one unit: the throughput of a good node of the tier. Pay is linear in units up to the cap. In the GPU
pools the cap sits far above one machine (s1-fast: 640/s against 55–74/s per GPU on the reference path and 226/s for
vLLM bf16 on an H100). A node twice as fast as the target earns twice as much behind one hotkey; above the cap, a second
node is paid through its own hotkey. In s1-lite and s1-cpu the cap is 4 units (32/s and 4/s), near the best honest node
of the tier (an RTX 5070 Ti runs s1-lite at 33/s): their small models run 10–100× faster on a big GPU, and a high cap
there would let one GPU take most of a cheap pool.

**Mechanism v2: higher cap for s1-fast and s1-pro (`tiers.CAPACITY_CAP_MULTIPLIER`; x2 first, x4 later).** **Currently x2 since 1.2.0rc9 (s1-fast 1280/s, s1-pro 320/s; 1.2.0rc8 shipped the plumbing at x1); x4 after a live probe bench, with its own dated #92 notice.** Effective miner defaults at x2: per-caller concurrency 32, 456 requests per minute, `REAX_MAX_CONCURRENT` 8, max pending per IP 64. Live data (17 Oct 2026) had 78 of
146 s1-pro hotkeys exactly at 160/s and 95 of 234 inference hotkeys as same-IP siblings of one operator: a multi-GPU
box was split into several hotkeys only because one hotkey could not show more than the cap. The multiplier (a release
constant, 1.0 = the numbers above) raises the cap of s1-fast and s1-pro to 1280/s and 320/s at x2 and to 2560/s and 640/s at x4, so one hotkey can
stand for several GPUs or machines. Pay stays linear in verified throughput, so this is sybil-neutral, and it frees UIDs.
s1-lite and s1-cpu keep their caps. What scales with the cap:
- **Lanes per UID** follow the UID's probe budget (`CapacityParams.lanes_for`): budget × expected answer time (s1-fast
  0.4 s, s1-pro 0.8 s, the time at which today's 16 lanes just carry the old cap) / items per request, at least 16,
  at most 64 (`max_inflight`; the cap itself needs 32 lanes at x2 and 64 at x4). This assumes answers take about
  0.4 s (s1-fast) / 0.8 s (s1-pro) per request: a miner with a longer round trip or slower batches shows less than the
  cap with the same lanes, as it does today. A new UID still gets 16 lanes; a UID measured above the old cap gets
  more budget next probe and more lanes with it. At x4 the requests per probe at the cap are 864 and 704 (x2: 432
  and 352).
- **Preparation ceiling** (`s1validator.PROBE_MAX_PREPARED_REQUESTS`, 90 000: the live-size fleet fully ramped at x2 fits unscaled; re-size from a bench before x4): a probe prepares and serialises every
  request before the window (0.34-0.64 ms each). Today's fleet needs ~45k requests (~21 s of the 60 s preparation
  budget); a fully ramped x4 fleet would need ~179k (~84 s) and a swarm of junk-fast UIDs could force that. When the
  sum over all probed UIDs exceeds the ceiling, every UID's budget is scaled down by the same factor (never below its
  target; `tiers.scale_budgets_to_ceiling`). Measured throughput is still compared with each UID's own window; a
  scaled budget only limits how high a UID can show in that probe, equally for everybody, so it stays sybil-neutral.
  Today's x1 fleet is below the ceiling and unaffected. A scaled probe only shows the factor times the budget, so
  it must not drag the capacity average down: `tiers.probe_budgets` divides the last scale out of the previous
  measurement, and `tiers.fold_capacity` lets a scaled probe in which the UID showed at least 90 % of its budget
  raise the average but never lower it (it is only a floor). A UID that answers less than its budget, and any strike,
  is averaged as before. Once the fleet shrinks below the ceiling the factor is 1 and the next budget is 4x the
  average, so there is nothing to recover (tests: 20 probes under a 500-UID swarm, then 3 probes after it leaves, all
  at 90 % of the true capacity). Griefing note: a swarm of genuinely fast UIDs lowers everybody's scaled budget (and
  so how high anyone can show in those probes, and the validator's preparation time); it costs the attacker real
  capacity and one registration per UID, gains nothing (the factor is common), and the held average means honest UIDs
  are not marked down. Whether the send side (~36 cores at a fully ramped x4) fits
  the validator host must be benched (`scripts/bench_probe.py`) before x4.
- **Verified sample** grows with the measured factor: `verified_max × max(1, (factor / 16) ** 0.75)` (s1-fast 40 → 68
  at x2 and 114 at x4, s1-pro 10 → 17 and 29 at the new cap), still 25 % of the answers, at least 3. Up to the old
  cap nothing changes. The checked share of the answers falls (s1-fast at the cap 1.25 % → 0.9 % at x4), but the
  number of checked items per probe grows, so the chance to catch a miner that cheats on a share of its answers
  (1 − (1 − f)^k) stays at or above today's. The GPU pools have no reference cap entry (`REFERENCE_CAP` covers
  s1-lite and s1-cpu only), so nothing else bounds this; the extra reference load per probe at the cap is about
  1.5-2× today's.
- **Probe shards** are still planned by each UID's prepared requests per second (`probe_request_load`), so a
  2560/s UID (864 requests in 5 s) is a heavy piece of its own and the balance and skew guards hold; the shard count
  uses the total lanes (`PROBE_LANES_PER_SHARD` 192). The probe response cap (32 KiB per answer) is unchanged.
- **Miner defaults** (`config.py`) are derived from the table: per-validator concurrency = lanes at the cap, requests
  per minute = requests at the cap + 24, `REAX_MAX_CONCURRENT` so that the runtime queue (5 × this) holds one probe's
  lanes. x1: 16 / 240 / 8 (unchanged); x2: 32 / 456 / 8; x4: 64 / 888 / 16. `REAX_MAX_PENDING_PER_IP` defaults to
  `max(32, 2 × REAX_PER_CALLER_CONCURRENCY)` (x2: 64, x4: 128), because all lanes of one probe start within a few
  ms from one validator IP and a refused lane ends for the whole probe. A miner on old limits is measured at most
  what its limits allow (a fast machine on the old 240 requests/min and 16 lanes can still show up to ~768/s on
  s1-fast and ~240/s on s1-pro, a slow or distant one less); refused lanes are logged by the validator and the miner
  is never struck.

**Rollout order (mechanism v2).** The multiplier is a release constant that all validators must agree on. During a
mixed fleet a consolidated multi-machine hotkey is paid more by validators on the higher cap than by the others,
and Yuma consensus clips it towards the stake-weighted median; rolling back after consolidation cuts it at once
(stored capacities above the lower cap are paid as the cap, so nothing breaks). Therefore: (1) ship the plumbing at
x1 first (every number is identical to 1.2.0rc7.post4; this is 1.2.0rc8); (2) enable x2 as its own release with a
`WEIGHTS_VERSION_KEY` bump by the release owner and a dated #92 notice; (3) x4 follows only after a live probe bench
at the ramped load. Miners should not consolidate hotkeys before the enable is live on the stake majority.

## 4. Anti-sybil and anti-copy

### 4.1 One backend behind many UIDs: neutralised by simultaneous probes
- A capacity probe loads **every UID of a pool at the same moment** for the whole window. Each UID gets up to 16
  concurrent request lanes (s1-lite 8, s1-cpu 4), with enough requests to stay busy past the window.
- If N hotkeys share one backend, each one measures about 1/N of that backend, and only answers inside each UID's
  window count. The windows of all UIDs open within the send skew (at most 250 ms, ~30 ms measured at 175 UIDs), so
  together they never measure more than the machine plus skew / window (at most 5 % of a 5 s window, ~0.6 % measured). Because pay is linear in capacity up to a cap
  far above one machine, together they earn what one hotkey on that machine earns, minus N − 1 extra registrations.
- **This needs both parts.** 1.2.0rc2 capped the factor at the target (40/s): a GPU doing 160/s split over four hotkeys
  measured 40/s each and earned four full shares (on 15 Oct 2026 96 UIDs from 22 coldkeys earned within
  ±4 % of each other). And the probe must be able to *see* a fast machine through one hotkey: with two lanes of 16
  decisions a distant miner shows at most 32 decisions per round trip, so four hotkeys (eight lanes) measured more than
  one. The probe now offers 16 lanes (more for a UID with a bigger budget under the v2 cap); the miner defaults accept them.
- No detection or ban is needed, and nothing can be "falsely accused".
- Prototype test (`tests/test_tiers.py`): three hotkeys on one simulated machine measured 5.33 + 5.33 + 4.00 =
  **14.67 decisions/s**. The same machine behind one hotkey measured **14.67/s**. Their combined weight was within
  15 % of the solo node's weight.
- The probe counts only decisions answered **inside the window**. A first version counted to each UID's last answer,
  so a UID that finished early freed the shared machine for its siblings, and three sybils measured 1.14× the solo
  node. The test caught this and the probe was fixed.

### 4.2 Copying answers or proxying
- **Fresh items, secret verification:** every item is newly generated, so there is no answer key to copy. The
  verified subset is chosen after the answers arrive.
- **Responses are signed** by the miner hotkey and bound to the validator and the request id (existing btauth). If a
  hotkey republishes another miner's axon, it fails authentication (existing test `axon_copy`).
- **Forwarding requests to another miner:** that miner's capacity is then shared (4.1). The copier pays a
  registration and earns roughly nothing.
- **Fast garbage** fails the verification sample and gets a strike. In the probe test, a "tiny model" with 10× the
  speed measured capacity 0 and its moving average was reset.

### 4.3 Hardware is not attested, but a mismatch is not profitable
- We cannot prove what hardware sits behind a UID. A GPU owner can run the CPU model on a GPU, and an s1-lite miner
  can run bf16 on a big GPU if its answers stay within the s1-lite tolerance.
- **Every pool's total is a fixed budget**, and s1-lite / s1-cpu count a hotkey for at most 4 units. So moving big
  hardware into a cheap pool earns at most 4 units' share of that pool's small budget.
- The tier-order rule keeps the cheap pools' average per miner at or below the expensive ones (ratio 1.0 since 1.2.0rc7.post4).
- Result: in a fleet with many cheap nodes (e.g. 20 / 50 / 200 in section 5) a GPU in s1-lite or s1-cpu earns no more
  than a target-speed s1-fast hotkey (`tests/test_tiers.py`). While the cheap pools have only a handful of nodes, the
  per-miner rule leaves them a generous average, and a GPU-speed hotkey there can earn more than an s1-fast target
  hotkey; it is still bounded by the cheap tier's budget.

### 4.4 Monitoring (not scoring)
These signals are logged for operators, never used to score:
- latency correlation between UIDs over many rounds;
- identical failure times;
- many UIDs per IP.

Their purpose is to spot a validator-side bug or a new attack early. Optional extension: a pairwise "solo vs joint"
probe for UIDs flagged by these signals.

### 4.5 Registration burn
- Every hotkey pays the subnet registration burn, which is the natural per-node cost.
- With the rules above, a sybil hotkey earns nothing extra, so its registration is pure cost.

## 5. Budgets: the numbers behind 80 / 15 / 5

Example of per-miner shares of the inference part. All miners are healthy at the target capacity; the shares are
release constants, not a forecast.

| Miners (gpu / small / cpu) | Per GPU miner | Per small miner | Per CPU miner | Tier totals (gpu / small / cpu) |
|---|---|---|---|---|
| 20 / 50 / 200 | 4.0 % | 0.30 % | 0.025 % | 80 / 15 / 5 (rule not needed) |
| 10 / 10 / 10 | 8.0 % | 1.5 % | 0.5 % | 80 / 15 / 5 |
| 5 / 40 / 400 | 16 % | 0.375 % | 0.013 % | 80 / 15 / 5 |
| 20 / 5 / 2 | 4.0 % | 3.0 % | 2.5 % | 80 / 15 / 5 (rule not needed at ratio 1.0; at the old ratio 0.5 it moved 7 points up: 87.4 / 10.6 / 2.0) |
| 50 / 0 / 1 | 1.68 % | — | 0.8 % (would be 5 %) | 84.2 / 0 / 0.8 in the prototype (15 % burned); in 1.2 the empty small pool's 15 % is shared pro rata and the cap moves the excess up: 98.1 / 0 / 1.9 at ratio 1.0 (1.96 % per GPU miner, 1.88 % for the CPU miner; 99.1 / 0 / 0.9 at the old ratio 0.5) |

Computed with `compute_tiered_weights` (all miners healthy at the capacity target).

- Running cost per node is roughly: CPU VPS a few EUR/month; a Jetson Orin Nano Super is about USD 250 once plus
  about 25 W; a rented 24 GB GPU is USD 0.3–0.7 per hour. The 80 / 15 / 5 split keeps the GPU pool clearly the main
  pool, and a Jetson or CPU node still sees non-zero, measurable rewards.
- No public text may state or imply returns or token prices (MiCAR). Public docs only state the shares and rules.
- Interaction with the inference/competition split: the pool budgets divide the **inference**
  part. The competition part (engine and specialists) is unchanged, so all numbers here scale with the inference part.
- **1.2:** an empty pool's share goes pro rata to the other pools (inside its tier first, then across tiers), never to the
  owner UID; the tier-order cap then moves any excess up. Only when no pool has a miner does the owner UID get it (as in 1.1).

## 6. Calibration across backends

The tolerances must cover honest differences between backends that run the **same GGUF file**: x86 CPU, ARM CPU,
CUDA on different cards, and the validator's own runtime. They must still catch a cheaper substitute such as another
quantisation.
- Method: `scripts/calibration_gguf/run.py` on 200 generated subnet items (seed 7), with the runtime's own prompt and
  readout. Distances use the validator's metric (`s1scoring.logit_distance`).
- Pods: RTX A6000 (USD 0.42/h) and RTX 5070 Ti 16 GB (USD 0.40/h) on Lium. CPUs: a Ryzen 5 3600 build host and the pod hosts.
- Raw data is kept with the calibration run, outside the repository.

**s1-cpu (Gutsy 0.8B Q8_0), distance to x86 CPU (Ryzen 5 3600), T = 1.0**

| run | items | mean | p95 | max | top-answer flips | s/decision |
|---|---:|---:|---:|---:|---:|---:|
| x86 CPU, other machine (A6000 pod host), same portable image | 200 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.99 (8 threads) |
| x86 CPU, 2 threads instead of 4 | 200 | 0.0000 | 0.0000 | 0.0000 | 0 | 1.75 |
| real sidecar container (HTTP contract) | 40 | 0.0000 | 0.0000 | 0.0000 | 0 | — |
| CUDA, RTX A6000 (llama.cpp CUDA build) | 200 | 0.032 | 0.081 | 0.113 | 2 | 0.014 |
| arm64 CPU (NEON, qemu emulation: real ARM arithmetic, not real speed) | 30 | 0.035 | 0.081 | 0.087 | 0 | (emulated) |
| **Q4_K_M file instead of Q8_0** | 200 | **0.251** | 0.641 | 1.338 | 17 | 0.91 |

**s1-lite (Plumb-4B v5 Q8_0), distance to CUDA on RTX A6000, T = 2.07**

| run | items | mean | p95 | max | top-answer flips | s/decision |
|---|---:|---:|---:|---:|---:|---:|
| same card, second run | 200 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.041 |
| CUDA, RTX 5070 Ti (real sidecar over HTTP) | 50 | 0.078 | 0.198 | 0.205 | 2 | 0.030 |
| x86 CPU (pod host, 8 threads) | 100 | 0.073 | 0.187 | 0.246 | 1 | 5.1 |
| arm64 CPU (NEON, qemu emulation; small sample: 5.4 min per item emulated) | 4 | 0.108 | 0.319 | 0.319 | 0 | (emulated) |
| **Q4_K_M file instead of Q8_0** | 200 | **0.604** | 1.539 | 3.097 | 27 | 0.041 |
| bf16 s1-fast reference runtime (the s1-fast pool's model) | 200 | 0.115 | 0.275 | 0.347 | 3 | 0.031 |

Consequences:
- **Tolerances:** s1-cpu keeps the s1 values, `tau_mean 0.20 / tau_strike 0.50 / tau_item 2.0`. s1-lite uses
  **`tau_mean 0.30`**, so a validator may use its existing **bf16 s1-fast runtime as the s1-lite reference**: the same
  weights, no extra GPU memory, about 30 ms per decision. Against it:
  - honest Q8 miners measure mean 0.107–0.115;
  - P(round fails) is at most 0.0001 on CUDA and on CPU;
  - a Q4 substitute still fails 92 % of rounds and is struck in 61 %.
  Against the GGUF reference itself the honest distances are smaller still.
  Resampling the measured distances into rounds of 4 verified items gives these failure rates:
  - every honest cross-backend pair above: P(round mean > 0.20) ≈ 0 and P(strike) ≈ 0;
  - a Q4_K_M substitute in s1-lite: P(round scores 0) = 98.6 % and P(strike) = 62 % per round.
- **One file per pool.** Q4 cannot hide behind Q8.
- **The validator can run both references on CUDA:**
  - for s1-cpu, CUDA vs CPU is 0.03 mean;
  - for s1-lite, CPU vs CUDA is 0.07 mean;
  - so miners on any backend are judged fairly.
- **Finding for the existing s1-fast pool:** the Q8_0 GGUF of the s1-fast model sits at mean 0.115 from the bf16
  reference, and with 8 verified items it passes the s1-fast tolerance in 99.8 % of rounds.
  - That gap is the same size as honest bf16 drift between cards (CALIBRATION.md: median 0.063–0.075, p99 about
    0.22–0.31). It **cannot be excluded without also striking honest miners on other cards**.
  - In practice, 8–16 GB GPUs can already mine s1-fast through llama.cpp Q8 (the `s1-gguf` runtime). The "24 GB"
    in the miner docs is advice, not a rule the validator enforces.
  - Customers get the same model within the tolerance, and the capacity probes decide the pay.
  - Proposal: accept it and document it as an allowed s1-fast runtime once it has a JevBench row,
    instead of pretending it is excluded.
- **Speeds and capacity targets** (verified decisions per second):

  | Hardware | Measured or estimated speed |
  |---|---|
  | s1-fast bf16 on A6000 | about 32/s (sidecar) |
  | s1-lite on RTX 5070 Ti | 33/s (sidecar) |
  | s1-lite on A6000 | 24/s |
  | s1-lite on Jetson Orin Nano 8 GB | estimated 0.7–2/s |
  | s1-cpu, 4–8 x86 cores | about 1/s |

  Targets: s1-fast 40/s, s1-lite **8/s** (a desktop GPU fills the target and may run more hotkeys; an Orin reaches
  about 10–25 %), s1-cpu 1/s.

## 6a. What a validator needs

- **s1-lite reference:** either of these.
  - Recommended: the **existing bf16 s1-fast runtime** (`REAX_S1_LITE_RUNTIME_URL` = the s1-fast sidecar URL). It
    has the same weights and the same contract, needs no extra GPU memory and takes about 30 ms per decision. The
    s1-lite tolerance (`tau_mean 0.30`) is set for this.
  - Alternatively, the GGUF CUDA sidecar (`runtimes/s1-gguf/Dockerfile.cuda`, MODEL=s1-lite, about 5.5 GB VRAM).
- **s1-cpu reference:** the CPU image (`runtimes/s1-gguf/Dockerfile`, MODEL=s1-cpu) on 8 threads, about 3.7
  decisions/s on a server CPU. Or the CUDA image with MODEL=s1-cpu, about 1.5 GB VRAM: CUDA vs CPU is mean 0.03.
- **Isolation:** each pool's references run concurrently with their own time budget and their own failure share.
  A slow, hung or failing pool leaves only its own miners unscored that round; the round is skipped only if every
  pool fails.
- **Reference cap and rotation:** at most 240 (s1-lite) / 160 (s1-cpu) reference decisions per round per pool.
  - With more miners than that, miners are queried in fair rotation (least recently verified first); resting miners
    are not decayed.
  - Capacity probes share the same cap.
  - Validators can lower or raise it (`REAX_REFERENCE_CAP_S1_LITE`, `REAX_REFERENCE_CAP_S1_CPU`). It changes how
    often a miner is verified, never how it is scored.
- **Cost per tiered miner:** 4 verified decisions per round plus probes (about one round in 10, a quarter of the probe
  answers), so about 7–8 reference decisions per round on average.

## 7. Real customer traffic (decisionmodels.io)

| Pool | Paid traffic at start | Later option (needs a product decision) |
|---|---|---|
| s1-fast (GPU) | yes, as today: the P2P route serves `s1-fast-1` = Plumb-4B through the router (`models=("s1-fast",)`) | — |
| s1-lite | **none** | "s1-fast, 8-bit" overflow or batch tier, if Q8 stays within the measured distance of bf16 on JevBench and in production canaries. The model is the same, so no new product page is needed, only a disclosed quantisation. |
| s1-cpu | **none** | a free or community tier, or slow batch jobs for a small 0.8B model. No paid product without its own JevBench row and page. |

The router keeps refusing anything other than s1-fast/s1-pro until such a decision. No routing code changes in this
prototype.

## 8. Changing budgets later

- The tier budgets and capacity targets are **release constants**. Validators refuse overrides on finney, as with all
  scoring parameters, so every validator runs the same numbers.
- A change is announced **at least 14 days in advance** in the release notes, the miner docs and on
  mine.decisionmodels.io / reax.co. This follows the 80/20 rule. A change starts with a release that validators adopt.
- Data used to tune the budgets: miners per pool, measured capacity per pool, router demand per model. The ratio rule
  (≤ ½ per tier step) stays fixed unless it is announced the same way.

## 9. Rollout

1. Now: prototype on localnet behind the flag (done), GPU and CPU calibration (done), ARM build (done), Jetson recipe
   (written, untested).
2. Testnet with ≥ 2 validators, real GGUF sidecars on x86 CPU, ARM CPU and a CUDA card, and the first outside beta testers
   (for example Jetson owners).
3. Release (1.2): manifests pinned by digest, kit support, docs, the 14-day budget notice, then mainnet with an Owner
   decision. (The tiered pools are part of the 1.2 release candidate on every network; see the status note at the top.)
