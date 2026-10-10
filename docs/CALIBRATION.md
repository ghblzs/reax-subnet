# reax-s1 GPU calibration (release 1.1.0rc1, 3 Oct 2026)

Where the scoring constants in `src/reax_subnet/s1validator.py` (`POOL_PARAMS`) come from. Every number below was
measured with the exact reference runtimes on rented, isolated GPU pods; the scripts are in `scripts/calibration_s1/`.

## Setup

* **Items:** 871 decisions = 640 from the validator's own generator (`s1items.py`, seed 20261003) + the 231 public
  JevBench items (public set only). The same items on every GPU and every run.
* **s1-fast runtime:** the reference fast image recipe (pytorch 2.13.0-cuda13.0 by digest, jevk5 at the pinned commit,
  runtime source hashes verified), Plumb-4B at the pinned revision, reference readout (CUDA-graph batch-1 replay,
  letter logits / 2.07, float32 softmax). Variants: same path in a new process; the engine's eager micro-batch path (8 per
  batch); exact-length eager; vLLM 0.30.0 bf16 with continuous batching (an honest alternative engine).
* **s1-pro runtime:** the reference Rune recipe (vLLM 0.30.0 by digest + the reference decision adapter, deterministic
  flags: `VLLM_BATCH_INVARIANT=1`, no chunked prefill, `-O0`), Rune weights at the pinned revision (all 18 file hashes
  verified). Runs at client concurrency 1, 16 and 64, after a container restart, and stock vLLM without the
  deterministic flags.
* **Adversaries:** Plumb FP8 (vLLM online FP8), base Qwen3.5-4B with the same prompt/readout, JevK5 v0.2 (the closest
  relative: same base, same recipe family) at its own and at the best-fitting temperature; Rune FP8 (vLLM online FP8,
  deterministic flags).
* **GPUs:** RTX PRO 6000 Blackwell Workstation (96 GB, driver 595), H100 80GB HBM3 (driver 580), RTX 4090 (24 GB,
  s1-fast and rc3 only; driver 595).
* **Metric:** exactly the validator's: `d = T · max_i |centred ln p_i − centred ln q_i|`, probabilities floored at 1e-6,
  in logit units of the model. Round statistic = mean of d over the verified items (8 for s1-fast, 3 for s1-pro);
  round distributions from 50,000 bootstrap rounds over the measured items.

## s1-fast (Plumb-4B)

Per item, distance to the reference (reference path on the reference GPU):

| Pair | median | p99 | max | top choice agrees |
|---|---|---|---|---|
| reference path, new process, same GPU (both GPUs) | 0 | 0 | 0 | 100 % |
| reference path H100 vs RTX PRO 6000 | 0.071 | 0.281 | 0.708 | 98.7 % |
| reference path RTX 4090 vs RTX PRO 6000 / vs H100 | 0.063 / 0.071 | 0.275 / 0.313 | 0.469 / 0.719 | 98.7 / 98.4 % |
| engine eager micro-batch, same GPU | 0.071 | 0.275 | 0.542 | 99.5 % |
| vLLM bf16 batched, same GPU / cross-GPU (any pair incl. RTX 4090) | 0.063 | 0.20–0.29 | 0.48–1.06 | 98.4–99.2 % |
| **FP8 (same weights)** | 0.250 | 1.03 | 3.00 | 96.8 % |
| JevK5 v0.2 (sibling), best temperature | 0.89 | 4.55 | 5.79 | 83.2 % |
| base Qwen3.5-4B | 1.43 | 6.94 | 10.70 | 74.4 % |

Round mean over 8 verified items:

| Setup | P(D ≤ 0.20) = round scores | P(D > 0.50) = strike | worst D seen |
|---|---|---|---|
| reference path, any pair of RTX PRO 6000 / H100 / RTX 4090 | 0.9998–1.0 | 0 | 0.24 |
| reference path RTX A6000 / L40S / RTX 5090 vs each of the three above (Oct 2026 addendum) | 0.9990–1.0 | 0 | 0.27 |
| other honest engines (eager, vLLM bf16), cross-GPU | ≥ 0.9999 | 0 | 0.24 |
| **FP8** | **0.095–0.098** | 0.02 | — |
| JevK5 sibling, other models | 0 | ≥ 0.997 | — |

So an honest miner on another GPU type loses about one round in 10,000 (≈ one round per two weeks per validator) and is
never struck; an FP8 build scores about one round in ten (moving average ≈ 0.1, below the 0.3 needed for any weight) and
other models are struck almost every round. Raising τ_mean to 0.25 would let FP8 score 28 % of rounds — too close to
the 0.3 line — which is why 0.20 was chosen.

Throughput of the reference path (one decision at a time, as the reference engine): RTX PRO 6000 55.6, H100 60.5, RTX 4090 30.4
decisions/s; vLLM bf16 batched 135 / 226 / 57 decisions/s (honest, passes, but not the reference path — operators may use it at their own
risk; FP8 does not pass).

## s1-pro (Rune 26B-A4B v3)

| Pair | median | p99 | max |
|---|---|---|---|
| deterministic build: new process, concurrency 1 / 16 / 64, same GPU (both GPUs) | 0 | 0 | 0 |
| **deterministic build: H100 vs RTX PRO 6000 (every run pair)** | **0** | **0** | **0** |
| stock vLLM (no deterministic flags), concurrency 16 | 0.146 | 1.22 | 3.94 |
| **FP8 (same weights), deterministic flags** | 0.313 | 2.5–3.4 | 4.1–5.0 |

The deterministic Rune build is bit-identical across the two measured GPU architectures, restarts and load. Round mean
over 3 verified items: deterministic build P(D ≤ 0.20) = 1, never struck; FP8 0.12 (struck in 22 % of rounds); stock vLLM
0.65 (it loses a third of its rounds, so miners must run the pinned deterministic runtime).

Throughput (deterministic build, 871 decisions): RTX PRO 6000 17 / 73 / 93 decisions/s at concurrency 1 / 16 / 64
(p50 54 / 138 / 343 ms), H100 15 / 77 / 108 decisions/s (p50 63 / 136 / 301 ms).

Validator load: the s1-fast reference verifies 8 items per miner and round with the reference path (one decision at a
time). At 30 decisions/s (RTX 4090) 255 miners need about 68 s of the 120 s round, at 55–60 decisions/s (RTX PRO 6000,
H100) about 35 s. A validator on a 24 GB card is fine for up to roughly 150 s1-fast miners; above that use a faster card.

**Shipped generator, shipped runtime image:** the first calibration items were drawn before the generator was widened
The initial calibration items were drawn before the generator was widened. A second run used 640 items from the shipped generator through the REAX s1-fast runtime image itself
(run as a non-root operator uid with a 0600 token): RTX PRO 6000 vs RTX 4090 per item median 0.075, p99 0.22, max 0.53;
round mean of 8 at most 0.185, P(D ≤ 0.20) = 1.0. Same behaviour as above. Runtime throughput through its HTTP
endpoint, one decision at a time: 74 decisions/s (RTX PRO 6000), 49 (RTX 4090).

## Constants chosen

| | s1-fast | s1-pro |
|---|---|---|
| temperature (reference) | 2.07 | 2.0 |
| items per round / verified | 8 / 8 | 8 / 3 (secret, after the answers) |
| τ_mean (round scores if D ≤) | 0.20 | 0.20 |
| τ_strike (strike if D >) | 0.50 | 0.50 |
| τ_item (strike if any item >) | 2.0 | 2.0 |
| latency bounds | 0.5–8 s | 1–12 s |

s1-pro could use much tighter limits on the measured hardware (it is exact), but A100 and other architectures are not
measured; the shared limits keep FP8 and stock vLLM unprofitable either way.

## rc3 checks (mechanism mpm1, Qwen3-4B), measured on the same pods

* **Cross-GPU drift (bf16):** H100 and RTX 4090 each vs the RTX PRO 6000 reference on 256 items at T = 4: mean TV
  0.012, p99 0.045 / 0.057, max 0.062 / 0.062 (criterion ≤ 0.10 with τ 0.15); 10-of-12 round pass probability 1.0. **Pass.**
* **Validator round time:** reference computation for 64 miners × 12 items: 3.4 s on RTX PRO 6000, 5.3 s on H100,
  6.1 s on RTX 4090 (≤ 5.1 % of the 120 s round). **Pass.**
* Found on the way: the rc3 Docker image crashed on any GPU (torch 2.14 compiles Triton helpers with a C compiler the slim
  base lacked); fixed in 1.1.0rc1 (gcc in the image). CPU runs were unaffected, which is why rc3's CPU tests missed it.

## Addendum (Oct 2026): RTX A6000, L40S, RTX 5090

Same kit, same 871 items, same pinned weights (`model.safetensors` sha256 `89e119ea…e1e1` on every pod), reference path
(`graph`) run twice per card, compared with the reference-path results of the three release GPUs:

| Card (driver) | decisions/s | same card, new process: max d | vs RTX PRO 6000 / H100 / RTX 4090: median, p99, max | top choice agrees | P(round scores) | strikes |
|---|---|---|---|---|---|---|
| RTX A6000 48 GB, Ampere (580) | 19.3 | 0 | 0.071–0.075, 0.25–0.30, 0.46–0.81 | 98.9–99.0 % | 0.9993–1.0 | 0 |
| L40S 48 GB, Ada (580) | 31.0 | 0 | 0.063–0.075, 0.25–0.28, 0.55–0.75 | 98.7–98.9 % | 0.9998–1.0 | 0 |
| RTX 5090 32 GB, Blackwell (595) | 41.6 | 0.475 | 0.063–0.075, 0.28–0.31, 0.48–0.91 | 98.7–99.1 % | 0.9990–1.0 | 0 |

All three stay far inside the limits: no item above τ_item = 2.0, worst bootstrap round mean 0.27 (strike line 0.50).
The RTX 5090 reference path is not bit-reproducible between processes (median 0.063 between two runs on the same card),
which is the same size as the cross-GPU drift and does not change any round outcome. The RTX A6000 uses the same GA102
chip as the RTX 3090; the 3090 itself was not measured.

A larger cross-check on 5,000 fresh items from the release generator (seed 20261011, one reference-path run per card):

| Pair | median | p99 | p99.9 | max | P(round scores) | strikes | worst round mean |
|---|---|---|---|---|---|---|---|
| RTX 4090 vs RTX A6000 | 0.071 | 0.219 | 0.313 | 0.500 | 1.0000 | 0 | 0.17 |
| RTX 4090 vs RTX 5090 | 0.071 | 0.225 | 0.354 | 0.911 | 0.9998 | 0 | 0.22 |
| RTX A6000 vs RTX 5090 | 0.063 | 0.219 | 0.339 | 1.375 | 0.9982 | 0 | 0.27 |

The largest single-item distance (1.375, a seven-option choice item) is one isolated bf16 drift; the next largest is
0.46. Across the least similar pair an honest miner loses about one round in 550 and is never struck.

Adversaries on the same 5,000 items (RTX 4090; reference = reference path on RTX 4090 / RTX A6000):

| Run | median | p99 | max | P(round scores) | P(strike) |
|---|---|---|---|---|---|
| vLLM bf16 batched (honest, not the reference path) | 0.063 | 0.23 | 1.66 | 0.9984 | 0 |
| **FP8 (same weights, vLLM online FP8)** | 0.250 | 1.00–1.03 | 1.97–2.03 | **0.114** | 0.008–0.010 |
| base Qwen3.5-4B | 1.25–1.27 | 3.8 | 5.6 | 0 | 0.9996–0.9997 |

FP8 scores slightly more often here than in the 871-item release set (0.114 vs about 0.10). A 60-day simulation of the
moving average (α = 0.1, strikes reset it) keeps an FP8 miner at or above the 0.3 eligibility line in 1.2 % of rounds,
i.e. about 0.4 % of an honest miner's weight: still not worth the saving.

## Not measured

A100, L4, H200, RTX 3090 and other cards; FP8/INT4 builds other than vLLM online FP8; a model distilled on the
public item generator. Operators on other hardware should run `scripts/calibration_s1` against a reference result
before mining.
