# REAX v1.1 — miners serve the exact System1 production models (design)

Status: design for release 1.1.0rc1 (3 Oct 2026). Numbers marked **[cal]** are set from the GPU calibration
(CALIBRATION.md) and are copied into the code as release constants.

## 1. Goal and non-goals

Goal: a subnet miner answers a decision exactly as System1 production does for `s1-fast` (Plumb-4B) or `s1-pro`
(Surogate Rune 26B-A4B v3): the same weights, the same prompt, the same one-pass readout, the same answer schema.
A validator can prove this cheaply by recomputing the answer itself.

Non-goals for 1.1: images (`s1-vision`), multi-question requests, organic third-party traffic (design only) and capacity
audits.

## 2. Licences (checked 3 Oct 2026)

| Model | Weights | Licence | Third-party commercial serving | Duties |
|---|---|---|---|---|
| s1-fast | `crh225/plumb-4b@55de037801a8a9b9de3db5c0e16cef86210c2186`, public | Apache-2.0 (base Qwen3.5-4B Apache-2.0; prompt/readout from SemIf, MIT) | allowed | consult the model repository's LICENSE, NOTICE and NOTICE-jevk5 when obtaining or redistributing the weights; no trademark use |
| s1-pro | `surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921` (bf16 safetensors despite the repo name) | Apache-2.0 (base Gemma 4 26B-A4B-it, Apache-2.0) | allowed | Hugging Face gate ("auto": accept by sharing contact details). Every miner downloads with **its own** HF account; we do not mirror the weights. Keep the licence and attribution. |

Neither licence blocks the plan. The runtime code miners run is ours (MIT/Apache) plus `jevk5` (Apache-2.0,
github.com/allebee/jevk5 at a pinned commit) and vLLM (Apache-2.0).

## 3. Roles

* **Miner role = one model per hotkey.** A miner publishes an on-chain commitment
  `reax-s1/1:<model>:<manifest_id>` (Commitments pallet, plaintext). `model` ∈ {`s1-fast`, `s1-pro`}.
  A hotkey without a valid commitment for an enabled model is not queried and earns nothing.
  Changing the commitment resets that UID's score (state is keyed by hotkey + model + manifest).
* **Hardware.** s1-fast: one NVIDIA GPU with ≥ 24 GB (8.4 GB weights + CUDA graphs); measured on RTX PRO 6000
  Blackwell and H100 80 GB (RTX 4090: see CALIBRATION.md). s1-pro: ≥ 80 GB class (52 GB bf16 weights + KV
  cache); measured on RTX PRO 6000 96 GB and H100 80 GB with bit-identical results. s1-fast was later also
  measured on RTX 5090, RTX 3090, L40S and RTX A6000. Other cards (L4, A100, H200 …) should work but are not yet measured — run scripts/calibration_s1 first.
* **Validator.** Runs the reference of every *enabled* model on its own GPU. s1-fast reference: 24 GB GPU.
  s1-pro reference: an 80–96 GB GPU, which is why the s1-pro pool is a separate switch (§8).

## 4. Model manifests (versioned, in the repository)

`src/reax_subnet/manifests/<model>-<version>.json`, canonical JSON; `manifest_id` = first 16 hex chars of the
sha256 of the canonical bytes. A manifest pins:

* weights: HF repo, revision, sha256 of every file the runtime reads (miners verify before serving, exactly like
  the reference `_verify_files`);
* runtime: base image digest (s1-fast `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f…`;
  s1-pro `vllm/vllm-openai@sha256:8a69ffad…` = v0.30.0), Python package pins, `jevk5` source hashes;
* readout: prompt template id, option labels, temperature (s1-fast 2.07 from `jevk5_config.json`; s1-pro 2.0),
  deterministic engine flags (s1-pro: `VLLM_BATCH_INVARIANT=1`, no chunked prefill, `-O0`);
* limits: options per question (s1-fast 2–16, s1-pro 2–64; the subnet generator uses 2–8), state ≤ 16 KiB,
  ≤ 4,096 input tokens.

The reference pins are the same (`rune-26b-a4b-v3-bf16-77e7068c…` = sha256 of the same 18-file list).

## 5. Wire protocol `reax-s1/1`

`POST /v1/s1/decide`, btauth-signed both ways exactly as rc3 (request bound to the miner hotkey, response bound to
the validator hotkey and request id).

```json
{"protocol": "reax-s1/1", "request_id": "…", "model": "s1-fast", "manifest_id": "…",
 "decisions": [{"id": "…", "state": <string|object|array>,
                "questions": {"q": {"type": "choice|noul|score", "instructions": …, "criteria": …}}}]}
```
→
```json
{"protocol": "reax-s1/1", "request_id": "…", "miner_version": "1.1.0rc1",
 "results": [{"id": "…", "answers": {"q": <reference answer object>}}]}
```

Each `decisions[i]` is a typed-decision body without `model` (one question, no images: a limit of the reference contract). Each answer is byte-for-byte the reference `normalize_answer` output: `type`, `probabilities`
(every option key), plus `choice`+`confidence` / `noul` / `score`+`confidence`+`legend`. The validator rejects a
response whose derived fields disagree with its own probabilities (choice ≠ argmax, noul ≠ p(true),
score ≠ expected value), so a miner cannot return reference-looking fields over wrong probabilities.

## 6. Verification

### 6.1 Fresh items, no answer key
As rc3: items come from a cryptographically seeded generator per miner and round; there is no correct answer to
leak. The 1.1 generator produces typed decisions shaped like real decision traffic: a JSON or text *state*
(short messages, product notes, schedules, small records) and one `choice` (2–8 options), `noul`, or
`score` (3–5 levels) question. Questions are judgement calls on that state, so the reference distribution is
spread out and specific to the model.

### 6.2 Distance in logit space (why not rc3's T=4)
rc3 raised the protocol temperature to 4 so that a model-free "copy the top choice" answer fails. 1.1 must keep the
reference temperature, otherwise the answers are not reference answers. Instead the validator compares in logit
space: from probabilities p (miner) and q (reference) it computes centred log-ratios

    d(p, q) = T · max_i | (ln p_i − mean_j ln p_j) − (ln q_i − mean_j ln q_j) |      (p, q floored at 1e-6)

i.e. the largest disagreement between the miner's and the reference's option logits, in the model's own logit
units. The floor is 1e-6, not tiny: the reference engine returns float32 probabilities and on extremely confident items
the small side is pure rounding; below 1e-6 an option carries no usable fingerprint anyway. The distance is
invariant to the softmax temperature and does not saturate when the model is confident: a near-one-hot answer
still carries the exact logit gaps, and those gaps are a fingerprint of the weights and the prompt. TV distance
is reported for comparison.

### 6.3 Round scoring per pool
Per miner and round: n = 8 items (both pools). Each item is *verified* or not; the miner cannot tell which
(the validator draws the verification set with its own secret after the responses arrive).

* s1-fast: all 8 items are verified (reference cost ≈ 15 ms per item on the validator GPU).
* s1-pro: a secret random 3 of 8 items are verified **(spot check)**; the rest are real work the miner must
  do anyway.

Per verified item the validator computes the distance d of §6.2 against its own reference runtime; D is the
**mean d over the round's verified items**. A malformed, late, unsigned or inconsistent response scores 0 for
the round. Otherwise:

* **D ≤ 0.20** → the round scores `0.8 + 0.2 × speed` (lw = 0.2; speed from the validator-measured latency,
  rc3 formula, bounds per model: s1-fast 0.5–8 s, s1-pro 1–12 s);
* **D > 0.20** → the round scores 0 (round lost);
* **D > 0.50**, or **any single verified item with d > 2.0** → **strike**: the round scores 0 and the EMA is
  reset to 0.

The tolerances and latency bounds are release constants from the GPU calibration **[cal]** (CALIBRATION.md);
the reference runtime on the same GPU type gives D = 0 exactly.

### 6.4 Why the spot-check rate makes cheating unprofitable (s1-pro)
A cheater answers a fraction x of items with something cheaper (quantised model, cache, other model). With k of n
verified, the chance a round catches at least one cheated item is P(x) = 1 − C(n − xn, k)/C(n, k):
x = 1/8 → 0.375, x = 1/4 → 0.64, x = 1/2 → 0.93. A catch resets the EMA; with α = 0.1 and the eligibility
threshold 0.3, rebuilding to the honest steady state takes ≈ 22 rounds and the first 4 rounds earn nothing.
Expected loss per cheating round ≥ P(x) · (≈ 15 rounds of reward) vs a saving of at most x · (compute cost of a
round), which is below one round of reward for any miner whose reward covers its GPU. Example x = 1/8:
loss ≈ 0.375 · 15 = 5.6 rounds of reward against a saving of 1/8 of one round of compute. The honest false-strike
rate must be ≈ 0, which is what τ is calibrated for (§7).

### 6.5 Attacks and answers
| Attack | Defence |
|---|---|
| wrong / cheaper model | logit-space distance on fresh items (calibration measures base Qwen3.5-4B, JevK5 sibling, FP8 builds) |
| quantised substitute (FP8/INT8/NVFP4 of the right weights) | same; FP8 Plumb/Rune measured explicitly |
| cached answers | items are fresh per miner and round; exact repeats are rare (< 1 % of state–question pairs in 200k draws) and a repeat can only replay an answer the real model produced |
| copied answers (relay another miner) | rc3 hotkey binding of request and response; per-miner items double the victim's work |
| latency spoofing | latency is measured by the validator's own clock around the signed exchange; only real speed helps. Small weight (0.2) and only on rounds that score (D ≤ 0.20) |
| answering only verified items well | verified set drawn after the response; spot-check maths §6.4 |
| inconsistent fields (reference-looking choice over wrong probabilities) | validator recomputes derived fields |
| model hopping to dodge strikes | state keyed by hotkey+model+manifest; new commitment = start from 0 |

## 7. Determinism and tolerances (calibration)

Measured with the exact reference runtimes (data and thresholds: CALIBRATION.md, `scripts/calibration_s1`):

* s1-fast — measured on RTX PRO 6000 Blackwell and H100 80 GB (RTX 4090: see CALIBRATION.md): reference path
  (jevk5 CUDA-graph batch-1), repeat run, new process, eager micro-batches, vLLM batched (alternative engine),
  and adversaries FP8, Qwen3.5-4B base, JevK5 v0.2 sibling.
* s1-pro — measured on RTX PRO 6000 96 GB and H100 80 GB (the deterministic build gave **bit-identical**
  answers on both): deterministic BF16 vLLM at concurrency 1 and 16, new process, stock (non-deterministic)
  vLLM, and adversary FP8.

τ_model is set above the largest honest cross-GPU distance with margin, and must sit clearly below the adversary
distribution. Other GPU cards (L4, A100, H200 …) should work but are not yet measured: run
`scripts/calibration_s1` on the card first. The calibration also covers the two rc3 checks
(Qwen3-4B cross-GPU drift; validator round time with many miners).

## 8. Switches and defaults (not-yet-robust features stay OFF)

| Setting | Default 1.1.0rc1 | Meaning |
|---|---|---|
| `REAX_MECHANISM` | `s1` | `mpm1` = rc3 mechanism (Qwen3-4B, `/v1/decide`, unchanged) as fallback |
| pool `s1-fast` | ON, share 1.0 when alone | release constant |
| pool `s1-pro` | **OFF** (`REAX_FEATURE_S1_PRO_POOL`) | needs every validator to hold an 80–96 GB reference GPU; turning it on needs a new release. When on: shares s1-fast 0.35 / s1-pro 0.65 |
| unfilled pool share | owner UID | as rc3 (no miners → owner) |
| images, multi-question requests, external-client routing, capacity audits | OFF (startup refuses) | unchanged from rc3 |

All scoring constants (τ, ε, n, k, shares, latency bounds) are release constants; validators refuse overrides on
finney (rc3 rule).

## 9. What changes in the code

* `manifests/` + `manifest.py` (canonical id, file verification).
* `s1items.py` typed generator; `s1protocol.py` strict parser + answer consistency check;
  `s1backends.py`: `PlumbProductionBackend` (jevk5 graph batch-1, reference readout), `RuneVllmBackend`
  (reference adapter readout against a local deterministic vLLM), `S1HashBackend` for tests.
* `s1scoring.py`: logit distance, spot-check selection, strike.
* `chain.py`: read commitments; miner CLI publishes its commitment.
* `validator.py`/`miner.py`: mechanism switch; pools; rc3 paths untouched under `mpm1`.
* Docker: `runtimes/s1-fast/Dockerfile` (reference fast base + jevk5 pins), `runtimes/s1-pro/Dockerfile` (vLLM digest).
