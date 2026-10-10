# REAX Minimal Production Mechanism (MPM), protocol `reax-mpm/1`

Release 1.0 ships the smallest incentive mechanism we can defend today. Everything else from the long-term
REAX design is switched off (see [FEATURE-FLAGS.md](FEATURE-FLAGS.md)).

## In one paragraph

Miners serve one approved open model. Validators send each miner a batch of fresh multiple-choice questions,
signed with the validator hotkey. The miner returns one probability distribution per question, signed with the
miner hotkey. The validator runs the same model itself and checks how close each answer is to its own result.
A miner that runs the approved model matches; a miner that runs something else, guesses or copies does not.
Matching miners split the weight equally, with a small bonus for speed.

## Roles

| Role | Runs | Needs |
|---|---|---|
| Miner | `reax-miner`: HTTP endpoint `POST /v1/decide`, publishes `ip:port` on chain | GPU with the approved model, public IP/port, registered hotkey |
| Validator | `reax-validator`: query loop, reference model, `set_weights` | GPU with the approved model, registered hotkey with validator permit and stake |
| Subnet owner | chain hyperparameters, releases | owner coldkey (cold storage) |

No other service is involved: no gateway, no database and no external API.

## Model and probabilities

* Catalog (closed): `qwen3-4b-instruct-2507` = `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0). The development models in
  `src/reax_subnet/backends.py` are refused unless `REAX_ALLOW_DEV_MODELS=1`.
* Probability extraction `label_logits_v1`: the options are rendered with letters A–H in a fixed chat prompt
  (`PROMPT_TEMPLATE_ID = label_logits_v1/2`); one forward pass; `p` = float32 softmax over the logits of the
  letter tokens at the answer position, **divided by the model's protocol temperature T** (catalog value; T = 4 for
  the production model). No sampling, so seeds
  and generation settings do not matter. Why T = 4: the approved model is very confident (mean top
  probability 0.92 at T = 1). Measured on Qwen3-4B, a cheat that knew only the model's top choice matched
  80 % of items at T = 1, but 30 % at T = 4, because the full logit gaps then have to be reproduced.
* Miners and validators run the same code (`HFBackend`) on GPU in bfloat16 (the defaults). One request
  (12 items) is computed as batches of 8 and 4 on both sides, so batching cannot make honest results differ.

## Wire protocol

`POST /v1/decide`, body (UTF-8 JSON, at most 256 KiB):

```json
{"protocol": "reax-mpm/1", "request_id": "<1-64 [A-Za-z0-9_-]>", "model": "qwen3-4b-instruct-2507",
 "items": [{"id": "<id>", "question": "<= 2000 chars", "options": ["2..8 unique strings, <= 200 chars"]}]}
```

1–16 items per request. Response (at most 64 KiB):

```json
{"protocol": "reax-mpm/1", "request_id": "<same>", "miner_version": "1.0.0rc3",
 "results": [{"id": "<id>", "probs": [0.61, 0.27, 0.12]}]}
```

Exactly one result per item; `probs` has one finite number in [0, 1] per option and sums to 1 within 1e-3
(the validator renormalises). Unknown fields, missing fields, NaN/Infinity, booleans, duplicates or a different
`request_id` make the whole response invalid.

**Authentication.** Requests carry `btauth/1` headers (Bittensor SDK `http_auth`): sender hotkey, receiver
hotkey (the miner), nanosecond nonce, sr25519 signature over method, path, body hash, nonce, sender and
receiver. The miner accepts only senders with a validator permit and at least `REAX_MIN_VALIDATOR_STAKE`
stake on the subnet, rejects stale (> 10 s) and replayed nonces, and checks the signature before parsing.
Responses are signed by the miner hotkey under method `REAX-RESPONSE`, path `/v1/decide/<request_id>` and
receiver = the validator; the validator rejects a response signed by any other hotkey.

## Validator round (every `REAX_ROUND_INTERVAL_S`, default 120 s)

1. Read the metagraph. If the chain is unreachable, skip the round (no state change).
2. Drop departed UIDs; reset the score of any UID whose hotkey changed (no score inheritance).
3. For **each** miner with a public endpoint, draw 12 fresh items from a cryptographically seeded generator
   and compute the reference distributions locally.
4. Query all miners at the same moment (up to `REAX_MAX_PARALLEL` = 256, i.e. every UID), timeout 20 s.
5. Score (below), update the moving average, save state atomically. A metagraph read that shrank by more than
   10 % is accepted only after a second identical read (protects against truncated RPC answers, still follows an
   owner trim); scores follow hotkeys, so renumbered UIDs keep their score.
6. After a warm-up of 6 rounds (only when starting without saved state), and when at least
   `max(weights_rate_limit, REAX_WEIGHTS_INTERVAL_BLOCKS)` blocks passed since this validator's last update,
   call `set_weights` (SDK 11 picks commit-reveal automatically when the subnet has it on). A submission that
   hangs is abandoned after 300 s; scoring continues meanwhile.

The item generator uses only open preference/plausibility templates. Computable templates (arithmetic,
number sequences) were removed after review: a few lines of code answered them without any model.

Items are generated, not stored: there is no answer key and no private pool, so nothing can leak or be
memorised. A different item set per miner means a miner cannot reuse another hotkey's answer.

## Scoring (`src/reax_subnet/scoring.py`, pure and deterministic)

```
item passes   <=> TV(p_miner, p_ref) = 0.5 * sum |p_miner - p_ref| <= tau          (tau = 0.15)
fidelity       = passed items / items sent
round score    = 0                                        if fidelity < 0.8   (fewer than 10 of 12 items)
               = fidelity * (0.8 + 0.2 * speed)           otherwise
speed          = clamp((20 s - latency) / (20 s - 2 s), 0, 1)     latency measured by the validator
EMA            = 0.9 * EMA + 0.1 * round score            per UID, persisted
weight(uid)    ∝ EMA(uid)  for EMA >= 0.3, owner UID excluded
owner UID      receives REAX_BURN_FRACTION (default 0), and everything when no miner qualifies
```

No response, timeout, wrong signature, malformed JSON or protocol violation = round score 0.
Registered UIDs without a reachable public endpoint also score 0.

### Calibrating tau

Production model `Qwen3-4B-Instruct-2507`, 128 items, CPU (measured during release hardening;
raw data not published). Share of items within tau = 0.15 of the float32 reference, and the probability that the answer
source passes a whole round (10 of 12 items):

| Answer source | T = 1 (old) item pass / round pass | T = 4 (release) item pass / round pass |
|---|---|---|
| same model, batch of 1 vs batch of 8 | 100 % / 100 % | 100 % / 100 % |
| knows the reference's top choice exactly, shape tuned per option count (needs the model itself) | ~81 % / ~60 % | 57 % / 5.6 % |
| one-hot on the reference's top choice | 80 % / 58 % | 39 % / 0.2 % |
| Qwen3-0.6B (cheaper model of the same family), at the cheater's best temperature | 20 % / ~0 | 15 % / ~0 |
| Qwen2.5-0.5B, at the cheater's best temperature | 16 % / ~0 | 16 % / ~0 |
| model-free logit table fitted to 608 reference answers (held out) | – | 22 % / ~0 |
| uniform distribution | 2 % / 0 | 6 % / 0 |

Honest numeric drift shrinks with T as well. bfloat16 against float32 (Qwen2.5-0.5B, 96 items, CPU): largest
distance 0.179 at T = 1 (one item would have failed) and 0.043 at T = 4 (mean 0.013). The same comparison for
the 4B model on CPU was too slow to finish (CPU bfloat16 is emulated); it has to be done on GPUs anyway. A model-free lookup table reached 47 % of items on the
0.5B test model at T = 1 (round pass 1 % with 12 items and the 10/12 gate). The same attacks were
re-run against the 4B model at T = 4 (numbers above). A miner only gets weight once its moving
average reaches 0.3, which needs roughly 70 % of items per round; every cheat that does not run the model is
far below that.

T is a property of the model, not a universal constant: on the much less confident 0.5B test model, T = 4
flattened the distributions so much that the uniform cheat matched 58 % of items in one end-to-end round (its
average stayed below the weight threshold). The test model therefore uses T = 1. Any new catalog model needs its
own T from `scripts/calibrate.py` (uniform and oracle-top cheats both < 1 % round-pass probability).

**Calibration required:** the honest distance between real GPU types in bfloat16 is not measured yet. Run
`scripts/calibrate.py` on at least two GPU types (see CALIBRATION.md). Pass: honest max ≤ 0.10, every
cheat's round-pass probability < 1 %.

## Fixed parameters

All scoring constants (T, tau, gate, latency bounds and weight, EMA alpha, weight threshold, items per round,
timeout, warm-up) are part of the release. A validator refuses to start if any of them is overridden, except on
development networks (`REAX_ALLOW_DEV_MODELS=1`). Only `REAX_BURN_FRACTION` is configurable, and it must be
identical on all validators.

## Versioning

* `reax_subnet.version.PROTOCOL = "reax-mpm/1"`: miners reject any other value.
* `WEIGHTS_VERSION_KEY = 1000` is sent with `set_weights`. The owner sets the `weights_version` hyperparameter
  to the oldest accepted key, so outdated validators are rejected by the chain.
* Any change to scoring, items, prompt or model bumps the version key and the minor version.

## What MPM does not claim

* Output equivalence is not identity: any computation whose answers stay within tau is indistinguishable
  from the approved model (see THREAT-MODEL.md, R1).
* Rewards are per hotkey. MPM does not measure capacity; one operator with many hotkeys and enough hardware
  earns for each hotkey that answers correctly and fast (R2).
* No third-party traffic flows through the subnet in this release.
