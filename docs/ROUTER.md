# REAX router

`reax-router` (also `python -m reax_subnet.router`) is an optional, separate, long-running
client-facing router that forwards signed requests to miners. It uses the **validator's existing
hotkey** and reads its score file without modifying it. It never sets weights, serves
an axon, commits a role, creates a wallet, or submits any chain transaction. The
validator continues its usual audits independently. The router is not needed to run a
miner or a validator, and it is disabled until an operator configures and enables it.

## Client contract

A trusted client on the router's private network calls `POST /internal/v1/decide`
with the `X-S1M-Internal-Token` header and the fields `request_id`, `model`, `state`,
`questions`, `deadline_unix_ms`. Images receive 422; malformed requests/unsupported
models receive 400; authentication failures receive 401. Requests carry 1-16 questions,
split into one subnet decision per question in a single signed request.

Responses carry the request ID, model, original question IDs, strict miner
answer objects, `usage.input_tokens`, `usage.output_tokens=0`, `usage.decisions`,
`latency_ms`, `tokenizer_version`, and `model_revision`. The response also carries the headers
`X-Reax-Served-By: miner|reference`, `X-Reax-Verified: 1|0` and, only for miner-served answers,
`X-Reax-Miner-Uid: <uid>` (public chain data). They are never part of the JSON body, whose shape is
identical for both paths.

Access control and consent rules belong to the client that calls the router. A 503 with
`{"error":{"message":"no eligible miner"}}` means no miner meets the router's current eligibility checks; the caller
should apply its configured fallback (it also means a request selected for verification could not be
verified, see below). `GET /health` is liveness. `GET /health/ready?model=s1-fast` returns a model
state, revision, `miners` (eligible count) and `reference_fallback` (200 when ready, 503 when
unavailable); omitting the query returns all configured models. A model is ready when the token counter
is available **and** (eligible miners >= `REAX_ROUTER_MIN_MINERS` **or** the reference fallback is on and a reference
backend is configured for it).

## Configuration

| Variable | Default / requirement |
| --- | --- |
| `REAX_ROUTER_TOKEN_FILE` | Required regular file, mode **0600**, ASCII token of at least 32 bytes; startup refuses missing/weak/accessible tokens. |
| `REAX_ROUTER_MODEL_REVISION`, `REAX_ROUTER_TOKENIZER_VERSION` | Required exact pins for `s1-fast` (the single pair is the s1-fast default). |
| `REAX_ROUTER_MODEL_REVISION_<MODEL>`, `REAX_ROUTER_TOKENIZER_VERSION_<MODEL>` | Per-model pins, `<MODEL>` = `S1_FAST` or `S1_PRO`. Required for every other configured model (s1-pro never inherits the s1-fast pins); for s1-fast they override the single pair. Responses, `/health/ready` and `/status` carry the pin of the model that answered. |
| `REAX_ROUTER_MODELS` | `s1-fast`; optional comma-separated `s1-pro` requires its release manifest and its own pins and token counter (see "Token counting"). |
| `REAX_ROUTER_ENABLED` | `1`; `0` is the kill switch: readiness unavailable, decide returns 503, **and the reference fallback is disabled too**. Restart after changing env. |
| `REAX_ROUTER_HOST`, `REAX_ROUTER_PORT` | `127.0.0.1`, `8080`; expose only on a private network. |
| `REAX_ROUTER_MIN_MINERS` | `1`. Mainnet launch day has only the anchor miner, so launch with `1`; raise to `3` once more miners are eligible. With the reference fallback on, fewer miners no longer make the router unready. |
| `REAX_ROUTER_MIN_EMA` | `0.9` (track record, see "Who gets customer traffic"). Joined to the validator's current manifest role and hotkey. |
| `REAX_ROUTER_MIN_ROUNDS` | `720` validator rounds with the current hotkey and role (720 x 120 s = 24 h). `0` switches the rule off. |
| `REAX_ROUTER_STRIKE_COOLDOWN_S` | `604800` (7 days): how long a dated strike keeps a miner out. |
| `REAX_ROUTER_INFLIGHT_CAP` | `1` concurrent router request per miner hotkey. Keep it at 1: miners allow 2 concurrent requests per caller and the router signs with the validator hotkey, so the validator's scoring rounds need the second slot. |
| `REAX_ROUTER_PER_MINER_PER_MINUTE` | `12` (range 1..19): sliding 60 s window of router requests per miner hotkey. Miners allow 20/min per caller, shared with the validator's own rounds. |
| `REAX_ROUTER_VERIFY_RATE` | `1.0` (0..1): share of requests, **after probation**, sampled for synchronous verification against the reference before a miner answer is served. |
| `REAX_ROUTER_VERIFY_FLOOR` | `0.1` (0..1, effectively capped at the verify rate): the part of the sampled share that is verified fail-closed whatever the reference load. The rest of the sampled share is skipped, and counted, when no reference slot is free (see "Probation, sampling and the reference load guard"). |
| `REAX_ROUTER_PROBATION_ANSWERS` | `20` (0..100000): a miner's first N customer answers (per hotkey, in memory) are always verified. `0` switches probation off. |
| `REAX_ROUTER_TIMEOUT_QUARANTINE_S` | `60` (range 5..600): after a miner attempt ends in a timeout, disconnect or other transport error the miner is not routed to for this long (see below). |
| `REAX_ROUTER_MINER_TIMEOUT_S` | `1.5` (range 0.05..120): cap for one miner attempt, so a slow miner cannot use up the request budget that the (already running) reference answer needs. |
| `REAX_ROUTER_REFERENCE_CONCURRENCY` | `2` (range 1..16): router-wide limit on concurrent reference computations, counted in decisions (see below). |
| `REAX_ROUTER_MIN_REFERENCE_BUDGET_S` | `0.2` (range 0..5): a reference computation is not started when less than this remains before the request deadline; the request is then treated as `reference_unavailable` (503). |
| `REAX_ROUTER_REFERENCE_FALLBACK` | `1`: answer from the reference runtime when no miner answer can be served. |
| `REAX_ROUTER_TIMEOUT_S` | `3.0`; total counting and miner budget is additionally bounded by deadline minus 300 ms. |
| `REAX_ROUTER_TOKENIZER_DIR` | Offline directory with the pinned Plumb tokenizer artifacts, including its chat template/config; no weights needed. |
| `REAX_ROUTER_TOKENIZER_DIR_S1_PRO` | Offline directory with the Rune (Gemma 4) `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` of the pinned s1-pro weights; no weights needed. All three files are hash-checked against the s1-pro manifest. |
| `REAX_ROUTER_SHADOW_RATE` | `0.05`; async after-the-fact sampling of **one** decision, used only for requests not selected for synchronous verification. `0` disables it. |
| `REAX_ROUTER_STRIKES_FILE` | `~/.reax/router-strikes.jsonl`; separate durable exclusion ledger. |

The wallet/network/state settings are **identical to the validator's** (`config.py`):
`REAX_NETWORK`, `REAX_NETUID`, `REAX_WALLET_NAME`, `REAX_WALLET_HOTKEY`,
`REAX_WALLET_PATH`, `REAX_STATE_PATH`, and development-only `REAX_ALLOW_PRIVATE_IPS`.
Use the s1 mechanism. For shadow verification configure the validator's loopback
`REAX_S1_FAST_RUNTIME_URL` (and, if applicable, `REAX_S1_PRO_RUNTIME_URL`) and shared
`REAX_RUNTIME_TOKEN_FILE`. The same runtime serves synchronous verification and the reference fallback. No coldkey is needed or mounted.

## Token counting

The router counts input tokens itself; miner usage is never accepted. The pinned
`jevk5.runtime.JevK5.encode` uses only `self.tok`: the same JSON evidence/criterion/
lettered-options prompt, chat template with thinking disabled, then encode without
special tokens. `router_tokens.py` reproduces these operations on CPU. A startup
hash check verifies `tokenizer.json` against the release manifest, then loads the
local tokenizer with remote code disabled. Install the repo's model extra if using
a bare Python environment; the validator image already contains transformers.

Supply the tokenizer config/chat template from the **same immutable HF revision** as
the reference runtime. The existing manifest pins tokenizer.json but does not hash
these additional template artifacts; that existing provenance limitation remains.
An optional test compares 50 generated decisions against the pinned JevK5 encode
method, bypassing its model-loading constructor and using a deterministic tokenizer
to verify exact template inputs/flags/IDs; it skips when JevK5 is absent. This proves
encoder equivalence for a shared tokenizer, not the provenance of a mounted template.
Missing transformers/artifacts or failed counting results in 503 (and unavailable
readiness). Inputs over the manifest token limit receive 400. No guessed or zero
input token counts are returned.

**s1-pro** is counted by `RuneTokenCounter`: it builds the same text-only prompt as
`runtimes/s1-pro/rune_adapter.py` (system prompt, `SHARED STATE (JSON string):` state, `QUESTION:`/`OPTIONS:` branch,
Gemma 4 chat template with thinking off, encode without special tokens) and counts it with the tokenizer from
`REAX_ROUTER_TOKENIZER_DIR_S1_PRO`. `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` must match the
hashes in `s1-pro-v1.json`; otherwise the counter does not load and s1-pro stays unready (503) while s1-fast keeps
working. Inputs over the manifest's 4096 tokens get 400. Limitation: the production reference image is built with
`chat_template_nospace.jinja`, which the manifest does not hash; if that template renders differently from the pinned
`chat_template.jinja`, the counted (billed) number can differ by a few tokens and an input right at the limit can be
accepted here and refused by the reference. Such a request ends in 503 (never a wrong answer).
Counters are per model; a model whose counter failed to load is unready, the others are unaffected.

## Selection, failure and verification

Every 60 seconds the router joins a chain snapshot (including commitments) with the
validator's atomically saved JSON state. Eligibility requires the committed release
model/manifest, matching stored role/hotkey, finite EMA at least the threshold, a
valid permitted IP and a recent score snapshot. The owner and router's own validator
hotkey are excluded. Score age uses the earlier of `saved_at` and file mtime; a failed
refresh never renews it. Last good chain/score snapshots expire after one hour.

The current validator state stores cumulative strike counts **without timestamps**;
it cannot establish when a past strike happened. If `last_strike_at` or `last_strike`
(epoch seconds) is present it is honored for a 24-hour exclusion. Router shadow
strikes always have timestamps and survive restart. No validator state format or
scoring behavior is changed.

Candidates are ordered by descending EMA, then recent router latency, then UID. Miners at
their in-flight cap or per-minute budget are skipped, as are miners that answer HTTP 429 ("busy").
A 429 is recorded (`outcome="busy"`) but never counts toward the breaker or the two-attempt limit. At most two are tried **sequentially**; the first gets half the
remaining budget when another miner exists. There is no hedging. Three consecutive
failures open that hotkey's breaker for 60 seconds; success resets the failure count.
A timeout, disconnect or other transport error where the miner may still be computing quarantines that miner
for `REAX_ROUTER_TIMEOUT_QUARANTINE_S` (no strike, no ledger entry): the miner holds its per-caller slot until its
inference returns and a client timeout does not free it, but the validator's scoring rounds need that slot. An HTTP
status or an invalid reply proves the miner finished and does not quarantine; a 429 (busy) never does.
Signed response verification and the existing strict `parse_s1_response` must pass.
The router's tighter probability-sum tolerance (1e-4) is also enforced.

### Per-miner budget

The router signs with the validator hotkey, so each miner's per-caller limits (2 concurrent, 20 per
minute) are shared with the validator's scoring rounds. The router therefore uses at most 1 concurrent and
12 requests per minute per miner (both configurable, see above) and leaves the rest to the validator; a
starved validator would score honest miners 0.

### Probation, sampling and the reference load guard

Verification has three triggers, checked in this order:

1. **Probation.** A hotkey's first `REAX_ROUTER_PROBATION_ANSWERS` customer answers (default 20) are always verified
   synchronously. The counter is per hotkey and in memory; it counts verified matching answers. Any mismatch (strike or
   grey zone, synchronous or router shadow check) resets it to 0. A restart puts every miner back on probation, which
   fails safe. A request whose first candidate is on probation starts the reference immediately, otherwise it starts
   as soon as an answering miner turns out to be on probation.
2. **Floor share.** One secret draw per request (`secrets`, 1 in 1,000,000 resolution, independent of the content, the
   caller, the miner and every earlier draw) decides the rest. A draw below `min(VERIFY_FLOOR, VERIFY_RATE)` is verified
   and **fails closed** whatever the reference load.
3. **Best-effort share.** A draw between the floor and `VERIFY_RATE` is verified when a reference slot is free. When
   none is free the check is skipped: the miner answer is served unverified, counted as `skipped` and
   `skipped_reference_busy`, and still falls under the asynchronous shadow check.

Fail-closed means 503 (no miner answer is returned, no strike) and applies to probation, the floor share, and every
case where no reference runtime exists for the model (a missing reference is a misconfiguration, not load). Reference
timeouts and errors after a slot was obtained also stay fail-closed in all shares.

Integrity argument. (a) A miner cannot tell or influence which requests are checked: the draw is secret and
content-independent, and probation is keyed to its own hotkey, so an answer is verified with probability at least
`VERIFY_FLOOR` (10 % by default) at any time, at any load, and with certainty for its first 20 answers. A miner that
returns even a single answer outside `tau_*` is caught within about `1/VERIFY_FLOOR` answers on average (the chance of
going `k` answers undetected is at most `(1 - floor)^k`, 12 answers per minute and miner at most): it is struck, loses
customer traffic for 24 h, and on return starts probation again. (b) Skips only remove best-effort checks, never the
floor. A miner cannot cause reference saturation by itself (slots are held only by this router's own requests, one
miner attempt at a time per hotkey, never longer than the request deadline), so it cannot steer skips onto its own
answers; saturation comes from customer load, which is the same for honest and dishonest miners. (c) The skip shows up
in the counters (`skipped_reference_busy`) so an operator sees how much of the sampled share is lost; if it stays high,
raise the reference capacity instead of lowering the floor. (d) A new or reset miner can only reach the unverified share
after 20 clean verified answers; the asynchronous shadow check (`REAX_ROUTER_SHADOW_RATE`) covers a share of the
unverified answers on top, without needing a slot. The residual risk is a miner that is correct for 20 answers and then
cheats on a small share; the floor bounds how long that lasts, not that it happens.

### Synchronous verification

A request is selected for verification as described above; if it is selected and there is no backend for the model the router
answers 503 before asking any miner (outcome `reference_unavailable`). The
reference computation (one owned `backends[model].decide([decision])` task per decision, over **all** decisions) starts concurrently with the
miner query and is bounded by the same deadline (`min(timeout, deadline - 0.3 s)`). Every decision is then
compared with `score_round` and the release `POOL_PARAMS`, exactly as the validator scores a round with
all items verified. A miner answer is accepted only in the validator's passing zone: **no strike**
(mean <= `tau_strike` and every item <= `tau_item`) **and mean <= `tau_mean`**. Above `tau_mean` is the
validator's grey zone: the round would score 0 without a strike.

* match: the miner answer is served (`X-Reax-Verified: 1`).
* strike-level mismatch: strike recorded as below (`verify_strike`), the reference answer is served.
* grey-zone mismatch: no strike, counted (`mismatch_grey`), the reference answer is served.
* reference unavailable, too slow or without a free slot (see admission below): **fail closed**. The miner answer is not returned; the router answers
  503 `no eligible miner` so the gateway falls back to its own GPUs. A reference outage is never a strike.

If the reference fallback is off, the two mismatch cases also end in 503.

### Reference admission

At most `REAX_ROUTER_REFERENCE_CONCURRENCY` reference computations run at once in the router, counted in decisions: a
request with k decisions needs k free slots at the moment its reference is needed (at the start for a request selected
for verification, otherwise when falling back). With too few free slots the request is answered 503 immediately: a
verification request never serves the miner answer (outcome `reference_unavailable`, no strike), a fallback-only request
just gets 503. Nothing waits in a queue. A request with more decisions than the limit therefore always gets 503 on the
reference paths; size the limit to the largest accepted request (up to 16). Slots are released only when the
per-decision task has finished or has been cancelled and awaited, so a failed, timed-out or cancelled request leaves no
router-side reference task or slot behind. The reference runtime itself (FastAPI/uvicorn) does not stop a request when the
router abandons it, so it finishes abandoned requests; that work still occupies the runtime until done. Reference answers must also satisfy the 1e-4 probability-sum check (503 otherwise; never
renormalized). No 200 is returned after the request deadline, and strike ledger writes run off the event loop.
**This limit does not isolate the shared validator runtime.** Sharing the validator's runtime is acceptable only while
the gateway's node capacity is <= 2 and `REAX_ROUTER_REFERENCE_CONCURRENCY` is <= 2. Within that bound each validator
item waits behind at most about 3 router decisions (about 0.2 s on the A6000 at 55-75 ms per decision), and the anchor
miner's measured latency may rise (roughly 0.5 s to 2 s, which lowers its speed factor but cannot fail rounds). Above
either value, run a separate reference runtime (or a reservation mechanism) for the router first.

### Reference fallback

When no miner answer can be served (no eligible miner, all attempts failed, busy or over budget, or a
verification mismatch), the router answers from the reference runtime (our own GPU running the identical
pinned model) if that is still inside the deadline, otherwise 503. The answer objects are built as a miner
builds them (`S1Response` bytes, strict `parse_s1_response`, then the same per-decision answer object), so the
body is identical in shape to a miner-served one; token counting is unchanged. Reference-served responses have
`X-Reax-Served-By: reference` and `X-Reax-Verified: 1` (the answer is the reference).

### Asynchronous shadow check

For requests not selected for synchronous verification, after returning an answer a sampled decision is recomputed on the validator's
reference backend, using `score_round` and the validator's release thresholds:
mean distance > `tau_strike` (0.50) or item distance > `tau_item` (2.0) causes a strike.
The non-striking grey zone above `tau_mean` is treated exactly as by the validator.
A reference outage does not penalize a miner. Verification tasks are bounded to 32
in flight and a configured timeout; sampling may be skipped when all slots are busy.
A strike opens the miner's breaker for 24 hours and appends only timestamp, UID,
hotkey, subnet request ID, model and outcome to the separate JSONL file. **Ingestion
by the validator is not implemented**; an operator may build a reviewed ingestion
path later. **Open: router strikes are not read by the validator, so they have no weight impact yet;
they only exclude the miner from this router for 24 hours.** Persistence failures are logged and the in-memory exclusion remains;
operators must monitor them because exclusion would not survive restart.

## Metrics and status

`GET /metrics` (Prometheus text) and `GET /status` (JSON: eligible miners and UIDs per model, counts since
start, start time) require the same `X-S1M-Internal-Token` (401 otherwise). They are payload-free:
`reax_router_requests_total{model,served_by="miner|reference|none"}`,
`reax_router_verify_total{model,outcome="match|mismatch_strike|mismatch_grey|reference_unavailable|skipped|skipped_reference_busy"}`,
`reax_router_miner_attempts_total{model,outcome="ok|error|busy|budget|timeout"}`,
`reax_router_served_by_uid_total{model,uid}`, gauges `reax_router_eligible_miners{model}` and
`reax_router_probation_miners{model}` (eligible miners still on probation) and the summary
`reax_router_latency_seconds{model,served_by}` (`_count`, `_sum`). Counters reset on restart.

`GET /status` also includes `last_24h` per model with `miner`, `reference`, `none`,
`verify_match` and `verify_mismatch` (both strike and grey mismatches), `verified` (miner answers compared with the
reference, match or mismatch), of which `probation` (forced by probation) and `sampled` (drawn by the sampling, no
probation), `skipped` (miner answers served unverified) of which `skipped_reference_busy` (the sampled check was dropped
because no reference slot was free), `window_seconds=86400`,
and `since=max(started_at, now-86400)` in Unix seconds. The top-level
`miner_uids_served_24h` maps each model to sorted distinct public UIDs that actually served
an answer, excluding attempted miners whose answers were discarded. These in-memory counters
reset on restart and use at most 1,440 minute buckets per model, pruned on writes and status
reads; the current minute and preceding 1,439 minutes are retained (the oldest partial
minute expires up to 59 seconds early). Per model, `served_by_uid_24h` maps each public UID to the number of answers it
served in the window (JSON keys are strings), and `probation_miners` is the number of eligible miners still on
probation; the top level adds `verify_floor` and `probation_answers`. UID storage is bounded to the subnet's 256 UIDs.
Authentication and lifetime Prometheus metrics are unchanged.

## Privacy and operations

The miner sees the state, instructions and options in clear text and can retain them;
this service cannot prevent that. Never send content without the caller's consent
check. Each hop uses a fresh random request ID and decision IDs, and a constant `q`
question ID. No caller request ID, caller question ID, API key, IP or account data
is sent as transport metadata. Caller identifiers deliberately placed in content
remain visible to the miner. Payloads are held in memory for request/shadow handling;
logs and strike files never contain state, instructions, answers or probabilities.
Access logs are disabled. Routing logs contain only subnet request ID, UID, latency
and outcome; attributable shadow strike lines also contain the hotkey.

`deploy/docker-compose.router.yml` adds an independent service using the validator
image and separate writable router-state volume. It reads the validator's **directory**
read-only (so atomic state replacement is visible), a staged **hotkey-only** wallet
tree, the tokenizer directory and 0600 token files. Existing services are unchanged.
Copy `deploy/router.env.example` to `deploy/router.env`; it defaults to **disabled**.
Create the token locally with `openssl rand -hex 32 > deploy/secrets/router-token`
and `chmod 0600 deploy/secrets/router-token`. Run as the owning UID/GID.
The reference runtime must already be running: it is needed for verification, the fallback and shadow checks.

For Finney mainnet set `REAX_NETWORK=finney`, `REAX_NETUID=92`, min miners 3 once enough miners are eligible, the
registered/staked validator hotkey and exact pins (launch values in `deploy/router.env.example`). First verify readiness
with generated internal traffic, then enable routing for real traffic. Disable
`REAX_ROUTER_ENABLED` and restart to stop subnet routing.
No chain write is performed by the router or this compose example.

## Who gets customer traffic (track record)

Customer requests only go to miners that have passed our own validator correctly for a while, never to a fresh
registration. The router re-reads the validator state on every refresh and lets a miner through only if all of this holds:

1. The validator scored this hotkey with the miner's current committed role (model and manifest). A recycled UID or a
   role change starts from zero rounds, so it has to earn the track record again.
2. At least `REAX_ROUTER_MIN_ROUNDS` scored rounds (default 720, one day at the 120 s round interval). Rounds where the
   miner failed to answer count too; they hurt the average instead.
3. Moving average (EMA) at least `REAX_ROUTER_MIN_EMA` (default 0.9). With the validator's alpha of 0.1 a brand-new miner
   with perfect scores needs 22 rounds to reach 0.9, so the rounds rule is the binding one. A miner that is always
   correct but slow settles near 0.8 and stays out; the router abandons a miner after 1.5 s anyway.
4. The latest round passed: score at least `1 - latency_weight` (0.8 for `s1-fast`). Zero means a grey-zone, failed or
   struck round.
5. No strike. The validator state only keeps a strike count, no date, so a miner with any strike is excluded until its
   hotkey or role changes. If the state gains a `last_strike_at` timestamp, a strike older than
   `REAX_ROUTER_STRIKE_COOLDOWN_S` no longer blocks. The router's own 24 h strike breaker still applies on top.

Anything unreadable in a score entry fails closed. Verification (probation, then sampling, see above) is separate; this rule only
decides who is asked. The `/status` page reports per model `seen_miners` (right manifest, reachable and scored or not),
`eligible_miners` (passes the rule, not in a breaker) and `excluded` (counts per reason: `no_score`, `bad_score`,
`recent_strike`, `strike`, `rounds`, `ema`, `last_round`). `/metrics` has `reax_router_seen_miners` next to
`reax_router_eligible_miners`. The public provider count must be taken from `eligible_miners` only.
