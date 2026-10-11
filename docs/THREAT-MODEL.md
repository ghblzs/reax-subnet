# REAX threat model and adversarial review

Release 1.1 adds the reax-s1 mechanism (exact System1 production models); its attacks are in the next section.
The rest of this document is the 1.0 analysis of the rc3 mechanism (`REAX_MECHANISM=mpm1`), still valid for it.

## reax-s1 (1.1): attacks and how they are handled

Miners serve `s1-fast` (Plumb-4B) or `s1-pro` (Rune 26B-A4B v3) with the reference readout; validators recompute
answers on their own runtime of the same pinned model and compare in logit space (docs/DESIGN-v1.1.md,
docs/CALIBRATION.md). Evidence names: `tests/test_s1.py`, `scripts/localnet/e2e_s1.py` (real chain, real models).

| # | Attack | Status | How it is handled | Evidence |
|---|---|---|---|---|
| S1 | Serve another model under the role (base Qwen3.5-4B, the JevK5 sibling, any cheaper model) | M | Per-item logit distance; measured round means 1.4–1.6 logits vs honest ≤ 0.22 (s1-fast); far above the strike line 0.5. | calibration; e2e `m_other_model` strike, 0 incentive |
| S2 | Quantised substitute of the right weights (FP8) | M | FP8 Plumb: mean of 8 items ≤ 0.20 in 9.8 % of rounds (11.4 % on a 5,000-item recheck; about 0.4–0.5 % of an honest miner's weight over time); FP8 Rune: mean of 3 ≤ 0.20 in 12 %. A miner below the 0.3 eligibility average earns nothing; frequent strikes reset it. | calibration |
| S3 | Use stock (non-deterministic) vLLM for Rune | M (by rule) | The manifest pins the deterministic flags; stock vLLM drifts up to 3.9 logits and fails 35 % of rounds. Honest operators must run the pinned runtime. | calibration |
| S4 | Partial cheating: real model on some items, something cheap on others | M | s1-fast verifies all 8 items; s1-pro verifies a secret 3 of 8 drawn after the answers arrive. One cheap item among 8 is caught in 37.5 % of s1-pro rounds; a catch is a strike (average reset). | `test_spot_check_catches_partial_cheater`; e2e `m_pro_partial` |
| S5 | Plausible-looking fields over wrong probabilities | M | Validator recomputes `choice`, `confidence`, `noul`, `score`, `legend` from the probabilities; any mismatch zeroes the round. | e2e `m_inconsistent` |
| S6 | Cached or replayed answers | M | Fresh typed items per miner and round (128-bit seeds); responses signed and bound to the request id. | `test_fresh_rounds_never_repeat` |
| S7 | Relay / axon copy / answer with another hotkey | M | Same as A6–A8: receiver-bound signatures both ways. | e2e `m_axon_copy` = 401 |
| S8 | Latency spoofing | M | Validator clock; speed is at most 20 % of a round and only counts in passing rounds. | e2e `m_slow` timeout |
| S9 | Hop between roles to shed a strike | M | State is keyed by hotkey + committed role; a new commitment restarts from zero. | `test_role_change_resets_score` |
| S10 | Commit to s1-pro but serve s1-fast (cheaper hardware) | M | Requests name the model and manifest; a miner that cannot answer them fails the round. | `test_s1_only_honest_miners_earn` (role mismatch) |
| S11 | Imitate the model (distillation on the public generator) | R – Medium | A distilled model must reproduce all option logits within 0.2 on average; the closest relative we have (JevK5 v0.2, same base and recipe) averages 1.4. Not measured for a model distilled on this generator; running the real model is cheap for s1-fast. | calibration |
| S12 | Honest miner struck by GPU differences | M (measured GPUs), L (others) | Measured on RTX PRO 6000, H100, RTX 4090, RTX 5090, RTX 3090, L40S and RTX A6000 (CALIBRATION.md): the s1-fast reference path loses at most about 1 round in 550 across GPU types (worst simulated round mean 0.27 and worst possible 0.48 vs strike line 0.50). Per-item outliers are rare but exist: one item in 20,000 reached 1.999999, right at the per-item strike line of 2.0, none exceeded it in 50,000 cross-card comparisons on 25,000 distinct items, so an occasional honest strike (about 20 minutes of moving average) cannot be ruled out. Individual hosts can drift more than their card type (one RTX 4090 host scored about 93 % of rounds): miners and validators run `reax-selfcheck` on each host; Rune's deterministic build is bit-identical across H100 and RTX PRO 6000. Unmeasured cards: run scripts/calibration_s1 first. | calibration |
| S13 | Validators without an 80 GB GPU cannot verify s1-pro | L | The s1-pro pool is OFF in 1.1 and can only be enabled on development networks; enabling it on mainnet needs a new release once validators have the hardware. | `test_config.py` |
| S15 | **Relay validator items through a hosted API** | **R – accepted residual risk** | A hosted API that serves the same reference model may return answers that pass the validator checks. This release has no technical control that prevents relaying. Operators should consider API cost, latency and the provider's terms. Possible controls such as validator-specific items, latency analysis or repeat-item checks are not implemented. | design analysis |
| S14 | Runtime sidecar exposed to the internet | M/L | Sidecars bind 127.0.0.1 and require a shared token; REAX refuses non-loopback runtime URLs. Operators must not publish the sidecar port. | `config.py`, runtime Dockerfiles |

# REAX 1.0 threat model (mpm1)

Scope: the code in this repository at release 1.0 (MPM, `reax-mpm/1`). Assume miners are rational attackers
who read this code, that some validators are lazy or hostile, and that anyone on the internet can reach a
miner's port. "Test" names refer to `tests/` and `scripts/localnet/e2e.py`.

Status: **M** = mitigated in code and tested, **R** = accepted residual risk (with severity), **L** = needs an
action before mainnet use.

## Attacks on the incentive mechanism

| # | Attack | Status | How it is handled | Evidence |
|---|---|---|---|---|
| A1 | Serve a cheaper or different model, or only predict the top choice | M | Answers must be within total variation 0.15 of the validator's own reference (probabilities at temperature T = 4) on ≥ 10 of 12 fresh items per round. On Qwen3-4B, Qwen3-0.6B matched at most 15 % of items (at the cheater's best temperature). Exact knowledge of the reference's top choice (which itself needs the model) matched up to 57 % with a tuned shape (round pass 5.6 %, moving average stays far below 0.3); at T = 1 it would have matched 80 % — found by our own production-model measurement and fixed in rc3. | calibration table in MPM.md; e2e `m_other_model` = 0 incentive |
| A2 | Constant answers (uniform, first option, always the "obvious" choice) | M | Same tolerance; uniform matched 11.5 %, first option 9.4 %, one-hot of the reference's top choice 20.8 % of items. | `test_only_honest_miners_earn`; e2e `m_uniform`, `m_first` |
| A3 | Abstain on hard items, answer only easy ones | M | A missing item counts as failed; a missing or invalid response fails the whole round. | `test_round_score_missing_items_and_length_mismatch` |
| A4 | Format tricks (NaN, Infinity, booleans, negative or >1 numbers, 400-digit integers, wrong length, extra fields, duplicate ids, huge bodies) | M | Strict parser; any violation zeroes that miner's round. 64 KiB response cap enforced while streaming. Any unexpected exception while handling one miner is caught per miner, so one response can never abort the round for everybody. | `test_protocol.py`, adversaries `nan`, `wrong_length`, `oversize`, `garbage`, `overflow` |
| A5 | Replay or cache old answers | M | Items are random per round and per miner; responses are bound to the request id by the miner's signature; requests carry nonces checked against a 10 s window. | adversary `stale_request_id`; `test_auth.py` |
| A6 | Publish another miner's endpoint (axon copying) | M | Requests name the receiving hotkey; the real owner of the endpoint rejects a request addressed to someone else. | adversary `axon_copy`; e2e `m_axon_copy` = 0 |
| A7 | Relay requests to an honest miner | M | Same receiver binding (401), and the honest miner's answer would be signed with the wrong hotkey. | adversary `relay` |
| A8 | Answer with someone else's signature or unsigned | M | Validator verifies that the response is signed by exactly the hotkey it queried. | adversaries `unsigned`, `wrong_signer` |
| A9 | Spoof latency | M | Latency is measured by the validator's own clock, from sending the request to the last byte. Speed is at most 20 % of the score. | `test_slow_miner_loses_at_most_latency_weight` |
| A10 | Inherit a good score by taking over a recycled UID | M | A UID whose hotkey changes starts again from zero. | `test_recycled_uid_does_not_inherit_score`, `test_state.py` |
| A11 | Steal, leak or memorise validator items | M (by design) | There is no item pool and no answer key; items are generated fresh, so a logged item has no future value. | `test_items.py` |
| A12 | Many hotkeys on one machine or one endpoint (Sybil) | R – Medium | Each hotkey gets its own items, all queried at the same moment, so N hotkeys need N× the compute inside the latency bound. Registration cost and the 256-UID cap limit the rest. MPM rewards per hotkey, not per GPU: one process can serve many hotkeys (measured: 12 hotkeys behind one endpoint each earned 1/12). This is the expected equilibrium for 1.0; capacity audits are off (`REAX_FEATURE_CAPACITY_PROBES`). Monitor UIDs per coldkey and per IP. | design property (per-hotkey items); no dedicated test |
| A13 | Imitate the reference without the model (lookup tables, heuristics, distilled small model) | R – Medium | The item generator is public and narrower than general text. Model-free lookup-table experiments reached 38 % and 47 % of items on the 0.5B test model at T = 1, and 22 % on held-out items from the 4B reference model at T = 4 with 608 training answers. These baselines did not reach the per-round eligibility threshold of about 70 %. A table trained on more 4B answers, or a distilled small model, could get closer; this has not been measured. Mitigation options include broader and validator-private item sources. | design analysis |
| A14 | Collusion between miners | R – Low | Miners gain nothing by sharing answers: every miner gets different items. Colluding operators are equivalent to A12. | design |
| A16 | Serve only the validators holding most of the stake | R – Low | A miner can refuse or time out for small validators and still earn most of its weight, because consensus is stake-weighted; it saves the work for those validators. Generic to Bittensor subnets; small validators' weights for that miner drop to 0, which costs the miner its share of their weight. Not mitigated. | protocol design |
| A15 | Owner capture at validator start | M | A fresh validator scores 6 rounds before its first weights, so honest miners already have non-zero averages. | `test_fresh_validator_waits_for_warmup` |

## Attacks by or on validators

| # | Attack | Status | How it is handled |
|---|---|---|---|
| V1 | Weight copying (lazy validators copy published weights) | R – Medium, L | Commit-reveal must be on (`commit_reveal_weights_enabled=true`, reveal period ≥ 1 tempo); SDK 11 commits timelocked weights automatically. MPM weights are stable (equal split among honest miners), so a copier who lags by one reveal period is still close to correct. This hurts dividend fairness between validators, not miners. |
| V2 | Validator favours its own miners | R – Low | Standard Yuma consensus clipping limits single-validator influence; honest validators score independently with private random items. |
| V3 | Validators disagree because hardware gives slightly different numbers | L – **High until measured** | tau = 0.15 covers bfloat16 vs float32 on CPU with the 0.5B test model, but only just (max distance 0.143). Production uses bfloat16 on both sides and identical batching, and the code refuses silent CPU fallback, which removes the two largest known sources of drift. **The production model on production GPUs is not measured by this script's default tests.** Run `scripts/calibrate.py` on at least two GPU types first. |
| V3b | A validator runs lax or strict scoring parameters | M | Scoring constants are fixed by the release; overriding them aborts startup unless development mode is on, and development mode is refused on finney. |
| V4 | Hostile validator floods a miner | M | Per-validator limits: 2 concurrent requests, 20 per minute, 256 KiB bodies, global queue of 4× the concurrency limit. Inference is one batch at a time per process, so a permitted validator can still add queueing delay within those limits (review M4, accepted). |
| V4b | Anyone holds miner connections open (slow bodies, slow headers, many connections) | R – Medium, L | Bodies must arrive within 5 s; at most 8 unfinished bodies per remote IP (global cap 4096, high on purpose so it cannot be used to lock out validators); rejected and idle connections are closed immediately. aiohttp has no header timeout and an attacker with many IPs can still exhaust file descriptors: like any public HTTP service, production miners should sit behind a reverse proxy with header/body timeouts and connection limits, and raise the open-file limit (MINER.md). |
| V5 | Non-validator uses miners as free inference | M | Only hotkeys with a validator permit and ≥ `REAX_MIN_VALIDATOR_STAKE` stake are served; the check runs before signature verification, so junk traffic is cheap to refuse. |
| V6 | Miner-supplied IP used to attack the validator's network (SSRF) | M | Validators only contact public unicast IPs (`ip_allowed`); link-local/metadata addresses are refused even in local test mode; redirects are not followed; proxies from the environment are ignored. |
| V7 | Miner exhausts validator memory or time | M | 64 KiB cap while streaming, total timeout per request, bounded parallelism, one connection per request (`force_close`). |
| V8 | Validators expose an attack surface | M | The validator opens no port. |

## Operational risks

| # | Risk | Status | How it is handled |
|---|---|---|---|
| O1 | Hot keys or secrets in the repository or logs | M | Servers only need the hotkey; the e2e test runs every miner and validator with hotkey-only wallet copies. `.gitignore`/`.dockerignore` exclude wallets, env files and state. Logs print only public ss58 addresses and RPC URLs without credentials; the validator state file holds only scores. |
| O2 | Supply chain | M/L | Exact dependency pins in `pyproject.toml`; the Docker image installs from a hash-locked file (`requirements/lock.txt`, CPython 3.12 x86_64) without build isolation; base image pinned by digest; model pinned to a Git commit of the Hugging Face repo; weights loaded from safetensors only (no pickle); `trust_remote_code` is never enabled. GitHub Actions are pinned by commit SHA. Runtime sidecar images (1.1): base images pinned by digest; the s1-fast sidecar pins its top-level packages, bounds every PyPI resolution by upload date (`--uploaded-prior-to`) and checks the jevk5 source tarball by SHA-256, but its transitive dependencies are not hash-locked (residual, Medium; a hash lock compiled against the pinned base image is the fix). Published images (GHCR) are built, smoke-tested and only then pushed; tags are mutable, so deployments pin images by digest (`.github/workflows/docker.yml`). |
| O3 | Version mismatch between miners and validators | M/L | Miners reject any protocol except `reax-mpm/1`; validators send `WEIGHTS_VERSION_KEY`; the owner raises `weights_version` after each scoring release so outdated validators are refused by the chain. A protocol change requires a coordinated window (validators first). |
| O4 | Chain RPC failures | M | Metagraph failures and suspiciously short metagraphs skip the round without changing scores; `set_weights` failures are logged and retried next round; a hanging submission is abandoned after 300 s while scoring continues; the miner keeps its last validator list and keeps serving if publishing its endpoint fails. |
| O5 | Crash or restart | M | Validator state is written atomically after every round; a corrupt or foreign state file is moved aside and scoring restarts from zero. The e2e run kills a validator with SIGKILL and verifies it resumes. |
| O6 | Clock drift | L | Signed requests must be younger than 10 s (2 s future skew). Run NTP on every miner and validator. |
| O7 | Mainnet MEV shield | L | SDK 11 sends registration through the MEV shield. On the fast-block local chain that path failed and the e2e uses the raw call; on finney use the normal `btcli`/SDK path and verify. |
| O8 | Owner key compromise | L | Owner coldkey in cold storage; hyperparameter changes are rare. |

## Out of scope in 1.0 (switched off)

GDPR/EU admission, third-party traffic routing, images, capacity audits, multiple models, score/noul question
types and labelled items. See [FEATURE-FLAGS.md](FEATURE-FLAGS.md). Each one adds an attack surface that is
not yet implemented and reviewed to production quality.
