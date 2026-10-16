# Sampling and admission policy of the REAX router

Status: implemented in the router branch p2p/a140-router-scale. Numbers are the recommended launch settings; every
number is an environment variable, listed in docs/ROUTER.md.

## 1. What is decided

The router sends customer requests to miners of the s1 pool. For each answer it decides whether to check it first against
our own reference runtime ("verify") or to serve it unverified. A miner answer that does not pass the check is never
returned; the reference answer is returned instead.

## 2. Policy in one table (recommended launch values)

| Rule | Setting | Value |
| --- | --- | --- |
| Eligibility for customer traffic | MIN_ROUNDS, MIN_EMA, last round, no strike | 720 rounds, EMA 0.9, passing, none |
| Probation | `REAX_ROUTER_PROBATION_ANSWERS` | first 20 customer answers of a hotkey always verified |
| Base sampling rate | `REAX_ROUTER_VERIFY_RATE` | 0.30 |
| Floor (verified share of served answers, rolling 1 h) | `REAX_ROUTER_VERIFY_FLOOR` | 0.25 |
| Longest unverified run | `REAX_ROUTER_MAX_UNVERIFIED_RUN` | ceil(1/0.25) = 4 |
| Asynchronous shadow check of unverified answers | `REAX_ROUTER_SHADOW_RATE` | 0.10 |
| Grey quarantine | `GREY_RATE_MAX`, `GREY_QUARANTINE_S` | Wilson 95 % lower bound > 0.10 with >= 20 checks; 3600 s |
| Strike (gross deviation) | validator thresholds | mean > 0.5 or any item > 2.0: 24 h exclusion |

Default when nothing is configured: `VERIFY_RATE=1.0`, floor = rate: every answer is verified and a missing reference slot
is a 503, as before this policy existed. Two defaults are new even then: the grey quarantine (`GREY_RATE_MAX` 0.10; the
old router only answered from the reference on a grey result) and the 5 s skip after a 429 (`BUSY_BACKOFF_S`). Set
`GREY_RATE_MAX=1` and `BUSY_BACKOFF_S=0` for the exact old behaviour.

## 3. Unpredictable selection

* Per request one draw from `secrets` (operating-system randomness, resolution 1 in 1,000,000). It does not depend on the
  content, the caller, the miner, the time, the request size, or earlier draws; the verification starts in parallel with the
  miner query, so the miner sees no difference in the request or in the timing of the request it receives.
* The forced share (probation, floor) depends only on the miner's own served history. A miner can compute it, but cannot
  use it: an answer it serves is checked whenever its budget is short, and an answer it does not serve cannot hurt anyone.
* Probation is per hotkey and in memory. After a router restart every miner is on probation again (fails safe).
* Floor and run cap are counted on served answers over 1 h, not on arriving requests, so reference saturation (customer
  load, validator rounds on a shared GPU, s1-pro load) cannot lower them. Without a free reference slot a forced check
  skips the miner (next candidate) or answers 503; it never serves unverified. Models have separate reference slots.
  Only the sampled share above the floor (empty by default) is skipped without a slot, and it leaves one slot free for
  forced checks.

## 4. Grey zone, quarantine, strike

Honest reference-vs-reference runs differ: mean distance 0.061 on average, maximum 0.34, against tau_mean 0.20, tau_strike
0.50, tau_item 2.0. An honest miner therefore lands in the grey zone (above 0.20, below 0.50) now and then.

* One grey: the reference answer is served; nothing else happens (no quarantine, no probation reset).
* Per hotkey the router counts, over 24 h, verification checks (synchronous and shadow) and greys. The hotkey is
  quarantined (no routing for 3600 s) only when the Wilson 95 % lower bound of its grey rate exceeds 0.10 and at least 20
  checks exist. 5 greys in 20 checks, 7 in 30, 10 in 50, 16 in 100 trigger it; 4 in 20 does not. An honest miner with a grey
  rate of a few percent practically never reaches it.
* A strike (mean > 0.5 or an item > 2.0, in a synchronous check or in the shadow check) excludes the hotkey for 24 h, resets
  probation and is written to the strike ledger. Router strikes are not read by the validator; they only exclude here.
* Requalification: after a quarantine or a strike the hotkey is on probation again (20 verified answers) and its grey
  statistics start from zero. A hotkey that changes registration needs the full 720 scored rounds again.

## 5. Bad answers served before exclusion (recommended settings)

Model: a miner honest through probation, then wrong on every answer; rate 0.30, floor 0.25, run cap 4, shadow 0.10
(simulation of the rules above, 40,000 runs; formula for the strike-level case: 0.7 x (1 + 0.63 + 0.63^2 + 0.63^3) = 1.6).

| Case | Mean bad answers served | 95th percentile | Hard maximum |
| --- | --- | --- | --- |
| Strike-level wrong answers, free reference | 1.6 | 4 | 4 |
| Strike-level wrong answers, reference saturated (no check fits) | up to 4, then 503 / skip | 4 | 4 |
| Grey-level answers on every answer | 4 | 4 | 4, then every answer is checked until the 20th check quarantines it |
| Miner wrong on one decision of a multi-decision request | requests with more decisions than the model's slots never reach a miner | | |

The bound of 4 is per episode: it restarts after every strike or quarantine (the miner requalifies through probation),
so a persistent cheater can serve up to 4 bad answers per episode (strike = 24 h out, so at most about 4 per day for
strike-level cheating). A miner whose grey-level answers stay below about 10 % of its checked answers is never quarantined
(that is the honest-variance allowance); it is not a guarantee that no degraded answer is ever served. Within an episode
the bound of 4 holds for every mix of load, history and draws; the means assume the draws are independent. A miner that
cheats on a small share f of its answers is checked just as often, so it needs about 1/f times as many answers for the same
number of bad ones; the run cap and the checks do not depend on f. Each bad answer is one customer answer that the pool's
quality bar (tau thresholds) would reject; none is served on probation.

Cost: at rate 0.30 about 30 % of served answers (more while the budget refills) and 20 answers per miner after a restart
or requalification use the reference GPU; the shadow adds 10 % of the unverified answers, one decision each.

## 6. Privacy

* Verification is a re-run of the same request on our own reference GPU, inside the same processing scope as the router. No
  customer content goes to any other party for verification, sampling or shadow checks.
* Nothing is stored: the router keeps payloads in memory only while a request runs. Logs, `/status`, `/metrics` and the
  strike ledger contain only router-generated request ids, UID, hotkey, model, outcome and counts, never state, questions,
  answers, probabilities or caller identifiers.
* The miner that answers does see the request content (see docs/ROUTER.md, Privacy and operations); sampling does not change
  what any miner sees.

## 7. Validator scoring traffic and customer traffic are separate

* The validator scores miners with its own rounds (its own item generator, its own hotkey signature). Customer requests
  are routed by this separate service and never feed validator scores. Router strikes and quarantines do not change weights.
* Per miner the router uses at most one concurrent request and 12 per minute (limits are 2 and 20 for the shared caller),
  and backs off 5 s after a 429, so the validator's rounds keep their share of the miner's limit.
* Customer answers are selected only from miners that the validator has scored for a day with a high average; the router
  can only exclude additionally.
* The reference GPU is shared with validator scoring today; the router's concurrency is capped per model, and a customer
  check never queues. Raising the router's slots above 2 requires its own reference runtime.

## 8. Today and later

Today (this branch): in-memory probation and statistics, 1 h credit window, strike ledger on disk, quarantine by
statistics, per-model slots, validator and router independent.

Later (not built): persist probation and grey statistics across restarts; feed router strikes and quarantines into validator
scoring; a separate reference runtime for the router; a longer exclusion for repeated strikes; confidence-based requalification
instead of a fixed 20 answers; a hidden-family check that does not rely on public templates.
