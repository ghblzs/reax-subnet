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
| Which eligible miner is asked | random, weighted by remaining per-minute budget (`secrets`) | EMA does not order candidates |
| Probation | `REAX_ROUTER_PROBATION_ANSWERS` | first 20 customer answers of a hotkey always verified |
| Base sampling rate | `REAX_ROUTER_VERIFY_RATE` | 0.5 |
| Floor (verified share of served answers, rolling 1 h) | `REAX_ROUTER_VERIFY_FLOOR` | 0.3 |
| Longest unverified run | `REAX_ROUTER_MAX_UNVERIFIED_RUN` | ceil(1/0.3) = 4 |
| Asynchronous shadow check of unverified answers | `REAX_ROUTER_SHADOW_RATE` | 0.10 (uses a reference slot, skipped and counted when none is free) |
| Grey quarantine | `GREY_RATE_MAX`, `GREY_QUARANTINE_S` | Wilson 95 % lower bound > 0.10 with >= 20 checks; 3600 s |
| Strike (gross deviation) | per item | item distance > tau_item (2.0); 24 h exclusion |
| Repeat strike | `REAX_ROUTER_REPEAT_STRIKE_EXCLUSION_S` | a strike within 7 days after the end of a previous exclusion: 7 days out (stays 7 days while it repeats) |
| Reference slots | `REFERENCE_CONCURRENCY`, `_S1_PRO` | 2 and 2, never shared between models |
| After a 429 | `REAX_ROUTER_BUSY_BACKOFF_S` | 5 s skip, no failure, no budget used |

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

The validator scores a round of 8 items: it strikes on a round mean above tau_strike (0.50) or any item above tau_item
(2.0), and a mean above tau_mean (0.20) scores the round 0 without a strike. A routed request carries one decision, so the
mean line would be a single-item line of 0.5. Honest per-item drift between cards exceeds 0.5 now and then (calibration
maxima 0.5 to 2.0; reference-vs-reference means average 0.061), so the router judges each **item**: above tau_item (2.0) is
a strike, above tau_mean (0.20) up to tau_item is grey, at most tau_mean is a match. This is the validator's own per-item
line; the round-mean line is not applied to single items. For a request with two decisions the worst item decides.

* One grey: the reference answer is served; nothing else happens (no quarantine, no probation reset).
* Per hotkey the router counts, over 24 h, verification checks (synchronous and shadow) and greys. The hotkey is
  quarantined (no routing for 3600 s) only when the Wilson 95 % lower bound of its grey rate exceeds 0.10 and at least 20
  checks exist. 5 greys in 20 checks, 7 in 30, 10 in 50, 16 in 100 trigger it; 4 in 20 does not. An honest miner with a grey
  rate of a few percent practically never reaches it (about 0.2 % within 2,000 checks at 2 % honest grey rate).
* A strike (an item above 2.0, in a synchronous check or in the shadow check) excludes the hotkey for 24 h, resets
  probation and is written to the strike ledger. A **strike within 7 days after the previous exclusion ended** excludes the hotkey for 7 days (so 1 d, 7 d, 7 d, ...; a quiet week resets it)
  (`REAX_ROUTER_REPEAT_STRIKE_EXCLUSION_S`); the exclusion end is written into the ledger entry and honoured after a restart.
  Router strikes are not read by the validator; they only exclude here.
* Requalification: after a quarantine or a strike the hotkey is on probation again (20 verified answers) and its grey
  statistics start from zero. A hotkey that changes registration needs the full 720 scored rounds again.
* Known limit: probation, credit and grey statistics are keyed by hotkey only. A miner that changes its committed role or
  artifact without a router restart keeps its probation state (the validator makes it earn 720 rounds on the new role first,
  but the router does not notice the change). A restart, a strike or a quarantine starts probation again.

## 5. Bad answers served before exclusion (live settings)

Model: a miner honest through probation, then wrong at strike level on every answer; rate 0.5, floor 0.3, run cap 4, shadow
0.10 (simulation of the rules above; the means equal the closed form q x (1 + c + c^2 + c^3) with q the share not checked and
c the chance that an unverified bad answer also escapes the shadow). Exposure starts after the 20 probation answers; no
probation answer is served unverified.

| Case | Mean bad answers served | P(maximum of 4) | Hard maximum |
| --- | --- | --- | --- |
| Reference free: drawn share 0.5 verifies, shadow 0.1 | 0.87 | 4.6 % | 4 |
| One slot taken by another request (above-floor share skipped, so only the floor share 0.3 verifies) | 1.59 | 17.5 % | 4 |
| Reference saturated (all slots taken: no above-floor check, and the shadow, which needs a slot, is skipped) | 4 | 100 % | 4, then 503 / skip until a check fits |
| Grey-level answers on every answer | 4 | | 4, then every answer is checked until the 20th check quarantines it |

The hard maximum of 4 is per episode and holds for a miner that is wrong on **every** answer (f = 1). An episode restarts
after the 24 h strike plus 20 probation answers; a strike within 7 days of the previous exclusion's end costs 7 days, so a persistent strike-level
cheater serves at most about 4 bad answers per 8 days after the second episode. A miner that is wrong on only a share f of its
answers resets its run with every honest verified answer, so the bound of 4 does not apply to it: its bad answers before the
first catch are geometric (mean about 0.9 at a free reference, about 1.6 when only the floor share verifies; 10 or more
happen in about 0.02 % and 0.2 to 0.4 % of episodes, simulated for f = 0.1 to 0.5; an independent analytic estimate gives up to 0.1 % and 1.6 %). A miner with an oracle for "this query is unverified" could keep up to 70 %
of its served answers bad; only the shadow stops it. For s1-fast that is not reachable in the gateway topology (capacity 2,
one-decision requests, 2 slots; the shadow never takes the last free slot, so a forced check finds one); keep the gateway's capacity at or below the slots and
alert on `verify reference_unavailable` and `shadow skipped_no_slot`. A miner whose grey-level answers stay below about 10 %
of its checked answers is never quarantined (the honest-variance allowance); that is not a guarantee that no degraded
answer is ever served.

Cost: at rate 0.5 about 50 % of served answers (30 % at least) and 20 answers per miner after a restart
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
