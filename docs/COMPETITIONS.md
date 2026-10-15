# Competitions: how the miner weights are shared (release 1.2)

Besides serving, the subnet pays for two kinds of competition. This page is the single description of the split. The
code is `src/reax_subnet/competition_split.py`; every number here is a **release constant**, not an operator setting.

## Start time

`COMPETITIONS_START_UTC` = **2026-10-16 12:00 UTC**. Before that instant a 1.2 validator sets exactly the 1.1 weights and
runs no engine evaluation. From it on the competitions below are active. (Validators use their own UTC clock; a few
seconds of difference is irrelevant for a 14-day rotation.)

## Profiles

| Profile | Inference (serving) | Engine champion | Use-case tracks |
|---|---|---|---|
| **`50/50` (active, `SPLIT_PROFILE`; approved by the subnet owner on 15 Oct 2026)** | 0.50 | 0.20 | 0.15 + 0.15: **two** active tracks |
| `80/20` (ships ready, inactive) | 0.80 | 0.10 | 0.10 in total: exactly **one** active track |

The shares are fractions of the miner part of the weights; the owner's burn fraction is unchanged. Changing the profile
needs the subnet owner's approval and a new release: it is not configurable on finney, so all validators agree.

## What goes back to inference

Unclaimed or inactive competition share always goes back to the **inference** (serving) part, in proportion to the
serving weights. It is never burned and never paid to the owner UID. This covers: no active track, a track with no valid
result, a missing rank, no engine champion, a struck or withdrawn champion, and the decayed part of the champion's share.
With nothing to pay the weights are exactly the 1.1 weights. The weights always sum to the same total as the serving
weights.

## Engine competition

Competitors submit engine images; validators measure them ([ENGINE-COMPETITION.md](ENGINE-COMPETITION.md)). The
champion gets the engine share at once; it **halves linearly over 14 days** (100 % when crowned, 75 % after 7 days, 50 %
from day 14 on), and the decayed part goes to inference. A new champion starts at 100 % again.

## Use-case tracks (specialist competitions)

A track is a use case (for example `demand_forecasting`) with a start and end time and an evaluator that scores
miners. Inside a track the share is **winner-take-most**: rank 1 gets 75 %, rank 2 17 %, rank 3 8 %; ranks that do not
exist go to inference. Ties are broken by the lower UID.

Tracks run for **14 days and rotate**. The schedule is a release constant (`TRACK_SCHEDULE`):

| Track | From | Until | Profile |
|---|---|---|---|
| `demand_forecasting` | 2026-10-16 12:00 UTC | 2026-10-30 12:00 UTC | both (the only track of `80/20`) |
| `risk_assessment` | 2026-10-16 12:00 UTC | 2026-10-30 12:00 UTC | `50/50` (the second slot) |

Only as many tracks as the profile has slots are paid. **The next pair is not in the schedule yet** (a TODO in
`competition_split.py`): after 30 Oct 12:00 UTC no track is active, and its share stays with inference, until a release
adds the next pair (rotation every 14 days). The schedule is the single source of truth: the specialist evaluator
(`reax_subnet.spec.config`) derives its tracks from it.

**Entry floor:** `TRACK_ENTRY_FLOOR` (release constant, per track; `None` = no floor). A submission whose score is at or
below its track's floor earns nothing from that track. Both floors are `None` until the baseline has been measured.

**The specialist evaluator** ([SPECIALIST-TRACKS.md](SPECIALIST-TRACKS.md)) is wired in: `SpecEvaluator` results
(`{track_id: [(uid, score), ...]}`, best first) go through `SpecialistAdapter` into `combine_weights` (ranks 75/17/8). The
adapter hands out the ranked leaders of the last cycle and re-reads them from the evaluator's saved state every round, so
a restart does not interrupt the payout.

## One GPU at a time

Engine measurements and specialist evaluations never run at the same time (`competitions.CompetitionDriver` holds one
GPU lock). They run as a **background task**, so serving rounds are not paused: a round never waits for a measurement.
Each keeps its own free-memory check before a cycle (engine: 2 x 14 GB; specialist: up to 30 GB) and skips the cycle
cleanly when the card has less free, recording nothing against any submission.

## A validator without the competition setup

On finney the competitions are fixed on, but they need the host side: the sealed pools (`REAX_SPEC_SEALED_DIR`), the
engine weights (`REAX_ENGINE_WEIGHTS_DIR`) and the published reference engine image (the release constant
`config.ENGINE_REFERENCE_IMAGE`; `REAX_ENGINE_REFERENCE` is a development-network override only). If one is missing the
validator does **not** stop: it logs one warning and serves only, which means it sets 100 % inference weights.
Such a validator disagrees with the others on the competition share, so operators should set it up
([deploy/validator-1.2/README.md](../deploy/validator-1.2/README.md)).

## Where the weights are computed

`S1Validator.compute_weights` builds the 1.1 serving weights, then `EngineCompetition.weights` calls
`competition_split.combine_weights(serving, engine_champion, engine_share_frac, tracks, profile)`.
