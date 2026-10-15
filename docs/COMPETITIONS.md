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
| **`80/20` (active, `SPLIT_PROFILE`)** | 0.80 | 0.10 | 0.10 in total: exactly **one** active track |
| `50/50` (ships ready, inactive) | 0.50 | 0.20 | 0.15 + 0.15: **two** active tracks |

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
| `risk_assessment` | 2026-10-16 12:00 UTC | 2026-10-30 12:00 UTC | only used by `50/50` |

Only as many tracks as the profile has slots are paid (one in `80/20`). Later rotations are added to the schedule with
each release.

**Plug-in:** the evaluator is a `SpecialistEvaluator` (`track_results(track_ids, now) -> {track_id: [(uid, score), ...]}`,
best score wins). The default evaluates nothing, so until an evaluator is wired in, every track share stays with inference.
`EngineCompetition(..., specialist=...)` takes the evaluator; nothing else needs to change.

## Where the weights are computed

`S1Validator.compute_weights` builds the 1.1 serving weights, then `EngineCompetition.weights` calls
`competition_split.combine_weights(serving, engine_champion, engine_share_frac, tracks, profile)`.
