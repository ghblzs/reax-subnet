# Competitions: how the miner weights are shared (release 1.2)

Besides serving, the subnet pays for two kinds of competition. This page is the single description of the split. The
code is `src/reax_subnet/competition_split.py`; every number here is a **release constant**, not an operator setting.

## Start time

`COMPETITIONS_START_UTC` = **2026-10-16 12:00 UTC**. Before that instant a 1.2 validator sets exactly the 1.1 weights and
runs no engine evaluation. From it on the use-case tracks below are active. (Validators use their own UTC clock; a few
seconds of difference is irrelevant for a 14-day rotation.)

**The engine track has its own start and is held.** `ENGINE_TRACK_START_UTC` is `None` in release 1.2.0rc5: on finney no
engine evaluation runs (neither in the validator nor in `reax-competition-worker`), `reax-engine-submit` publishes
nothing, and a validator pays no engine champion, even if a state file names one. The engine share goes to inference.
The subnet owner opens the engine track in a later release by setting a start time.

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

## Owner approval

A winning entry is paid only after the subnet owner has checked and approved it. This holds for the engine champion and
for every paid rank of a use-case track (ranks 1, 2 and 3). Until the approval exists, that entry's share goes to
inference (it is not burned, and the next rank does **not** move up), and the validator logs `awaiting owner approval`.

An approval covers exactly one entry: the track, the hotkey and the submission (the image digest `sha256:<64 hex>` for
the engine, `<hf repo>@<40 hex revision>` for a use-case track). A new image, a new model revision or another hotkey
needs a new approval. An approval may carry an expiry date.

How it works for validators:

* The owner signs approvals with an Ed25519 key that never leaves the owner's machine (`reax-award-approve keygen`,
  `reax-award-approve sign --key <file> --track <id> --hotkey <ss58> --submission <id> --approvals approvals.json`). The
  owner publishes the signed `approvals.json`; a signature cannot be forged or changed, so any channel can carry it.
* The validator reads the file named by `REAX_AWARD_APPROVALS` (in `deploy/validator-1.2`: `/approvals/approvals.json`,
  host directory `REAX_AWARD_APPROVALS_DIR`, default `deploy/approvals`) and checks every entry against the owner's
  public key (release constant `award.AWARD_APPROVAL_PUBKEY`, or `REAX_AWARD_APPROVAL_PUBKEY` as 64 hex characters).
  The file is re-read when it changes; no restart is needed.
* Keep the approvals directory **out of** `REAX_COMPETITION_DIR` and never mount it into the competition worker: the
  worker must not be able to write approvals.
* Fail closed: no public key, no file, an unreadable file, a bad signature, another key or an expired approval means no
  award. On finney the check is always on; on development networks only when `REAX_AWARD_APPROVALS` is set.
* `reax-award-approve verify --approvals approvals.json` lists every entry and whether its signature is valid.

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
below its track's floor earns nothing from that track. The floor is the score of the s1-fast reference model (Plumb-4B) on the
full sealed pool plus 1.5 points, measured on the reference validator: **demand_forecasting 24.8** (baseline 23.28),
**risk_assessment 33.7** (baseline 32.19). A specialist has to beat the general model that is already served.

**The specialist evaluator** ([SPECIALIST-TRACKS.md](SPECIALIST-TRACKS.md)) is wired in: `SpecEvaluator` results
(`{track_id: [(uid, score), ...]}`, best first) go through `SpecialistAdapter` into `combine_weights` (ranks 75/17/8). The
adapter hands out the ranked leaders of the last cycle and re-reads them from the evaluator's saved state every round, so
a restart does not interrupt the payout.

## One GPU at a time

Engine measurements and specialist evaluations never run at the same time (`competitions.CompetitionDriver` holds one
GPU lock). They run as a **background task** (in the worker process in `remote` mode), so serving rounds are not paused:
a round never waits for a measurement. Serving and the measurement therefore share the card; the engine session measures
the reference alone first and treats later reference runs more than 12 % away from that as inconclusive.
Each keeps its own free-memory check before a cycle (engine: 2 x 12.5 GB; specialist: up to 25 GB) and skips the cycle
cleanly when the card has less free, recording nothing against any submission.

## Where the competitions run: `REAX_COMPETITIONS_MODE`

| Mode | What the validator does | Needs |
|---|---|---|
| `local` (default on development networks; **on finney only when set explicitly**) | runs the engine and specialist cycles inside its own process | the docker socket, the sealed pools, the engine weights, a GPU |
| `remote` | runs **no** cycle; each round it reads `engine-competition.json` and `spec-state.json` from `REAX_COMPETITIONS_STATE_DIR` (read-only) | only that directory |
| `off` | no competitions at all: 100 % inference (one warning on finney) | nothing |

**Why `remote`.** The competitions run code and models that miners chose, in containers on a GPU, under a root docker
daemon. The process that holds that socket is root on its machine. With `remote` the socket, the sealed data and the
GPU work live in a separate process, the **competition worker**, and the validator (which holds the hotkey) has none of
them. Since 1.2.0rc2 `deploy/validator-1.2` runs it that way.

**The worker** is `reax-competition-worker [--engine] [--spec]` (both when neither flag is given). It reads the chain
read-only, has no wallet and no token, never sets weights, and runs only the cycles, with the same GPU lock (engine and
specialist never at the same time), the same free-memory preflights and the same anti-cheat as `local` mode. It writes the
two state files atomically into `REAX_COMPETITIONS_STATE_DIR` and refreshes their `saved_at` every 5 minutes while it is
alive. It runs unchanged (a) as a container next to the validator without wallets or tokens
(`deploy/validator-1.2/compose.yml`) and (b) on a separate keyless GPU box
([deploy/competition-worker](../deploy/competition-worker/README.md)), from which the two files are copied to the
validator host.

**What the validator does with the files** (`reax_subnet/remote.py`): it pays exactly like `local` mode (engine champion
with the 14-day decay and the confirmations; track leaders with the entry floors and 75/17/8; a withdrawn or deregistered
competitor drops out to inference). On every read it checks against the chain snapshot that the champion's hotkey still
holds its uid and that its commitment still names the measured image or model. A file that is missing, unreadable, of
another version, or older than **26 hours** (its own `saved_at`) pays nothing: that competition's share goes to the
inference part, with **one** warning per outage (and one note when it recovers). The two files fail independently. The
validator never writes, moves or deletes them.

**Finney default.** On finney an unset `REAX_COMPETITIONS_MODE` means serving-only (100 % inference, one warning). If
the old in-process setup is detected (`REAX_ENGINE_WEIGHTS_DIR` or `REAX_SPEC_SEALED_DIR` set) the validator logs an
**error** and still does not run the competitions: a 1.2.0rc1 compose file with newer code must never put the docker
socket next to the hotkey by accident. Choose `remote` (recommended) or write `REAX_COMPETITIONS_MODE=local` yourself.

**What the heartbeat means.** The worker refreshes `saved_at` every 5 minutes whatever its cycles do, so "older than 26 hours
means inference" detects a **dead worker**, not stale results. A worker that is alive but whose cycles all fail (broken docker
daemon, revoked registry, no GPU) keeps its last champion paid: the champion still decays to the floor over 14 days and
still has to hold its uid and commitment on every read, but nothing new is evaluated. Watch the worker log
("competition cycle failed") and the age of the newest `results` entry, not only the state file's age.

**Trust boundary of the worker.** The worker parses miner-controlled input (Hugging Face metadata, safetensors headers,
registry manifests, engine and vLLM responses) **while it holds the docker socket**. A remote-code-execution bug in that
parsing is root on the host, including the hotkey, the router and the WireGuard keys; the container flags do not change that.
`remote` mode removes the hotkey from the *validator's* parser, not from the machine. The target setup is the worker on a
separate keyless GPU box ([deploy/competition-worker](../deploy/competition-worker/README.md)).

Variables: `REAX_COMPETITIONS_MODE`, `REAX_COMPETITIONS_STATE_DIR` (absolute; required for `remote`, and for the worker).
In `remote` mode the validator ignores `REAX_ENGINE_WEIGHTS_DIR`, `REAX_SPEC_SEALED_DIR` and the other worker variables
(one warning). On finney `REAX_FEATURE_ENGINE_COMPETITION` is refused in every mode; `remote` needs no flag on a
development network.

## A validator without the competition setup

On finney the competitions are fixed on, but they need the host side: the sealed pools (`REAX_SPEC_SEALED_DIR`), the
engine weights (`REAX_ENGINE_WEIGHTS_DIR`) and the published reference engine image (the release constant
`config.ENGINE_REFERENCE_IMAGE`; `REAX_ENGINE_REFERENCE` is a development-network override only). If one is missing the
validator does **not** stop: it logs one warning and serves only, which means it sets 100 % inference weights. (This is
`local` mode; in `remote` mode the validator needs none of these, the worker does, and refuses to start without them.)
Such a validator disagrees with the others on the competition share, so operators should set it up
([deploy/validator-1.2/README.md](../deploy/validator-1.2/README.md)).

## Where the weights are computed

`S1Validator.compute_weights` builds the 1.1 serving weights, then `EngineCompetition.weights` calls
`competition_split.combine_weights(serving, engine_champion, engine_share_frac, tracks, profile)`.
