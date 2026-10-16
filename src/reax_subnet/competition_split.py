"""Competition share layer (docs/COMPETITIONS.md): how the miner weights are split between serving and competitions.

Pure and deterministic. Release constants only; nothing here is operator-configurable on finney.

    inference (serving)  +  engine champion  +  use-case (specialist) tracks

Profiles (``SPLIT_PROFILE`` is the active one):

* ``50/50`` (ACTIVE since the owner approved it on 15 Oct 2026): inference 0.50, engine 0.20, TWO active tracks
  0.15 + 0.15.
* ``80/20`` (inactive, kept ready): inference 0.80, engine 0.10, ONE active use-case track 0.10.

Rule that never changes: a competition share that is unclaimed or inactive (no active track, no champion, a struck
champion, the decayed part of the champion, a missing track rank) goes back to the inference part. It is never
burned and never paid to the owner UID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional, Protocol, Sequence

SPLIT_PROFILE = "50/50"
# Before this instant the weights are exactly the 1.1 weights; from it on the competitions are active.
COMPETITIONS_START_UTC = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
ENGINE_DECAY_DAYS = 14  # the champion's share falls linearly from full to ENGINE_DECAY_FLOOR over this time
ENGINE_DECAY_FLOOR = 0.5
TRACK_RANK_SPLIT = (0.75, 0.17, 0.08)  # winner-take-most inside a use-case track, ranks 1-3
ROTATION_DAYS = 14


@dataclass(frozen=True)
class SplitProfile:
    name: str
    engine: float  # fraction of the miner part reserved for the engine champion
    tracks: tuple[float, ...]  # one entry per active use-case track slot

    @property
    def competition(self) -> float:
        return self.engine + math.fsum(self.tracks)

    @property
    def inference(self) -> float:
        return 1.0 - self.competition


PROFILES: dict[str, SplitProfile] = {
    "80/20": SplitProfile("80/20", engine=0.10, tracks=(0.10,)),
    "50/50": SplitProfile("50/50", engine=0.20, tracks=(0.15, 0.15)),
}


@dataclass(frozen=True)
class TrackSchedule:
    track_id: str
    start: datetime
    end: datetime  # exclusive

    def active(self, now: datetime) -> bool:
        return self.start <= now < self.end


_PAIR_1_END = COMPETITIONS_START_UTC + timedelta(days=ROTATION_DAYS)  # 2026-10-30T12:00Z

# Release constant and the ONE source of truth for which use-case tracks run when (the specialist evaluator,
# reax_subnet.spec.config, derives its schedule from this list). First entry = the only track of the 80/20 profile.
# TODO(next release): add the next pair for 2026-10-30T12:00Z -> 2026-11-13T12:00Z (rotation every ROTATION_DAYS days);
# until an entry exists, no track is active after the first pair and its share goes to the inference part.
TRACK_SCHEDULE: tuple[TrackSchedule, ...] = (
    TrackSchedule("demand_forecasting", COMPETITIONS_START_UTC, _PAIR_1_END),
    TrackSchedule("risk_assessment", COMPETITIONS_START_UTC, _PAIR_1_END),
)

# Release constant, per track: submissions scoring at or below the floor earn nothing from the track; None = no floor.
# The lead sets numbers after measuring the s1-fast / Plumb-4B baseline on each track's dev split.
# Entry floor = the s1-fast reference model's (Plumb-4B) score on the full sealed pool + 1.5 points, measured on the
# reference validator (RTX PRO 6000, release 1.2.0rc2 evaluator): demand_forecasting 23.28, risk_assessment 32.19.
# A specialist must beat the general model that is already served; at or below the floor a submission is not ranked.
TRACK_ENTRY_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 24.8, "risk_assessment": 33.7}



class RankedLeaders(list):
    """``[(uid, score), ...]`` already in rank order, rank 1 first (docs/SPECIALIST-TRACKS.md, "Ranks"): the specialist
    evaluator's champion, then its former champions, most recently crowned first. ``track_payout`` pays it in this
    order and never re-sorts it by score; the scores come from different draws and are not comparable."""


# Per track: [(uid, score)]. A ``RankedLeaders`` (what the specialist evaluator returns) is paid in its order; any
# other list or mapping is ranked by score, best first.
TrackResults = dict[str, list[tuple[int, float]]]


class SpecialistEvaluator(Protocol):
    """Plug-in point for the use-case tracks. ``track_results`` must be cheap (read the evaluator's own state)."""

    def track_results(self, track_ids: Sequence[str], now: datetime) -> TrackResults: ...


class NoSpecialistEvaluator:
    """Default: no track has results, so every track share goes to the inference part."""

    def track_results(self, track_ids: Sequence[str], now: datetime) -> TrackResults:
        return {}


def competitions_active(now: datetime) -> bool:
    return now >= COMPETITIONS_START_UTC


def profile_for(name: str = SPLIT_PROFILE) -> SplitProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown split profile {name!r}; choose from {', '.join(PROFILES)}") from None


def active_tracks(now: datetime, profile: str = SPLIT_PROFILE,
                  schedule: Sequence[TrackSchedule] = TRACK_SCHEDULE) -> list[str]:
    """Track ids that are active at ``now``, in schedule order, at most one per slot of the profile."""
    slots = len(profile_for(profile).tracks)
    return [s.track_id for s in schedule if s.active(now)][:slots]


def engine_decay_fraction(age_s: float) -> float:
    """The champion's part of the engine share: 1.0 when crowned, linear to 0.5 after 14 days, then flat."""
    frac = min(1.0, max(0.0, age_s) / (ENGINE_DECAY_DAYS * 86400.0))
    return 1.0 - (1.0 - ENGINE_DECAY_FLOOR) * frac


def track_payout(scores: Mapping[int, float] | Sequence[tuple[int, float]],
                 floor: Optional[float] = None) -> dict[int, float]:
    """75/17/8 for ranks 1-3. A ``RankedLeaders`` keeps its order (a uid counts at its first place); anything else is
    ranked best score first, ties by lower uid. Missing ranks are NOT redistributed: the returned fractions sum to at
    most 1 and the caller sends the rest to the inference part. A score at or below ``floor`` (``TRACK_ENTRY_FLOOR``;
    None = no floor), or not a positive finite number, is not ranked; the entries after it move up one rank."""
    in_order = isinstance(scores, RankedLeaders)
    items = scores.items() if isinstance(scores, Mapping) else scores
    best: dict[int, float] = {}
    for uid, s in items:
        if not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 or (floor is not None and s <= floor):
            continue
        if uid in best and (in_order or s <= best[uid]):
            continue
        best[uid] = float(s)
    ranked = list(best) if in_order else sorted(best, key=lambda u: (-best[u], u))
    return {uid: share for uid, share in zip(ranked[: len(TRACK_RANK_SPLIT)], TRACK_RANK_SPLIT)}


def combine_weights(serving: Mapping[int, float], engine_champion: Optional[int], engine_share_frac: float,
                    tracks: Mapping[str, Mapping[int, float] | Sequence[tuple[int, float]]],
                    profile: str = SPLIT_PROFILE, *, owner_uid: Optional[int] = None,
                    burn_fraction: float = 0.0,
                    floors: Mapping[str, Optional[float]] = TRACK_ENTRY_FLOOR) -> dict[int, float]:
    """Mix the competitions into the serving weights.

    ``serving``: the 1.1 weight vector (may contain the owner's burn entry). ``engine_champion``: uid or None;
    ``engine_share_frac``: the champion's part of the engine share (1.0 .. 0.5, see ``engine_decay_fraction``).
    ``tracks``: scores per ACTIVE track (only the first ``len(profile.tracks)`` track ids in sorted order are paid).

    The competition fractions apply to the miner part (the owner's burn stays as it is, as in 1.1). Everything not
    claimed stays with ``serving``. The result is normalised to sum 1; an empty ``serving`` returns ``{}``.
    """
    prof = profile_for(profile)
    base_total = math.fsum(w for w in serving.values() if w > 0)
    if base_total <= 0:
        return {}
    base = {uid: w / base_total for uid, w in serving.items() if w > 0}
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn

    paid: dict[int, float] = {}  # fractions of the miner part
    if engine_champion is not None and engine_share_frac > 0:
        paid[engine_champion] = prof.engine * min(1.0, engine_share_frac)
    for slot_share, track_id in zip(prof.tracks, sorted(tracks)):
        for uid, share in track_payout(tracks[track_id], floors.get(track_id)).items():
            paid[uid] = paid.get(uid, 0.0) + slot_share * share
    paid.pop(owner_uid, None)  # never to the owner UID: its share stays with the inference part
    claimed = math.fsum(paid.values())
    if claimed <= 0:
        return dict(sorted(base.items()))

    keep = 1.0 - claimed
    out: dict[int, float] = {}
    for uid, w in base.items():
        out[uid] = burn + (w - burn) * keep if uid == owner_uid else w * keep
    for uid, f in paid.items():
        out[uid] = out.get(uid, 0.0) + f * miner_part
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items())}
