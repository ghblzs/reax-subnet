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
from typing import Callable, Mapping, Optional, Protocol, Sequence

SPLIT_PROFILE = "50/50"
# Before this instant the weights are exactly the 1.1 weights; from it on the competitions are active.
COMPETITIONS_START_UTC = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
# The use-case (specialist) tracks start with the competitions.
SPECIALIST_TRACKS_START_UTC = COMPETITIONS_START_UTC
# The engine track has its own start. None = HELD: no engine cycle runs, no engine submission is accepted and the engine
# share goes to the inference part, whatever a state file says. The owner opens it in a later release.
ENGINE_TRACK_START_UTC: Optional[datetime] = None
# Stands for "never" where a datetime is needed (a held engine track).
ENGINE_HELD = datetime.max.replace(tzinfo=timezone.utc)
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
# Entry floor = the s1-fast reference model's (Plumb-4B) score + 1.5 points. 1.2.0rc6 (draws of 300 items, 15 % proc,
# owner decision 16 Oct 2026): measured with the validator's evaluator (reax-spec-eval, reference validator RTX PRO 6000)
# on six consecutive 300-item draws of the release sampler from the deployed pools (writer-note rows removed) and the
# live use counts of 16 Oct 14:05Z: demand_forecasting 38.17, risk_assessment 31.66 (1800 items each). 1.2.0rc5 (600
# items, 50 % proc, full pool): 23.28 / 32.19 -> 24.8 / 33.7.
# A specialist must beat the general model that is already served; at or below the floor a submission is not ranked.
TRACK_ENTRY_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 39.7, "risk_assessment": 33.2}

# Participation keep-alive (docs/SPECIALIST-TRACKS.md, "Keep-alive"), release constant: this fraction of each active
# track's share is split evenly among the track's keep-alive entries (copy-checked submissions, still committed, that
# scored above the entry floor on a real draw; at most 2 per coldkey), BEFORE the ranks are paid from the rest. A
# competition hotkey cannot serve inference, so without it an entrant earns 0 while it is evaluated, waits for its
# confirmation or for the owner's approval, and is deregistered once its immunity ends. Not an award: the owner-approval
# gate does not apply. No entry: nothing is reserved and the weights are exactly the rc5 weights. 0 = off.
KEEPALIVE_SHARE = 0.05

# Keep-alive non-proc floor (1.2.0rc6), release constant: a keep-alive entry must also score above this on the NON-proc
# items of its latest draw (same_family, new_domain, new_family; tier != "proc_sealed"), so learning the public
# procedural families alone never earns it. Value: the reference model's (Plumb-4B = s1-fast) score on the non-proc items
# of the same six release draws as TRACK_ENTRY_FLOOR + 1.5 points: demand_forecasting 48.34, risk_assessment 36.27
# (1530 items each).
KEEPALIVE_NONPROC_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 49.8, "risk_assessment": 37.8}

# Keep-alive legacy floor (1.2.0rc7), release constant: an entry whose latest score was taken before 1.2.0rc6 has no
# non-proc score (``Submission.nonproc`` None and no per-tier scores); it qualifies when that total score is above this
# instead. Value: the score of a model that is perfect on the procedural items and uniform elsewhere under the rc5 mix
# (50 % proc, 600 items, the deployed pools: demand_forecasting 49.8, risk_assessment 31.8, workers/leak-audit), i.e. the
# best a rules-only model reaches without any skill on the non-proc tiers, plus a margin (10.2 / 18.2 points).
KEEPALIVE_LEGACY_FLOOR: dict[str, Optional[float]] = {"demand_forecasting": 60.0, "risk_assessment": 50.0}



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


def release_engine_start() -> datetime:
    """The engine track's start on finney: ``ENGINE_HELD`` while ``ENGINE_TRACK_START_UTC`` is None."""
    return ENGINE_HELD if ENGINE_TRACK_START_UTC is None else ENGINE_TRACK_START_UTC


def engine_track_open(now: datetime) -> bool:
    return now >= release_engine_start()


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


def keepalive_uids(entries: Sequence[tuple[int, float]], floor: Optional[float] = None,
                   owner_uid: Optional[int] = None) -> list[int]:
    """The uids of a track's keep-alive entries ``[(uid, score), ...]`` (the specialist evaluator's ``keepalive``) that
    get a slot: a positive finite score above ``floor``, never the owner UID, each uid once, in the given order."""
    out: list[int] = []
    for uid, s in entries:
        if not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 or (floor is not None and s <= floor):
            continue
        if uid == owner_uid or uid in out:
            continue
        out.append(uid)
    return out


def combine_weights(serving: Mapping[int, float], engine_champion: Optional[int], engine_share_frac: float,
                    tracks: Mapping[str, Mapping[int, float] | Sequence[tuple[int, float]]],
                    profile: str = SPLIT_PROFILE, *, owner_uid: Optional[int] = None,
                    burn_fraction: float = 0.0,
                    floors: Mapping[str, Optional[float]] = TRACK_ENTRY_FLOOR,
                    award_ok: Optional[Callable[[str, int], bool]] = None,
                    keepalive: Optional[Mapping[str, Sequence[tuple[int, float]]]] = None,
                    keepalive_share: Optional[float] = None) -> dict[int, float]:
    """Mix the competitions into the serving weights.

    ``serving``: the 1.1 weight vector (may contain the owner's burn entry). ``engine_champion``: uid or None;
    ``engine_share_frac``: the champion's part of the engine share (1.0 .. 0.5, see ``engine_decay_fraction``).
    ``tracks``: scores per ACTIVE track (only the first ``len(profile.tracks)`` track ids in sorted order are paid).

    ``award_ok(track_id, uid)``: the owner-approval gate (award.py), asked AFTER the ranking: an entry without approval
    keeps its rank, and its share stays with the inference part (the next rank is not promoted). None = no gate.

    ``keepalive``: per active track, the keep-alive entries ``[(uid, score), ...]`` (``keepalive_uids``). When a track
    has any, ``keepalive_share`` (default ``KEEPALIVE_SHARE``) of its share is split evenly among them, without the
    approval gate, and the ranks are paid from the rest. A track without entries pays exactly as before. An entry the
    track's ranks pay and ``award_ok`` approves is paid by its rank only: it gets no keep-alive slot.

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
    ka_share = KEEPALIVE_SHARE if keepalive_share is None else keepalive_share
    alive: dict[str, list[int]] = {}
    if ka_share > 0:
        for track_id, entries in (keepalive or {}).items():
            uids = keepalive_uids(entries, floors.get(track_id), owner_uid)
            if award_ok is not None and uids:
                # an approved entry is paid by its rank: it does not also hold a keep-alive slot
                ranked = track_payout(tracks.get(track_id, ()), floors.get(track_id))
                uids = [uid for uid in uids if not (uid in ranked and award_ok(track_id, uid))]
            if uids:
                alive[track_id] = uids
    for slot_share, track_id in zip(prof.tracks, sorted(set(tracks) | set(alive))):
        rank_share = slot_share
        if track_id in alive:
            each = slot_share * ka_share / len(alive[track_id])
            for uid in alive[track_id]:
                paid[uid] = paid.get(uid, 0.0) + each
            rank_share = slot_share * (1.0 - ka_share)
        for uid, share in track_payout(tracks.get(track_id, ()), floors.get(track_id)).items():
            if award_ok is not None and not award_ok(track_id, uid):
                continue
            paid[uid] = paid.get(uid, 0.0) + rank_share * share
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
