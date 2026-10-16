"""Persisted state of the specialist tracks: champions, pending confirmations, queue, hash registry, used items.

One JSON file in the validator state directory, written atomically (temp file, fsync, rename) like state.py. No sealed
row content is stored, only item ids. A corrupt or foreign file is moved aside and the evaluator starts empty, which
is safe: submissions are re-detected from the chain and re-evaluated.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

log = logging.getLogger("reax.spec.state")
SPEC_STATE_VERSION = 1

QUEUED = "queued"                  # waiting for its first evaluation
PENDING_CONFIRM = "pending_confirm"  # beat the incumbent on draw 1, waits for the second draw
CHAMPION = "champion"
FORMER = "former_champion"
NOT_BETTER = "not_better"          # evaluated, did not clear the margin (or the confirmation)
REJECTED = "rejected"              # rule violation, copy, expired, too many failures
WITHDRAWN = "withdrawn"            # the hotkey no longer holds this commitment
TERMINAL = (NOT_BETTER, REJECTED, WITHDRAWN)


@dataclass
class Submission:
    key: str                      # f"{hotkey}|{commitment text}"
    uid: int
    hotkey: str
    coldkey: str
    track_id: str
    repo: str
    revision: str
    first_seen_block: int
    first_seen_ts: float
    status: str = QUEUED
    reason: str = ""
    attempts: int = 0
    hash_named: str = ""
    hash_unnamed: str = ""
    base_model: str = ""
    derived_from: str = ""                # key of the earlier submission this one is a near-duplicate of
    delta_checked: bool = False           # False: no base weights were available, only exact hashes were compared
    score: float | None = None            # latest score on a shared draw
    delta: float | None = None            # margin the first pass required
    first_gain: float | None = None
    draw1: str = ""                       # draw id of the first pass
    first_incumbent: str | None = None    # key of the incumbent the first pass was measured against ("" = vacant track)
    confirm_after_ts: float | None = None
    last_eval_ts: float | None = None
    champion_since_ts: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Draw:
    draw_id: str
    ts: float
    block: int
    seed: int
    item_ids: list[str]
    fresh: bool = True


@dataclass
class TrackState:
    track_id: str
    submissions: dict[str, Submission] = field(default_factory=dict)
    champion: str = ""
    hashes: dict[str, str] = field(default_factory=dict)       # tensor hash (named or unnamed) -> owning submission key
    used: dict[str, int] = field(default_factory=dict)         # sealed item id -> times drawn
    draws: dict[str, Draw] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)            # short audit lines, newest last (capped)

    def log(self, ts: float, text: str) -> None:
        self.events.append(f"{int(ts)} {text}")
        del self.events[:-200]

    def ordered(self, status: str) -> list[Submission]:
        """Submissions with this status, earliest commitment first (block, then uid, then key)."""
        subs = [s for s in self.submissions.values() if s.status == status]
        return sorted(subs, key=lambda s: (s.first_seen_block, s.uid, s.key))


@dataclass
class SpecState:
    tracks: dict[str, TrackState] = field(default_factory=dict)
    eval_times: list[float] = field(default_factory=list)      # wall-clock time of each served evaluation (rate limit)

    def track(self, track_id: str) -> TrackState:
        return self.tracks.setdefault(track_id, TrackState(track_id))


def _to_json(state: SpecState) -> dict[str, Any]:
    return {"version": SPEC_STATE_VERSION, "saved_at": int(time.time()), "eval_times": state.eval_times,
            "tracks": {tid: asdict(t) for tid, t in sorted(state.tracks.items())}}


def _build(cls, raw: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _from_json(raw: dict[str, Any]) -> SpecState:
    if raw.get("version") != SPEC_STATE_VERSION:
        raise ValueError("state belongs to a different version")
    st = SpecState(eval_times=[float(x) for x in raw.get("eval_times", [])])
    for tid, t in raw["tracks"].items():
        ts = TrackState(track_id=tid, champion=str(t.get("champion", "")), hashes=dict(t.get("hashes", {})),
                        used={k: int(v) for k, v in t.get("used", {}).items()}, events=list(t.get("events", [])))
        ts.submissions = {k: _build(Submission, s) for k, s in t.get("submissions", {}).items()}
        ts.draws = {k: _build(Draw, d) for k, d in t.get("draws", {}).items()}
        st.tracks[tid] = ts
    return st


from_json = _from_json   # public name for the validator's remote mode (a state file written by the competition worker)


def save(state: SpecState, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".spec-state-", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(_to_json(state), fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load(path: str) -> SpecState:
    if not os.path.exists(path):
        return SpecState()
    try:
        with open(path) as fh:
            return _from_json(json.load(fh))
    except Exception as exc:  # noqa: BLE001
        aside = f"{path}.corrupt-{int(time.time())}"
        log.warning("ignoring unusable spec state (%s); moved to %s", exc, aside)
        try:
            os.replace(path, aside)
        except OSError:
            pass
        return SpecState()
