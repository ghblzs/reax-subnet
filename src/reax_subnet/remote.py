"""Competitions outside the validator process (``REAX_COMPETITIONS_MODE=remote``, docs/COMPETITIONS.md "Remote mode").

``reax-competition-worker`` (worker.py) runs the engine and specialist cycles with docker and writes
``engine-competition.json`` and ``spec-state.json`` atomically into a state directory. The validator mounts that directory
read-only and, each round, only READS it: it needs no docker socket, no sealed data and no GPU for the competitions.

Payment is computed exactly as in local mode (engvalidator.EngineCompetition.weights: champion + 14-day decay +
confirmations; competition_split: track leaders + entry floors + 75/17/8). On top of that, every read re-checks against the
chain snapshot that the champion's hotkey still holds its uid and still commits to the measured image / model (a
withdrawn or deregistered competitor drops out at once). A state file that is missing, unreadable, from another version or
older than ``REMOTE_MAX_AGE_S`` (26 h, by its own ``saved_at``) pays nothing: that competition's share goes back to the
inference part, with ONE warning per outage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from .chain import Snapshot
from .competition_split import SPLIT_PROFILE, COMPETITIONS_START_UTC, SpecialistEvaluator
from .competitions import SpecialistAdapter
from .config import ENGINE_STATE_FILE, REMOTE_MAX_AGE_S, SPEC_STATE_FILE
from .engcomp import RewardParams, TrackParams, TrackState
from .engvalidator import STATE_VERSION, TRACKS, EngineCompetition
from .spec import config as spec_config
from .spec import state as spec_state
from .spec.evaluator import compute_leaders

log = logging.getLogger("reax.remote")

MAX_STATE_BYTES = 64 * 1024 * 1024
CLOCK_SKEW_S = 600


def read_state(path: str, *, now: float, max_age_s: float = REMOTE_MAX_AGE_S) -> tuple[Optional[dict[str, Any]], str]:
    """The parsed state file, or ``(None, reason)`` when it is missing, unreadable, too big, not an object, or stale."""
    try:
        if os.path.getsize(path) > MAX_STATE_BYTES:
            return None, "file too large"
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return None, "file missing"
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    saved = raw.get("saved_at")
    saved = float(saved) if isinstance(saved, (int, float)) and not isinstance(saved, bool) else mtime
    age = now - saved
    if age > max_age_s:
        return None, f"stale: written {age / 3600:.1f} h ago, limit {max_age_s / 3600:.0f} h"
    if age < -CLOCK_SKEW_S:
        return None, "saved_at lies in the future"
    return raw, ""


class _Outage:
    """One warning per outage of a state file (and one note when it recovers), not one per round."""

    def __init__(self, label: str, path: str):
        self.label, self.path, self.down = label, path, False

    def update(self, ok: bool, reason: str = "") -> None:
        if not ok and not self.down:
            log.warning("%s: state %s unusable (%s); this competition's share goes to the inference part until the "
                        "competition worker writes a fresh file", self.label, self.path, reason)
        elif ok and self.down:
            log.info("%s: state %s is readable and fresh again", self.label, self.path)
        self.down = not ok


class _RemoteRunner:
    """Stands in for the runner: the validator never starts an engine here."""

    binds_hotkey = True


class RemoteEngineCompetition(EngineCompetition):
    """``EngineCompetition`` whose results come from the worker's ``engine-competition.json``. No cycle ever runs here."""

    def __init__(self, state_path: str, *, tracks: Optional[Mapping[str, TrackParams]] = None,
                 reward: Optional[RewardParams] = None, start_utc: Optional[datetime] = COMPETITIONS_START_UTC,
                 profile: str = SPLIT_PROFILE, specialist: Optional[SpecialistEvaluator] = None,
                 max_age_s: float = REMOTE_MAX_AGE_S, wall: Callable[[], float] = time.time,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.max_age_s, self._wall = max_age_s, wall
        self._outage = _Outage("engine competition", state_path)
        tr = dict(tracks or TRACKS)
        super().__init__(_RemoteRunner(), {t: "remote" for t in tr}, state_path, tracks=tr, reward=reward,
                         start_utc=start_utc, profile=profile, specialist=specialist, utcnow=utcnow)

    # the worker owns the file: nothing is loaded at construction (a bad file must never be moved aside from here) or saved
    def _load(self) -> None:
        return None

    def save(self) -> None:
        return None

    def due(self) -> bool:
        return False

    async def run_cycle(self, snap: Snapshot, exclude: set[int], between: Any = None) -> dict[str, list[dict]]:
        return {}

    def refresh_state(self) -> bool:
        """Re-read the state file; on any problem every track is empty (nobody earns the engine share)."""
        empty = {t: TrackState() for t in self.tracks}
        raw, why = read_state(self.state_path, now=self._wall(), max_age_s=self.max_age_s)
        if raw is not None and raw.get("version") != STATE_VERSION:
            raw, why = None, f"version {raw.get('version')!r}, expected {STATE_VERSION}"
        states = empty
        if raw is not None:
            try:
                states = {**empty, **{t: TrackState.from_json(st) for t, st in raw.get("tracks", {}).items() if t in empty}}
            except Exception as exc:  # noqa: BLE001 - whatever a corrupt file contains
                raw, why = None, f"invalid content ({type(exc).__name__})"
        if raw is None:
            states = empty
        else:
            self.runner.binds_hotkey = bool(raw.get("binds_hotkey", True))
        self.states = states
        self._outage.update(raw is not None, why)
        return raw is not None

    def weights(self, base: Mapping[int, float], snap: Snapshot, exclude: set[int], burn_fraction: float = 0.0) -> dict[int, float]:
        if self.active():
            self.refresh_state()
        return super().weights(base, snap, exclude, burn_fraction)


class RemoteSpecEvaluator:
    """The part of ``SpecEvaluator`` the share layer needs (``cfg.active_tracks``, ``clock``, ``leaders``), over the
    worker's ``spec-state.json``. It cannot evaluate anything."""

    def __init__(self, state_path: str, tracks: Sequence[spec_config.TrackDef] = spec_config.DEFAULT_TRACKS, *,
                 max_age_s: float = REMOTE_MAX_AGE_S, clock: Callable[[], float] = time.time):
        self.cfg = spec_config.SpecConfig(sealed_dir="", work_dir="", state_path=state_path, tracks=tuple(tracks))
        self.max_age_s, self.clock = max_age_s, clock
        self._outage = _Outage("specialist tracks", state_path)

    def leaders(self, active: Sequence[spec_config.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        ids = [t.track_id for t in active]
        if not ids:
            return {}
        raw, why = read_state(self.cfg.state_path, now=self.clock(), max_age_s=self.max_age_s)
        state = None
        if raw is not None:
            try:
                state = spec_state.from_json(raw)
            except Exception as exc:  # noqa: BLE001
                why = f"invalid content ({type(exc).__name__})"
        self._outage.update(state is not None, why)
        if state is None:
            return {t: [] for t in ids}
        return compute_leaders(state, ids, snapshot)

    async def evaluate_cycle(self, snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:  # pragma: no cover
        raise RuntimeError("the validator runs no specialist cycles in remote mode")


class RemoteSpecAdapter(SpecialistAdapter):
    """Never due, never runs: refreshes its leaders from the state file every round (SpecialistAdapter.refresh)."""

    def due(self) -> bool:
        return False

    async def run_cycle(self, snap: Snapshot) -> bool:
        return False
