"""Persisted state of the specialist tracks: champions, pending confirmations, queue, hash registry, used items.

One JSON file in the validator state directory, written atomically (temp file, fsync, rename) like state.py. No sealed
row content is stored, only item ids. A corrupt or foreign file is moved aside and the evaluator starts empty, which
is safe: submissions are re-detected from the chain and re-evaluated.

Rollback compatibility, both directions: the loader ignores keys it does not know (a file written by a newer release
loads in an older one after a rollback) and every field added after 1.2.0rc4 has a default (a file written by an older
release loads in a newer one). Add fields only that way; ``SPEC_STATE_VERSION`` changes only for an incompatible
format, because a version mismatch moves the file aside and every champion is lost.
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
PENDING_CONFIRM = "pending_confirm"  # beat the incumbent on draw 1, waits for the second draw (or, confirmed on a vacant
                                     # track, for its cohort's playoff: ``confirm_draw`` set)
PENDING_VERIFY = "pending_verification"  # declared base model not available here: not served, re-checked every cycle
CHAMPION = "champion"
FORMER = "former_champion"
NOT_BETTER = "not_better"          # evaluated, did not clear the margin (or the confirmation)
REJECTED = "rejected"              # rule violation, copy, expired, too many failures
WITHDRAWN = "withdrawn"            # the hotkey no longer holds this commitment
TERMINAL = (NOT_BETTER, REJECTED, WITHDRAWN)
# why a PENDING_VERIFY entry waits (``Submission.wait_kind``, 1.2.0rc7): set where the wait starts, never read from the
# reason text (which can echo the miner-chosen base id)
WAIT_INFRA = "infra"               # this host's: base download deferred by the cycle's budget or the disk, in progress or
                                   # not storable; sketch renewal waiting for disk. Holds back its cohort's first crowning
WAIT_MINER = "miner"               # the miner's side: base not on the hub, gated, refused; weights not comparable


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
    attempts: int = 0                  # model failures in the last START_FAIL_WINDOW scheduled rounds
    serve_log: list[list[Any]] = field(default_factory=list)  # [timestamp, ok/model/infra], newest last, at most 12
    serve_retry_pending: bool = False  # champion must be re-measured next cycle, even without challengers
    unavailable_fetches: int = 0       # miner-side refetch failures, independent of start-up strikes
    last_serve_ok_ts: float | None = None
    champion_rounds_without_ok: int = 0
    pay_suspended_since_ts: float | None = None  # requires validator payment-gate support; crown is retained
    hash_named: str = ""
    hash_unnamed: str = ""
    base_model: str = ""
    derived_from: str = ""                # key of the earlier submission this one is a near-duplicate of
    derived_how: str = ""                 # "weights" (delta sketch) or "behaviour" (same answers on a shared draw)
    delta_checked: bool = False           # True: compared with the earlier submissions on the same base (near-duplicates)
    base_id: str = ""                     # tensor hash of the resolved base model ("" = not resolved yet)
    sketch_rule: int = 0                  # tensors.SKETCH_RULE of the last near-duplicate check (0 = before rc5 fixes)
    score: float | None = None            # latest score on a shared draw
    nonproc: float | None = None          # the same draw's score on its non-proc items only (None = before 1.2.0rc6)
    by_tier: dict[str, float] = field(default_factory=dict)   # the same draw's score per tier (monitoring)
    delta: float | None = None            # margin the first pass required
    first_gain: float | None = None
    draw1: str = ""                       # draw id of the first pass
    first_incumbent: str | None = None    # key of the incumbent the first pass was measured against ("" = vacant track)
    confirm_after_ts: float | None = None
    confirm_draw: str = ""                # vacant track (1.2.0rc7): draw id of the passed confirmation, waiting for the cohort playoff
    confirm_score: float | None = None    # the score on that draw (orders the playoff when the budget cannot serve every entry)
    wait_kind: str = ""                   # PENDING_VERIFY (1.2.0rc7): WAIT_INFRA = waits for this host, holds back its cohort
    last_eval_ts: float | None = None
    last_fetch_ts: float | None = None    # fetch retry age; preparing a model does not count as serving it
    provisional_since_ts: float | None = None   # mechanism v2: when the entry first qualified for the provisional keep-alive (set once by the evaluator)
    champion_since_ts: float | None = None
    withdrawn_ts: float | None = None
    withdrawn_kind: str = ""              # "pruned" (hotkey absent) or "changed" (still registered)
    withdrawn_status: str = ""            # status immediately before withdrawal; empty in old state
    carried_from: str = ""                # withdrawn entry whose first draw was restored
    carried_to: str = ""                  # that first draw can be restored only once
    # decision rule v3 (eval mechanism v3, stage 1). Add-only with defaults: an old state loads (missing = old rule), and an
    # older release that gets this file back after a rollback ignores them (``_build`` drops unknown keys).
    decision_rule: str = ""               # "v3" = first stage decided under rule v3; "" = old rule (confirmed under the old rule)
    close_draws: list[str] = field(default_factory=list)    # first-stage draws of a close call (the first one included) while queued
    confirm_draws: list[str] = field(default_factory=list)  # confirmation draws already taken that left it pending (extra draw)
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
    playoff_aborts: int = 0                                    # cohort playoffs abandoned in a row for a host reason (1.2.0rc7)
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
    """A dataclass from a JSON object; keys a newer release added are ignored."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _from_json(raw: dict[str, Any]) -> SpecState:
    if raw.get("version") != SPEC_STATE_VERSION:
        raise ValueError("state belongs to a different version")
    from . import config as C

    st = SpecState(eval_times=[float(x) for x in raw.get("eval_times", [])])
    for tid, t in raw["tracks"].items():
        ts = TrackState(track_id=tid, champion=str(t.get("champion", "")), hashes=dict(t.get("hashes", {})),
                        used={k: int(v) for k, v in t.get("used", {}).items()}, events=list(t.get("events", [])),
                        playoff_aborts=int(t.get("playoff_aborts", 0)))
        ts.submissions = {k: _build(Submission, s) for k, s in t.get("submissions", {}).items()}
        for sub in ts.submissions.values():
            sub.serve_log = sub.serve_log[-C.START_FAIL_LOG_SIZE:]
            sub.attempts = sum(entry[1] == "model" for entry in sub.serve_log[-C.START_FAIL_WINDOW:])
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
