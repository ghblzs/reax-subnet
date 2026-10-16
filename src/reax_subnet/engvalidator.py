"""The validator side of the engine competition (unreleased; docs/ENGINE-COMPETITION.md).

Hooked into the s1 validator loop: after a serving round, when an evaluation cycle is due, the validator measures
up to ``max_evals`` new engine submissions (oldest commitment first) plus the current champion, each against the
reference engine in the same session, updates the champion and mixes the competition share into the next weight
vector. Serving rounds go on during a cycle (background mode): the GPU is shared with the serving runtimes, which is
why every session measures the reference alone first and treats later reference runs far from that baseline as
inconclusive (engrunner.reference_baseline).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional

from .award import engine_submission_id, spec_submission_id
from .chain import Snapshot
from .competition_split import (
    COMPETITIONS_START_UTC, SPLIT_PROFILE, NoSpecialistEvaluator, SpecialistEvaluator, active_tracks, combine_weights,
)
from .engcomp import (
    EvalResult, RewardParams, Submission, TrackParams, TrackState, competition_weights,
    needs_remeasurement, observe, parse_engine_commitment, pending, update_champion,
)
from .spec.commitment import parse_commitment as parse_spec_commitment
from .engrunner import EngineError, InfraError, ReferenceError_, ReferenceFailedWithCandidate, Runner, clean, evaluate, reference_alone_ok

log = logging.getLogger("reax.engcomp")

# Release candidates; calibrate on the track's GPU class before activation (docs/ENGINE-COMPETITION.md section 8).
TRACKS: dict[str, TrackParams] = {
    "s1-fast": TrackParams("s1-fast", pool="s1-fast", temperature=2.07),
}
STATE_VERSION = 1


class EngineCompetition:
    def __init__(self, runner: Runner, reference_refs: Mapping[str, str], state_path: str,
                 tracks: Optional[Mapping[str, TrackParams]] = None, reward: Optional[RewardParams] = None,
                 interval_s: float = 6 * 3600, max_evals: int = 4, clock=time.monotonic,
                 start_utc: Optional[datetime] = COMPETITIONS_START_UTC, profile: str = SPLIT_PROFILE,
                 specialist: Optional[SpecialistEvaluator] = None,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 engine_start_utc: Optional[datetime] = None, award_gate: Optional[Any] = None):
        """``start_utc``: before it the weights are exactly the serving (1.1) weights and no cycle runs; None = always
        active (development networks and tests only; finney uses the release constant). ``profile``: split profile
        (release constant, competition_split.py). ``specialist``: plug-in for the use-case tracks.
        ``engine_start_utc``: the engine track's own start (finney: ``competition_split.release_engine_start()``, never
        while the track is held); None = with ``start_utc``. Before it no engine cycle runs and no champion is paid.
        ``award_gate``: ``award.AwardGate`` (finney); every award needs the owner's signed approval. None = no gate
        (development networks without REAX_AWARD_APPROVALS, tests)."""
        self.start_utc = start_utc
        self.engine_start_utc = engine_start_utc
        self.award_gate = award_gate
        self._held_logged = False
        self.profile = profile
        self.specialist = specialist or NoSpecialistEvaluator()
        self._utcnow = utcnow
        self.runner = runner
        self.tracks = dict(tracks or TRACKS)
        self.reference_refs = {t: r for t, r in reference_refs.items() if t in self.tracks}
        missing = [t for t in self.tracks if t not in self.reference_refs]
        if missing:
            raise ValueError(f"no reference engine image for track(s): {', '.join(missing)}")
        self.reward = reward or RewardParams()
        self.interval_s = interval_s
        self.max_evals = max_evals
        self.state_path = state_path
        self._clock = clock
        self._last_cycle: Optional[float] = None
        self.states: dict[str, TrackState] = {t: TrackState() for t in self.tracks}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("version") != STATE_VERSION:
                raise ValueError("version")
            for t, st in raw.get("tracks", {}).items():
                if t in self.states:
                    self.states[t] = TrackState.from_json(st)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001 - a bad file restarts the competition state, never the validator
            log.warning("engine competition state %s unreadable (%s); starting empty", self.state_path, exc)
            try:
                os.replace(self.state_path, self.state_path + ".bad")
            except OSError:
                pass

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.state_path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".engstate-", dir=directory)
        with os.fdopen(fd, "w") as fh:
            # saved_at: the validator in remote mode treats a file older than 26 h as missing (remote.py);
            # binds_hotkey: whether the runner that produced the results proved that an image belongs to its committer
            json.dump({"version": STATE_VERSION, "saved_at": int(time.time()),
                       "binds_hotkey": bool(getattr(self.runner, "binds_hotkey", False)),
                       "tracks": {t: s.to_json() for t, s in self.states.items()}},
                      fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ chain view
    def submissions(self, snap: Snapshot, exclude: set[int]) -> dict[str, list[Submission]]:
        out: dict[str, list[Submission]] = {t: [] for t in self.tracks}
        for n in snap.neurons:
            if n.uid in exclude:
                continue
            parsed = parse_engine_commitment(n.commitment)
            if parsed is None or parsed[0] not in self.tracks:
                continue
            track, image, digest = parsed
            out[track].append(Submission(n.uid, n.hotkey, track, image, digest, n.commitment_block))
        return out

    def is_engine_commitment(self, commitment: str) -> bool:
        return parse_engine_commitment(commitment) is not None

    # ------------------------------------------------------------------ evaluation cycle
    def active(self) -> bool:
        return self.start_utc is None or self._utcnow() >= self.start_utc

    def engine_active(self) -> bool:
        return self.active() and (self.engine_start_utc is None or self._utcnow() >= self.engine_start_utc)

    def due(self) -> bool:
        if not self.engine_active():
            return False
        return self._last_cycle is None or self._clock() - self._last_cycle >= self.interval_s

    async def run_cycle(self, snap: Snapshot, exclude: set[int],
                        between: Optional[Callable[[], Awaitable[object]]] = None) -> dict[str, list[dict]]:
        """One evaluation cycle. ``between`` runs after every evaluation (the validator keeps its weights fresh)."""
        self._last_cycle = self._clock()
        report: dict[str, list[dict]] = {}
        preflight = getattr(self.runner, "preflight", None)
        if preflight is not None:
            try:
                await preflight()
            except InfraError as exc:
                log.warning("engine competition: cycle skipped, retried next interval: %s", exc)
                return report
        bind = bool(getattr(self.runner, "binds_hotkey", False))
        for track, subs in self.submissions(snap, exclude).items():
            st = self.states[track]
            params = self.tracks[track]
            eligible = observe(st, subs, snap.block, first_committer_only=not bind)
            todo = pending(st, eligible, self.max_evals)
            again = needs_remeasurement(st, eligible)
            ch = st.champion
            if ch is not None and todo and not any(s.digest == ch.digest for s in again):
                again += [s for s in eligible if s.digest == ch.digest and s.hotkey == ch.hotkey]
            todo += [s for s in again if s not in todo]  # challenger confirmation; champion re-measured when challenged
            report[track] = []
            for sub in todo:
                res = await self._evaluate_one(st, track, params, sub, snap.block)
                if res == "stop":
                    break
                if isinstance(res, EvalResult):
                    st.results[sub.digest] = res
                    st.attempts.pop(sub.digest, None)
                    report[track].append({"uid": sub.uid, **res.to_json()})
                    log.info("engine competition [%s] uid %d %s: %s score %.3f (%.2f vs %.2f dec/s, mean_d %s)%s",
                             track, sub.uid, sub.digest[:12], "VALID" if res.ok else "REJECTED", res.score,
                             res.throughput, res.ref_throughput,
                             "-" if res.mean_distance == float("inf") else f"{res.mean_distance:.3f}",
                             f" reason={res.reason}" if res.reason else "")
                self.save()
                if between is not None:
                    try:
                        await between()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("engine competition: callback between evaluations failed: %r", exc)
            eligible = observe(st, subs, snap.block, first_committer_only=not bind)
            before = st.champion.digest if st.champion else None
            ch = update_champion(st, eligible, params, snap.block)
            if (ch.digest if ch else None) != before:
                log.info("engine competition [%s]: champion is now %s", track,
                         f"uid {ch.uid} {ch.digest[:12]} score {ch.score:.3f}" if ch else "vacant")
        self.save()
        return report

    async def _evaluate_one(self, st: TrackState, track: str, params: TrackParams, sub: Submission, block: int):
        """An EvalResult to record, None (inconclusive, retried later) or "stop" (our reference is broken)."""
        reference = self.reference_refs[track]

        def fail(reason: str) -> EvalResult:
            return EvalResult(digest=sub.digest, track=track, ok=False, reason=reason, block=block)

        def inconclusive(reason: str):
            n = st.attempts[sub.digest] = st.attempts.get(sub.digest, 0) + 1
            log.warning("engine competition: uid %d %s inconclusive (%d/%d): %s", sub.uid, sub.digest[:12], n,
                        params.max_infra_attempts, reason)
            return fail(f"inconclusive {n} times: {reason}") if n >= params.max_infra_attempts else None

        try:
            if getattr(self.runner, "binds_hotkey", False):
                owner = await asyncio.wait_for(self.runner.owner_of(sub.ref), params.eval_budget_s)
                if owner != sub.hotkey:
                    discard = getattr(self.runner, "discard", None)
                    if discard is not None:
                        await discard(sub.ref)  # somebody else's image: give the disk back
                    st.not_owner.add(f"{sub.digest}:{sub.hotkey}")
                    log.info("engine competition: uid %d committed %s, which is bound to another hotkey (label %r); "
                             "ignored", sub.uid, sub.digest[:12], (owner or "")[:12])
                    return None
            return await asyncio.wait_for(
                evaluate(self.runner, reference, sub.ref, track, params, block=block, digest=sub.digest),
                params.eval_budget_s)
        except asyncio.TimeoutError:
            return fail(f"evaluation exceeded its {params.eval_budget_s:.0f} s budget")
        except EngineError as exc:  # e.g. an image above the size limit, a registry that is not allowed
            return fail(clean(str(exc)))
        except InfraError as exc:
            return inconclusive(str(exc))
        except ReferenceFailedWithCandidate as exc:
            if await reference_alone_ok(self.runner, reference, params):
                return inconclusive(f"reference failed while this candidate was loaded ({exc})")
            log.error("engine competition: reference engine is broken (%s); cycle stopped, nothing recorded", exc)
            return "stop"
        except ReferenceError_ as exc:
            log.error("engine competition: reference engine failed (%s); cycle stopped, nothing recorded", exc)
            return "stop"
        except Exception as exc:  # noqa: BLE001 - a submission must never stop the validator
            log.warning("engine competition: evaluation of uid %d crashed: %r", sub.uid, exc)
            return inconclusive(f"internal: {type(exc).__name__}")

    # ------------------------------------------------------------------ weights
    def weights(self, base: Mapping[int, float], snap: Snapshot, exclude: set[int],
                burn_fraction: float = 0.0) -> dict[int, float]:
        """Mix the competitions into ``base`` (competition_split.combine_weights). Before the start time, or with
        nothing to pay, ``base`` is returned unchanged (= the 1.1 weights)."""
        if not self.active():
            return dict(base)
        champion: Optional[int] = None
        frac = 0.0
        if not self.engine_active():
            if not self._held_logged:
                log.info("engine competition: the engine track is held (not open); no engine champion is paid and the "
                         "engine share goes to the inference part")
                self._held_logged = True
            tracks: list[str] = []
        else:
            self._held_logged = False
            tracks = list(self.tracks)
        subs = self.submissions(snap, exclude) if tracks else {}
        for track in tracks:
            st = self.states[track]
            bind = bool(getattr(self.runner, "binds_hotkey", False))
            eligible = [s for s in subs[track] if f"{s.digest}:{s.hotkey}" not in st.not_owner
                        and (bind or st.first_seen.get(s.digest, (0, s.hotkey))[1] == s.hotkey)]
            ch = st.champion
            live = next((s for s in eligible if ch is not None and s.digest == ch.digest and s.hotkey == ch.hotkey),
                        None)
            if ch is None or live is None:
                continue  # champion withdrew since the last cycle: no competition weight until re-evaluated
            ch.uid = live.uid  # UIDs can be renumbered; the title belongs to the hotkey
            comp = competition_weights(st, eligible, self.tracks[track], self.reward, snap.block)
            if track == "s1-fast" and ch.uid in comp:
                if self.award_gate is not None and not self.award_gate.approved(track, ch.hotkey,
                                                                                engine_submission_id(ch.digest)):
                    continue  # awaiting owner approval: the engine share goes to the inference part
                champion, frac = ch.uid, comp[ch.uid]
        ids = active_tracks(self._utcnow(), self.profile)
        results = self.specialist.track_results(ids, self._utcnow()) if ids else {}
        scores = {t: results[t] for t in ids if results.get(t)}
        if champion is None and not scores:
            return dict(base)  # nothing to pay: exactly the 1.1 weights
        return combine_weights(base, champion, frac, scores, self.profile, owner_uid=snap.owner_uid,
                               burn_fraction=burn_fraction, award_ok=self._spec_award_ok(snap))

    def _spec_award_ok(self, snap: Snapshot) -> Optional[Callable[[str, int], bool]]:
        """The gate for a specialist rank: the uid's hotkey and the model it commits to now must be approved."""
        gate = self.award_gate
        if gate is None:
            return None
        by_uid = {n.uid: n for n in snap.neurons}

        def ok(track_id: str, uid: int) -> bool:
            n = by_uid.get(uid)
            c = parse_spec_commitment(n.commitment) if n is not None else None
            if c is None or c.track_id != track_id:
                return False
            return gate.approved(track_id, n.hotkey, spec_submission_id(c.repo, c.revision))
        return ok
