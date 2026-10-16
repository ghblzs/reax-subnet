"""Runs the competitions next to the serving rounds (docs/COMPETITIONS.md, "One GPU at a time").

* One GPU lock: engine measurements and specialist evaluations never run at the same time on the validator GPU.
  Each keeps its own free-memory check (engine: 2 x 12.5 GB, specialist: <= 25 GB) and skips the cycle cleanly, without
  recording anything, when the memory is short.
* Background mode (the real validator): a cycle runs as a task, so serving rounds go on at their normal interval while
  a measurement takes hours. The weights are computed from the competition state as it is at that moment.
* ``SpecialistAdapter`` turns the async ``SpecEvaluator`` into the synchronous ``SpecialistEvaluator`` the share layer
  asks: it hands out the ranked leaders of the last cycle (refreshed from the evaluator's saved state every round).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Sequence

from .chain import Snapshot
from .competition_split import RankedLeaders, SpecialistEvaluator, TrackResults

log = logging.getLogger("reax.competitions")


def _copy(r: Any) -> list:
    """A copy that keeps the evaluator's rank order marker (``RankedLeaders``)."""
    return RankedLeaders(r) if isinstance(r, RankedLeaders) else list(r)

SPEC_CYCLE_INTERVAL_S = 3600  # the evaluator rate-limits itself (6 served evaluations per hour); this is only the poll


async def gpu_free_mib(gpu: str = "0", timeout_s: float = 30.0) -> Optional[int]:
    """Free GPU memory in MiB, or None when nvidia-smi cannot be read (the check is then skipped)."""
    from .engrunner import _run

    try:
        code, out, _ = await _run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], timeout_s)
        return int(out.strip().splitlines()[0]) if code == 0 else None
    except (ValueError, IndexError, OSError):
        return None


class SpecialistAdapter:
    """``SpecialistEvaluator`` view of a ``reax_subnet.spec.evaluator.SpecEvaluator`` (or any object with an async
    ``evaluate_cycle(snapshot) -> {track_id: [(uid, score), ...]}`` and a ``leaders``-style refresh)."""

    def __init__(self, evaluator: Any, *, interval_s: float = SPEC_CYCLE_INTERVAL_S, gpu: str = "0",
                 need_mib: Optional[int] = None, free_mib: Callable[[str], Awaitable[Optional[int]]] = gpu_free_mib,
                 clock: Callable[[], float] = time.monotonic):
        self.evaluator = evaluator
        self.interval_s = interval_s
        self.gpu = gpu
        if need_mib is None:
            from .spec.vllm import GPU_BUDGET_GB

            need_mib = int(GPU_BUDGET_GB * 1024)
        self.need_mib = need_mib
        self._free_mib = free_mib
        self._clock = clock
        self._last_cycle: Optional[float] = None
        self._results: TrackResults = {}

    # --- SpecialistEvaluator protocol (synchronous, cheap)
    def track_results(self, track_ids: Sequence[str], now: datetime) -> TrackResults:
        return {t: _copy(self._results[t]) for t in track_ids if self._results.get(t)}

    # --- cycle
    def due(self) -> bool:
        return self._last_cycle is None or self._clock() - self._last_cycle >= self.interval_s

    def refresh(self, snap: Snapshot) -> None:
        """Re-read the leaders from the evaluator's saved state (no GPU, no network): after a restart the tracks pay
        again at once instead of after the first full cycle."""
        refresh = getattr(self.evaluator, "leaders", None)
        cfg = getattr(self.evaluator, "cfg", None)
        if refresh is None or cfg is None:
            return
        try:
            active = cfg.active_tracks(getattr(self.evaluator, "clock", time.time)())
            self._results = {t: _copy(r) for t, r in refresh(active, snap).items()}
        except Exception as exc:  # noqa: BLE001 - a bad state file must not stop the validator
            log.warning("specialist tracks: could not read the leaders: %r", exc)

    async def run_cycle(self, snap: Snapshot) -> bool:
        """One evaluation cycle; False when it was skipped for lack of GPU memory."""
        self._last_cycle = self._clock()
        free = await self._free_mib(self.gpu)
        if free is not None and free < self.need_mib:
            log.warning("specialist tracks: cycle skipped, retried next interval: GPU %s has %d MiB free, an "
                        "evaluation needs about %d MiB", self.gpu, free, self.need_mib)
            return False
        self._results = {t: _copy(r) for t, r in (await self.evaluator.evaluate_cycle(snap)).items()}
        return True


class CompetitionDriver:
    """Schedules the engine competition and the specialist evaluator on the one GPU."""

    def __init__(self, engine: Any, spec: Optional[SpecialistAdapter] = None, *, background: bool = True,
                 gpu_lock: Optional[asyncio.Lock] = None,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.engine = engine
        self.spec = spec
        self.background = background
        self.gpu_lock = gpu_lock or asyncio.Lock()
        self._utcnow = utcnow
        self._task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _spec_active(self) -> bool:
        start = getattr(self.engine, "start_utc", None)
        return self.spec is not None and (start is None or self._utcnow() >= start)

    def due(self) -> bool:
        return self.engine.due() or (self._spec_active() and self.spec.due())

    async def run(self, snap: Snapshot, exclude: set[int],
                  between: Optional[Callable[[], Awaitable[object]]] = None) -> dict[str, Any]:
        """Engine cycle (if due), then specialist cycle (if due), each under the GPU lock."""
        report: dict[str, Any] = {}
        if self.engine.due():
            async with self.gpu_lock:
                report = await self.engine.run_cycle(snap, exclude, between=between)
        if self._spec_active() and self.spec.due():
            async with self.gpu_lock:
                await self.spec.run_cycle(snap)
            if between is not None:
                try:
                    await between()
                except Exception as exc:  # noqa: BLE001
                    log.warning("competitions: callback after the specialist cycle failed: %r", exc)
        return report

    def refresh(self, snap: Snapshot) -> None:
        if self.spec is not None and not self.running:
            self.spec.refresh(snap)

    def kick(self, snap: Snapshot, exclude: set[int],
             between: Optional[Callable[[], Awaitable[object]]] = None) -> Optional[asyncio.Task]:
        """Background mode: start a cycle task unless one is running or nothing is due. Never raises, never awaits."""
        if self.running or not self.due():
            return None
        self._task = asyncio.ensure_future(self._guarded(snap, exclude, between))
        return self._task

    async def _guarded(self, snap, exclude, between) -> dict[str, Any]:
        try:
            return await self.run(snap, exclude, between)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("competition cycle failed; serving rounds continue")
            return {}
