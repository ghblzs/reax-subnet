"""Model backends for ``reax-s1/1``.

Miners and validators do not load the reference models in this process. The
model runs in a *runtime sidecar* container that serves the reference runtime's
decision contract on loopback (``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header):

* s1-fast: ``runtimes/s1-fast`` = the reference fast runtime (pytorch 2.13 image digest,
  jevk5 at the pinned commit) serving the batch-1 CUDA-graph readout of the reference engine;
* s1-pro: ``runtimes/s1-pro`` = the reference Rune image recipe (vLLM 0.30.0 digest,
  deterministic flags) with the reference decision adapter, unchanged.

This module sends each decision to the sidecar exactly as the reference engine
does (``Question.author_dict``) and applies the reference ``normalize_answer``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import time
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from .manifest import Manifest, load_manifest
from .s1contract import Decision, normalize_answer, parse_decision


class S1Backend(Protocol):
    model: str
    manifest: Manifest

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]: ...

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]: ...

    async def probe(self) -> bool: ...

    async def close(self) -> None: ...


class SidecarError(RuntimeError):
    """A runtime sidecar failure. ``answer_started``: the runtime had already begun sending its answer (a request that
    failed before that point is safe to try on another backend). ``backend_fault``: the failure says something about
    the backend's health (a 4xx caused by one bad request does not)."""

    def __init__(self, message: str, *, answer_started: bool = True, backend_fault: bool = True):
        super().__init__(message)
        self.answer_started = answer_started
        self.backend_fault = backend_fault


MAX_RUNTIME_RESPONSE_BYTES = 1 << 20
PROBE_DECISION = parse_decision(
    {"id": "probe", "state": "Probe: the parcel arrived two days late.",
     "questions": {"q": {"type": "noul", "instructions": "Was the delivery late?"}}},
    "s1-fast",
)


async def _read_capped(stream: Any, limit: int) -> bytes:
    """The whole body up to ``limit`` bytes. ``StreamReader.read(n)`` alone returns only what has arrived
    so far, so a body split across TCP segments would be cut short and fail to parse."""
    chunks = []
    size = 0
    async for chunk in stream.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > limit:
            raise SidecarError("runtime response too large")
        chunks.append(chunk)
    return b"".join(chunks)


class SidecarBackend:
    """Client for a runtime sidecar on loopback."""

    def __init__(self, model: str, url: str, token: str, max_inflight: int = 16, timeout_s: float = 60.0):
        self.model = model
        self.manifest = load_manifest(model)
        self.url = url.rstrip("/")
        self._headers = {"X-S1M-Internal-Token": token, "Content-Type": "application/json"}
        self._sem = asyncio.Semaphore(max_inflight)
        self.max_inflight = max_inflight
        self.inflight = 0  # requests of this process on this backend, running or waiting for a slot
        self._timeout = timeout_s
        self._session = None

    async def _client(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), trust_env=False,
                connector=aiohttp.TCPConnector(limit=64),
            )
        return self._session

    async def ready(self) -> bool:
        try:
            s = await self._client()
            async with s.get(self.url + "/health") as r:
                return r.status == 200
        except Exception:
            return False

    async def _one(self, d: Decision) -> dict[str, Any]:
        self.inflight += 1
        started = [False]  # set once the runtime answered HTTP 200: from then on answer bytes may exist
        try:
            return await self._send(d, started)
        except Exception as exc:
            # Failover (PooledBackend) needs to know whether the runtime had begun to answer.
            if not hasattr(exc, "answer_started"):
                exc.answer_started = started[0]  # type: ignore[attr-defined]
            raise
        finally:
            self.inflight -= 1

    async def _send(self, d: Decision, started: list) -> dict[str, Any]:
        async with self._sem:
            # The deadline is set once a slot is free, so queued items never reach the runtime already expired.
            body = {
                "request_id": secrets.token_hex(12),
                "deadline_unix_ms": int(time.time() * 1000) + int(self._timeout * 1000),
                "state": d.state,
                "questions": {d.question.ident: d.question.author_dict()},
            }
            import aiohttp

            s = await self._client()
            data = json.dumps(body, ensure_ascii=False)
            # A pooled keep-alive connection can be closed by the runtime between requests; the first
            # write then fails with ServerDisconnectedError (seen in the 10 Oct GPU rehearsal: 2 of 5
            # sparse requests). Decisions are pure, so one retry on a fresh connection is safe.
            for attempt in (0, 1):
                try:
                    async with s.post(self.url + "/internal/v1/decide", data=data, headers=self._headers) as r:
                        if r.status != 200:
                            raise SidecarError(f"runtime returned HTTP {r.status}", answer_started=False,
                                               backend_fault=r.status >= 500 or r.status in (408, 429))
                        started[0] = True
                        raw = await _read_capped(r.content, MAX_RUNTIME_RESPONSE_BYTES)
                    break
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                    if attempt or started[0]:
                        raise
        return normalize_answer(d.question, json.loads(raw)["answers"][d.question.ident])

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._one(d) for d in decisions)))

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        """Like ``decide`` but per item: each entry is an answer or the exception of its last attempt."""

        async def attempt(d: Decision) -> Any:
            last: Exception = SidecarError("not attempted")
            for _ in range(retries + 1):
                try:
                    return await self._one(d)
                except Exception as exc:  # noqa: BLE001 - reported per item
                    last = exc
            return last

        return list(await asyncio.gather(*(attempt(d) for d in decisions)))

    async def probe(self) -> bool:
        """Authenticated end-to-end check: one fixed decision through the runtime."""
        try:
            await self._one(PROBE_DECISION)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()


FAILURES_TO_REMOVE = 2  # consecutive backend faults (connection, timeout, HTTP 5xx) before a backend is taken out
SUCCESSES_TO_READMIT = 2  # consecutive good /health answers before a removed backend takes requests again
RETRY_AFTER_S = 10.0  # a removed backend is re-checked (health loop) or tried once (nothing else left) after this long
HEALTH_INTERVAL_S = 5.0

log = logging.getLogger("reax.s1backends")


class _Member:
    def __init__(self, backend: SidecarBackend):
        self.backend = backend
        self.healthy = True
        self.failures = 0
        self.good_checks = 0
        self.retry_at = 0.0
        self.trial = False  # a single request is being tried on this removed backend


class PooledBackend:
    """Several runtime sidecars of the SAME model behind one hotkey (docs/MINER.md, "Several GPUs or machines").

    Every decision goes to the healthy backend with the fewest requests in flight (ties go round-robin), never beyond
    what a backend's own ``max_inflight`` allows unless all of them are full. A backend that fails twice in a row is
    removed; the health loop (``GET /health``) re-admits it after it answered twice and, when ``readmit_check`` is set, passed
    the identity check against the release references. A request that failed on one
    backend before any answer byte arrived is tried once on another; it is never tried a third time, and never again
    once the runtime had begun to answer, so a sick fleet cannot be flooded with retries. Signing happens in the miner
    process only: the backends never see the hotkey.
    """

    def __init__(self, backends: Sequence[SidecarBackend], *, health_interval_s: float = HEALTH_INTERVAL_S,
                 retry_after_s: float = RETRY_AFTER_S):
        if not backends:
            raise ValueError("a backend pool needs at least one backend")
        if len({b.model for b in backends}) != 1 or len({b.manifest.manifest_id for b in backends}) != 1:
            raise ValueError("all backends of a pool must serve the same model manifest")
        self.model = backends[0].model
        self.manifest = backends[0].manifest
        self._members = [_Member(b) for b in backends]
        self._health_interval = health_interval_s
        self._retry_after = retry_after_s
        self._turn = 0
        self._monitor: Optional[asyncio.Task] = None
        # Set by the miner CLI: ``await readmit_check(backend)`` returns the identity problems of one backend (empty =
        # matches the release references). With it, a removed backend returns only after passing it, and never
        # through a trial request.
        self.readmit_check: Optional[Callable[[SidecarBackend], Awaitable[list[str]]]] = None

    # --- state -----------------------------------------------------------------------------------------------
    @property
    def backends(self) -> list[SidecarBackend]:
        return [m.backend for m in self._members]

    @property
    def size(self) -> int:
        return len(self._members)

    @property
    def healthy_count(self) -> int:
        return sum(1 for m in self._members if m.healthy)

    def describe(self) -> list[dict[str, Any]]:
        return [{"url": m.backend.url, "healthy": m.healthy, "inflight": m.backend.inflight,
                 "max_inflight": m.backend.max_inflight} for m in self._members]

    def _mark_bad(self, m: _Member) -> None:
        m.failures += 1
        m.good_checks = 0
        if m.healthy and m.failures >= FAILURES_TO_REMOVE:
            m.healthy = False
            log.warning("runtime backend %s removed from rotation after %d failures (%d of %d backends healthy)",
                        m.backend.url, m.failures, self.healthy_count, self.size)
        if not m.healthy:
            m.retry_at = time.monotonic() + self._retry_after

    def _mark_good(self, m: _Member) -> None:
        m.failures = 0
        if not m.healthy:
            m.healthy = True
            log.info("runtime backend %s is back in rotation (%d of %d backends healthy)",
                     m.backend.url, self.healthy_count, self.size)

    # --- dispatch --------------------------------------------------------------------------------------------
    def _pick(self, exclude: set[int]) -> Optional[_Member]:
        live = [m for m in self._members if m.healthy and id(m) not in exclude]
        if live:
            free = [m for m in live if m.backend.inflight < m.backend.max_inflight]
            pool = free or live  # everything full: queue at the least loaded one
            low = min(m.backend.inflight for m in pool)
            tied = [m for m in pool if m.backend.inflight == low]
            self._turn += 1
            return tied[self._turn % len(tied)]
        # Nothing healthy: let one request at a time test a removed backend once its wait is over, so the miner
        # recovers even without the health loop, but a dead fleet sees at most one request per backend per wait.
        if self.readmit_check is not None:
            return None  # an unchecked backend must not get validator traffic; the health loop re-admits it
        now = time.monotonic()
        for m in self._members:
            if id(m) not in exclude and not m.trial and now >= m.retry_at:
                m.trial = True
                return m
        return None

    async def _one(self, d: Decision) -> dict[str, Any]:
        tried: set[int] = set()
        for attempt in (0, 1):
            m = self._pick(tried)
            if m is None:
                raise SidecarError("no healthy runtime backend", answer_started=False, backend_fault=False)
            trial = m.trial
            try:
                answer = await m.backend._one(d)
            except Exception as exc:
                fault = getattr(exc, "backend_fault", True)
                if fault:
                    self._mark_bad(m)
                if trial:
                    m.trial = False
                    m.retry_at = time.monotonic() + self._retry_after
                if attempt or not fault or getattr(exc, "answer_started", True):
                    raise
                tried.add(id(m))
                continue
            if trial:
                m.trial = False
            self._mark_good(m)
            return answer
        raise AssertionError("unreachable")  # pragma: no cover

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._one(d) for d in decisions)))

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        async def attempt(d: Decision) -> Any:
            last: Exception = SidecarError("not attempted")
            for _ in range(retries + 1):
                try:
                    return await self._one(d)
                except Exception as exc:  # noqa: BLE001 - reported per item
                    last = exc
            return last

        return list(await asyncio.gather(*(attempt(d) for d in decisions)))

    # --- health ----------------------------------------------------------------------------------------------
    async def check_health_once(self) -> None:
        """One round of ``GET /health`` on every backend: removes failing ones, re-admits recovered ones."""
        results = await asyncio.gather(*(m.backend.ready() for m in self._members))
        for m, ok in zip(self._members, results):
            if ok:
                if m.healthy:
                    m.failures = 0
                else:
                    m.good_checks += 1
                    if m.good_checks >= SUCCESSES_TO_READMIT:
                        m.good_checks = 0
                        if await self._passes_identity(m):
                            self._mark_good(m)
            else:
                self._mark_bad(m)

    async def _passes_identity(self, m: _Member) -> bool:
        if self.readmit_check is None:
            return True
        try:
            problems = await self.readmit_check(m.backend)
        except Exception as exc:  # noqa: BLE001 - fail closed
            problems = [f"identity check failed ({type(exc).__name__}: {exc})"]
        if problems:
            log.warning("runtime backend %s is healthy again but stays out of rotation: %s", m.backend.url,
                        "; ".join(problems))
        return not problems

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self._health_interval)
            try:
                await self.check_health_once()
            except Exception:  # noqa: BLE001 - the loop must survive
                log.exception("backend health check failed")

    async def start(self) -> None:
        if self._monitor is None:
            self._monitor = asyncio.ensure_future(self._monitor_loop())

    async def probe(self) -> bool:
        """Start-up check: every backend answers an authenticated test decision (a misconfigured one is an error)."""
        results = await asyncio.gather(*(m.backend.probe() for m in self._members))
        for m, ok in zip(self._members, results):
            if not ok:
                log.error("runtime backend %s did not answer an authenticated test decision", m.backend.url)
        return all(results)

    async def close(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            self._monitor = None
        await asyncio.gather(*(m.backend.close() for m in self._members))


class S1HashBackend:
    """Deterministic pseudo-model for tests: ``salt`` plays the role of the weights."""

    def __init__(self, model: str = "s1-fast", salt: str = "reference", logit_noise: float = 0.0, seed: int = 0):
        self.model = model
        self.manifest = load_manifest(model)
        self.salt = salt
        self.noise = logit_noise
        self._rng_seed = seed
        self.calls = 0

    def logits(self, d: Decision) -> list[float]:
        base = json.dumps([d.state, d.question.author_dict()], sort_keys=True, ensure_ascii=False)
        out = []
        for k in d.question.keys:
            h = hashlib.sha256(f"{self.salt}\x00{base}\x00{k}".encode()).digest()
            z = int.from_bytes(h[:4], "big") / 2**32 * 12.0
            if self.noise:
                n = hashlib.sha256(f"noise{self._rng_seed}\x00{base}\x00{k}".encode()).digest()
                z += (int.from_bytes(n[:4], "big") / 2**32 - 0.5) * 2 * self.noise
            out.append(z)
        return out

    async def decide(self, decisions: Sequence[Decision]) -> list[dict[str, Any]]:
        self.calls += len(decisions)
        T = self.manifest.temperature
        answers = []
        for d in decisions:
            z = self.logits(d)
            m = max(z)
            e = [math.exp((x - m) / T) for x in z]
            s = math.fsum(e)
            probs = {k: v / s for k, v in zip(d.question.keys, e)}
            raw: dict[str, Any] = {"type": d.question.kind, "probabilities": probs}
            if d.question.kind == "noul":
                raw = {"type": "noul", "noul": probs["true"]}
            answers.append(normalize_answer(d.question, raw))
        return answers

    async def decide_each(self, decisions: Sequence[Decision], retries: int = 1) -> list[Any]:
        return await self.decide(decisions)

    async def probe(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def load_s1_backend(model: str, url: Optional[str], token: str, allow_dev: bool, max_inflight: int = 16,
                    endpoints: Sequence[tuple[str, str]] = ()) -> S1Backend:
    """One sidecar from ``url``, or a ``PooledBackend`` over ``endpoints`` ((url, token) pairs; an empty token falls
    back to ``token``). A single endpoint is a plain ``SidecarBackend``."""
    if endpoints:
        members = []
        for ep_url, ep_token in endpoints:
            if not (ep_token or token):
                raise ValueError("REAX_RUNTIME_TOKEN_FILE must name a non-empty token file shared with the runtime sidecar")
            members.append(SidecarBackend(model, ep_url, ep_token or token, max_inflight=max_inflight))
        return members[0] if len(members) == 1 else PooledBackend(members)
    if url == "hash" and allow_dev:
        return S1HashBackend(model)
    if not url:
        var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL"
        raise ValueError(f"{var} is required (the loopback URL of the {model} runtime sidecar)")
    if not token:
        raise ValueError("REAX_RUNTIME_TOKEN_FILE must name a non-empty token file shared with the runtime sidecar")
    return SidecarBackend(model, url, token, max_inflight=max_inflight)
