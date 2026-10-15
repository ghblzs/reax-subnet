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
import math
import secrets
import time
from typing import Any, Optional, Protocol, Sequence

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
    pass


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
                            raise SidecarError(f"runtime returned HTTP {r.status}")
                        raw = await _read_capped(r.content, MAX_RUNTIME_RESPONSE_BYTES)
                    break
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                    if attempt:
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


def load_s1_backend(model: str, url: Optional[str], token: str, allow_dev: bool, max_inflight: int = 16) -> S1Backend:
    if url == "hash" and allow_dev:
        return S1HashBackend(model)
    if not url:
        var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL"
        raise ValueError(f"{var} is required (the loopback URL of the {model} runtime sidecar)")
    if not token:
        raise ValueError("REAX_RUNTIME_TOKEN_FILE must name a non-empty token file shared with the runtime sidecar")
    return SidecarBackend(model, url, token, max_inflight=max_inflight)
