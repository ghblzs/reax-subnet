"""The REAX miner: an HTTP endpoint that answers signed validator requests.

Request handling order (cheapest checks first):
1. size limit, 2. claimed sender must be a current validator (permit and
minimum stake), 3. btauth signature, freshness and replay, 4. per-validator
concurrency, 5. strict payload parsing, 6. inference under a global
concurrency limit, 7. signed response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from aiohttp import web

from . import auth
from .backends import Backend
from .chain import Chain, Snapshot
from .protocol import MAX_REQUEST_BYTES, DecideResponse, ProtocolError, parse_request
from .s1protocol import MAX_REQUEST_BYTES as S1_MAX_REQUEST_BYTES
from .s1protocol import PROTOCOL_S1, S1_DECIDE_PATH, S1Response, parse_s1_request
from .version import PROTOCOL, __version__

log = logging.getLogger("reax.miner")


def allowed_validators(snapshot: Snapshot, min_stake: float) -> frozenset[str]:
    return frozenset(n.hotkey for n in snapshot.neurons if n.validator_permit and n.stake >= min_stake)


class Miner:
    path = auth.DECIDE_PATH
    max_request_bytes = MAX_REQUEST_BYTES

    def __init__(
        self,
        wallet: Any,
        backend: Backend,
        *,
        max_concurrent: int = 4,
        per_caller_concurrency: int = 2,
        per_caller_per_minute: int = 20,
        max_queue: Optional[int] = None,
        body_timeout_s: float = 5.0,
        max_pending_per_ip: int = 8,
        max_pending_total: int = 4096,
    ):
        self.wallet = wallet
        self.hotkey: str = wallet.hotkey.ss58_address
        self.backend = backend
        self.validators: Optional[frozenset[str]] = None  # None = not loaded yet
        self.nonce_store = auth.NonceStore(retention=60.0)
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_queue = max_queue if max_queue is not None else 4 * max_concurrent
        self._waiting = 0
        self._per_caller = per_caller_concurrency
        self._inflight: dict[str, int] = defaultdict(int)
        # Sliding one-minute window of accepted request times per validator.
        self._per_minute = per_caller_per_minute
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        # Bodies are read under a deadline and with a cap on unfinished reads, so
        # slow-drip clients cannot hold connections open.
        self._body_timeout = body_timeout_s
        self._max_pending_ip = max_pending_per_ip
        self._max_pending_total = max_pending_total
        self._pending: dict[str, int] = defaultdict(int)
        self._pending_total = 0
        self._last_warning: dict[int, float] = {}
        self.stats = {"ok": 0, "rejected": 0, "errors": 0}
        self.rejections: dict[int, int] = defaultdict(int)

    def set_validators(self, hotkeys: frozenset[str]) -> None:
        self.validators = hotkeys

    def app(self) -> web.Application:
        app = web.Application(client_max_size=self.max_request_bytes)
        app.router.add_post(self.path, self.handle_decide)
        app.router.add_get("/health", self.handle_health)
        return app

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(self.health_info())

    def health_info(self) -> dict:
        return {"protocol": PROTOCOL, "version": __version__, "model": self.backend.model_alias}

    def _reject(self, status: int, reason: str) -> web.Response:
        self.stats["rejected"] += 1
        self.rejections[status] += 1
        now = time.monotonic()
        if status in (401, 503) and now - self._last_warning.get(status, -1e9) > 60.0:
            # Rate-limited so floods cannot fill the log, but operators see e.g. clock-skew problems.
            self._last_warning[status] = now
            log.warning("rejected a request with %d: %s (further ones this minute are counted only)", status, reason)
        else:
            log.debug("rejected with %d: %s", status, reason)
        return web.json_response({"error": reason}, status=status)

    async def handle_decide(self, request: web.Request) -> web.Response:
        if request.content_length is None or request.content_length > self.max_request_bytes:
            return self._reject(413, "body missing or too large")
        if self.validators is None:
            return self._reject(503, "metagraph not loaded yet")
        claimed = auth.claimed_hotkey(request.headers)
        if not claimed or claimed not in self.validators:
            return self._reject(403, "sender is not a validator on this subnet")
        peer = request.remote or "?"
        if self._pending[peer] >= self._max_pending_ip or self._pending_total >= self._max_pending_total:
            return self._reject(503, "too many unfinished requests")
        self._pending[peer] += 1
        self._pending_total += 1
        try:
            body = await asyncio.wait_for(request.read(), timeout=self._body_timeout)
        except asyncio.TimeoutError:
            return self._reject(408, "request body too slow")
        finally:
            self._pending_total -= 1
            self._pending[peer] -= 1
            if self._pending[peer] <= 0:
                del self._pending[peer]
        try:
            caller = auth.verify_request(request.headers, body, self.hotkey, self.nonce_store, path=self.path)
        except auth.AuthError as exc:
            return self._reject(401, f"authentication failed: {type(exc).__name__}")
        # The verified signer must be the sender that passed the allow-list.
        if caller != claimed or caller not in self.validators:
            return self._reject(403, "sender is not a validator on this subnet")
        if self._inflight[caller] >= self._per_caller:
            return self._reject(429, "too many concurrent requests from this validator")
        now = time.monotonic()
        recent = self._recent[caller]
        while recent and recent[0] <= now - 60.0:
            recent.popleft()
        if len(recent) >= self._per_minute:
            return self._reject(429, "request rate limit for this validator exceeded")
        recent.append(now)
        self._inflight[caller] += 1
        try:
            return await self._decide(caller, body)
        finally:
            self._inflight[caller] -= 1
            if self._inflight[caller] <= 0:
                del self._inflight[caller]

    async def _decide(self, caller: str, body: bytes) -> web.Response:
        try:
            req = parse_request(body)
        except ProtocolError as exc:
            return self._reject(400, str(exc))
        if req.model != self.backend.model_alias:
            return self._reject(400, f"this miner serves {self.backend.model_alias}")
        if self._waiting >= self._max_queue:
            return self._reject(503, "miner is overloaded")
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
        started = time.monotonic()
        # The slot is released when inference really finishes, even if the
        # validator disconnects and this handler is cancelled meanwhile.
        task = asyncio.ensure_future(asyncio.to_thread(self.backend.predict, list(req.items)))
        task.add_done_callback(lambda _t: self._sem.release())
        try:
            probs = await asyncio.shield(task)
            if len(probs) != len(req.items):
                raise RuntimeError("backend returned the wrong number of results")
        except Exception:
            self.stats["errors"] += 1
            log.exception("inference failed")
            return web.json_response({"error": "inference failed"}, status=500)
        resp = DecideResponse(
            request_id=req.request_id,
            results=tuple((it.item_id, tuple(p)) for it, p in zip(req.items, probs)),
            miner_version=__version__,
        )
        raw = resp.to_bytes()
        headers = auth.sign_response(self.wallet, raw, req.request_id, caller)
        headers["Content-Type"] = "application/json"
        self.stats["ok"] += 1
        log.debug("answered %d items for %s in %.2fs", len(req.items), caller[:8], time.monotonic() - started)
        return web.Response(body=raw, headers=headers)


class S1Miner(Miner):
    """reax-s1 miner: answers typed decisions with the pinned reference model in its runtime sidecar."""

    path = S1_DECIDE_PATH
    max_request_bytes = S1_MAX_REQUEST_BYTES

    def health_info(self) -> dict:
        return {"protocol": PROTOCOL_S1, "version": __version__, "model": self.backend.model,
                "manifest_id": self.backend.manifest.manifest_id}

    async def _decide(self, caller: str, body: bytes) -> web.Response:
        try:
            req = parse_s1_request(body)
        except ProtocolError as exc:
            return self._reject(400, str(exc))
        if req.model != self.backend.model or req.manifest_id != self.backend.manifest.manifest_id:
            return self._reject(400, f"this miner serves {self.backend.model} manifest {self.backend.manifest.manifest_id}")
        if self._waiting >= self._max_queue:
            return self._reject(503, "miner is overloaded")
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
        started = time.monotonic()
        task = asyncio.ensure_future(self.backend.decide(list(req.decisions)))
        task.add_done_callback(lambda _t: self._sem.release())
        try:
            answers = await asyncio.shield(task)
            if len(answers) != len(req.decisions):
                raise RuntimeError("backend returned the wrong number of answers")
        except Exception:
            self.stats["errors"] += 1
            log.exception("inference failed")
            return web.json_response({"error": "inference failed"}, status=500)
        resp = S1Response(
            request_id=req.request_id,
            miner_version=__version__,
            results=tuple((d.ident, {d.question.ident: a}) for d, a in zip(req.decisions, answers)),
        )
        raw = resp.to_bytes()
        headers = auth.sign_response(self.wallet, raw, req.request_id, caller, path=self.path)
        headers["Content-Type"] = "application/json"
        self.stats["ok"] += 1
        log.debug("answered %d decisions for %s in %.2fs", len(req.decisions), caller[:8], time.monotonic() - started)
        return web.Response(body=raw, headers=headers)


async def refresh_validators_forever(
    miner: Miner, chain: Chain, min_stake: float, interval_s: int, on_snapshot: Optional[Callable[[Snapshot], None]] = None
) -> None:
    while True:
        try:
            snap = await asyncio.to_thread(chain.snapshot)
            miner.set_validators(allowed_validators(snap, min_stake))
            if snap.uid_of(miner.hotkey) is None:
                log.warning("hotkey %s is not registered on this subnet", miner.hotkey)
            if on_snapshot is not None:
                on_snapshot(snap)
            log.info(
                "metagraph block %d: %d neurons, %d accepted validators; served ok=%d rejected=%d %s errors=%d",
                snap.block, len(snap.neurons), len(miner.validators or ()), miner.stats["ok"],
                miner.stats["rejected"], dict(sorted(miner.rejections.items())), miner.stats["errors"],
            )
        except Exception as exc:  # keep serving with the last known validator set
            log.warning("metagraph refresh failed: %s", exc)
        await asyncio.sleep(interval_s)
