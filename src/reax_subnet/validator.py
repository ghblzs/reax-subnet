"""The REAX validator loop.

Each round:
1. read the metagraph (skip the round if the chain is unreachable),
2. forget UIDs that left and reset UIDs whose hotkey changed,
3. draw fresh items *separately for every miner* and compute the reference
   distributions locally,
4. query all miners at the same moment (one signed batch each),
5. score every miner (see scoring.py) and update its moving average,
6. save state atomically, and set weights when the rate limit allows.

Fresh items per miner mean an operator running many hotkeys on one GPU must
compute every batch separately, and all batches arrive at once, so shared
hardware shows up as latency and timeouts.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from . import auth
from .backends import Backend
from .chain import Chain, Neuron, Snapshot
from .config import ValidatorConfig
from .items import generate_items
from .protocol import MAX_RESPONSE_BYTES, DecideRequest, ProtocolError, parse_response
from .scoring import compute_weights, round_score
from .state import ValidatorState, load, save
from .version import WEIGHTS_VERSION_KEY

log = logging.getLogger("reax.validator")
SET_WEIGHTS_TIMEOUT_S = 300


def ip_allowed(ip: str, allow_private: bool) -> bool:
    """Refuse to send requests to addresses inside our own network.

    Miners choose the IP they publish. Without this check a miner could make
    validators send requests to loopback, private or cloud metadata addresses.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_unspecified or addr.is_multicast or addr.is_reserved or addr.is_link_local:
        return False
    if allow_private:
        return True
    return addr.is_global


def endpoint_url(ip: str, port: int) -> str:
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}{auth.DECIDE_PATH}"


@dataclass
class QueryResult:
    uid: int
    answers: Optional[dict[str, tuple[float, ...]]]
    latency_s: float
    error: Optional[str] = None


async def _read_limited(resp: aiohttp.ClientResponse, limit: int) -> bytes:
    if resp.content_length is not None and resp.content_length > limit:
        raise ProtocolError("response too large")
    chunks = []
    size = 0
    async for chunk in resp.content.iter_chunked(16 * 1024):
        size += len(chunk)
        if size > limit:
            raise ProtocolError("response too large")
        chunks.append(chunk)
    return b"".join(chunks)


class Validator:
    def __init__(self, cfg: ValidatorConfig, wallet: Any, chain: Chain, backend: Backend):
        self.cfg = cfg
        self.wallet = wallet
        self.hotkey: str = wallet.hotkey.ss58_address
        self.chain = chain
        self.backend = backend
        self.params = cfg.scoring
        self.state: ValidatorState = load(cfg.state_path, cfg.common.netuid, self.hotkey)
        self.nonce_store = auth.NonceStore(retention=cfg.request_timeout_s + 60.0)
        self.last_weights: dict[int, float] = {}
        self._last_neuron_count = 0
        self._shrunk_count: Optional[int] = None
        self._weights_task: Optional[asyncio.Future] = None

    # ------------------------------------------------------------------ query
    async def _query(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, neuron: Neuron, req: DecideRequest
    ) -> QueryResult:
        assert neuron.ip is not None and neuron.port is not None
        body = req.to_bytes()
        async with sem:
            headers = auth.sign_request(self.wallet, body, neuron.hotkey)
            headers["Content-Type"] = "application/json"
            started = time.monotonic()
            try:
                async with session.post(
                    endpoint_url(neuron.ip, neuron.port),
                    data=body,
                    headers=headers,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=self.cfg.request_timeout_s),
                ) as resp:
                    if resp.status != 200:
                        return QueryResult(neuron.uid, None, math.inf, f"http {resp.status}")
                    raw = await _read_limited(resp, MAX_RESPONSE_BYTES)
                    latency = time.monotonic() - started
                    auth.verify_response(
                        resp.headers, raw, req.request_id, self.hotkey, neuron.hotkey,
                        self.nonce_store, max_age=self.cfg.request_timeout_s + 10.0,
                    )
                    answers = parse_response(raw, req)
                    return QueryResult(neuron.uid, answers, latency)
            except asyncio.TimeoutError:
                return QueryResult(neuron.uid, None, math.inf, "timeout")
            except auth.AuthError as exc:
                return QueryResult(neuron.uid, None, math.inf, f"auth: {type(exc).__name__}")
            except ProtocolError as exc:
                return QueryResult(neuron.uid, None, math.inf, f"protocol: {exc}")
            except (aiohttp.ClientError, OSError, ValueError) as exc:
                return QueryResult(neuron.uid, None, math.inf, f"network: {type(exc).__name__}")
            except Exception as exc:  # a miner must never be able to abort the round
                log.warning("unexpected error querying uid %d: %r", neuron.uid, exc)
                return QueryResult(neuron.uid, None, math.inf, f"internal: {type(exc).__name__}")

    # ------------------------------------------------------------------ round
    def targets(self, snap: Snapshot) -> list[Neuron]:
        out = []
        for n in snap.neurons:
            if n.hotkey == self.hotkey or n.uid == snap.owner_uid:
                continue
            if n.ip is None or n.port is None or not ip_allowed(n.ip, self.cfg.allow_private_ips):
                continue
            out.append(n)
        return out

    async def run_round(self, session: aiohttp.ClientSession) -> Optional[dict[str, Any]]:
        try:
            snap = await asyncio.to_thread(self.chain.snapshot)
        except Exception as exc:
            log.warning("chain unavailable, skipping round: %s", exc)
            return None
        known = self._last_neuron_count
        count = len(snap.neurons)
        if count == 0:
            log.warning("metagraph read returned no neurons; skipping round")
            return None
        if known and count < 0.9 * known and self._shrunk_count != count:
            # UIDs are normally recycled, not removed, so a sudden shrink is more likely a
            # bad read than an owner trim. Accept it only when the next read confirms it.
            self._shrunk_count = count
            log.warning("metagraph shrank from %d to %d neurons; skipping round until a second read confirms", known, count)
            return None
        self._shrunk_count = None
        self._last_neuron_count = count
        reset = self.state.sync(snap)
        if reset:
            log.info("hotkey changed on uids %s; their scores were reset", reset)

        targets = self.targets(snap)
        requests: dict[int, DecideRequest] = {}
        references: dict[int, dict[str, tuple[float, ...]]] = {}
        for n in targets:
            items = generate_items(self.cfg.items_per_round)
            requests[n.uid] = DecideRequest(secrets.token_hex(16), self.backend.model_alias, tuple(items))
        # Reference computation happens before any miner is contacted, so all
        # queries leave at the same time and see no validator-side delay.
        try:
            for uid, req in requests.items():
                probs = await asyncio.to_thread(self.backend.predict, list(req.items))
                references[uid] = {it.item_id: p for it, p in zip(req.items, probs)}
        except Exception:
            log.exception("reference computation failed; skipping round without scoring anyone")
            return None

        sem = asyncio.Semaphore(self.cfg.max_parallel)
        gathered = await asyncio.gather(
            *(self._query(session, sem, n, requests[n.uid]) for n in targets), return_exceptions=True
        )
        results = [
            r if isinstance(r, QueryResult) else QueryResult(n.uid, None, math.inf, f"internal: {type(r).__name__}")
            for n, r in zip(targets, gathered)
        ]

        queried = set()
        summary = {"block": snap.block, "targets": len(targets), "ok": 0, "failed": 0, "scores": {}}
        for res in results:
            queried.add(res.uid)
            score, fidelity = round_score(res.answers, references[res.uid], res.latency_s, self.params)
            self.state.update(res.uid, score, fidelity, self.params.ema_alpha)
            summary["ok" if res.answers is not None else "failed"] += 1
            summary["scores"][res.uid] = {
                "score": round(score, 4),
                "fidelity": round(fidelity, 3),
                "latency_s": None if math.isinf(res.latency_s) else round(res.latency_s, 3),
                "error": res.error,
            }
            log.info(
                "uid %d: score %.3f fidelity %.3f latency %s ema %.3f%s",
                res.uid, score, fidelity, "-" if math.isinf(res.latency_s) else f"{res.latency_s:.2f}s",
                self.state.miners[res.uid].ema, f" error={res.error}" if res.error else "",
            )
        # Registered neurons without a reachable endpoint earn nothing this round.
        for n in snap.neurons:
            if n.uid not in queried and n.hotkey != self.hotkey and n.uid != snap.owner_uid:
                self.state.update(n.uid, 0.0, 0.0, self.params.ema_alpha)
        self.state.rounds += 1
        try:
            save(self.state, self.cfg.state_path)
        except OSError as exc:
            log.error("could not save state: %s", exc)
        log.info(
            "round %d at block %d: %d miners queried, %d answered, %d failed",
            self.state.rounds, snap.block, len(targets), summary["ok"], summary["failed"],
        )
        await self.maybe_set_weights(snap)
        return summary

    # ---------------------------------------------------------------- weights
    async def maybe_set_weights(self, snap: Snapshot) -> bool:
        if self.state.rounds < self.cfg.warmup_rounds:
            # A fresh validator has near-zero averages for everyone; submitting now
            # would hand the whole interval to the owner UID.
            log.info("warm-up: %d/%d rounds before the first weights", self.state.rounds, self.cfg.warmup_rounds)
            return False
        if self._weights_task is not None and not self._weights_task.done():
            log.warning("previous set_weights call still running; not submitting again")
            return False
        weights = self.compute_weights(snap)
        if not weights:
            log.warning("no qualifying miner and no owner uid; not setting weights")
            return False
        self.last_weights = weights
        if not self.cfg.set_weights:
            log.info("dry run (REAX_SET_WEIGHTS=0): weights %s", _fmt(weights))
            return False
        my_uid = snap.uid_of(self.hotkey)
        if my_uid is None:
            log.warning("validator hotkey is not registered on subnet %d; cannot set weights", self.cfg.common.netuid)
            return False
        try:
            since = await asyncio.to_thread(self.chain.blocks_since_last_update, my_uid)
            limit = await asyncio.to_thread(self.chain.weights_rate_limit)
        except Exception as exc:
            log.warning("could not read weight rate limit: %s", exc)
            return False
        if since < max(limit, self.cfg.weights_interval_blocks):
            return False
        uids = sorted(weights)
        self._weights_task = asyncio.ensure_future(
            asyncio.to_thread(self.chain.set_weights, uids, [weights[u] for u in uids], WEIGHTS_VERSION_KEY)
        )
        try:
            # Bounded wait so a stalled chain cannot freeze the scoring loop; the
            # submission thread keeps running and blocks new submissions until it ends.
            await asyncio.wait_for(asyncio.shield(self._weights_task), timeout=SET_WEIGHTS_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("set_weights still pending after %ds; continuing with scoring", SET_WEIGHTS_TIMEOUT_S)
            return False
        except Exception as exc:
            log.warning("setting weights failed, will retry next round: %s", exc)
            return False
        log.info("weights set at block %d: %s", snap.block, _fmt(weights))
        return True

    def compute_weights(self, snap: Snapshot) -> dict[int, float]:
        return compute_weights(self.state.ema(), snap.owner_uid, self.params)

    async def run_forever(self) -> None:
        connector = aiohttp.TCPConnector(limit=self.cfg.max_parallel, force_close=True)
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            while True:
                started = time.monotonic()
                try:
                    await self.run_round(session)
                except Exception:
                    log.exception("round failed unexpectedly")
                elapsed = time.monotonic() - started
                await asyncio.sleep(max(1.0, self.cfg.round_interval_s - elapsed))


def _fmt(weights: dict[int, float]) -> str:
    return ", ".join(f"{uid}:{w:.4f}" for uid, w in sorted(weights.items()))
