"""Read-only REAX subnet front end for a private client contract."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import math
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web

from . import auth
from .config import ConfigError, _bool, _float, _int, load_validator
from .manifest import load_manifest, parse_commitment
from .protocol import ProtocolError, _loads
from .s1contract import ContractError, parse_decision
from .s1protocol import MAX_DECISIONS, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, S1_DECIDE_PATH, S1Request, parse_s1_response
from .s1scoring import score_round
from .s1validator import POOL_PARAMS, s1_endpoint_url
from .state import STATE_VERSION
from .router_tokens import InputTokenLimitError
from .validator import _read_limited, ip_allowed

log = logging.getLogger("reax.router")
MAX_SNAPSHOT_AGE = 3600


def read_token(path):
    # Check the opened inode, not a stat followed by a second open.
    with open(path, "rb") as fh:
        info = os.fstat(fh.fileno())
        token = fh.read(4097).strip()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or not 32 <= len(token) <= 4096:
        raise ConfigError("router token must be a regular 0600 file containing 32..4096 bytes")
    try:
        return token.decode("ascii")
    except UnicodeDecodeError:
        raise ConfigError("router token must be ASCII") from None


@dataclass(frozen=True)
class RouterConfig:
    token: str
    revision: str
    tokenizer_version: str
    models: tuple[str, ...] = ("s1-fast",)
    enabled: bool = True
    min_miners: int = 1
    min_ema: float = 0.5
    inflight_cap: int = 4
    timeout_s: float = 3.0
    shadow_rate: float = 0.05
    strikes_file: str = "router-strikes.jsonl"
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls, env=os.environ):
        token = read_token(env.get("REAX_ROUTER_TOKEN_FILE", ""))
        models = tuple(x.strip() for x in env.get("REAX_ROUTER_MODELS", "s1-fast").split(","))
        if not models or len(set(models)) != len(models) or any(m not in ("s1-fast", "s1-pro") for m in models):
            raise ConfigError("invalid router models")
        for m in models:
            load_manifest(m)
        revision = env.get("REAX_ROUTER_MODEL_REVISION", "")
        tokenizer = env.get("REAX_ROUTER_TOKENIZER_VERSION", "")
        if not revision or not tokenizer:
            raise ConfigError("router model revision and tokenizer version are required (client pins)")
        return cls(token, revision, tokenizer, models,
                   _bool(env, "REAX_ROUTER_ENABLED", True),
                   _int(env, "REAX_ROUTER_MIN_MINERS", 1, 1, 10000),
                   _float(env, "REAX_ROUTER_MIN_EMA", 0.5, 0, 1),
                   _int(env, "REAX_ROUTER_INFLIGHT_CAP", 4, 1, 1024),
                   _float(env, "REAX_ROUTER_TIMEOUT_S", 3.0, 0.01, 120),
                   _float(env, "REAX_ROUTER_SHADOW_RATE", 0.05, 0, 1),
                   os.path.expanduser(env.get("REAX_ROUTER_STRIKES_FILE", "~/.reax/router-strikes.jsonl")),
                   env.get("REAX_ROUTER_HOST", "127.0.0.1"),
                   _int(env, "REAX_ROUTER_PORT", 8080, 1, 65535))


@dataclass
class MinerHealth:
    inflight: int = 0
    failures: int = 0
    open_until: float = 0.0
    latency: float = math.inf


class Router:
    def __init__(self, cfg, validator_cfg, wallet, chain, counter=None, backends=None):
        self.cfg, self.validator_cfg = cfg, validator_cfg
        self.wallet, self.chain, self.counter = wallet, chain, counter
        self.backends = dict(backends or {})
        self.hotkey = wallet.hotkey.ss58_address
        self.manifests = {m: load_manifest(m) for m in cfg.models}
        self.miners = {m: [] for m in cfg.models}
        self.fresh_at = 0.0
        self.health = {}  # hotkey, so a recycled UID cannot inherit a breaker
        self.nonce_store = auth.NonceStore()
        self.session = None
        self.tasks = set()
        self._restore_strikes()

    def _restore_strikes(self):
        path = Path(self.cfg.strikes_file)
        if not path.exists():
            return
        try:
            with path.open() as fh:
                for line in fh:
                    entry = json.loads(line)
                    struck_at = entry["time"]
                    if (isinstance(struck_at, bool) or not isinstance(struck_at, (int, float))
                            or not math.isfinite(struck_at) or not isinstance(entry["hotkey"], str)):
                        raise ValueError("invalid strike entry")
                    until = struck_at + 86400
                    if until > time.time():
                        h = self.health.setdefault(entry["hotkey"], MinerHealth())
                        h.open_until = max(h.open_until, until)
        except (OSError, ValueError, KeyError, TypeError):
            # A damaged strike ledger must not silently allow previously excluded miners.
            raise ConfigError("unreadable router strike ledger") from None

    def _read_scores(self):
        # state.load deliberately renames corrupt files; routers MUST NOT use it.
        with open(self.validator_cfg.state_path, "rb") as fh:
            info = os.fstat(fh.fileno())
            raw = _loads(fh.read(4 * 1024 * 1024 + 1), 4 * 1024 * 1024)
        if (raw.get("version") != STATE_VERSION or raw.get("netuid") != self.validator_cfg.common.netuid
                or raw.get("hotkey") != self.hotkey or not isinstance(raw.get("miners"), dict)):
            raise ValueError("foreign validator state")
        saved = raw.get("saved_at")
        if isinstance(saved, bool) or not isinstance(saved, (int, float)) or not math.isfinite(saved):
            raise ValueError("invalid score timestamp")
        fresh = min(float(saved), info.st_mtime, time.time())
        if saved > time.time() + 60 or time.time() - fresh > MAX_SNAPSHOT_AGE:
            raise ValueError("stale scores")
        return raw["miners"], fresh

    async def refresh(self):
        try:
            snap = await asyncio.to_thread(self.chain.snapshot)
            scores, fresh = await asyncio.to_thread(self._read_scores)
            selected = {m: [] for m in self.cfg.models}
            for n in snap.neurons:
                role = parse_commitment(n.commitment)
                if not role or role[0] not in self.manifests or role[1] != self.manifests[role[0]].manifest_id:
                    continue
                if (n.uid == snap.owner_uid or n.hotkey == self.hotkey or not n.ip or not n.port
                        or not ip_allowed(n.ip, self.validator_cfg.allow_private_ips)):
                    continue
                score = scores.get(str(n.uid), {})
                ema = score.get("ema")
                if (score.get("hotkey") != n.hotkey or score.get("role") != f"{role[0]}:{role[1]}"
                        or isinstance(ema, bool) or not isinstance(ema, (int, float))
                        or not math.isfinite(ema) or not self.cfg.min_ema <= ema <= 1):
                    continue
                # Current state.py has only cumulative strikes. Future timestamp fields are honored.
                struck = score.get("last_strike_at", score.get("last_strike", 0))
                if (isinstance(struck, bool) or not isinstance(struck, (int, float))
                        or not math.isfinite(struck) or (struck and time.time() - struck < 86400)):
                    continue
                selected[role[0]].append((n, float(ema)))
                self.health.setdefault(n.hotkey, MinerHealth())
            self.miners, self.fresh_at = selected, fresh
        except Exception:
            log.warning("snapshot refresh outcome=unavailable")
            # The last good joined chain/score snapshot expires based on score age, never renewed on failure.

    def candidates(self, model):
        if (not self.cfg.enabled or model not in self.miners or self.fresh_at <= 0
                or time.time() - self.fresh_at > MAX_SNAPSHOT_AGE):
            return []
        return sorted(((n, s) for n, s in self.miners[model]
                       if self.health[n.hotkey].open_until <= time.time()),
                      key=lambda ns: (-ns[1], self.health[ns[0].hotkey].latency, ns[0].uid))

    def ready(self, model):
        return (self.counter is not None and model in getattr(self.counter, "models", self.cfg.models)
                and len(self.candidates(model)) >= self.cfg.min_miners)

    async def _query(self, n, req, timeout):
        body = req.to_bytes()
        headers = auth.sign_request(self.wallet, body, n.hotkey, path=S1_DECIDE_PATH)
        headers["Content-Type"] = "application/json"
        async with self.session.post(s1_endpoint_url(n.ip, n.port), data=body, headers=headers,
                                     allow_redirects=False, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise ProtocolError("miner unavailable")
            raw = await _read_limited(resp, MAX_RESPONSE_BYTES)
            auth.verify_response(resp.headers, raw, req.request_id, self.hotkey, n.hotkey,
                                 self.nonce_store, max_age=timeout + 10, path=S1_DECIDE_PATH)
            probabilities = parse_s1_response(raw, req)
            results = json.loads(raw)["results"]
            answers = {r["id"]: r["answers"]["q"] for r in results}
            # The router's normalization boundary is tighter than the subnet (1e-4 vs 1e-3).
            if any(abs(math.fsum(p) - 1) > 1e-4 for p in probabilities.values()):
                raise ProtocolError("probability sum exceeds router tolerance")
            return answers, probabilities

    async def shadow(self, n, req, probabilities):
        backend = self.backends.get(req.model)
        if backend is None:
            return
        d = secrets.choice(req.decisions)
        try:
            result = await asyncio.wait_for(backend.decide([d]), self.cfg.timeout_s)
            ref = {d.ident: tuple(result[0]["probabilities"][k] for k in d.question.keys)}
            outcome = score_round(probabilities, ref, 0, POOL_PARAMS[req.model])
            if not outcome.strike:
                return
        except Exception:
            log.warning("shadow uid=%d request_id=%s outcome=reference_unavailable", n.uid, req.request_id)
            return  # reference failures are never attributed to a miner
        self.health[n.hotkey].open_until = time.time() + 86400
        entry = {"time": time.time(), "uid": n.uid, "hotkey": n.hotkey,
                 "request_id": req.request_id, "model": req.model, "outcome": "shadow_strike"}
        # No decisions/probabilities/caller ids ever persist. Separate from validator state.
        try:
            path = Path(self.cfg.strikes_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            log.error("shadow uid=%d request_id=%s outcome=strike_persistence_failed", n.uid, req.request_id)
        log.warning("shadow uid=%d hotkey=%s request_id=%s outcome=strike", n.uid, n.hotkey, req.request_id)

    @staticmethod
    def error(status, message):
        return web.json_response({"error": {"message": message}}, status=status)

    async def decide(self, request):
        tokens = request.headers.getall("X-S1M-Internal-Token", [])
        if len(tokens) != 1 or not hmac.compare_digest(tokens[0].encode(), self.cfg.token.encode()):
            return self.error(401, "unauthorized")
        started = time.monotonic()
        try:
            body = _loads(await request.read(), MAX_REQUEST_BYTES)
            if not isinstance(body, dict):
                raise ValueError()
            if "images" in body:
                return self.error(422, "images are not supported")
            if (set(body) != {"request_id", "model", "state", "questions", "deadline_unix_ms"}
                    or not isinstance(body["request_id"], str) or not 1 <= len(body["request_id"]) <= 128
                    or not isinstance(body["model"], str) or body["model"] not in self.manifests):
                raise ValueError()
            deadline = body["deadline_unix_ms"]
            if isinstance(deadline, bool) or not isinstance(deadline, int) or not 0 < deadline <= (1 << 63) - 1:
                raise ValueError()
            questions = body["questions"]
            if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_DECISIONS:
                raise ValueError()
            # Both decision and question ids are replaced: even user-chosen qids can identify a caller.
            qids = list(questions)
            decisions = tuple(parse_decision({"id": secrets.token_hex(16), "state": body["state"],
                                               "questions": {"q": q}}, body["model"]) for q in questions.values())
            if any(not isinstance(qid, str) or not qid or len(qid.encode()) > 128 for qid in qids):
                raise ValueError()
        except (ValueError, TypeError, KeyError, ContractError, ProtocolError, web.HTTPRequestEntityTooLarge):
            return self.error(400, "bad request")
        model = body["model"]
        if not self.ready(model) or self.counter is None:
            return self.error(503, "no eligible miner")
        end = started + min(self.cfg.timeout_s, (deadline / 1000 - time.time() - 0.3))
        try:
            budget = end - time.monotonic()
            if budget <= 0:
                return self.error(503, "no eligible miner")
            count = await asyncio.wait_for(asyncio.to_thread(self.counter, model, decisions), budget)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RuntimeError()
        except InputTokenLimitError:
            return self.error(400, "input exceeds model token limit")
        except Exception:
            return self.error(503, "no eligible miner")
        req = S1Request(secrets.token_hex(16), model, self.manifests[model].manifest_id, decisions)
        candidates = self.candidates(model)
        attempts = 0
        for n, _score in candidates:
            h = self.health[n.hotkey]
            if h.inflight >= self.cfg.inflight_cap or h.open_until > time.time():
                continue
            budget = end - time.monotonic()
            if budget <= 0 or attempts >= 2:
                break
            attempts += 1
            # Reserve half for a sequential fallback when another miner is available.
            timeout = budget / 2 if attempts == 1 and len(candidates) > 1 else budget
            h.inflight += 1
            before = time.monotonic()
            try:
                answers, probabilities = await self._query(n, req, timeout)
                if time.monotonic() > end:
                    raise asyncio.TimeoutError()
                h.failures, h.latency = 0, time.monotonic() - before
                if (model in self.backends and len(self.tasks) < 32
                        and secrets.randbelow(1_000_000) < self.cfg.shadow_rate * 1_000_000):
                    task = asyncio.create_task(self.shadow(n, req, probabilities))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                log.info("request_id=%s uid=%d latency_ms=%.1f outcome=ok", req.request_id, n.uid,
                         (time.monotonic() - started) * 1000)
                return web.json_response({"request_id": body["request_id"], "model": model,
                    "answers": {qid: answers[d.ident] for qid, d in zip(qids, decisions)},
                    "usage": {"input_tokens": count, "output_tokens": 0, "decisions": len(decisions)},
                    "latency_ms": (time.monotonic() - started) * 1000,
                    "tokenizer_version": self.cfg.tokenizer_version, "model_revision": self.cfg.revision})
            except Exception:
                h.failures += 1
                if h.failures >= 3:
                    h.open_until = max(h.open_until, time.time() + 60)
                log.info("request_id=%s uid=%d latency_ms=%.1f outcome=failed", req.request_id, n.uid,
                         (time.monotonic() - before) * 1000)
            finally:
                h.inflight -= 1
        return self.error(503, "no eligible miner")

    async def _refresh_loop(self):
        while True:
            await asyncio.sleep(60)
            await self.refresh()

    async def _lifecycle(self, app):
        self.session = aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(limit=256))
        await self.refresh()
        loop = asyncio.create_task(self._refresh_loop())
        try:
            yield
        finally:
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop
            for t in self.tasks:
                t.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            await self.session.close()
            for backend in self.backends.values():
                await backend.close()

    async def health_ready(self, request):
        model = request.query.get("model")
        models = (model,) if model else self.cfg.models
        states = {m: {"state": "ready" if self.ready(m) else "unavailable", "revision": self.cfg.revision}
                  for m in models}
        ok = all(s["state"] == "ready" for s in states.values())
        return web.json_response({"ok": ok, "models": states}, status=200 if ok else 503)

    async def health_live(self, request):
        return web.json_response({"ok": True})

    def app(self):
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.cleanup_ctx.append(self._lifecycle)
        app.router.add_post("/internal/v1/decide", self.decide)
        app.router.add_get("/health/ready", self.health_ready)
        app.router.add_get("/health", self.health_live)
        return app


def main():
    from .chain import SdkChain
    from .cli import _s1_backend, _wallet, setup_logging
    from .router_tokens import PlumbTokenCounter

    try:
        cfg = RouterConfig.from_env()
        vcfg = load_validator()
        if vcfg.common.mechanism != "s1":
            raise ConfigError("router requires the s1 validator configuration")
        wallet, _ = _wallet(vcfg.common)
    except (ConfigError, OSError) as exc:
        raise SystemExit(f"router configuration error: {exc}") from None
    setup_logging(vcfg.common)
    counter = None
    try:
        counter = PlumbTokenCounter.from_directory(os.environ.get("REAX_ROUTER_TOKENIZER_DIR", ""))
    except Exception:
        log.error("token counter outcome=unavailable (decide will fail closed)")
    backends = {m: _s1_backend(vcfg.common, m) for m in cfg.models} if cfg.shadow_rate else {}
    chain = SdkChain(vcfg.common.network, vcfg.common.netuid, wallet, commitments=True)
    router = Router(cfg, vcfg, wallet, chain, counter, backends)
    # Disable aiohttp access logging, which would include URL/query data.
    web.run_app(router.app(), host=cfg.host, port=cfg.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
