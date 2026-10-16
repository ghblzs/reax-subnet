"""Read-only REAX subnet front end for a private client contract."""
from __future__ import annotations

import asyncio
import collections
import contextlib
import hmac
import json
import logging
import math
import os
import secrets
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

from . import auth
from .config import ConfigError, _bool, _float, _int, load_validator
from .manifest import load_manifest, parse_commitment
from .protocol import ProtocolError, _loads
from .s1contract import ContractError, parse_decision
from .s1protocol import (MAX_DECISIONS, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, S1_DECIDE_PATH, S1Request,
                         S1Response, parse_s1_response)
from .s1scoring import score_round
from .s1validator import POOL_PARAMS, s1_endpoint_url
from .state import STATE_VERSION
from .router_tokens import InputTokenLimitError
from .validator import _read_limited, ip_allowed

log = logging.getLogger("reax.router")
MAX_SNAPSHOT_AGE = 3600
STATUS_WINDOW_SECONDS = 86400
STATUS_WINDOW_MINUTES = STATUS_WINDOW_SECONDS // 60
CREDIT_WINDOW_MINUTES = 60  # window of the verified/served ratio
GREY_MIN_VERIFIED = 20  # verified answers inside 24 h before the grey rate can quarantine a hotkey
GREY_WILSON_Z = 1.96  # 95 % confidence


def judge(outcome, params):
    """Per-item verdict of one routed check: "strike" above tau_item, "grey" above tau_mean, else "match".

    The validator strikes on a round mean above tau_strike, which averages over its 8 items; a routed request has one
    item, so the mean line would turn honest per-item drift (0.5 and up, rare) into 24 h exclusions. The router uses the
    validator's per-item line (tau_item) for strikes; anything between tau_mean and tau_item feeds the grey statistics.
    """
    worst = outcome.max_distance
    if not math.isfinite(worst) or worst > params.tau_item:
        return "strike"
    return "grey" if worst > params.tau_mean else "match"


def wilson_lower(bad, total, z=GREY_WILSON_Z):
    """Lower bound of the Wilson score interval of a rate bad/total."""
    if total <= 0:
        return 0.0
    p = bad / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / (1 + z * z / total))
STATUS_COUNTERS = ("miner", "reference", "none", "verify_match", "verify_mismatch",
                   "verified", "sampled", "skipped", "skipped_reference_busy", "probation")


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
    min_ema: float = 0.9  # track record: a fresh perfect UID needs 22 rounds, a slow-but-correct one (0.8) never gets here
    min_rounds: int = 720  # scored rounds with the current hotkey+role; 720 x 120 s = 24 h of validator rounds
    strike_cooldown_s: float = 7 * 86400  # a struck miner stays out this long; strikes without a timestamp = out for good
    inflight_cap: int = 1  # miners allow 2 per caller; the validator's scoring rounds need the other slot
    timeout_s: float = 3.0
    shadow_rate: float = 0.05
    strikes_file: str = "router-strikes.jsonl"
    host: str = "127.0.0.1"
    port: int = 8080
    per_miner_per_minute: int = 12  # miners allow 20/min per caller, shared with the validator's own rounds
    verify_rate: float = 1.0  # share of requests verified synchronously against the reference before serving
    reference_fallback: bool = True  # answer from our own reference runtime when no miner answer can be served
    timeout_quarantine_s: float = 60.0  # a miner whose attempt timed out/disconnected may still hold its caller slot
    reference_concurrency: int = 2  # router-wide concurrent reference computations (one slot per decision)
    min_reference_budget_s: float = 0.2  # do not start a reference computation with less time than this left
    miner_timeout_s: float = 1.5  # per-attempt cap: a slow miner must not eat the budget the reference answer needs
    probation_answers: int = 20  # a miner's first N customer answers (per hotkey, in memory) are always verified
    verify_floor: float | None = None  # served-answer share that must be verified; None = verify_rate (no best-effort share)
    grey_quarantine_s: float = 3600.0  # a hotkey whose grey rate is significantly high stays out this long
    grey_rate_max: float = 0.10  # quarantine when the Wilson 95 % lower bound of the 24 h grey rate exceeds this
    max_unverified_run: int | None = None  # consecutive unverified serves per hotkey; None = ceil(1 / floor)
    repeat_strike_exclusion_s: float = 7 * 86400  # a second strike for the same hotkey inside 7 days excludes it this long
    busy_backoff_s: float = 5.0  # after a 429 the busy miner is skipped this long (no failure, no breaker)
    reference_concurrency_models: dict = field(default_factory=dict)  # per model; a model not listed uses the global value
    model_revisions: dict = field(default_factory=dict)  # per model; a model not listed uses `revision`
    tokenizer_versions: dict = field(default_factory=dict)

    @property
    def floor(self):
        """Verified share of served miner answers the router guarantees; never above verify_rate."""
        floor = self.verify_rate if self.verify_floor is None else self.verify_floor
        return min(floor, self.verify_rate)

    @property
    def unverified_run_cap(self):
        if self.max_unverified_run is not None:
            return self.max_unverified_run
        return math.ceil(1 / self.floor - 1e-9) if self.floor > 0 else 1 << 62

    def reference_slots(self, model):
        return self.reference_concurrency_models.get(model, self.reference_concurrency)

    def revision_for(self, model):
        return self.model_revisions.get(model, self.revision)

    def tokenizer_version_for(self, model):
        return self.tokenizer_versions.get(model, self.tokenizer_version)

    @classmethod
    def from_env(cls, env=os.environ):
        token = read_token(env.get("REAX_ROUTER_TOKEN_FILE", ""))
        models = tuple(x.strip() for x in env.get("REAX_ROUTER_MODELS", "s1-fast").split(","))
        if not models or len(set(models)) != len(models) or any(m not in ("s1-fast", "s1-pro") for m in models):
            raise ConfigError("invalid router models")
        for m in models:
            load_manifest(m)
        # The single vars are the s1-fast pins; every other model needs its own REAX_ROUTER_*_<MODEL> pair.
        revisions, tokenizers = {}, {}
        for m in models:
            suffix = m.upper().replace("-", "_")
            revisions[m] = env.get(f"REAX_ROUTER_MODEL_REVISION_{suffix}", "")
            tokenizers[m] = env.get(f"REAX_ROUTER_TOKENIZER_VERSION_{suffix}", "")
            if m == "s1-fast":
                revisions[m] = revisions[m] or env.get("REAX_ROUTER_MODEL_REVISION", "")
                tokenizers[m] = tokenizers[m] or env.get("REAX_ROUTER_TOKENIZER_VERSION", "")
            if not revisions[m] or not tokenizers[m]:
                raise ConfigError(f"router model revision and tokenizer version are required for {m} (client pins)")
        revision = revisions.get("s1-fast", "")
        tokenizer = tokenizers.get("s1-fast", "")
        verify_rate = _float(env, "REAX_ROUTER_VERIFY_RATE", 1.0, 0, 1)
        reference_concurrency = _int(env, "REAX_ROUTER_REFERENCE_CONCURRENCY", 2, 1, 16)
        floor_raw = env.get("REAX_ROUTER_VERIFY_FLOOR")
        if verify_rate > 0 and floor_raw is not None and _float(env, "REAX_ROUTER_VERIFY_FLOOR", verify_rate, 0, 1) <= 0:
            # A zero floor would switch off the credit budget and the unverified-run cap.
            raise ConfigError("REAX_ROUTER_VERIFY_FLOOR must be above 0 while REAX_ROUTER_VERIFY_RATE is above 0")
        return cls(token, revision, tokenizer, models,
                   _bool(env, "REAX_ROUTER_ENABLED", True),
                   _int(env, "REAX_ROUTER_MIN_MINERS", 1, 1, 10000),
                   _float(env, "REAX_ROUTER_MIN_EMA", 0.9, 0, 1),
                   _int(env, "REAX_ROUTER_MIN_ROUNDS", 720, 0, 10_000_000),
                   _float(env, "REAX_ROUTER_STRIKE_COOLDOWN_S", 7 * 86400, 0, 365 * 86400),
                   _int(env, "REAX_ROUTER_INFLIGHT_CAP", 1, 1, 1024),
                   _float(env, "REAX_ROUTER_TIMEOUT_S", 3.0, 0.01, 120),
                   _float(env, "REAX_ROUTER_SHADOW_RATE", 0.05, 0, 1),
                   os.path.expanduser(env.get("REAX_ROUTER_STRIKES_FILE", "~/.reax/router-strikes.jsonl")),
                   env.get("REAX_ROUTER_HOST", "127.0.0.1"),
                   _int(env, "REAX_ROUTER_PORT", 8080, 1, 65535),
                   per_miner_per_minute=_int(env, "REAX_ROUTER_PER_MINER_PER_MINUTE", 12, 1, 19),
                   verify_rate=verify_rate,
                   reference_fallback=_bool(env, "REAX_ROUTER_REFERENCE_FALLBACK", True),
                   timeout_quarantine_s=_float(env, "REAX_ROUTER_TIMEOUT_QUARANTINE_S", 60.0, 5, 600),
                   reference_concurrency=reference_concurrency,
                   miner_timeout_s=_float(env, "REAX_ROUTER_MINER_TIMEOUT_S", 1.5, 0.05, 120),
                   min_reference_budget_s=_float(env, "REAX_ROUTER_MIN_REFERENCE_BUDGET_S", 0.2, 0, 5),
                   probation_answers=_int(env, "REAX_ROUTER_PROBATION_ANSWERS", 20, 0, 100000),
                   verify_floor=_float(env, "REAX_ROUTER_VERIFY_FLOOR", verify_rate, 0, 1),
                   grey_quarantine_s=_float(env, "REAX_ROUTER_GREY_QUARANTINE_S", 3600.0, 0, 86400),
                   grey_rate_max=_float(env, "REAX_ROUTER_GREY_RATE_MAX", 0.10, 0, 1),
                   max_unverified_run=(_int(env, "REAX_ROUTER_MAX_UNVERIFIED_RUN", 1, 1, 100000)
                                       if env.get("REAX_ROUTER_MAX_UNVERIFIED_RUN") else None),
                   busy_backoff_s=_float(env, "REAX_ROUTER_BUSY_BACKOFF_S", 5.0, 0, 600),
                   repeat_strike_exclusion_s=_float(env, "REAX_ROUTER_REPEAT_STRIKE_EXCLUSION_S", 7 * 86400, 86400,
                                                    365 * 86400),
                   reference_concurrency_models={
                       m: _int(env, f"REAX_ROUTER_REFERENCE_CONCURRENCY_{m.upper().replace('-', '_')}",
                               reference_concurrency, 1, 16) for m in models},
                   model_revisions=revisions, tokenizer_versions=tokenizers)


@dataclass
class MinerHealth:
    inflight: int = 0
    failures: int = 0
    open_until: float = 0.0
    latency: float = math.inf
    recent: collections.deque = field(default_factory=collections.deque)  # router request times, sliding 60 s
    busy_until: float = 0.0  # a 429 backs the miner off for busy_backoff_s


class Rolling:
    """Two counters over a sliding window of whole minutes."""

    def __init__(self, minutes):
        self.minutes = minutes
        self.buckets = collections.deque()  # [minute, a, b]
        self.a = self.b = 0

    def totals(self, now):
        minute = int(now // 60)
        while self.buckets and self.buckets[0][0] <= minute - self.minutes:
            _m, a, b = self.buckets.popleft()
            self.a -= a
            self.b -= b
        return self.a, self.b

    def add(self, now, a, b):
        self.totals(now)
        minute = int(now // 60)
        if not self.buckets or self.buckets[-1][0] != minute:
            self.buckets.append([minute, 0, 0])
        self.buckets[-1][1] += a
        self.buckets[-1][2] += b
        self.a += a
        self.b += b


class Credit(Rolling):
    """Customer answers per hotkey over the last hour: served (miner answers sent) and verified_served, plus the run
    of consecutive unverified serves since the last verified one."""

    def __init__(self, minutes=CREDIT_WINDOW_MINUTES):
        super().__init__(minutes)
        self.run = 0

    served = property(lambda self: self.a)
    verified = property(lambda self: self.b)

    def add(self, now, verified):
        super().add(now, 1, 1 if verified else 0)
        self.run = 0 if verified else self.run + 1


class ReferenceUnavailable(RuntimeError):
    """No reference answer can be produced for this request (no backend, no free slots, failure or timeout)."""


class MinerBusy(ProtocolError):
    """Miner answered 429: skip it, but it is not a failure for the breaker."""


class Router:
    def __init__(self, cfg, validator_cfg, wallet, chain, counter=None, backends=None):
        self.cfg, self.validator_cfg = cfg, validator_cfg
        self.wallet, self.chain, self.counter = wallet, chain, counter
        self.backends = dict(backends or {})
        self.hotkey = wallet.hotkey.ss58_address
        self.manifests = {m: load_manifest(m) for m in cfg.models}
        self.miners = {m: [] for m in cfg.models}
        self.excluded = {m: collections.Counter() for m in cfg.models}  # last refresh: seen + reasons
        self.fresh_at = 0.0
        self.health = {}  # hotkey, so a recycled UID cannot inherit a breaker
        self.nonce_store = auth.NonceStore()
        self.session = None
        self.tasks = set()
        self.reference_used = collections.Counter()  # reference computations in flight per model, in decisions
        self.credits = {}  # hotkey -> Credit
        self.greys = {}  # hotkey -> Rolling(24 h): verification checks (sync and shadow), grey results
        self.started_at = time.time()
        self.counts = collections.Counter()  # payload-free metric counters keyed by (name, *labels)
        self.status_buckets = {m: collections.deque(maxlen=STATUS_WINDOW_MINUTES) for m in cfg.models}
        self.latency = {}  # (model, served_by) -> [count, sum_seconds]
        self.probation = {}  # hotkey -> verified-OK customer answers since the last reset; in memory, restart = all in probation
        self.strike_times = {}  # hotkey -> (time, exclusion end) of ledger strikes, for the repeat-strike rule
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
                    until = entry.get("until", struck_at + 86400)  # entries from before F5 have no "until"
                    if isinstance(until, bool) or not isinstance(until, (int, float)) or not math.isfinite(until):
                        raise ValueError("invalid strike entry")
                    until = max(until, struck_at + 86400)  # an edited ledger line cannot shorten the base exclusion
                    self.strike_times.setdefault(entry["hotkey"], []).append((struck_at, until))
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

    def exclusion(self, score, hotkey, role, now):
        """Why this miner may not get customer traffic, or None when it has earned it.

        Track record: same hotkey and role as the validator scored, at least min_rounds scored rounds, EMA >=
        min_ema, a passing latest round and no strike inside the cooldown. Anything unreadable is excluded.
        """
        def num(v):
            return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)

        if score.get("hotkey") != hotkey or score.get("role") != f"{role[0]}:{role[1]}":
            return "no_score"
        ema, last, rounds, strikes = score.get("ema"), score.get("last_score"), score.get("rounds"), score.get("strikes", 0)
        if not (num(ema) and num(last) and num(rounds) and num(strikes)):
            return "bad_score"
        struck = score.get("last_strike_at", score.get("last_strike", 0))
        if not num(struck):
            return "bad_score"
        if struck and now - struck < self.cfg.strike_cooldown_s:
            return "recent_strike"
        if strikes > 0 and not struck:  # state.py keeps only a count; without a date we cannot tell it is old
            return "strike"
        if rounds < self.cfg.min_rounds:
            return "rounds"
        if not self.cfg.min_ema <= ema <= 1:
            return "ema"
        lw = POOL_PARAMS[role[0]].latency_weight
        if last < 1.0 - lw - 1e-9:  # a passing round scores at least 1 - latency_weight; 0 = grey zone, strike or failure
            return "last_round"
        return None

    async def refresh(self):
        try:
            snap = await asyncio.to_thread(self.chain.snapshot)
            scores, fresh = await asyncio.to_thread(self._read_scores)
            selected = {m: [] for m in self.cfg.models}
            excluded = {m: collections.Counter() for m in self.cfg.models}
            for n in snap.neurons:
                role = parse_commitment(n.commitment)
                if not role or role[0] not in self.manifests or role[1] != self.manifests[role[0]].manifest_id:
                    continue
                if (n.uid == snap.owner_uid or n.hotkey == self.hotkey or not n.ip or not n.port
                        or not ip_allowed(n.ip, self.validator_cfg.allow_private_ips)):
                    continue
                excluded[role[0]]["seen"] += 1
                reason = self.exclusion(scores.get(str(n.uid), {}), n.hotkey, role, time.time())
                if reason:
                    excluded[role[0]][reason] += 1
                    continue
                selected[role[0]].append((n, float(scores[str(n.uid)]["ema"])))
                self.health.setdefault(n.hotkey, MinerHealth())
            self.miners, self.fresh_at, self.excluded = selected, fresh, excluded
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

    def order_candidates(self, model):
        """Eligible miners in random order, weighted by their remaining per-minute budget, so that load spreads.

        Miners that are busy, backed off or out of budget go last (the request loop skips and counts them). EMA only
        decides eligibility, not the order. Uses `secrets`.
        """
        now, mono = time.time(), time.monotonic()
        ready, rest = [], []
        for item in self.candidates(model):
            h = self.health[item[0].hotkey]
            while h.recent and mono - h.recent[0] >= 60:
                h.recent.popleft()
            weight = self.cfg.per_miner_per_minute - len(h.recent)
            if h.inflight >= self.cfg.inflight_cap or h.busy_until > now or weight <= 0:
                rest.append(item)
            else:
                ready.append((item, weight))
        out = []
        while ready:
            point = secrets.randbelow(sum(w for _, w in ready))
            for i, (item, weight) in enumerate(ready):
                if point < weight:
                    out.append(item)
                    del ready[i]
                    break
                point -= weight
        return out + rest

    def fallback_available(self, model):
        return self.cfg.enabled and self.cfg.reference_fallback and model in self.backends

    def ready(self, model):
        return (self.counter is not None and model in getattr(self.counter, "models", self.cfg.models)
                and (len(self.candidates(model)) >= self.cfg.min_miners or self.fallback_available(model)))

    def _budget_ok(self, h, now):
        while h.recent and now - h.recent[0] >= 60:
            h.recent.popleft()
        return len(h.recent) < self.cfg.per_miner_per_minute

    async def _query(self, n, req, timeout):
        body = req.to_bytes()
        headers = auth.sign_request(self.wallet, body, n.hotkey, path=S1_DECIDE_PATH)
        headers["Content-Type"] = "application/json"
        async with self.session.post(s1_endpoint_url(n.ip, n.port), data=body, headers=headers,
                                     allow_redirects=False, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 429:
                raise MinerBusy("miner busy")
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
        model = req.model
        if self.reference_used[model] + 1 + 1 > self.cfg.reference_slots(model):
            # Shadows use the same per-model reference slots as verification, never queue behind it and never take
            # the last free slot: that one stays for forced and probation checks.
            self.counts["shadow", model, "skipped_no_slot"] += 1
            return
        self.reference_used[model] += 1
        d = secrets.choice(req.decisions)
        try:
            result = await asyncio.wait_for(backend.decide([d]), self.cfg.timeout_s)
            ref = {d.ident: tuple(result[0]["probabilities"][k] for k in d.question.keys)}
            params = POOL_PARAMS[model]
            verdict = judge(score_round(probabilities, ref, 0, params), params)
        except Exception:
            self.counts["shadow", model, "reference_unavailable"] += 1
            log.warning("shadow uid=%d request_id=%s outcome=reference_unavailable", n.uid, req.request_id)
            return  # reference failures are never attributed to a miner
        finally:
            self.reference_used[model] -= 1
        self.counts["shadow", model, verdict] += 1
        if verdict == "strike":
            await self.strike(n, req, "shadow_strike")
        await self.check_result(n, req, "shadow_grey", verdict == "grey")

    def on_probation(self, hotkey):
        return self.probation.get(hotkey, 0) < self.cfg.probation_answers

    def credit_forces(self, hotkey):
        """True when serving one more unverified answer would break the budget: verified_served (last hour) below
        floor * (served + 1), or the run of unverified serves since the last verified one already at its cap."""
        floor = self.cfg.floor
        if floor <= 0:
            return False
        credit = self.credits.get(hotkey)
        served, verified = credit.totals(time.time()) if credit else (0, 0)
        if credit and credit.run >= self.cfg.unverified_run_cap:
            return True
        return verified + 1e-9 < floor * (served + 1)

    async def check_result(self, n, req, kind, grey):
        """Book one verification of this hotkey (sync or shadow) and quarantine it when its grey rate is significantly high.

        Honest runs land in the grey zone now and then, so one grey changes nothing: only the Wilson 95 % lower bound of
        the 24 h grey rate above grey_rate_max, with at least GREY_MIN_VERIFIED checks, quarantines the hotkey; it then
        starts probation again once the quarantine is over.
        """
        now = time.time()
        stats = self.greys.setdefault(n.hotkey, Rolling(STATUS_WINDOW_MINUTES))
        stats.add(now, 1, 1 if grey else 0)
        checks, greys = stats.totals(now)
        if grey:
            log.info("%s uid=%d request_id=%s outcome=grey greys=%d checks=%d", kind, n.uid, req.request_id, greys, checks)
        if checks >= GREY_MIN_VERIFIED and wilson_lower(greys, checks) > self.cfg.grey_rate_max:
            h = self.health[n.hotkey]
            h.open_until = max(h.open_until, now + self.cfg.grey_quarantine_s)
            self.probation[n.hotkey] = 0
            self.greys[n.hotkey] = Rolling(STATUS_WINDOW_MINUTES)  # requalification starts from clean statistics
            self.counts["quarantine", req.model, "grey_rate"] += 1
            log.warning("%s uid=%d request_id=%s outcome=grey_quarantine greys=%d checks=%d", kind, n.uid, req.request_id,
                        greys, checks)

    async def strike(self, n, req, kind):
        now = time.time()
        h = self.health[n.hotkey]
        self.probation[n.hotkey] = 0
        history = [(t, u) for t, u in self.strike_times.get(n.hotkey, ()) if now < u + 7 * 86400]
        if any(now < u for _t, u in history):
            # Already excluded by an earlier strike (a shadow that was pending, or a burst of one incident): this is
            # the same episode, not a repeat. No escalation and no second ledger entry.
            self.strike_times[n.hotkey] = history
            log.warning("%s uid=%d hotkey=%s request_id=%s outcome=already_excluded", kind, n.uid, n.hotkey, req.request_id)
            return
        # A second strike within 7 days after the previous exclusion ended is a pattern, not an accident: long
        # exclusion, persisted with the entry. Measured from the end of the exclusion so that 7 d stays 7 d.
        duration = self.cfg.repeat_strike_exclusion_s if history else 86400
        self.strike_times[n.hotkey] = history + [(now, now + duration)]
        h.open_until = max(h.open_until, now + duration)
        entry = {"time": now, "until": now + duration, "uid": n.uid, "hotkey": n.hotkey,
                 "request_id": req.request_id, "model": req.model, "outcome": kind}
        # No decisions/probabilities/caller ids ever persist. Separate from validator state.
        try:
            await asyncio.to_thread(self._write_strike, entry)  # fsync must not block the event loop
        except OSError:
            log.error("%s uid=%d request_id=%s outcome=strike_persistence_failed", kind, n.uid, req.request_id)
        log.warning("%s uid=%d hotkey=%s request_id=%s outcome=strike", kind, n.uid, n.hotkey, req.request_id)

    def _write_strike(self, entry):
        path = Path(self.cfg.strikes_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def reserve_reference(self, backend, req, spare=0):
        """Start one owned task per decision, or refuse. Never queues: slots are counted in decisions, per model.

        `spare` slots must remain free afterwards: best-effort sampling leaves them for forced/probation verification.
        """
        count, model = len(req.decisions), req.model
        if backend is None or self.reference_used[model] + count + spare > self.cfg.reference_slots(model):
            raise ReferenceUnavailable("no reference capacity")
        self.reference_used[model] += count
        tasks = []

        def release(_task):  # fires only when the task has really finished or was cancelled
            self.reference_used[model] -= 1
        try:
            for d in req.decisions:
                task = asyncio.ensure_future(backend.decide([d]))
                task.add_done_callback(release)
                tasks.append(task)
        except BaseException:
            self.reference_used[model] -= count - len(tasks)  # slots of tasks that never started
            for task in tasks:
                task.cancel()
            raise
        return tasks

    @staticmethod
    async def _reference(tasks, req, remaining):
        """Reference probabilities per decision id plus the raw runtime results, bounded by the request deadline.

        Owns `tasks`: on failure, timeout or cancellation every unfinished one is cancelled and awaited.
        """
        try:
            _done, pending = await asyncio.wait(tasks, timeout=max(remaining, 0.001),
                                                return_when=asyncio.FIRST_EXCEPTION)
            for t in tasks:
                if t.done() and t.cancelled():
                    raise RuntimeError("reference task was cancelled")
                if t.done() and t.exception() is not None:
                    raise t.exception()
            if pending:
                raise asyncio.TimeoutError()
            results = []
            for t in tasks:
                one = t.result()
                if len(one) != 1:
                    raise RuntimeError("reference returned the wrong number of answers")
                results.append(one[0])
            probabilities = {d.ident: tuple(r["probabilities"][k] for k in d.question.keys)
                             for d, r in zip(req.decisions, results)}
            # Same tighter normalization boundary as the miner path (1e-4); never renormalized.
            if any(abs(math.fsum(p) - 1) > 1e-4 for p in probabilities.values()):
                raise RuntimeError("reference probability sum exceeds router tolerance")
            return probabilities, results
        finally:
            for t in tasks:
                t.cancel()
            while not all(t.done() for t in tasks):
                try:
                    await asyncio.wait(tasks)
                except asyncio.CancelledError:
                    pass  # children are already cancelled; finish the cleanup before propagating
            await asyncio.sleep(0)  # let the slot-release callbacks run

    @staticmethod
    def _reference_answers(req, results):
        # Same path a miner response takes: S1Response bytes -> strict parse -> JSON answer objects.
        raw = S1Response(req.request_id, "reference",
                         tuple((d.ident, {d.question.ident: a}) for d, a in zip(req.decisions, results))).to_bytes()
        parse_s1_response(raw, req)
        return {r["id"]: r["answers"]["q"] for r in json.loads(raw)["results"]}

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
            return self.unavailable(model, started)
        end = started + min(self.cfg.timeout_s, (deadline / 1000 - time.time() - 0.3))
        try:
            budget = end - time.monotonic()
            if budget <= 0:
                return self.unavailable(model, started)
            count = await asyncio.wait_for(asyncio.to_thread(self.counter, model, decisions), budget)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RuntimeError()
        except InputTokenLimitError:
            return self.error(400, "input exceeds model token limit")
        except Exception:
            return self.unavailable(model, started)
        req = S1Request(secrets.token_hex(16), model, self.manifests[model].manifest_id, decisions)
        backend = self.backends.get(model)
        # Sampling never depends on the backend or the content: one secret draw per request. The floor share is nested in
        # the sampled share (floor <= rate): a hard draw fails closed, a sampled draw above the floor is best effort.
        draw = secrets.randbelow(1_000_000)
        sampled = draw < self.cfg.verify_rate * 1_000_000
        hard = draw < self.cfg.floor * 1_000_000
        ref_task = None
        ref_children = []

        def start_reference(spare=0):
            nonlocal ref_task, ref_children
            if ref_task is None:  # one reference computation per request, shared by verification and fallback
                if end - time.monotonic() < self.cfg.min_reference_budget_s:
                    raise ReferenceUnavailable("too little time left for a reference computation")
                ref_children = self.reserve_reference(backend, req, spare)
                ref_task = asyncio.ensure_future(self._reference(ref_children, req, end - time.monotonic()))
            return ref_task

        async def reference_result():
            task = start_reference()
            try:
                return await asyncio.wait_for(asyncio.shield(task), max(end - time.monotonic(), 0.001))
            except asyncio.CancelledError:
                if task.cancelled():  # the reference work was cancelled, not this request handler
                    raise ReferenceUnavailable("reference cancelled") from None
                raise

        def respond(answers, served_by, verified, uid=None, hotkey=None):
            if time.monotonic() > end:  # never return a 200 after the deadline
                return self.unavailable(model, started)
            if hotkey is not None:
                self.credits.setdefault(hotkey, Credit()).add(time.time(), verified)
            self.counts["requests", model, served_by] += 1
            self._record_status(model, served_by, uid)
            if uid is not None:
                self.counts["uid", model, str(uid)] += 1
            elapsed = time.monotonic() - started
            stat = self.latency.setdefault((model, served_by), [0, 0.0])
            stat[0] += 1
            stat[1] += elapsed
            headers = {"X-Reax-Served-By": served_by, "X-Reax-Verified": "1" if verified else "0"}
            if uid is not None:
                headers["X-Reax-Miner-Uid"] = str(uid)
            return web.json_response({"request_id": body["request_id"], "model": model,
                "answers": {qid: answers[d.ident] for qid, d in zip(qids, decisions)},
                "usage": {"input_tokens": count, "output_tokens": 0, "decisions": len(decisions)},
                "latency_ms": elapsed * 1000,
                "tokenizer_version": self.cfg.tokenizer_version_for(model),
                "model_revision": self.cfg.revision_for(model)},
                headers=headers)

        async def serve_reference(reason):
            # Our own GPU running the identical pinned model; the answer objects go through the miner's format path.
            if not self.fallback_available(model):
                return None
            try:
                _probs, results = await reference_result()
                answers = self._reference_answers(req, results)
            except Exception:
                log.warning("request_id=%s outcome=reference_unavailable reason=%s", req.request_id, reason)
                return None
            log.info("request_id=%s latency_ms=%.1f outcome=reference reason=%s", req.request_id,
                     (time.monotonic() - started) * 1000, reason)
            return respond(answers, "reference", True)

        try:
            candidates = self.order_candidates(model)
            if len(decisions) > self.cfg.reference_slots(model):
                candidates = []  # a request the reference cannot check in one go never reaches a miner
            if hard or (sampled and backend is None):
                try:
                    start_reference()
                except ReferenceUnavailable:  # no backend or no free slot: fail closed before asking any miner
                    self.counts["verify", model, "reference_unavailable"] += 1
                    log.warning("request_id=%s outcome=verify_reference_unavailable", req.request_id)
                    return self.unavailable(model, started)
            must_skipped = False
            chosen = None
            attempts = 0
            for n, _score in candidates:
                h = self.health[n.hotkey]
                if h.open_until > time.time():
                    continue
                now = time.monotonic()
                if h.inflight >= self.cfg.inflight_cap or h.busy_until > time.time():
                    self.counts["attempt", model, "busy"] += 1
                    continue
                if not self._budget_ok(h, now):
                    self.counts["attempt", model, "budget"] += 1
                    continue
                budget = end - now
                if budget <= 0 or attempts >= 2:
                    break
                # Verification is decided for the miner that actually gets the attempt.
                probation = self.on_probation(n.hotkey)
                must = hard or probation or self.credit_forces(n.hotkey)
                verify, skipped_busy = ref_task is not None, False
                if not verify and (must or sampled):
                    if backend is None:
                        self.counts["verify", model, "reference_unavailable"] += 1
                        log.warning("request_id=%s outcome=verify_reference_unavailable", req.request_id)
                        return self.unavailable(model, started)
                    try:
                        start_reference(0 if must else 1)
                        verify = True
                    except ReferenceUnavailable:
                        if must:  # probation or credit budget: this miner may not answer unverified, try the next one
                            must_skipped = True
                            self.counts["attempt", model, "no_reference"] += 1
                            continue
                        skipped_busy = True  # best-effort share: no spare slot, skip the check and count it
                attempts += 1
                # Reserve half for a sequential fallback when another miner is available.
                timeout = min(budget / 2 if attempts == 1 and len(candidates) > 1 else budget, self.cfg.miner_timeout_s)
                h.inflight += 1
                h.recent.append(now)
                before = time.monotonic()
                replied = False
                try:
                    answers, probabilities = await self._query(n, req, timeout)
                    replied = True
                    if time.monotonic() > end:
                        raise asyncio.TimeoutError()
                    h.failures, h.latency = 0, time.monotonic() - before
                    self.counts["attempt", model, "ok"] += 1
                    chosen = (n, answers, probabilities)
                except MinerBusy:
                    attempts -= 1  # busy is not an attempt and never counts toward the breaker
                    with contextlib.suppress(ValueError):
                        h.recent.remove(now)  # a refused request does not use up the miner's per-minute budget
                    h.busy_until = time.time() + self.cfg.busy_backoff_s
                    self.counts["attempt", model, "busy"] += 1
                    log.info("request_id=%s uid=%d outcome=busy", req.request_id, n.uid)
                except Exception as exc:
                    h.failures += 1
                    if h.failures >= 3:
                        h.open_until = max(h.open_until, time.time() + 60)
                    self.counts["attempt", model, "timeout" if isinstance(exc, asyncio.TimeoutError) else "error"] += 1
                    if not replied and not isinstance(exc, (ValueError, auth.AuthError)):
                        # Timeout/disconnect/transport error: the miner may still be computing and holds the
                        # per-caller slot until its inference returns; the client giving up does not free it, and
                        # the validator's scoring rounds need the other slot. Keep routing away from it. (No
                        # strike; an HTTP status or an invalid reply proves the miner finished and is not covered.)
                        h.open_until = max(h.open_until, time.time() + self.cfg.timeout_quarantine_s)
                    log.info("request_id=%s uid=%d latency_ms=%.1f outcome=failed", req.request_id, n.uid,
                             (time.monotonic() - before) * 1000)
                finally:
                    h.inflight -= 1
                if chosen:
                    break
            if chosen is None:
                answered = await serve_reference("no_miner_answer")
                if answered is None and must_skipped:
                    self.counts["verify", model, "reference_unavailable"] += 1
                return answered or self.unavailable(model, started)
            n, answers, probabilities = chosen
            if not verify:
                self.counts["verify", model, "skipped"] += 1
                self._record_status(model, "skipped")
                if skipped_busy:
                    self.counts["verify", model, "skipped_reference_busy"] += 1
                    self._record_status(model, "skipped_reference_busy")
                if (backend is not None and len(self.tasks) < 32
                        and secrets.randbelow(1_000_000) < self.cfg.shadow_rate * 1_000_000):
                    task = asyncio.create_task(self.shadow(n, req, probabilities))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                log.info("request_id=%s uid=%d latency_ms=%.1f outcome=ok", req.request_id, n.uid,
                         (time.monotonic() - started) * 1000)
                return respond(answers, "miner", False, n.uid, n.hotkey)
            # Synchronous verification: every decision against the reference, fail closed if it is unavailable.
            try:
                ref, _results = await reference_result()
            except Exception:
                self.counts["verify", model, "reference_unavailable"] += 1
                log.warning("request_id=%s uid=%d outcome=verify_reference_unavailable", req.request_id, n.uid)
                return self.unavailable(model, started)
            params = POOL_PARAMS[model]
            outcome = score_round(probabilities, ref, 0, params)
            self._record_status(model, "verified")
            self._record_status(model, "probation" if probation else "sampled")
            verdict = judge(outcome, params)
            if verdict == "match":
                await self.check_result(n, req, "verify_grey", False)
                self.counts["verify", model, "match"] += 1
                self._record_status(model, "verify_match")
                if probation:
                    self.probation[n.hotkey] = self.probation.get(n.hotkey, 0) + 1
                log.info("request_id=%s uid=%d latency_ms=%.1f outcome=ok verified=1", req.request_id, n.uid,
                         (time.monotonic() - started) * 1000)
                return respond(answers, "miner", True, n.uid, n.hotkey)
            if verdict == "strike":
                self.counts["verify", model, "mismatch_strike"] += 1
                self._record_status(model, "verify_mismatch")
                await self.strike(n, req, "verify_strike")  # also puts the miner back on probation
                await self.check_result(n, req, "verify_grey", False)
            else:  # grey zone above tau_mean: the validator scores the round 0 without a strike
                self.counts["verify", model, "mismatch_grey"] += 1
                self._record_status(model, "verify_mismatch")
                await self.check_result(n, req, "verify_grey", True)  # one grey: honest variance, no probation reset
            return (await serve_reference("verify_mismatch")) or self.unavailable(model, started)
        finally:
            if ref_task is not None:
                ref_task.cancel()  # no-op when done; otherwise it cancels and awaits its own per-decision tasks
                await asyncio.wait({ref_task})
                if not ref_task.cancelled():
                    ref_task.exception()  # mark retrieved
                # A ref_task cancelled before its first step never runs _reference's cleanup: own the children here.
                for child in ref_children:
                    child.cancel()
                await asyncio.gather(*ref_children, return_exceptions=True)
                await asyncio.sleep(0)  # let the slot-release callbacks run

    def unavailable(self, model, started):
        self.counts["requests", model, "none"] += 1
        self._record_status(model, "none")
        stat = self.latency.setdefault((model, "none"), [0, 0.0])
        stat[0] += 1
        stat[1] += time.monotonic() - started
        return self.error(503, "no eligible miner")

    def _prune_status(self, model, now):
        buckets = self.status_buckets[model]
        minute = int(now // 60)
        # Keep the current minute and the preceding 1439 whole-minute buckets.
        while buckets and buckets[0][0] <= minute - STATUS_WINDOW_MINUTES:
            buckets.popleft()
        return buckets

    def _record_status(self, model, kind, uid=None):
        now = time.time()
        buckets = self._prune_status(model, now)
        minute = int(now // 60)
        if not buckets or buckets[-1][0] != minute:
            buckets.append([minute, dict.fromkeys(STATUS_COUNTERS, 0), {}])
        bucket = buckets[-1]
        bucket[1][kind] += 1
        # Netuid 92 has at most 256 public UIDs; that bounds the per-bucket map.
        if uid is not None and 0 <= uid < 256:
            bucket[2][uid] = bucket[2].get(uid, 0) + 1

    def _status_window(self, model, now):
        counts = dict.fromkeys(STATUS_COUNTERS, 0)
        per_uid = collections.Counter()
        for _minute, bucket_counts, uids in self._prune_status(model, now):
            for kind in STATUS_COUNTERS:
                counts[kind] += bucket_counts[kind]
            per_uid.update(uids)
        counts.update(window_seconds=STATUS_WINDOW_SECONDS,
                      since=max(self.started_at, now - STATUS_WINDOW_SECONDS))
        return counts, dict(sorted(per_uid.items()))

    def _probation_miners(self, model):
        return sum(1 for n, _ in self.candidates(model) if self.on_probation(n.hotkey))

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
        states = {m: {"state": "ready" if self.ready(m) else "unavailable", "revision": self.cfg.revision_for(m),
                      "miners": len(self.candidates(m)), "reference_fallback": self.fallback_available(m)}
                  for m in models}
        ok = all(s["state"] == "ready" for s in states.values())
        return web.json_response({"ok": ok, "models": states}, status=200 if ok else 503)

    async def health_live(self, request):
        return web.json_response({"ok": True})

    def _authorized(self, request):
        tokens = request.headers.getall("X-S1M-Internal-Token", [])
        return len(tokens) == 1 and hmac.compare_digest(tokens[0].encode(), self.cfg.token.encode())

    async def metrics(self, request):
        if not self._authorized(request):
            return self.error(401, "unauthorized")
        out = []

        def emit(name, kind, rows):
            out.append(f"# TYPE {name} {kind}")
            for labels, value in rows:
                text = ",".join('%s="%s"' % kv for kv in labels)
                out.append(f"{name}{{{text}}} {value}")

        def counter(kind, names):
            return [(tuple(zip(names, key[1:])), v) for key, v in sorted(self.counts.items()) if key[0] == kind]

        emit("reax_router_requests_total", "counter", counter("requests", ("model", "served_by")))
        emit("reax_router_verify_total", "counter", counter("verify", ("model", "outcome")))
        emit("reax_router_miner_attempts_total", "counter", counter("attempt", ("model", "outcome")))
        emit("reax_router_served_by_uid_total", "counter", counter("uid", ("model", "uid")))
        emit("reax_router_eligible_miners", "gauge",
             [((("model", m),), len(self.candidates(m))) for m in self.cfg.models])
        emit("reax_router_probation_miners", "gauge",
             [((("model", m),), self._probation_miners(m)) for m in self.cfg.models])
        emit("reax_router_shadow_total", "counter", counter("shadow", ("model", "outcome")))
        emit("reax_router_seen_miners", "gauge",
             [((("model", m),), self.excluded.get(m, {}).get("seen", 0)) for m in self.cfg.models])
        out.append("# TYPE reax_router_latency_seconds summary")
        for (m, sb), (num, total) in sorted(self.latency.items()):
            labels = f'model="{m}",served_by="{sb}"'
            out.append(f"reax_router_latency_seconds_count{{{labels}}} {num}")
            out.append(f"reax_router_latency_seconds_sum{{{labels}}} {total:.6f}")
        return web.Response(text="\n".join(out) + "\n", content_type="text/plain", charset="utf-8")

    async def status(self, request):
        if not self._authorized(request):
            return self.error(401, "unauthorized")
        models = {m: {"eligible_miners": len(self.candidates(m)), "seen_miners": self.excluded.get(m, {}).get("seen", 0),
                      "excluded": {k: v for k, v in sorted(self.excluded.get(m, {}).items()) if k != "seen"}, "uids": [n.uid for n, _ in self.candidates(m)],
                      "ready": self.ready(m), "reference_fallback": self.fallback_available(m),
                      "probation_miners": self._probation_miners(m)}
                  for m in self.cfg.models}
        now = time.time()
        served_uids = {}
        for m in models:
            models[m]["last_24h"], per_uid = self._status_window(m, now)
            models[m]["served_by_uid_24h"] = {str(uid): n for uid, n in per_uid.items()}  # JSON keys are strings
            served_uids[m] = list(per_uid)
        counts = {}
        for key, v in self.counts.items():
            counts.setdefault(key[0], {})[":".join(key[1:])] = v
        return web.json_response({"started_at": self.started_at, "enabled": self.cfg.enabled,
                                  "verify_rate": self.cfg.verify_rate, "verify_floor": self.cfg.floor,
                                  "probation_answers": self.cfg.probation_answers, "models": models, "counts": counts,
                                  "miner_uids_served_24h": served_uids})

    def app(self):
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.cleanup_ctx.append(self._lifecycle)
        app.router.add_post("/internal/v1/decide", self.decide)
        app.router.add_get("/health/ready", self.health_ready)
        app.router.add_get("/health", self.health_live)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/status", self.status)
        return app


def main():
    from .chain import SdkChain
    from .cli import _s1_backend, _wallet, setup_logging
    from .router_tokens import MultiTokenCounter, PlumbTokenCounter, RuneTokenCounter

    try:
        cfg = RouterConfig.from_env()
        vcfg = load_validator()
        if vcfg.common.mechanism != "s1":
            raise ConfigError("router requires the s1 validator configuration")
        wallet, _ = _wallet(vcfg.common)
    except (ConfigError, OSError) as exc:
        raise SystemExit(f"router configuration error: {exc}") from None
    setup_logging(vcfg.common)
    loaders = {"s1-fast": (PlumbTokenCounter, "REAX_ROUTER_TOKENIZER_DIR"),
               "s1-pro": (RuneTokenCounter, "REAX_ROUTER_TOKENIZER_DIR_S1_PRO")}
    counters = {}
    for m in cfg.models:
        try:
            counters[m] = loaders[m][0].from_directory(os.environ.get(loaders[m][1], ""))
        except Exception:
            log.error("token counter model=%s outcome=unavailable (decide will fail closed)", m)
    counter = MultiTokenCounter(counters) if counters else None
    backends = {m: _s1_backend(vcfg.common, m) for m in cfg.models} if cfg.shadow_rate or cfg.verify_rate or cfg.reference_fallback else {}
    chain = SdkChain(vcfg.common.network, vcfg.common.netuid, wallet, commitments=True)
    router = Router(cfg, vcfg, wallet, chain, counter, backends)
    # Disable aiohttp access logging, which would include URL/query data.
    web.run_app(router.app(), host=cfg.host, port=cfg.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
