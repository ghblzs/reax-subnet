"""The reax-s1 validator: miners serve the exact REAX reference models.

Each round:
1. read the metagraph with commitments; a miner's role is its commitment
   ``reax-s1/1:<model>:<manifest_id>`` for an enabled pool and this release's manifest;
   a changed role restarts that UID from zero,
2. draw n fresh typed decisions per miner and query all miners at once (signed),
3. *after* the answers are in, draw a secret set of k items per miner and compute
   their references on the validator's own runtime of the same pinned model,
4. score in logit space (s1scoring.py): a round mean above tau_strike or any item above tau_item = strike
   (moving average to 0); a mean within tau_mean passes,
5. combine per-pool averages into weights with fixed pool shares; save state; set weights.
Unverified items still cost the miner full work because it cannot tell them apart.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import secrets
import time
from typing import Any, Mapping, Optional

import aiohttp
from multidict import CIMultiDict

from . import auth
from .chain import Neuron, Snapshot
from .manifest import load_manifest, parse_commitment
from .protocol import ProtocolError
from .s1backends import S1Backend
from .s1contract import Decision, parse_decision
from .s1items import generate_s1_items
from .s1protocol import MAX_RESPONSE_BYTES, S1_DECIDE_PATH, S1Request, parse_s1_request, parse_s1_response
from .s1scoring import PoolParams, choose_verified, compute_pool_weights, score_round
from .state import save
from .tiers import CAPACITY, TierBudgets, compute_tiered_weights, pool_shares
from .validator import CHAIN_READ_TIMEOUT_S, QueryResult, Validator, _read_limited, ip_allowed

log = logging.getLogger("reax.s1validator")

# Release constants (docs/DESIGN-v1.1.md, CALIBRATION.md). Validators refuse overrides on finney.
POOL_PARAMS: dict[str, PoolParams] = {
    "s1-fast": PoolParams("s1-fast", temperature=2.07, tau_mean=0.20, tau_strike=0.50, tau_item=2.0,
                          items_per_round=8, verified_per_round=8, latency_fast_s=0.5, latency_slow_s=8.0),
    "s1-pro": PoolParams("s1-pro", temperature=2.0, tau_mean=0.20, tau_strike=0.50, tau_item=2.0,
                         items_per_round=8, verified_per_round=3, latency_fast_s=1.0, latency_slow_s=12.0),
    # Tiered pools (REAX_FEATURE_TIERED_POOLS, on in 1.2). Tolerances from the cross-backend
    # calibration in docs/DESIGN-POOLS.md section 6 (llama.cpp CPU x86 / aarch64 / CUDA against each other);
    # the latency curve fits the tier's hardware (4 decisions per request).
    # s1-lite tau_mean 0.30: a validator may use its bf16 s1-fast runtime (the same weights, no extra GPU memory) as
    # the s1-lite reference; honest Q8 miners then sit at mean 0.11 (P(round fails) ~ 0), a Q4 substitute still fails
    # 92 % of rounds and is struck in 61 %. Against the GGUF reference itself honest distances are smaller still.
    "s1-lite": PoolParams("s1-lite", temperature=2.07, tau_mean=0.30, tau_strike=0.50, tau_item=2.0,
                          items_per_round=4, verified_per_round=4, latency_fast_s=1.0, latency_slow_s=12.0),
    "s1-cpu": PoolParams("s1-cpu", temperature=1.0, tau_mean=0.20, tau_strike=0.50, tau_item=2.0,
                         items_per_round=4, verified_per_round=4, latency_fast_s=2.0, latency_slow_s=30.0),
}
TIERED_POOLS = ("s1-lite", "s1-cpu")
SHARES_FAST_ONLY = {"s1-fast": 1.0}
# If our own runtime fails on more than this share of the verified items, nobody is scored this round.
MAX_REFERENCE_FAILURE_SHARE = 0.10
# A hung reference runtime (health green, decisions never return) must not stall the validator for hours:
# the whole reference step of a round gets this budget in multiples of the round interval.
REFERENCE_BUDGET_ROUNDS = 1.5
REFERENCE_BUDGET_MIN_S = 60.0
SHIELD_COOLDOWN_ROUNDS = 30
# Tiered pools: a capacity probe runs in about one round in this many.
PROBE_EVERY_ROUNDS = 10

SHARES_BOTH = {"s1-fast": 0.35, "s1-pro": 0.65}
# Tiered pools: reference decisions per round a validator spends per pool (normal rounds; capacity probes share
# the same number). With more miners than fit, miners are verified in fair rotation. Defaults fit a validator
# that runs the s1-lite reference on its GPU (~30 ms/decision) and the s1-cpu reference on 8 CPU threads
# (~3.7 decisions/s); override per validator with REAX_REFERENCE_CAP_S1_LITE / REAX_REFERENCE_CAP_S1_CPU.
REFERENCE_CAP = {"s1-lite": 240, "s1-cpu": 160}


# Capacity probes: the window's measurement is dropped when the validator itself was the bottleneck (review C1). The
# event loop is sampled every PROBE_TICK_S; a lag p95 above PROBE_LAG_P95_MAX_S means answers waited for the
# validator's CPU. REAX_PROBE_CEILING_DPS (validator-local, from scripts/bench_probe.py; 0 = off) drops a probe whose
# total throughput reaches what this host can take. Neither changes how a measured probe is scored.
PROBE_TICK_S = 0.010
PROBE_LAG_P95_MAX_S = 0.050
# Before the window each probe lane's connection is opened with a GET /health, bounded by this time.
PROBE_WARMUP_S = 2.0
# The preparation of a probe's requests (a worker thread) comes before its window and reference budget.
PROBE_PREPARE_BUDGET_S = 60.0


def _probe_ceiling_dps() -> float:
    raw = os.environ.get("REAX_PROBE_CEILING_DPS", "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


class _LoopLag:
    """Event-loop lag sampler: a ticker that sleeps PROBE_TICK_S and records how late it wakes up."""

    def __init__(self) -> None:
        self.samples: list[float] = []

    async def run(self) -> None:
        while True:
            t0 = time.monotonic()
            await asyncio.sleep(PROBE_TICK_S)
            self.samples.append(max(0.0, time.monotonic() - t0 - PROBE_TICK_S))

    def p95(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


class _HttpStatus(Exception):
    """A miner answered with a status other than 200 (a refusal, an overload or an error)."""

    def __init__(self, status: int):
        super().__init__(f"http {status}")
        self.status = status


def s1_endpoint_url(ip: str, port: int) -> str:
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}{S1_DECIDE_PATH}"


class S1Validator(Validator):
    def __init__(self, cfg, wallet, chain, backends: Mapping[str, S1Backend],
                 pools: Optional[Mapping[str, PoolParams]] = None, shares: Optional[Mapping[str, float]] = None,
                 engine: Optional[Any] = None, competitions: Optional[Any] = None):
        super().__init__(cfg, wallet, chain, backend=None)  # type: ignore[arg-type]
        self.tiered = bool(getattr(cfg, "tiered_pools", False))
        # Development networks may probe more often (localnet runs last minutes, not hours).
        self.probe_every = PROBE_EVERY_ROUNDS
        dev_every = os.environ.get("REAX_DEV_PROBE_EVERY_ROUNDS", "")
        if self.tiered and dev_every.isdigit() and cfg.common.allow_dev_models:
            self.probe_every = max(1, int(dev_every))
        self.budgets = TierBudgets()
        if self.tiered and shares is None:
            # A validator without the reference of a tiered pool leaves that pool unscored (its share goes to the
            # other pools pro rata, like an empty pool); the GPU pools' references are required as before.
            enabled = ["s1-fast", *(["s1-pro"] if cfg.s1_pro_pool else [])]
            for m in TIERED_POOLS:
                if m in backends:
                    enabled.append(m)
                else:
                    log.warning("no reference runtime for tiered pool %s: its miners are not scored by this validator", m)
            shares = pool_shares(enabled, self.budgets)
        self.shares = dict(shares or (SHARES_BOTH if cfg.s1_pro_pool else SHARES_FAST_ONLY))
        self.pools = {m: (pools or POOL_PARAMS)[m] for m in self.shares}
        missing = [m for m in self.pools if m not in backends]
        if missing:
            raise ValueError(f"no reference runtime for enabled pool(s): {', '.join(missing)}")
        self.backends = dict(backends)
        self.manifests = {m: load_manifest(m) for m in self.pools}
        self.last_round: dict[str, Any] = {}
        self._role_pending: dict[int, str] = {}
        self._unconfirmed_streak: dict[int, int] = {}
        self._last_shield: dict[int, int] = {}
        self._rejected_commitment: dict[int, str] = {}
        # Unreleased engine competition (engvalidator.EngineCompetition), None = off.
        self.engine = engine
        # competitions.CompetitionDriver: runs the engine and specialist cycles on one GPU. Given -> cycles run as a
        # background task and serving rounds never wait for them. Without it an engine is driven inline (tests).
        if competitions is None and engine is not None:
            from .competitions import CompetitionDriver

            competitions = CompetitionDriver(engine, background=False)
        self.competitions = competitions
        self._weights_lock = asyncio.Lock()
        self._latest_snap: Optional[Snapshot] = None
        self._last_verified: dict[int, int] = {}  # uid -> round of its last verified answer (rotation)
        self._probe_last: dict[int, float] = {}  # uid -> last verified probe measurement (sizes its next probe)
        self.last_probe_stats: dict[str, Any] = {}  # the last probe's total throughput and event-loop lag

    # --------------------------------------------------------------- roles
    def role_of(self, n: Neuron) -> Optional[str]:
        parsed = parse_commitment(n.commitment)
        if parsed is None:
            return None
        model, manifest_id = parsed
        if model not in self.pools or manifest_id != self.manifests[model].manifest_id:
            return None
        return model

    def _why_rejected(self, commitment: str) -> str:
        parsed = parse_commitment(commitment)
        if parsed is None:
            return "not a reax-s1/1 commitment"
        model, manifest_id = parsed
        if model not in self.pools:
            return f"the {model} pool is not enabled in this release"
        return (f"manifest {manifest_id} differs from this release's {model} manifest "
                f"{self.manifests[model].manifest_id} (miner and validator run different releases)")

    def _log_rejected_commitments(self, snap: Snapshot) -> None:
        """Say once per UID and commitment why a committed miner is not queried."""
        seen: dict[int, str] = {}
        for n in snap.neurons:
            if not n.commitment or n.hotkey == self.hotkey or n.uid == snap.owner_uid or self.role_of(n):
                continue
            if self.engine is not None and self.engine.is_engine_commitment(n.commitment):
                continue
            seen[n.uid] = n.commitment
            if self._rejected_commitment.get(n.uid) != n.commitment:
                log.warning("uid %d is not queried: commitment %r: %s", n.uid, n.commitment[:80],
                            self._why_rejected(n.commitment))
        self._rejected_commitment = seen

    # --------------------------------------------------------------- query
    async def _post_s1(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, neuron: Neuron, body: bytes,
                       timeout_s: float) -> tuple[bytes, CIMultiDict, float]:
        """Signed POST of a serialised request: (raw response, response headers, latency). Raises on a refusal
        (``_HttpStatus``), timeout or network error; the response is neither verified nor parsed here."""
        assert neuron.ip is not None and neuron.port is not None
        async with sem:
            headers = auth.sign_request(self.wallet, body, neuron.hotkey, path=S1_DECIDE_PATH)
            headers["Content-Type"] = "application/json"
            started = time.monotonic()
            async with session.post(s1_endpoint_url(neuron.ip, neuron.port), data=body, headers=headers,
                                    allow_redirects=False, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                if resp.status != 200:
                    raise _HttpStatus(resp.status)
                raw = await _read_limited(resp, MAX_RESPONSE_BYTES)
                return raw, CIMultiDict(resp.headers), time.monotonic() - started  # keeps duplicate headers

    def _check_s1(self, neuron: Neuron, req: S1Request, raw: bytes, headers: Mapping[str, str],
                  max_age: float) -> dict[str, tuple[float, ...]]:
        """Signature (bound to this validator, the miner's hotkey and the request id) and strict parse."""
        auth.verify_response(headers, raw, req.request_id, self.hotkey, neuron.hotkey, self.nonce_store,
                             max_age=max_age, path=S1_DECIDE_PATH)
        return parse_s1_response(raw, req)

    @staticmethod
    def _s1_error(neuron: Neuron, exc: BaseException) -> str:
        if isinstance(exc, _HttpStatus):
            return f"http {exc.status}"
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        if isinstance(exc, auth.AuthError):
            return f"auth: {type(exc).__name__}"
        if isinstance(exc, ProtocolError):
            return f"protocol: {exc}"
        if isinstance(exc, (aiohttp.ClientError, OSError, ValueError)):
            return f"network: {type(exc).__name__}"
        log.warning("unexpected error querying uid %d: %r", neuron.uid, exc)
        return f"internal: {type(exc).__name__}"

    async def _query_s1(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, neuron: Neuron,
                        req: S1Request, timeout_s: float) -> QueryResult:
        try:
            raw, headers, latency = await self._post_s1(session, sem, neuron, req.to_bytes(), timeout_s)
            return QueryResult(neuron.uid, self._check_s1(neuron, req, raw, headers, timeout_s + 10.0), latency)
        except Exception as exc:  # a miner must never be able to abort the round
            return QueryResult(neuron.uid, None, math.inf, self._s1_error(neuron, exc))

    # --------------------------------------------------------------- round
    def _new_request(self, model: str, items: Optional[int] = None) -> S1Request:
        params = self.pools[model]
        decisions = []
        for it in generate_s1_items(items or params.items_per_round):
            decisions.append(parse_decision(it.decision(), model))
        return S1Request(secrets.token_hex(16), model, self.manifests[model].manifest_id, tuple(decisions))

    async def _references(
        self, verify: dict[int, tuple[str, list[Decision]]]
    ) -> tuple[dict[int, dict[str, tuple[float, ...]]], set[int], int, int]:
        """Reference probabilities for every verified decision, batched per model.

        Returns (references, uids whose reference failed, failed items, total items). A failed item
        (after one retry) leaves only that miner unscored this round, never the whole round.
        """
        out: dict[int, dict[str, tuple[float, ...]]] = {uid: {} for uid in verify}
        failed_uids: set[int] = set()
        failed = total = 0
        for model in sorted(self.pools):
            flat = [(uid, d) for uid, (m, ds) in sorted(verify.items()) if m == model for d in ds]
            if not flat:
                continue
            results = await self.backends[model].decide_each([d for _, d in flat])
            for (uid, d), a in zip(flat, results):
                total += 1
                if isinstance(a, BaseException):
                    failed += 1
                    failed_uids.add(uid)
                    continue
                out[uid][d.ident] = tuple(float(a["probabilities"][k]) for k in d.question.keys)
        return out, failed_uids, failed, total

    async def _references_isolated(
        self, verify: dict[int, tuple[str, list[Decision]]], budget: float
    ) -> tuple[dict[int, dict[str, tuple[float, ...]]], set[int], list[str]]:
        """References pool by pool, concurrently, each pool with its own budget and its own failure share.

        One pool's slow, hung or failing runtime never costs another pool its round: it leaves only its own miners
        unscored (no score update either way). Returns (references, uids left unscored, pools that failed).
        """
        by_model: dict[str, dict[int, tuple[str, list[Decision]]]] = {}
        for uid, (model, ds) in verify.items():
            by_model.setdefault(model, {})[uid] = (model, ds)

        async def one(model: str, sub: dict[int, tuple[str, list[Decision]]]):
            try:
                refs, bad, failed, total = await asyncio.wait_for(self._references(sub), budget)
            except asyncio.TimeoutError:
                log.error("%s reference runtime did not finish %d miners' items within %.0f s; %s miners unscored this "
                          "round (check that sidecar: it may be hung while /health stays green)",
                          model, len(sub), budget, model)
                return model, None, set(sub)
            except Exception:
                log.exception("%s reference computation failed; %s miners unscored this round", model, model)
                return model, None, set(sub)
            if total and failed > MAX_REFERENCE_FAILURE_SHARE * total:
                log.error("%s reference runtime failed on %d of %d items; %s miners unscored this round",
                          model, failed, total, model)
                return model, None, set(sub)
            if failed:
                log.warning("%s reference runtime failed on %d of %d items; %d miner(s) left unscored this round",
                            model, failed, total, len(bad))
            return model, refs, bad

        references: dict[int, dict[str, tuple[float, ...]]] = {}
        unscored: set[int] = set()
        failed_pools: list[str] = []
        for model, refs, bad in await asyncio.gather(*(one(m, sub) for m, sub in sorted(by_model.items()))):
            if refs is None:
                failed_pools.append(model)
            else:
                references.update(refs)
            unscored |= bad
        return references, unscored, failed_pools

    def _reference_cap(self, model: str) -> Optional[int]:
        """Reference decisions per round this validator spends on a pool (None = no cap). Validator-local resource
        limit, not a scoring constant: it changes how often a miner is verified, never how it is scored."""
        if model not in REFERENCE_CAP:
            return None
        raw = os.environ.get(f"REAX_REFERENCE_CAP_{model.upper().replace('-', '_')}", "")
        return max(1, int(raw)) if raw.isdigit() else REFERENCE_CAP[model]

    def _rotate(self, model: str, candidates: list[Neuron]) -> tuple[list[Neuron], list[Neuron]]:
        """Fair rotation under the pool's reference cap: least recently verified first (random among equals).
        Returns (queried this round, rested this round). Rested miners are neither queried nor decayed."""
        cap = self._reference_cap(model)
        if cap is None:
            return candidates, []
        room = max(1, cap // self.pools[model].verified_per_round)
        if len(candidates) <= room:
            return candidates, []
        order = sorted(candidates, key=lambda n: (self._last_verified.get(n.uid, -1), secrets.randbits(32)))
        return order[:room], order[room:]

    async def run_round(self, session: aiohttp.ClientSession) -> Optional[dict[str, Any]]:
        try:
            snap = await self.read_snapshot()
        except asyncio.TimeoutError:
            log.warning("metagraph read did not finish within %.0f s, skipping round", CHAIN_READ_TIMEOUT_S)
            return None
        except Exception as exc:
            log.warning("chain unavailable, skipping round: %s", exc)
            return None
        count = len(snap.neurons)
        if count == 0:
            log.warning("metagraph read returned no neurons; skipping round")
            return None
        known = self._last_neuron_count
        if known and count < 0.9 * known and self._shrunk_count != count:
            self._shrunk_count = count
            log.warning("metagraph shrank from %d to %d neurons; skipping round until a second read confirms", known, count)
            return None
        self._shrunk_count = None
        self._last_neuron_count = count
        reset = self.state.sync(snap)
        if reset:
            log.info("hotkey changed on uids %s; their scores were reset", reset)
        for uid in set(self._last_shield) - set(self.state.miners) | set(reset):
            self._last_shield.pop(uid, None)  # a new hotkey starts without the previous one's cooldown
        self._log_rejected_commitments(snap)

        proposed: dict[int, tuple[Optional[str], str]] = {}
        for n in snap.neurons:
            if n.hotkey == self.hotkey or n.uid == snap.owner_uid:
                continue
            model = self.role_of(n)
            proposed[n.uid] = (model, f"{model}:{self.manifests[model].manifest_id}" if model else "")
        # Commitments have no "shrank" signal like the neuron count. A role change is applied only when two
        # consecutive reads agree on it for that UID; until then the UID is neither queried nor scored. One bad
        # read can therefore never reset anybody; a whole-round guard could be stalled by a
        # few rotating hotkeys). A UID that stays unconfirmed for a second read in a row (it keeps flipping its
        # commitment) is decayed like a miner that did not answer, so flipping cannot freeze an old score
        #.
        unconfirmed: set[int] = set()
        pending: dict[int, str] = {}
        streak: dict[int, int] = {}
        for uid, (_model, role) in proposed.items():
            if self.state.miners[uid].role != role and self._role_pending.get(uid) != role:
                unconfirmed.add(uid)
                pending[uid] = role
                streak[uid] = self._unconfirmed_streak.get(uid, 0) + 1
        self._role_pending = pending
        self._unconfirmed_streak = streak
        # The first unconfirmed read is shielded at most once per SHIELD_COOLDOWN_ROUNDS per UID, so alternating
        # between the stored role and anything else on every read cannot skip half the work (round 3, MECH-L5).
        shielded = {uid for uid in unconfirmed if streak[uid] == 1
                    and self.state.rounds - self._last_shield.get(uid, -SHIELD_COOLDOWN_ROUNDS) >= SHIELD_COOLDOWN_ROUNDS}
        for uid in shielded:
            if self.state.miners[uid].role:  # a first registration has nothing to protect and uses no shield
                self._last_shield[uid] = self.state.rounds
        if unconfirmed:
            log.info("role change on uids %s seen once; applied when the next read confirms", sorted(unconfirmed))

        roles: dict[int, str] = {}
        targets: list[Neuron] = []
        for n in snap.neurons:
            if n.uid not in proposed or n.uid in unconfirmed:
                continue
            model, role = proposed[n.uid]
            if self.state.set_role(n.uid, role):
                log.info("uid %d changed its committed role to %r; score restarts from zero", n.uid, role or "none")
            if model is None or n.ip is None or n.port is None or not ip_allowed(n.ip, self.cfg.allow_private_ips):
                continue
            roles[n.uid] = model
            targets.append(n)

        rested: set[int] = set()
        if self.tiered:
            for model in sorted(m for m in REFERENCE_CAP if m in self.pools):
                pool_targets = [n for n in targets if roles[n.uid] == model]
                keep, rest = self._rotate(model, pool_targets)
                if rest:
                    rested |= {n.uid for n in rest}
                    log.info("%s: %d miners, reference cap %d decisions/round: %d verified this round, %d rested "
                             "(rotation)", model, len(pool_targets), self._reference_cap(model), len(keep), len(rest))
        queryable = list(targets)
        if rested:
            targets = [n for n in targets if n.uid not in rested]
        requests = {n.uid: self._new_request(roles[n.uid]) for n in targets}
        sem = asyncio.Semaphore(self.cfg.max_parallel)
        timeouts = {m: p.latency_slow_s for m, p in self.pools.items()}
        gathered = await asyncio.gather(
            *(self._query_s1(session, sem, n, requests[n.uid], timeouts[roles[n.uid]]) for n in targets),
            return_exceptions=True,
        )
        results = [
            r if isinstance(r, QueryResult) else QueryResult(n.uid, None, math.inf, f"internal: {type(r).__name__}")
            for n, r in zip(targets, gathered)
        ]
        # The verified set is drawn only now, after every answer has arrived.
        verify: dict[int, tuple[str, list[Decision]]] = {}
        for res in results:
            if res.answers is None:
                continue
            req = requests[res.uid]
            chosen = set(choose_verified([d.ident for d in req.decisions], self.pools[req.model].verified_per_round))
            verify[res.uid] = (req.model, [d for d in req.decisions if d.ident in chosen])
        budget = max(REFERENCE_BUDGET_MIN_S, REFERENCE_BUDGET_ROUNDS * self.cfg.round_interval_s)
        references, unscored, failed_pools = await self._references_isolated(verify, budget)
        if verify and set(failed_pools) >= {m for m, _ds in verify.values()}:
            log.error("every pool's reference runtime failed or timed out; skipping the round without scoring anyone")
            return None

        summary: dict[str, Any] = {"block": snap.block, "targets": len(targets), "ok": 0, "failed": 0, "scores": {}}
        queried = set()
        for res in results:
            queried.add(res.uid)
            if res.uid in unscored:  # our own reference failed for this miner: no update either way
                summary["scores"][res.uid] = {"model": roles[res.uid], "score": None, "error": "reference unavailable"}
                continue
            params = self.pools[roles[res.uid]]
            refs = references.get(res.uid, {})
            if res.answers is None:
                outcome_score, strike, passed, verified, worst, mean = 0.0, False, 0, 0, math.inf, math.inf
            else:
                o = score_round(res.answers, refs, res.latency_s, params)
                outcome_score, strike, passed, verified, worst, mean = (
                    o.score, o.strike, o.passed, o.verified, o.max_distance, o.mean_distance)
            fidelity = passed / verified if verified else 0.0
            if strike:
                self.state.strike(res.uid, fidelity)
            else:
                self.state.update(res.uid, outcome_score, fidelity, self.params.ema_alpha)
            self._last_verified[res.uid] = self.state.rounds
            summary["ok" if res.answers is not None else "failed"] += 1
            summary["scores"][res.uid] = {
                "model": roles[res.uid], "score": round(outcome_score, 4), "verified": verified, "passed": passed,
                "strike": strike, "max_distance": None if not math.isfinite(worst) else round(worst, 4),
                "mean_distance": None if not math.isfinite(mean) else round(mean, 4),
                "latency_s": None if math.isinf(res.latency_s) else round(res.latency_s, 3), "error": res.error,
            }
            log.info("uid %d [%s]: score %.3f items %d/%d mean_d %s max_d %s latency %s ema %.3f%s%s",
                     res.uid, roles[res.uid], outcome_score, passed, verified,
                     "-" if not math.isfinite(mean) else f"{mean:.3f}",
                     "-" if not math.isfinite(worst) else f"{worst:.3f}",
                     "-" if math.isinf(res.latency_s) else f"{res.latency_s:.2f}s", self.state.miners[res.uid].ema,
                     " STRIKE" if strike else "", f" error={res.error}" if res.error else "")
        for n in snap.neurons:
            if (n.uid not in queried and n.uid not in shielded and n.uid not in rested and n.hotkey != self.hotkey
                    and n.uid != snap.owner_uid):
                self.state.update(n.uid, 0.0, 0.0, self.params.ema_alpha)
        answered = {res.uid for res in results if res.answers is not None and res.uid not in unscored}
        # A probe measures every queryable UID of each pool at once, rested by the rotation or not, answered this round
        # or not (a UID that is down measures 0): probing only part of a pool would let a machine's hotkeys be
        # measured apart, each at the whole machine (review H2). Its verification stays bounded per pool.
        probe_targets = [n for n in queryable if roles[n.uid] in CAPACITY]
        # A UID that has never been measured (new registration or new role) triggers a probe in the next round in
        # which it answers, so it does not wait ~10 rounds at zero capacity weight.
        unmeasured = any(self.state.miners[n.uid].capacity is None for n in probe_targets if n.uid in answered)
        if self.tiered and probe_targets and (unmeasured or self._probe_due()):
            # The probe's own time (preparation, connection warm-up, window + one latency bound) comes on top of the
            # reference budget.
            probe_budget = PROBE_PREPARE_BUDGET_S + PROBE_WARMUP_S + 1.0 + budget + max((CAPACITY[m].window_s + self.pools[m].latency_slow_s
                                                                  for m in self.pools if m in CAPACITY), default=0.0)
            try:
                summary["capacity"] = await asyncio.wait_for(
                    self._probe_capacity(session, probe_targets, roles, budget), probe_budget + 5.0)
            except asyncio.TimeoutError:
                log.error("capacity probe did not finish within %.0f s; no capacity update this round", probe_budget)
            except Exception:
                log.exception("capacity probe failed; no capacity update this round")
        self.state.rounds += 1
        try:
            save(self.state, self.cfg.state_path)
        except OSError as exc:
            log.error("could not save state: %s", exc)
        log.info("round %d at block %d: %d miners queried, %d answered, %d failed",
                 self.state.rounds, snap.block, len(targets), summary["ok"], summary["failed"])
        self.last_round = summary
        self._latest_snap = snap
        drv = self.competitions
        if drv is not None:
            drv.refresh(snap)
            between = lambda: self.maybe_set_weights(self._latest_snap or snap)  # noqa: E731
            if drv.background:
                drv.kick(snap, self._not_competitors(snap), between=between)
            elif drv.due():
                try:
                    summary["engine_competition"] = await drv.run(snap, self._not_competitors(snap), between=between)
                except Exception:
                    log.exception("engine competition cycle failed; serving rounds continue")
        await self.maybe_set_weights(snap)
        return summary

    async def maybe_set_weights(self, snap: Snapshot) -> bool:
        # The competition task and the serving round both end here: one at a time.
        async with self._weights_lock:
            return await super().maybe_set_weights(snap)

    def _not_competitors(self, snap: Snapshot) -> set[int]:
        out = {n.uid for n in snap.neurons if n.hotkey == self.hotkey}
        if snap.owner_uid is not None:
            out.add(snap.owner_uid)
        return out
    # --------------------------------------------------------------- capacity (tiered pools)
    def _probe_due(self) -> bool:
        """About one round in PROBE_EVERY_ROUNDS, at a time miners cannot predict."""
        every = min(self.probe_every, PROBE_EVERY_ROUNDS)  # tests patch the module constant
        return self.state.rounds >= 1 and secrets.randbelow(every) == 0

    async def _probe_capacity(self, session: aiohttp.ClientSession, targets: list[Neuron],
                              roles: Mapping[int, str], reference_budget: float = REFERENCE_BUDGET_MIN_S
                              ) -> dict[int, Any]:
        """Verified throughput of every target, all probed at the same moment.

        Each UID is offered ``inflight`` concurrent lanes of fresh requests (it accepts as many as its own limit allows;
        a refused lane just ends) and throughput = decisions answered inside one window common to all UIDs / window.
        Because every UID of a pool is loaded at once and only answers inside the common window count, UIDs served by
        one shared backend together measure at most what that backend does in the window, whenever any of them stops.

        Inside the window the validator only sends pre-serialised bodies and records (response, headers, arrival
        time); answers are counted by request size. Signatures and answers are checked after the window, so the
        validator's own CPU is not shared out per lane while it measures (review C1): any forged or malformed answer
        sets that UID's probe to 0. A secret sample of the answers is then verified against the reference like a
        normal round: a failed item is a strike, so fast garbage cannot buy capacity. If the validator's event loop
        still lags during the window (or the probe total reaches ``REAX_PROBE_CEILING_DPS``), the measurement is
        bounded by the validator, not the miners, and the probe is dropped without any capacity update.
        """
        lanes_total = max(1, sum(CAPACITY[roles[n.uid]].inflight for n in targets))
        # Own semaphore and connection pool sized for every lane: probe lanes never queue behind each other (a shared,
        # smaller limit would under-count UIDs by queue position, and hanging UIDs could starve honest lanes).
        sem = asyncio.Semaphore(lanes_total)

        def budget(uid: int) -> float:
            cap = CAPACITY[roles[uid]]
            ema = self.state.miners[uid].capacity  # None = new or reset UID: start from the target
            return cap.budget_dps(None if ema is None else max(ema, self._probe_last.get(uid, 0.0)))

        budgets = {n.uid: budget(n.uid) for n in targets}

        # Every request is prepared and serialised before the window opens (in a worker thread): generating them inside
        # the window would start the UIDs' windows at different moments, and only a common window keeps a shared
        # machine's hotkeys from taking turns. Only (decisions, body) is kept; the Decision objects of the requests
        # that were really sent are rebuilt from their bodies after the window (review M3: memory).
        def prepare() -> dict[int, list[tuple[int, bytes]]]:
            out = {}
            for n in targets:
                cap = CAPACITY[roles[n.uid]]
                out[n.uid] = [(cap.items_per_request, self._new_request(roles[n.uid], cap.items_per_request).to_bytes())
                              for _ in range(cap.requests_for(budgets[n.uid]))]
            return out

        prepared = await asyncio.to_thread(prepare)
        # The reference deadline starts after the preparation, which can take seconds at the cap (review M3).
        deadline = time.monotonic() + reference_budget + max(
            (CAPACITY[roles[n.uid]].window_s + self.pools[roles[n.uid]].latency_slow_s for n in targets), default=0.0)

        async def probe(n: Neuron, psession: aiohttp.ClientSession, started: float
                        ) -> tuple[int, list[tuple[bytes, bytes, CIMultiDict]], float, int]:
            model = roles[n.uid]
            cap, params = CAPACITY[model], self.pools[model]
            pending = iter(prepared[n.uid])
            done: list[tuple[bytes, bytes, CIMultiDict]] = []  # (request body, raw response, headers), checked later
            in_window = 0

            async def lane() -> None:
                nonlocal in_window
                for size, body in pending:  # one iterator shared by the lanes
                    if time.monotonic() - started > cap.window_s:
                        return
                    try:
                        raw, headers, _latency = await self._post_s1(psession, sem, n, body, 2 * params.latency_slow_s)
                    except Exception as exc:  # a refusal, timeout or network error ends the lane
                        log.debug("uid %d capacity probe lane ended: %s", n.uid, self._s1_error(n, exc))
                        return
                    arrived = time.monotonic()
                    done.append((body, raw, headers))
                    if arrived - started <= cap.window_s:
                        in_window += size

            # A request started just before the window ends may run up to its own timeout; stop waiting one latency
            # bound after the window so a slow miner cannot stretch the probe (its late answers would not count).
            lanes = [asyncio.ensure_future(lane()) for _ in range(cap.inflight)]
            try:
                await asyncio.wait(lanes, timeout=cap.window_s + params.latency_slow_s)
            finally:  # also when the whole probe is cancelled by its time budget
                for task in lanes:
                    task.cancel()
                await asyncio.gather(*lanes, return_exceptions=True)
            return n.uid, done, cap.window_s, in_window

        async def warm(n: Neuron, psession: aiohttp.ClientSession) -> None:
            assert n.ip is not None and n.port is not None
            host = f"[{n.ip}]" if ":" in n.ip else n.ip
            url = f"http://{host}:{n.port}/health"

            async def one() -> None:
                try:
                    async with psession.get(url, allow_redirects=False,
                                            timeout=aiohttp.ClientTimeout(total=PROBE_WARMUP_S)) as resp:
                        await resp.content.read(4096)
                except Exception:
                    pass  # this lane connects inside the window, as before

            await asyncio.gather(*(one() for _ in range(CAPACITY[roles[n.uid]].inflight)))

        connector = aiohttp.TCPConnector(limit=lanes_total)
        lag = _LoopLag()
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as psession:
            # Open the lanes' connections before the window (review L2): thousands of TCP connects at the window's start
            # would stall the validator's event loop and delay the first answers of the UIDs connected last.
            await asyncio.wait_for(asyncio.gather(*(warm(n, psession) for n in targets)), PROBE_WARMUP_S + 1.0)
            started = time.monotonic()  # one window for every UID
            ticker = asyncio.ensure_future(lag.run())
            try:
                probed = await asyncio.gather(*(probe(n, psession, started) for n in targets))
            finally:
                ticker.cancel()
                await asyncio.gather(ticker, return_exceptions=True)
        total_dps = math.fsum(items / seconds for _uid, _done, seconds, items in probed if seconds > 0)
        ceiling = _probe_ceiling_dps()
        p95 = lag.p95()
        skipped = p95 > PROBE_LAG_P95_MAX_S or bool(ceiling and total_dps >= ceiling)
        self.last_probe_stats = {"uids": len(targets), "total_dps": round(total_dps, 1),
                                 "lag_p95_ms": round(1000 * p95, 1), "skipped": skipped}
        if skipped:
            log.error("capacity probe skipped: the validator was the bottleneck (event-loop lag p95 %.0f ms, limit "
                      "%.0f ms; %.0f decisions/s in total%s); no capacity update this probe (see scripts/bench_probe.py)",
                      1000 * p95, 1000 * PROBE_LAG_P95_MAX_S, total_dps,
                      f", ceiling {ceiling:.0f}/s" if ceiling else "")
            return {}

        # Signatures and answers, checked now (in a worker thread, the nonce store is thread-safe). One forged or
        # malformed answer sets the UID's probe to 0; requests that were not answered are never rebuilt.
        def check() -> dict[int, Optional[tuple[dict, dict[str, Decision]]]]:
            out: dict[int, Optional[tuple[dict, dict[str, Decision]]]] = {}
            by_uid = {n.uid: n for n in targets}
            for uid, done, _seconds, _items in probed:
                if not done:
                    continue
                n, cap, params = by_uid[uid], CAPACITY[roles[uid]], self.pools[roles[uid]]
                max_age = cap.window_s + 2 * params.latency_slow_s + 10.0
                answers: dict = {}
                decisions: dict[str, Decision] = {}
                try:
                    for body, raw, headers in done:
                        req = parse_s1_request(body)
                        answers.update(self._check_s1(n, req, raw, headers, max_age))
                        decisions.update((d.ident, d) for d in req.decisions)
                except Exception as exc:
                    log.warning("uid %d [%s] capacity probe: invalid answer (%s); probe counts 0", uid, roles[uid],
                                self._s1_error(n, exc))
                    out[uid] = None
                    continue
                out[uid] = (answers, decisions)
            return out

        checked = await asyncio.to_thread(check)
        # The pool's reference cap also bounds a probe's verification: shared out over the probed UIDs (at least 2).
        per_pool = {m: sum(1 for n in targets if roles[n.uid] == m) for m in {roles[n.uid] for n in targets}}
        verify: dict[int, tuple[str, list[Decision]]] = {}
        for uid, got in checked.items():
            if got is None:
                continue
            decisions = got[1]
            k = CAPACITY[roles[uid]].verified(len(decisions))
            pool_cap = self._reference_cap(roles[uid])
            if pool_cap is not None:
                k = min(k, max(2, pool_cap // per_pool[roles[uid]]))
            chosen = choose_verified(sorted(decisions), k)
            verify[uid] = (roles[uid], [decisions[i] for i in chosen])
        left = max(1.0, deadline - time.monotonic())
        references, unscored, _failed_pools = await self._references_isolated(verify, left)
        report: dict[int, Any] = {}
        for uid, done, elapsed, items in probed:
            if uid in unscored:
                continue
            model = roles[uid]
            cap, params = CAPACITY[model], self.pools[model]
            got = checked.get(uid)
            dps = items / elapsed if got is not None and elapsed > 0 else 0.0
            o = score_round(got[0], references.get(uid, {}), 0.0, params) if got is not None else None
            if o is not None and o.strike:
                self.state.strike(uid, o.passed / o.verified if o.verified else 0.0)
                dps = 0.0
            elif o is None or o.mean_distance > params.tau_mean:
                dps = 0.0
            # Clamp before averaging: one burst far above the cap must not carry the miner for many probes.
            dps = min(dps, cap.cap_dps)
            self._probe_last[uid] = dps
            self.state.update_capacity(uid, dps, cap.alpha)
            report[uid] = {"model": model, "items": items, "seconds": round(elapsed, 3), "dps": round(dps, 3),
                           "budget_dps": round(budgets[uid], 3), "invalid": bool(done) and got is None,
                           "strike": bool(o and o.strike), "capacity": round(self.state.miners[uid].capacity or 0.0, 3)}
            log.info("uid %d [%s] capacity probe: %d decisions in %.2fs = %.2f/s%s; moving average %.2f/s",
                     uid, model, items, elapsed, dps, " STRIKE" if o and o.strike else "",
                     self.state.miners[uid].capacity or 0.0)
        return report

    def compute_weights(self, snap: Snapshot) -> dict[int, float]:
        ema_by_pool: dict[str, dict[int, float]] = {m: {} for m in self.pools}
        current = {f"{model}:{man.manifest_id}": model for model, man in self.manifests.items()}
        for uid, m in self.state.miners.items():
            model = current.get(m.role)  # only roles on the current manifest count (MECH-M1)
            if model is not None:
                ema_by_pool[model][uid] = m.ema
        if self.tiered:
            capacity_by_pool: dict[str, dict[int, float]] = {m: {} for m in self.pools}
            for uid, m in self.state.miners.items():
                model = current.get(m.role)
                if model is not None and m.capacity is not None:
                    capacity_by_pool[model][uid] = m.capacity
            base = compute_tiered_weights(ema_by_pool, capacity_by_pool, self.budgets, snap.owner_uid,
                                          self.params.min_ema, self.params.burn_fraction)
        else:
            base = compute_pool_weights(ema_by_pool, self.shares, snap.owner_uid, self.params.min_ema,
                                        self.params.burn_fraction)
        if self.engine is None:
            return base
        return self.engine.weights(base, snap, self._not_competitors(snap), self.params.burn_fraction)
