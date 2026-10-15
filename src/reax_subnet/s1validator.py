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

from . import auth
from .chain import Neuron, Snapshot
from .manifest import load_manifest, parse_commitment
from .protocol import ProtocolError
from .s1backends import S1Backend
from .s1contract import Decision, parse_decision
from .s1items import generate_s1_items
from .s1protocol import MAX_RESPONSE_BYTES, S1_DECIDE_PATH, S1Request, parse_s1_response
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
    # Tiered pools (REAX_FEATURE_TIERED_POOLS, development networks only). Tolerances from the cross-backend
    # calibration in docs/DESIGN-POOLS.md section 6 (llama.cpp CPU x86 / aarch64 / CUDA against each other);
    # the latency curve fits the tier's hardware (4 decisions per request).
    "s1-lite": PoolParams("s1-lite", temperature=2.07, tau_mean=0.20, tau_strike=0.50, tau_item=2.0,
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


def s1_endpoint_url(ip: str, port: int) -> str:
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}{S1_DECIDE_PATH}"


class S1Validator(Validator):
    def __init__(self, cfg, wallet, chain, backends: Mapping[str, S1Backend],
                 pools: Optional[Mapping[str, PoolParams]] = None, shares: Optional[Mapping[str, float]] = None):
        super().__init__(cfg, wallet, chain, backend=None)  # type: ignore[arg-type]
        self.tiered = bool(getattr(cfg, "tiered_pools", False))
        # Development networks may probe more often (localnet runs last minutes, not hours).
        self.probe_every = PROBE_EVERY_ROUNDS
        dev_every = os.environ.get("REAX_DEV_PROBE_EVERY_ROUNDS")
        if self.tiered and dev_every and cfg.common.allow_dev_models:
            self.probe_every = max(1, int(dev_every))
        self.budgets = TierBudgets()
        if self.tiered and shares is None:
            shares = pool_shares(["s1-fast", *(["s1-pro"] if cfg.s1_pro_pool else []), *TIERED_POOLS], self.budgets)
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
            seen[n.uid] = n.commitment
            if self._rejected_commitment.get(n.uid) != n.commitment:
                log.warning("uid %d is not queried: commitment %r: %s", n.uid, n.commitment[:80],
                            self._why_rejected(n.commitment))
        self._rejected_commitment = seen

    # --------------------------------------------------------------- query
    async def _query_s1(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, neuron: Neuron,
                        req: S1Request, timeout_s: float) -> QueryResult:
        assert neuron.ip is not None and neuron.port is not None
        body = req.to_bytes()
        async with sem:
            headers = auth.sign_request(self.wallet, body, neuron.hotkey, path=S1_DECIDE_PATH)
            headers["Content-Type"] = "application/json"
            started = time.monotonic()
            try:
                async with session.post(s1_endpoint_url(neuron.ip, neuron.port), data=body, headers=headers,
                                        allow_redirects=False, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                    if resp.status != 200:
                        return QueryResult(neuron.uid, None, math.inf, f"http {resp.status}")
                    raw = await _read_limited(resp, MAX_RESPONSE_BYTES)
                    latency = time.monotonic() - started
                    auth.verify_response(resp.headers, raw, req.request_id, self.hotkey, neuron.hotkey,
                                         self.nonce_store, max_age=timeout_s + 10.0, path=S1_DECIDE_PATH)
                    return QueryResult(neuron.uid, parse_s1_response(raw, req), latency)
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
        try:
            references, unscored, failed, total = await asyncio.wait_for(self._references(verify), budget)
        except asyncio.TimeoutError:
            log.error("reference runtime did not finish %d miners' items within %.0f s; skipping the round without "
                      "scoring anyone (check the runtime sidecar: it may be hung while /health stays green)",
                      len(verify), budget)
            return None
        except Exception:
            log.exception("reference computation failed; skipping round without scoring anyone")
            return None
        if total and failed > MAX_REFERENCE_FAILURE_SHARE * total:
            log.error("reference runtime failed on %d of %d items; skipping the round without scoring anyone "
                      "(check the runtime sidecar)", failed, total)
            return None
        if failed:
            log.warning("reference runtime failed on %d of %d items; %d miner(s) left unscored this round",
                        failed, total, len(unscored))

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
            if n.uid not in queried and n.uid not in shielded and n.hotkey != self.hotkey and n.uid != snap.owner_uid:
                self.state.update(n.uid, 0.0, 0.0, self.params.ema_alpha)
        if self.tiered and self._probe_due():
            answered = {res.uid for res in results if res.answers is not None and res.uid not in unscored}
            probe_targets = [n for n in targets if n.uid in answered and roles[n.uid] in CAPACITY]
            # The probe's own time (window + one latency bound) comes on top of the reference budget.
            probe_budget = budget + max((CAPACITY[m].window_s + self.pools[m].latency_slow_s
                                         for m in self.pools if m in CAPACITY), default=0.0)
            try:
                summary["capacity"] = await asyncio.wait_for(
                    self._probe_capacity(session, probe_targets, roles), probe_budget)
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
        await self.maybe_set_weights(snap)
        return summary

    # --------------------------------------------------------------- capacity (tiered pools)
    def _probe_due(self) -> bool:
        """About one round in PROBE_EVERY_ROUNDS, at a time miners cannot predict."""
        every = min(self.probe_every, PROBE_EVERY_ROUNDS)  # tests patch the module constant
        return self.state.rounds >= 1 and secrets.randbelow(every) == 0

    async def _probe_capacity(self, session: aiohttp.ClientSession, targets: list[Neuron],
                              roles: Mapping[int, str]) -> dict[int, Any]:
        """Verified throughput of every target, all probed at the same moment.

        Each UID gets ``inflight`` concurrent lanes of fresh requests until the window ends; throughput = decisions
        answered within the window / window (a UID that answered all ``max_requests`` before the window ended is above
        the target: decisions / time to its last answer). Because every UID of a pool is
        loaded at once, UIDs served by one shared backend split its throughput instead of each reporting all of
        it. A secret sample of the answers is verified against the reference like a normal round: a failed item
        is a strike, so fast garbage cannot buy capacity.
        """
        sem = asyncio.Semaphore(self.cfg.max_parallel)

        async def probe(n: Neuron) -> tuple[int, list[tuple[S1Request, dict]], float, int]:
            model = roles[n.uid]
            cap, params = CAPACITY[model], self.pools[model]
            pending = iter([self._new_request(model, cap.items_per_request) for _ in range(cap.max_requests)])
            done: list[tuple[S1Request, dict]] = []
            in_window = 0
            started = time.monotonic()
            last = started

            async def lane() -> None:
                nonlocal last, in_window
                for req in pending:  # one iterator shared by the lanes
                    if time.monotonic() - started > cap.window_s:
                        return
                    res = await self._query_s1(session, sem, n, req, 2 * params.latency_slow_s)
                    if res.answers is None:
                        return
                    done.append((req, res.answers))
                    last = time.monotonic()
                    if last - started <= cap.window_s:
                        in_window += len(req.decisions)

            # A request started just before the window ends may run up to its own timeout; stop waiting one latency
            # bound after the window so a slow miner cannot stretch the probe (its late answers would not count).
            lanes = [asyncio.ensure_future(lane()) for _ in range(cap.inflight)]
            _finished, late = await asyncio.wait(lanes, timeout=cap.window_s + params.latency_slow_s)
            for task in late:
                task.cancel()
            await asyncio.gather(*late, return_exceptions=True)
            exhausted = len(done) == cap.max_requests and last - started <= cap.window_s
            seconds = (last - started) if exhausted else cap.window_s
            return n.uid, done, seconds, sum(len(r.decisions) for r, _a in done) if exhausted else in_window

        probed = await asyncio.gather(*(probe(n) for n in targets))
        verify: dict[int, tuple[str, list[Decision]]] = {}
        answers: dict[int, dict] = {}
        for uid, done, _seconds, _items in probed:
            if not done:
                continue
            answers[uid] = {k: v for _req, a in done for k, v in a.items()}
            decisions = {d.ident: d for req, _a in done for d in req.decisions}
            chosen = choose_verified(sorted(decisions), CAPACITY[roles[uid]].verified)
            verify[uid] = (roles[uid], [decisions[i] for i in chosen])
        references, unscored, failed, total = await self._references(verify)
        if total and failed > MAX_REFERENCE_FAILURE_SHARE * total:
            log.error("reference runtime failed on %d of %d probe items; no capacity update", failed, total)
            return {}
        report: dict[int, Any] = {}
        for uid, done, elapsed, items in probed:
            if uid in unscored:
                continue
            model = roles[uid]
            cap, params = CAPACITY[model], self.pools[model]
            dps = items / elapsed if done and elapsed > 0 else 0.0
            o = score_round(answers.get(uid), references.get(uid, {}), 0.0, params) if done else None
            if o is not None and o.strike:
                self.state.strike(uid, o.passed / o.verified if o.verified else 0.0)
                dps = 0.0
            elif o is None or o.mean_distance > params.tau_mean:
                dps = 0.0
            self.state.update_capacity(uid, dps, cap.alpha)
            report[uid] = {"model": model, "items": items, "seconds": round(elapsed, 3), "dps": round(dps, 3),
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
            return compute_tiered_weights(ema_by_pool, capacity_by_pool, self.budgets, snap.owner_uid,
                                          self.params.min_ema, self.params.burn_fraction)
        return compute_pool_weights(ema_by_pool, self.shares, snap.owner_uid, self.params.min_ema,
                                    self.params.burn_fraction)
