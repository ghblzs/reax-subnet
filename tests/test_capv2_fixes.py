"""Fixes for the capacity-v2 review (1.2.0rc3): C1, H2, M1, M2, M3, L1. H1 and H3 are in test_tiers.py.

The probe tests reuse the fake-machine arena of test_tiers.py (real aiohttp miners on 127.0.0.1, a faked chain).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from reax_subnet import s1validator, tiers, version
from reax_subnet.miner import FairSlots, Overloaded, S1Miner
from reax_subnet.s1protocol import S1Request, parse_s1_request
from reax_subnet.s1validator import S1Validator
from test_tiers import (PROBE_V2, Machine, OverLink, _measured, _probe_arena,  # noqa: F401 (fixtures)
                        _release_floors_off, probe_v2)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _probe_reports(summaries):
    return [s["capacity"] for s in summaries if s and "capacity" in s]


# ----------------------------------------------------------------- C1: nothing but I/O inside the window
def _generator(tmp_path, wallet, fake_chain_cls) -> S1Validator:
    """A validator object used only to draw requests (all pools enabled)."""
    from reax_subnet.config import load_validator
    from reax_subnet.s1backends import S1HashBackend

    env = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_TIERED_POOLS": "1", "REAX_STATE_PATH": str(tmp_path / "gen-state.json")}
    return S1Validator(load_validator(env), wallet, fake_chain_cls(neurons=[]),
                       {m: S1HashBackend(m) for m in s1validator.POOL_PARAMS})


def test_a_prepared_body_rebuilds_the_same_request(make_wallet, fake_chain_cls, tmp_path):
    """The probe keeps only the serialised bodies and rebuilds the sent requests from them after the window."""
    val = _generator(tmp_path, make_wallet("//Validator"), fake_chain_cls)
    assert set(val.pools) == set(s1validator.POOL_PARAMS)
    for model in val.pools:
        for items in (1, 4, 16):
            req = val._new_request(model, items)
            assert parse_s1_request(req.to_bytes()) == req


async def test_probe_bodies_are_serialised_before_the_window(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                             monkeypatch):
    probe_v2()
    calls = {"main": 0, "worker": 0}
    original = S1Request.to_bytes

    def counting(self):
        calls["main" if threading.current_thread() is threading.main_thread() else "worker"] += 1
        return original(self)

    monkeypatch.setattr(S1Request, "to_bytes", counting)
    groups = [(Machine(200.0), [{}]), (Machine(200.0), [{}, {}])]
    _val, _by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert len(_probe_reports(summaries)) == 1
    # On the event loop only round 2's 3 requests (round 1 confirms the roles); every probe body was built in the
    # worker thread.
    assert calls["main"] == 3
    assert calls["worker"] >= 3 * PROBE_V2.requests_for(PROBE_V2.target_dps)


async def test_answers_are_checked_after_the_window_not_inside_it(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                                  monkeypatch):
    """Signature checks and parsing happen in the post-window worker thread, never on the event loop in the window."""
    probe_v2()
    where = {"main": 0, "worker": 0}
    original = S1Validator._check_s1

    def counting(self, *a, **kw):
        where["main" if threading.current_thread() is threading.main_thread() else "worker"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(S1Validator, "_check_s1", counting)
    groups = [(Machine(200.0), [{}, {}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert where["main"] == 2  # round 2's answers (round 1 only confirms the roles)
    assert where["worker"] > 20  # every probe answer, checked after the window
    assert 150.0 <= _measured(val, by_group[0]) <= 210.0


class TamperingMiner(S1Miner):
    """Answers normal rounds honestly; in a probe (16-decision requests) its ``nth`` answer is forged or malformed."""

    nth = 3
    mode = "forged"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.probe_answers = 0

    async def _decide(self, caller, body):
        resp = await super()._decide(caller, body)
        if resp.status != 200 or len(parse_s1_request(body).decisions) < 16:
            return resp
        self.probe_answers += 1
        if self.probe_answers != self.nth:
            return resp
        from aiohttp import web

        from reax_subnet import auth

        raw = bytes(resp.body)
        if self.mode == "forged":  # the signed body is changed afterwards: the signature no longer matches
            raw = raw.replace(b'"miner_version"', b'"miner_version" ', 1)
            headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
        else:  # correctly signed, but not an answer to this request
            req_id = parse_s1_request(body).request_id
            raw = raw.replace(req_id.encode(), b"f" * len(req_id), 1)
            headers = auth.sign_response(self.wallet, raw, req_id, caller, path=self.path)
            headers["Content-Type"] = "application/json"
        return web.Response(body=raw, headers=headers)


@pytest.mark.parametrize("mode", ["forged", "malformed"])
async def test_one_forged_or_malformed_probe_answer_sets_the_probe_to_zero(mode, probe_v2, make_wallet,
                                                                           fake_chain_cls, tmp_path, monkeypatch):
    probe_v2()
    monkeypatch.setattr(TamperingMiner, "mode", mode)
    groups = [(Machine(200.0), [{}]), (Machine(200.0), [{"miner_cls": TamperingMiner}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    (report,) = _probe_reports(summaries)
    honest, bad = by_group[0][0], by_group[1][0]
    assert report[bad]["invalid"] is True and report[bad]["dps"] == 0.0 and val.state.miners[bad].capacity == 0.0
    assert report[honest]["invalid"] is False and report[honest]["dps"] > 150.0
    assert val.state.miners[bad].ema > 0  # the normal rounds were honest: no strike, only the probe counts 0


class HealthCounter(S1Miner):
    health_hits = 0

    async def handle_health(self, request):
        type(self).health_hits += 1
        return await super().handle_health(request)


async def test_probe_lanes_connect_before_the_window(probe_v2, make_wallet, fake_chain_cls, tmp_path, monkeypatch):
    """Review L2: thousands of TCP connects at the window's start would stall the validator's loop; each lane's
    connection is opened with a GET /health before the window instead."""
    probe_v2()
    monkeypatch.setattr(HealthCounter, "health_hits", 0)
    groups = [(Machine(200.0), [{"miner_cls": HealthCounter}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert HealthCounter.health_hits == PROBE_V2.inflight
    assert _measured(val, by_group[0]) > 150.0


class LoopBlocker(OverLink):
    """A miner that blocks the shared event loop (in this arena the validator and miners share one loop), standing in
    for a validator whose CPU is saturated during the window."""

    async def decide(self, decisions):
        if len(decisions) >= 16:
            time.sleep(0.08)
        return await super().decide(decisions)


async def test_a_saturated_validator_skips_the_capacity_update(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                              monkeypatch, caplog):
    probe_v2()
    monkeypatch.setattr("test_tiers.OverLink", LoopBlocker)
    groups = [(Machine(200.0), [{}, {}])]
    with caplog.at_level("ERROR", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert _probe_reports(summaries) == [{}]
    assert all(val.state.miners[u].capacity is None for u in by_group[0])  # nothing recorded, not a low number
    assert "capacity probe skipped: the validator was the bottleneck" in caplog.text


async def test_a_dropped_probe_is_not_retried_every_round(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                           monkeypatch):
    """New UIDs normally trigger a probe in their first answered round; after a saturated (dropped) probe the
    validator waits for the normal schedule instead of loading itself again every round."""
    probe_v2()
    monkeypatch.setattr(s1validator, "PROBE_LAG_P95_MAX_S", -1.0)  # every probe counts as saturated
    monkeypatch.setattr(s1validator.S1Validator, "_probe_due", lambda self: False)
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 10)  # the normal schedule: about one round in 10
    groups = [(Machine(200.0), [{}])]
    _val, _by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=5)
    assert _probe_reports(summaries) == [{}]  # one probe (round 2), then none


async def test_a_probe_at_the_calibrated_ceiling_is_skipped(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                            monkeypatch):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_CEILING_DPS", "100")
    groups = [(Machine(200.0), [{}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert _probe_reports(summaries) == [{}] and val.state.miners[by_group[0][0]].capacity is None
    monkeypatch.setenv("REAX_PROBE_CEILING_DPS", "100000")
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert val.state.miners[by_group[0][0]].capacity > 150.0


def test_the_ceiling_setting_ignores_junk(monkeypatch):
    for raw, want in (("", 0.0), ("abc", 0.0), ("-5", 0.0), ("nan", 0.0), ("inf", 0.0), ("12000", 12000.0)):
        monkeypatch.setenv("REAX_PROBE_CEILING_DPS", raw)
        assert s1validator._probe_ceiling_dps() == want


# ----------------------------------------------------------------- H2: the whole pool, rested or not
async def test_rotation_never_lets_siblings_be_probed_apart(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                            monkeypatch):
    """Review H2 (probes/test_rotation_sybil.py): under a reference cap one of three UIDs rests every round. The probe
    still measures all three at once, so the machine behind two hotkeys never measures more than the solo machine."""
    probe_v2()
    monkeypatch.setattr(s1validator, "REFERENCE_CAP", {"s1-fast": 8})  # room = 8 // 4 = 2 UIDs per round
    groups = [(Machine(160.0), [{}]), (Machine(160.0), [{}, {}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=5)
    reports = _probe_reports(summaries)
    everyone = set(by_group[0]) | set(by_group[1])
    assert len(reports) == 4 and all(set(r) == everyone for r in reports)
    for r in reports:
        pair = sum(r[u]["dps"] for u in by_group[1])
        assert pair <= 1.15 * r[by_group[0][0]]["dps"]


async def test_a_down_uid_is_probed_and_measures_zero(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    """A UID that does not answer this round is still in the probe: it measures 0 instead of keeping its old number."""
    probe_v2()
    groups = [(Machine(200.0), [{}]), (Machine(200.0), [{"quit_after": 1}])]  # answers round 2's request only
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=3)
    reports = _probe_reports(summaries)
    down = by_group[1][0]
    assert all(down in r for r in reports) and reports[-1][down]["dps"] == 0.0
    assert val.state.miners[down].capacity == 0.0


# ----------------------------------------------------------------- M3: the reference deadline after preparation
async def test_the_reference_deadline_starts_after_the_preparation(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                                   monkeypatch):
    probe_v2()
    budgets: list[float] = []
    original_refs = S1Validator._references_isolated
    original_new = S1Validator._new_request
    slept = []

    async def capture(self, verify, budget):
        budgets.append(budget)
        return await original_refs(self, verify, budget)

    def slow(self, model, items=None):
        if threading.current_thread() is not threading.main_thread() and not slept:
            slept.append(1)
            time.sleep(1.5)  # a probe preparation that takes 1.5 s
        return original_new(self, model, items)

    monkeypatch.setattr(S1Validator, "_references_isolated", capture)
    monkeypatch.setattr(S1Validator, "_new_request", slow)
    groups = [(Machine(200.0), [{}])]
    await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert slept and len(budgets) == 3  # round 1, round 2, round 2's probe
    # deadline = end of preparation + budget + window + latency bound; the probe used about window (+ tail)
    assert budgets[-1] >= budgets[1] - 0.3


def test_the_round_allows_for_the_probe_preparation():
    assert s1validator.PROBE_PREPARE_BUDGET_S >= 30.0


# ----------------------------------------------------------------- M2: file descriptors for probe connections
def test_validator_services_raise_the_open_file_limit():
    yaml = pytest.importorskip("yaml")
    for f in ("deploy/validator-1.2/compose.yml", "deploy/docker-compose.s1-validator.yml"):
        svc = yaml.safe_load(open(os.path.join(ROOT, f)))["services"]["validator"]
        assert svc["ulimits"]["nofile"] == {"soft": 65536, "hard": 65536}, f


# ----------------------------------------------------------------- L1: release identifiers
def test_scoring_change_bumps_the_weights_version():
    assert version.WEIGHTS_VERSION_KEY == 1230 and version.__version__ == "1.2.0rc8.post2"
    changelog = open(os.path.join(ROOT, "CHANGELOG.md")).read()
    assert "## 1.2.0rc3" in changelog
    assert "REAX_PER_CALLER_CONCURRENCY=16" in changelog and "REAX_PER_CALLER_PER_MINUTE=240" in changelog


def test_items_per_request_fit_the_protocol():
    """Review L5: a probe request above MAX_DECISIONS would be a protocol error, so every UID would measure 0."""
    from reax_subnet.s1protocol import MAX_DECISIONS

    for pool, cap in tiers.CAPACITY.items():
        assert 1 <= cap.items_per_request <= MAX_DECISIONS, pool


# ----------------------------------------------------------------- M1: fair share of the miner between validators
async def test_fair_slots_alternate_between_callers():
    slots = FairSlots(1, 64)
    await slots.acquire("a")  # the only slot
    order: list[str] = []

    async def one(caller):
        await slots.acquire(caller)
        order.append(caller)

    tasks = [asyncio.ensure_future(one("attacker")) for _ in range(10)]
    await asyncio.sleep(0)
    tasks += [asyncio.ensure_future(one("prober")) for _ in range(3)]
    await asyncio.sleep(0)
    for _ in range(13):
        slots.release()
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    assert order[:6] == ["attacker", "prober"] * 3  # the prober is served every other slot, not after 10 requests


async def test_a_full_queue_makes_room_for_a_caller_below_its_share():
    slots = FairSlots(1, 4)
    await slots.acquire("x")
    flood = [asyncio.ensure_future(slots.acquire("attacker")) for _ in range(4)]
    await asyncio.sleep(0)
    with pytest.raises(Overloaded):  # the attacker itself gets no fifth place
        await slots.acquire("attacker")
    honest = asyncio.ensure_future(slots.acquire("prober"))
    for _ in range(3):
        await asyncio.sleep(0)
    refused = [t for t in flood if t.done()]
    assert len(refused) == 1 and isinstance(refused[0].exception(), Overloaded)
    slots.release()  # next in turn: the attacker's oldest, then the prober
    await asyncio.sleep(0)
    slots.release()
    await asyncio.sleep(0)
    assert honest.done() and honest.exception() is None
    for t in flood:
        if not t.done():
            t.cancel()
    await asyncio.gather(*flood, return_exceptions=True)


async def test_cancelled_waiters_do_not_leak_slots():
    slots = FairSlots(2, 8)
    await slots.acquire("a")
    await slots.acquire("a")
    w = asyncio.ensure_future(slots.acquire("b"))
    await asyncio.sleep(0)
    w.cancel()
    await asyncio.gather(w, return_exceptions=True)
    slots.release()
    slots.release()
    await asyncio.wait_for(slots.acquire("c"), 1)
    await asyncio.wait_for(slots.acquire("c"), 1)
    assert slots.waiting == 0


async def test_a_burst_from_a_second_validator_does_not_starve_a_probe(make_wallet, fake_chain_cls, tmp_path):
    """End to end on the real miner: a staked caller bursting 16 requests at once while a validator's 4 lanes probe.
    With FIFO admission the probe's requests would queue behind the whole burst."""
    import aiohttp

    from reax_subnet import auth
    from reax_subnet.s1protocol import S1_DECIDE_PATH
    from test_tiers import OnMachine, _serve

    machine = Machine(160.0)  # 16 decisions take 0.1 s
    miner_w, prober, attacker = make_wallet("//Miner/fair"), make_wallet("//V/prober"), make_wallet("//V/attacker")
    m = S1Miner(miner_w, OnMachine("s1-fast", machine), max_concurrent=1, max_queue=32, per_caller_concurrency=16,
                per_caller_per_minute=600, max_pending_per_ip=64)
    m.set_validators(frozenset({prober.hotkey.ss58_address, attacker.hotkey.ss58_address}))
    runner, port = await _serve(m.app())
    gen = _generator(tmp_path, prober, fake_chain_cls)
    url = f"http://127.0.0.1:{port}{S1_DECIDE_PATH}"
    done_at: dict[str, list[float]] = {"prober": [], "attacker": []}

    async def send(session, wallet, who):
        body = gen._new_request("s1-fast", 16).to_bytes()
        headers = auth.sign_request(wallet, body, miner_w.hotkey.ss58_address, path=S1_DECIDE_PATH)
        headers["Content-Type"] = "application/json"
        async with session.post(url, data=body, headers=headers) as resp:
            assert resp.status == 200
            await resp.read()
        done_at[who].append(time.monotonic())

    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            t0 = time.monotonic()
            burst = [asyncio.ensure_future(send(session, attacker, "attacker")) for _ in range(16)]
            await asyncio.sleep(0.05)
            await asyncio.gather(*(send(session, prober, "prober") for _ in range(4)), *burst)
    finally:
        await runner.cleanup()
    # FIFO: the 4 probe requests finish after all 16 attacker requests (~2.0 s). Round-robin: within ~1 s.
    assert max(done_at["prober"]) - t0 < 1.3
    assert max(done_at["attacker"]) > max(done_at["prober"])
