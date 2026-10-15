"""Tiered pools (REAX_FEATURE_TIERED_POOLS): budgets, tier-order cap, capacity probes, sybil splitting, readouts.

Miners are real aiohttp servers on 127.0.0.1 and the chain is faked, as in test_s1.py. Capacity is simulated with
a hash backend that holds a lock while it "computes": UIDs that share one backend share its throughput.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.util
import itertools
import math
import random
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import s1validator, tiers
from reax_subnet.chain import Neuron
from reax_subnet.config import ConfigError, load_common, load_validator
from reax_subnet.manifest import commitment_text, load_manifest
from reax_subnet.miner import S1Miner
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1items import generate_s1_items
from reax_subnet.s1scoring import PoolParams
from reax_subnet.s1validator import S1Validator
from reax_subnet.state import ValidatorState, load, save
from reax_subnet.tiers import CapacityParams, TierBudgets, compute_tiered_weights, pool_shares

ROOT = Path(__file__).resolve().parent.parent
DEV = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1"}


# ----------------------------------------------------------------- budgets
def test_pool_shares_follow_tier_budgets():
    b = TierBudgets()
    s = pool_shares(["s1-fast", "s1-lite", "s1-cpu"], b)
    assert s == pytest.approx({"s1-fast": 0.80, "s1-lite": 0.15, "s1-cpu": 0.05})
    s = pool_shares(["s1-fast", "s1-pro", "s1-lite", "s1-cpu"], b)
    assert s["s1-fast"] + s["s1-pro"] == pytest.approx(0.80)
    assert s["s1-pro"] / s["s1-fast"] == pytest.approx(0.65 / 0.35)
    assert math.fsum(s.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TierBudgets(gpu=0.5, small=0.3, cpu=0.3)


def _ema(n_fast, n_lite, n_cpu, start=10):
    uid = itertools.count(start)
    return {
        "s1-fast": {next(uid): 1.0 for _ in range(n_fast)},
        "s1-lite": {next(uid): 1.0 for _ in range(n_lite)},
        "s1-cpu": {next(uid): 1.0 for _ in range(n_cpu)},
    }


def _full_capacity(ema):
    return {p: {u: tiers.CAPACITY[p].target_dps for u in m} for p, m in ema.items()}


def _avg(w, uids):
    return math.fsum(w.get(u, 0.0) for u in uids) / len(uids)


def test_equal_counts_get_the_tier_budgets():
    ema = _ema(20, 20, 20)
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    for pool, share in (("s1-fast", 0.80), ("s1-lite", 0.15), ("s1-cpu", 0.05)):
        assert math.fsum(w[u] for u in ema[pool]) == pytest.approx(share)
    assert 0 not in w


def test_a_lone_cpu_miner_never_out_earns_gpu_miners():
    """Few CPU miners must not get a larger per-miner reward than the many GPU miners (tier-order cap)."""
    ema = _ema(50, 0, 1)
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    cpu_uid = next(iter(ema["s1-cpu"]))
    assert w[cpu_uid] <= 0.5 * _avg(w, list(ema["s1-fast"])) + 1e-12
    # the excess went up to the GPU tier, and the empty lite pool's share went to the other pools: nothing is burned
    assert 0 not in w
    assert w[cpu_uid] > 0.4 * _avg(w, list(ema["s1-fast"]))
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_an_unclaimed_pool_share_goes_to_the_other_pools_pro_rata():
    """1.2: an empty pool or tier is never burned: its share is shared out pro rata over the pools that have miners."""
    ema = _ema(20, 0, 20)  # the lite tier (15 %) is empty
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    assert 0 not in w
    assert math.fsum(w[u] for u in ema["s1-fast"]) == pytest.approx(0.80 / 0.85)
    assert math.fsum(w[u] for u in ema["s1-cpu"]) == pytest.approx(0.05 / 0.85)
    ema = _ema(0, 20, 20)  # the GPU tier is empty
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    assert 0 not in w and math.fsum(w.values()) == pytest.approx(1.0)
    assert math.fsum(w[u] for u in ema["s1-lite"]) == pytest.approx(0.75)  # 0.15/0.20; the cpu cap is not hit
    assert math.fsum(w[u] for u in ema["s1-cpu"]) == pytest.approx(0.25)


def test_an_empty_gpu_pool_is_shared_inside_the_gpu_tier_and_the_burn_fraction_is_kept():
    ema = {"s1-fast": {}, "s1-pro": {1: 1.0, 2: 1.0}, "s1-lite": {3: 1.0}, "s1-cpu": {}}
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1, burn_fraction=0.1)
    assert w[0] == pytest.approx(0.1)  # only the burn fraction goes to the owner
    assert w[1] + w[2] == pytest.approx(0.9 * 0.80 / 0.95) and w[3] == pytest.approx(0.9 * 0.15 / 0.95)
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_no_miners_at_all_pays_the_owner_only():
    ema = {"s1-fast": {}, "s1-lite": {}, "s1-cpu": {}}
    assert compute_tiered_weights(ema, {}, TierBudgets(), owner_uid=0, min_ema=0.1) == {0: 1.0}
    assert compute_tiered_weights(ema, {}, TierBudgets(), owner_uid=None, min_ema=0.1) == {}


@pytest.mark.parametrize("seed", range(40))
def test_tier_order_holds_for_any_miner_counts(seed):
    rng = random.Random(seed)
    ema = _ema(rng.randint(1, 200), rng.randint(1, 200), rng.randint(1, 200))
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    fast, lite, cpu = (_avg(w, list(ema[p])) for p in ("s1-fast", "s1-lite", "s1-cpu"))
    assert lite <= 0.5 * fast * (1 + 1e-9) and cpu <= 0.5 * lite * (1 + 1e-9)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert w.get(0, 0.0) == pytest.approx(0.0, abs=1e-12)  # all pools have miners: nothing burned


def test_capacity_factor_and_no_floor():
    cap = CapacityParams(target_dps=4.0)
    assert cap.multiplier(None) == 0.0 and cap.multiplier(2.0) == 0.5 and cap.multiplier(40.0) == 1.0
    ema = {"s1-fast": {}, "s1-lite": {1: 1.0, 2: 1.0, 3: 1.0}, "s1-cpu": {}}
    caps = {"s1-lite": {1: 4.0, 2: 1.0}}  # uid 3 never probed
    w = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
    assert 3 not in w and w[1] == pytest.approx(4 * w[2])


def test_pool_without_any_capacity_data_falls_back_to_correctness():
    ema = {"s1-fast": {1: 1.0}, "s1-lite": {2: 1.0, 3: 0.5}, "s1-cpu": {}}
    w = compute_tiered_weights(ema, {}, TierBudgets(), owner_uid=0, min_ema=0.1)
    assert w[2] == pytest.approx(2 * w[3]) and w[1] > w[2]
    assert 0 not in w  # the empty CPU pool's share went to the other pools, not to the owner


def test_sybil_split_earns_no_more_than_one_uid():
    """One machine behind 3 hotkeys is measured at 1/3 of its throughput each (probed together)."""
    one = compute_tiered_weights({"s1-cpu": {1: 1.0, 9: 1.0}}, {"s1-cpu": {1: 0.6, 9: 0.6}},
                                 TierBudgets(), owner_uid=0, min_ema=0.1)
    split = compute_tiered_weights({"s1-cpu": {1: 1.0, 2: 1.0, 3: 1.0, 9: 1.0}},
                                   {"s1-cpu": {1: 0.2, 2: 0.2, 3: 0.2, 9: 0.6}},
                                   TierBudgets(), owner_uid=0, min_ema=0.1)
    assert split[1] + split[2] + split[3] == pytest.approx(one[1])
    assert split[9] == pytest.approx(one[9])


# ----------------------------------------------------------------- config
def test_tiered_pools_are_on_by_default_on_every_network_and_fixed_on_finney():
    assert load_validator({}).tiered_pools  # finney defaults
    assert load_validator({"REAX_NETWORK": "finney"}).tiered_pools
    assert load_validator(DEV).tiered_pools
    with pytest.raises(ConfigError):  # a finney validator cannot switch them off
        load_validator({"REAX_NETWORK": "finney", "REAX_FEATURE_TIERED_POOLS": "0"})
    with pytest.raises(ConfigError):  # nor without the development switch
        load_validator({"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_FEATURE_TIERED_POOLS": "0"})
    assert not load_validator({**DEV, "REAX_FEATURE_TIERED_POOLS": "0"}).tiered_pools  # development networks may
    # miners may run the tiered models everywhere
    assert load_common({"REAX_NETWORK": "finney", "REAX_MODEL": "s1-cpu"}).model == "s1-cpu"
    assert load_common({"REAX_NETWORK": "finney", "REAX_MODEL": "s1-lite"}).model == "s1-lite"


def test_tiered_manifests_pin_one_gguf_each():
    for model in ("s1-lite", "s1-cpu"):
        m = load_manifest(model)
        files = m.weight_files
        assert list(files) == [m.data["weights"]["gguf"]] and all(len(h) == 64 for h in files.values())
        assert m.data["licence"]["spdx"] == "Apache-2.0"
    assert load_manifest("s1-cpu").temperature == pytest.approx(1.0627611560799335)


def test_the_development_switch_is_refused_on_any_finney_spelling():
    """Other spellings of finney's endpoint must not unlock the dev switch (which could turn the pools off)."""
    for network in ("wss://entrypoint-finney.opentensor.ai:443/", "wss://archive.chain.opentensor.ai:443",
                    "wss://entrypoint-finney.opentensor.ai", "finney"):
        with pytest.raises(ConfigError):
            load_validator({"REAX_NETWORK": network, "REAX_NETUID": "92", "REAX_ALLOW_DEV_MODELS": "1",
                            "REAX_FEATURE_TIERED_POOLS": "0"})
    # finney itself is not blocked by the endpoint spelling checks
    assert load_validator({"REAX_NETWORK": "wss://entrypoint-finney.opentensor.ai:443/", "REAX_NETUID": "92"}).tiered_pools
    assert load_common({**DEV, "REAX_NETWORK": "wss://test.finney.opentensor.ai:443"}).model == "s1-fast"  # testnet stays


def test_probe_verifies_a_quarter_of_the_answers():
    cap = CapacityParams(target_dps=8.0)
    assert [cap.verified(n) for n in (2, 4, 12, 40, 144, 400)] == [2, 3, 3, 10, 36, 40]
    assert tiers.CAPACITY["s1-cpu"].verified(34) == 8


def test_unusable_capacity_in_the_state_file_is_dropped_not_the_file(tmp_path):
    import json

    st = ValidatorState(netuid=9, hotkey="v")
    from reax_subnet.state import MinerState

    st.miners[3] = MinerState(hotkey="h3", ema=0.5, capacity=2.0)
    path = tmp_path / "s.json"
    save(st, str(path))
    raw = json.loads(path.read_text())
    raw["miners"]["3"]["capacity"] = 1e300
    path.write_text(json.dumps(raw))
    back = load(str(path), 9, "v")
    assert back.miners[3].ema == 0.5 and back.miners[3].capacity is None


def test_capacity_survives_a_state_reload(tmp_path):
    st = ValidatorState(netuid=9, hotkey="v")
    from reax_subnet.state import MinerState

    st.miners[3] = MinerState(hotkey="h3", ema=0.5)
    st.update_capacity(3, 2.0, 0.3)
    st.update_capacity(3, 4.0, 0.3)
    st.miners[4] = MinerState(hotkey="h4")
    save(st, str(tmp_path / "s.json"))
    back = load(str(tmp_path / "s.json"), 9, "v")
    assert back.miners[3].capacity == pytest.approx(2.6) and back.miners[4].capacity is None
    back.strike(3, 0.0)
    assert back.miners[3].capacity == 0.0


# ----------------------------------------------------------------- readouts
def _readouts():
    spec = importlib.util.spec_from_file_location("readouts", ROOT / "runtimes" / "s1-gguf" / "readouts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jevk5_prompt_is_the_s1_fast_reference_prompt():
    r = _readouts()
    prompt, keys, letters = r.build("jevk5", {"order": 7}, {"type": "choice", "instructions": "Route?",
                                                         "criteria": {"billing": "Money", "tech": None}})
    assert keys == ["billing", "tech"] and letters == ["A", "B"]
    # Rendered with Qwen3.5's chat_template.jinja (crh225/plumb-4b@55de037), thinking off.
    assert prompt == (
        "<|im_start|>system\nApply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>\n<|im_start|>user\n"
        '{"evidence": {"order": 7}, "criterion": "Route?", "options": [{"letter": "A", "description": '
        '"billing: Money"}, {"letter": "B", "description": "tech: tech"}]}<|im_end|>\n<|im_start|>assistant\n'
        "<think>\n\n</think>\n\n")
    _, keys, _ = r.build("jevk5", "s", {"type": "noul", "instructions": "x"})
    assert keys == ["true", "false"]


def test_gutsy_prompt_and_reject_slot():
    r = _readouts()
    prompt, keys, letters = r.build("gutsy", "Parcel late. ", {"type": "score", "instructions": "How bad?",
                                                              "criteria": ["fine", {"lvl": 2}]})
    assert keys == ["0", "1"] and letters == ["A", "B", "Z"]
    assert prompt == (
        "<|im_start|>system\nYou are a decision model. Read the state, then answer the question by choosing exactly "
        "one of the options. Reply with only the letter of your choice.<|im_end|>\n<|im_start|>user\nState:\n"
        "Parcel late.\n\nQuestion: How bad?\n\nOptions:\nA. fine\nB. {\"lvl\": 2}\n"
        "Z. None of the options fits, or the state does not say\n\nAnswer with a single letter.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prompt, keys, letters = r.build("gutsy", {}, {"type": "noul", "instructions": "Late?",
                                                  "criteria": {"true": "more than 2 days"}})
    assert letters == ["A", "B"] and "Late?\nYes means: more than 2 days\n\nOptions:\nA. Yes\nB. No" in prompt
    # The reject column is dropped and the options renormalised = softmax over the options alone.
    p = r.probabilities([1.0, 2.0, 5.0], ["0", "1"], 1.5)
    q = r.probabilities([1.0, 2.0], ["0", "1"], 1.5)
    assert p == pytest.approx(q) and math.fsum(p.values()) == pytest.approx(1.0)
    assert r.temperature({"temperature": {"yes_no": 0.7, "choice": 1.1, "score": 1.0}}, "noul") == 0.7


def test_generated_items_fit_the_gguf_readouts():
    r = _readouts()
    for model, kind in (("s1-lite", "jevk5"), ("s1-cpu", "gutsy")):
        for it in generate_s1_items(200, seed=3):
            d = parse_decision(it.decision(), model)
            q = dict(d.question.author_dict())
            q.setdefault("instructions", "")
            _, keys, letters = r.build(kind, d.state, q)
            assert list(keys) == list(d.question.keys) and len(letters) >= len(keys)


# ------------------------------------------------------------- arena
class Machine:
    """One physical backend: decisions are computed one at a time at ``dps`` decisions per second."""

    def __init__(self, dps: float):
        self.delay = 1.0 / dps
        self.lock = asyncio.Lock()


class OnMachine(S1HashBackend):
    def __init__(self, model: str, machine: Machine, salt: str = "reference"):
        super().__init__(model, salt=salt)
        self.machine = machine

    async def decide(self, decisions):
        async with self.machine.lock:
            await asyncio.sleep(self.machine.delay * len(decisions))
        return await super().decide(decisions)


async def _serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest.fixture
async def tiered_arena(tmp_path, make_wallet, fake_chain_cls, monkeypatch):
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 1)
    probe = dict(inflight=2, window_s=3.0, items_per_request=4, alpha=1.0)
    monkeypatch.setattr(tiers, "CAPACITY", {
        "s1-fast": CapacityParams(target_dps=400.0, **probe),
        "s1-lite": CapacityParams(target_dps=40.0, **probe),
        "s1-cpu": CapacityParams(target_dps=20.0, **probe),
    })
    monkeypatch.setattr(s1validator, "CAPACITY", tiers.CAPACITY)
    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vk, "c1", True, 5000.0, None, None)]
    runners, roles = [], {}

    async def add(role, uid, model, backend):
        w = make_wallet(f"//Miner/{role}")
        m = S1Miner(w, backend, max_concurrent=4, per_caller_concurrency=2, per_caller_per_minute=600)
        m.set_validators(frozenset({vk}))
        runner, port = await _serve(m.app())
        runners.append(runner)
        man = load_manifest(model)
        neurons.append(Neuron(uid, w.hotkey.ss58_address, f"c{uid}", False, 0.0, "127.0.0.1", port,
                              commitment=commitment_text(model, man.manifest_id)))
        roles[uid] = role

    await add("gpu", 2, "s1-fast", OnMachine("s1-fast", Machine(400.0)))
    await add("lite_big", 3, "s1-lite", OnMachine("s1-lite", Machine(40.0)))
    await add("lite_small", 4, "s1-lite", OnMachine("s1-lite", Machine(10.0)))
    await add("lite_fast_garbage", 5, "s1-lite", OnMachine("s1-lite", Machine(400.0), salt="tiny-model"))
    await add("cpu_solo", 6, "s1-cpu", OnMachine("s1-cpu", Machine(15.0)))
    shared = Machine(15.0)  # same hardware as cpu_solo, behind three hotkeys
    for uid in (7, 8, 9):
        await add(f"cpu_sybil_{uid}", uid, "s1-cpu", OnMachine("s1-cpu", shared))

    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    cfg = load_validator({**DEV, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"),
                          "REAX_EMA_ALPHA": "0.5", "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0",
                          "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "0"})
    pools = {m: PoolParams(m, p.temperature, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=4,
                           verified_per_round=4, latency_fast_s=0.0, latency_slow_s=4.0)
             for m, p in s1validator.POOL_PARAMS.items() if m != "s1-pro"}
    refs = {m: S1HashBackend(m, salt="reference") for m in pools}
    val = S1Validator(cfg, vwallet, chain, refs, pools=pools)
    session = aiohttp.ClientSession(trust_env=False)
    yield {"val": val, "chain": chain, "session": session, "roles": roles}
    await session.close()
    for r in runners:
        await r.cleanup()


async def test_tiered_round_measures_capacity_and_splits_sybils(tiered_arena):
    val, chain, roles = tiered_arena["val"], tiered_arena["chain"], tiered_arena["roles"]
    assert val.shares == pytest.approx({"s1-fast": 0.80, "s1-lite": 0.15, "s1-cpu": 0.05})
    summary = None
    for _ in range(3):
        summary = await val.run_round(tiered_arena["session"])
    assert summary is not None and "capacity" in summary
    cap = {roles[u]: val.state.miners[u].capacity for u in roles}
    # A big node measures more than a small one; garbage is struck and has no capacity.
    assert cap["lite_big"] > 2 * cap["lite_small"] > 0
    assert cap["lite_fast_garbage"] == 0.0 and val.state.miners[5].ema == 0.0
    # Three hotkeys on one machine are measured at about a third of the same machine alone each.
    sybil = [cap[f"cpu_sybil_{u}"] for u in (7, 8, 9)]
    assert sum(sybil) == pytest.approx(cap["cpu_solo"], rel=0.15)
    assert max(sybil) < 0.75 * cap["cpu_solo"]
    uids, weights, _ = chain.set_calls[-1]
    w = dict(zip(uids, weights))
    assert w[2] == pytest.approx(0.80, rel=1e-6)  # the only GPU miner holds the whole GPU budget
    assert w[3] + w[4] == pytest.approx(0.15, rel=1e-6) and w[3] > w[4]
    assert 5 not in w
    assert w[7] + w[8] + w[9] < 1.15 * w[6]
    assert math.fsum(w.values()) == pytest.approx(1.0)
    # Measurements are clamped at the pool target before they are averaged (the GPU machine runs at 10x target).
    assert cap["gpu"] <= tiers.CAPACITY["s1-fast"].target_dps


async def test_new_uid_is_probed_in_its_first_answered_round(tiered_arena, monkeypatch):
    val = tiered_arena["val"]
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 10_000)  # scheduled probes practically never fire
    monkeypatch.setattr(val, "probe_every", 10_000)
    for _ in range(2):  # round 1 confirms the roles, round 2 is the first answered round
        await val.run_round(tiered_arena["session"])
    assert all(val.state.miners[u].capacity is not None for u in tiered_arena["roles"])
    summary = await val.run_round(tiered_arena["session"])
    assert "capacity" not in summary  # everyone measured: back to the random schedule


def test_probes_fit_the_miner_default_rate_limit():
    """A probe plus the normal round request stays within a miner's default 20 requests/minute per validator,
    and a probe never carries more decisions per request than the protocol allows."""
    from reax_subnet.config import load_miner
    from reax_subnet.s1protocol import MAX_DECISIONS

    limit = load_miner({**DEV}).per_caller_per_minute
    for pool, cap in tiers.CAPACITY.items():
        assert cap.max_requests + 2 <= limit, pool  # the round's own request, and the previous round's if late
        assert cap.items_per_request <= MAX_DECISIONS and cap.inflight <= 2, pool


class HungReference(S1HashBackend):
    async def decide_each(self, decisions, retries=1):
        await asyncio.sleep(3600)


async def test_a_hung_tiered_reference_only_costs_its_own_pool(tiered_arena, monkeypatch):
    """A slow or hung s1-cpu reference (e.g. on the validator's CPU) must never stall s1-fast scoring."""
    val = tiered_arena["val"]
    monkeypatch.setattr(s1validator, "REFERENCE_BUDGET_MIN_S", 2.0)
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 10_000)
    monkeypatch.setattr(val, "probe_every", 10_000)
    val.cfg = replace(val.cfg, round_interval_s=1)
    val.backends["s1-cpu"] = HungReference("s1-cpu")
    for _ in range(3):
        summary = await val.run_round(tiered_arena["session"])
    assert summary is not None
    scores, roles = summary["scores"], tiered_arena["roles"]
    for uid, role in roles.items():
        if role.startswith("cpu"):
            assert scores[uid]["error"] == "reference unavailable", role
        elif role in ("gpu", "lite_big", "lite_small"):
            assert scores[uid]["score"] > 0.8, role


async def test_reference_cap_rotates_miners_fairly(tiered_arena, monkeypatch):
    """With room for one s1-cpu miner per round, the four CPU miners are verified in turn and never decayed while
    they rest; other pools are unaffected."""
    val = tiered_arena["val"]
    monkeypatch.setenv("REAX_REFERENCE_CAP_S1_CPU", "4")  # 4 decisions = one miner (4 verified per round)
    monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 10_000)
    monkeypatch.setattr(val, "probe_every", 10_000)
    cpu = [u for u, r in tiered_arena["roles"].items() if r.startswith("cpu")]
    seen: list[int] = []
    for _ in range(1 + 2 * len(cpu)):
        summary = await val.run_round(tiered_arena["session"])
        if summary and summary["scores"]:
            got = [u for u in cpu if u in summary["scores"]]
            assert len(got) <= 1
            seen += got
            assert all(u in summary["scores"] for u, r in tiered_arena["roles"].items() if r in ("gpu", "lite_big"))
    assert set(seen) == set(cpu)  # everyone got its turn
    assert all(val.state.miners[u].ema > 0 for u in cpu)  # resting never decayed anyone


def test_a_validator_without_a_tiered_reference_does_not_crash_and_leaves_that_pool_unscored(make_wallet, fake_chain_cls,
                                                                                             tmp_path, caplog):
    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    chain = fake_chain_cls(neurons=[Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None)],
                           owner_hotkey=owner.hotkey.ss58_address)
    cfg = load_validator({**DEV, "REAX_STATE_PATH": str(tmp_path / "state.json"), "REAX_FEATURE_S1_PRO_POOL": "0"})
    assert cfg.tiered_pools
    with caplog.at_level("WARNING", logger="reax.s1validator"):
        val = S1Validator(cfg, vwallet, chain, {"s1-fast": S1HashBackend("s1-fast"), "s1-lite": S1HashBackend("s1-lite")})
    assert sorted(val.pools) == ["s1-fast", "s1-lite"]  # s1-cpu has no reference here: not scored
    assert val.shares == pytest.approx({"s1-fast": 0.80 / 0.95, "s1-lite": 0.15 / 0.95})
    assert any("s1-cpu" in r.getMessage() for r in caplog.records)
    man = load_manifest("s1-cpu")
    cpu_miner = Neuron(5, "h5", "c5", False, 0.0, "127.0.0.1", 1, commitment=commitment_text("s1-cpu", man.manifest_id))
    assert val.role_of(cpu_miner) is None


def test_tier_budgets_apply_to_the_inference_part_of_the_50_50_split():
    """Order of the weight computation: tiers -> serving vector -> combine_weights(serving, engine, tracks)."""
    from reax_subnet.competition_split import combine_weights

    ema = _ema(20, 20, 20, start=10)
    serving = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    tracks = {"demand_forecasting": {200: 0.9, 201: 0.8, 202: 0.7}, "risk_assessment": {210: 0.9, 211: 0.8, 212: 0.7}}
    w = combine_weights(serving, 300, 1.0, tracks, "50/50", owner_uid=0)  # tracks pay 75/17/8 over three ranks
    assert w[300] == pytest.approx(0.20)
    assert math.fsum(w[u] for u in (200, 201, 202)) == pytest.approx(0.15)
    assert math.fsum(w[u] for u in (210, 211, 212)) == pytest.approx(0.15)
    for pool, share in (("s1-fast", 0.80), ("s1-lite", 0.15), ("s1-cpu", 0.05)):
        assert math.fsum(w[u] for u in ema[pool]) == pytest.approx(0.50 * share)
    # no champions: the whole competition share stays with the inference part (tiers 80/15/5 of 100 %)
    w = combine_weights(serving, None, 0.0, {}, "50/50", owner_uid=0)
    for pool, share in (("s1-fast", 0.80), ("s1-lite", 0.15), ("s1-cpu", 0.05)):
        assert math.fsum(w[u] for u in ema[pool]) == pytest.approx(share)
