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


@pytest.fixture(autouse=True)
def _release_floors_off(monkeypatch):
    """Synthetic track scores here are small; the release entry floors are tested in test_competitions.py."""
    from reax_subnet import competition_split as _cs
    for _k in list(_cs.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(_cs.TRACK_ENTRY_FLOOR, _k, None)


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


RATIO = TierBudgets().order_ratio


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
    assert w[cpu_uid] <= TierBudgets().order_ratio * _avg(w, list(ema["s1-fast"])) + 1e-12
    # the excess went up to the GPU tier, and the empty lite pool's share went to the other pools: nothing is burned
    assert 0 not in w
    assert w[cpu_uid] > 0.8 * _avg(w, list(ema["s1-fast"]))
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
    assert lite <= RATIO * fast * (1 + 1e-9) and cpu <= RATIO * lite * (1 + 1e-9)
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


def test_probes_fit_the_miner_default_limits():
    """A probe at the pool cap plus the normal round requests stays within a miner's default requests/minute per
    validator, its lanes within the default per-validator concurrency, and a request never carries more decisions
    than the protocol allows."""
    from reax_subnet.config import load_miner
    from reax_subnet.s1protocol import MAX_DECISIONS

    cfg = load_miner({**DEV})
    for pool, cap in tiers.CAPACITY.items():
        assert cap.max_requests + 2 <= cfg.per_caller_per_minute, pool  # the round's own request, and a late one
        assert cap.items_per_request <= MAX_DECISIONS and cap.inflight <= cfg.per_caller_concurrency, pool
        assert cap.inflight <= cfg.max_concurrent * 5, pool  # running + queued (queue = 4 x max_concurrent)
        assert cap.inflight <= cfg.max_pending_per_ip, pool  # every lane's body can be read at once


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


# ------------------------------------------------- capacity v2 (1.2.0rc3): linear factor, high cap, sybil neutrality
FAST = tiers.CAPACITY["s1-fast"]


def test_release_factor_is_linear_up_to_a_cap_far_above_one_machine():
    """One unit = the target; pay grows linearly up to max_factor units. The cap must sit above the fastest single
    machine of the tier (CALIBRATION.md: vLLM bf16 batched on an H100 226/s, reference path 55-74/s per GPU)."""
    assert FAST.factor(40.0) == 1.0 and FAST.factor(80.0) == 2.0 and FAST.factor(226.0) == pytest.approx(5.65)
    assert FAST.cap_dps >= 2 * 226.0 and FAST.factor(1e6) == FAST.max_factor
    for pool in ("s1-fast", "s1-pro"):
        assert tiers.CAPACITY[pool].max_factor >= 8.0, pool
    # the cheap pools cap near their best honest node (review H1): s1-lite on an RTX 5070 Ti does 33/s
    assert tiers.CAPACITY["s1-lite"].cap_dps == 32.0 and tiers.CAPACITY["s1-cpu"].cap_dps == 4.0
    assert CapacityParams(target_dps=4.0).max_factor == 1.0  # the dataclass default keeps the 1.2.0rc2 cap


def _fast_pool_weights(caps: dict[int, float], monkeypatch=None, max_factor=None) -> dict[int, float]:
    if max_factor is not None:
        monkeypatch.setitem(tiers.CAPACITY, "s1-fast", replace(FAST, max_factor=max_factor))
    return compute_tiered_weights({"s1-fast": {u: 1.0 for u in caps}}, {"s1-fast": caps}, TierBudgets(),
                                  owner_uid=0, min_ema=0.1)


OTHERS = {90: 60.0, 91: 40.0, 92: 226.0}  # other honest miners in the pool


@pytest.mark.parametrize("machine", [40.0, 80.0, 160.0, 320.0])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_n_hotkeys_on_one_machine_never_out_earn_one_hotkey(machine, n):
    """The simultaneous probe measures each of N hotkeys on one machine at about machine/N. With the linear factor
    their sum equals the solo hotkey's weight (registrations not counted), for every machine speed up to the cap."""
    solo = _fast_pool_weights({1: machine, **OTHERS})
    split = _fast_pool_weights({**{u: machine / n for u in range(1, n + 1)}, **OTHERS})
    assert math.fsum(split[u] for u in range(1, n + 1)) <= solo[1] * (1 + 1e-9)
    assert math.fsum(split[u] for u in range(1, n + 1)) == pytest.approx(solo[1])
    for u in OTHERS:  # nobody else is affected by the split
        assert split[u] == pytest.approx(solo[u])


def test_the_old_cap_at_the_target_paid_a_fast_machine_once_per_hotkey(monkeypatch):
    """Regression for the 1.2.0rc2 loophole: factor = min(1, dps / 40) paid a 160/s machine four
    times when it was split over four hotkeys."""
    solo = _fast_pool_weights({1: 160.0, **OTHERS}, monkeypatch, max_factor=1.0)
    split = _fast_pool_weights({1: 40.0, 2: 40.0, 3: 40.0, 4: 40.0, **OTHERS}, monkeypatch, max_factor=1.0)
    # 4 units instead of 1 against the other miners' 3: the pair of shares is 4/7 vs 1/4 of the pool
    assert math.fsum(split[u] for u in (1, 2, 3, 4)) == pytest.approx(4 / 7)
    assert solo[1] == pytest.approx(1 / 4)


def test_one_hotkey_with_more_gpus_earns_more():
    # up to the cap (640/s at x1, 1280/s at x2, 2560/s at x4)
    speeds = sorted({40.0, 80.0, 160.0, 320.0, 640.0, FAST.cap_dps})
    w = [_fast_pool_weights({1: c, **OTHERS})[1] for c in speeds if c <= FAST.cap_dps]
    # each doubling pays clearly more (a share of a fixed pool: the gain shrinks as the hotkey takes most of it)
    assert all(b > a * 1.1 for a, b in zip(w, w[1:]))
    # above the cap a single hotkey earns no more; a second node is then paid through its own hotkey
    assert _fast_pool_weights({1: 2 * FAST.cap_dps, **OTHERS})[1] == pytest.approx(w[-1])


def _fleet(spec: dict[str, list[float]], start: int = 10):
    ema, caps, uid = {}, {}, itertools.count(start)
    for pool, speeds in spec.items():
        ema[pool], caps[pool] = {}, {}
        for d in speeds:
            u = next(uid)
            ema[pool][u], caps[pool][u] = 1.0, d
    return ema, caps


def test_tier_order_stays_per_miner_so_faster_gpus_do_not_shrink_the_cheap_tiers():
    """1.2.0rc3 keeps the published rule: the average weight per miner in a tier is at most order_ratio (1.0 since 1.2.0rc7.post4) times the tier above
    (review H3). GPU miners getting faster moves pay inside the GPU tier only; Jetson and CPU pay per miner stays."""
    base = {"s1-lite": [8.0] * 5, "s1-cpu": [1.0] * 3}
    per_miner = []
    for fast, pro in ((40.0, 10.0), (120.0, 30.0), (226.0, 60.0)):
        ema, caps = _fleet({"s1-fast": [fast] * 85, "s1-pro": [pro] * 10, **base})
        w = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
        gpu = list(ema["s1-fast"]) + list(ema["s1-pro"])
        lite, cpu = _avg(w, list(ema["s1-lite"])), _avg(w, list(ema["s1-cpu"]))
        assert lite <= RATIO * _avg(w, gpu) * (1 + 1e-9) and cpu <= RATIO * lite * (1 + 1e-9)
        per_miner.append((lite, cpu))
    assert per_miner[1] == pytest.approx(per_miner[0]) and per_miner[2] == pytest.approx(per_miner[0])


@pytest.mark.parametrize("seed", range(20))
def test_tier_order_holds_per_miner_at_any_speeds(seed):
    rng = random.Random(seed)
    spec = {p: [rng.choice([0.1, 0.5, 1, 2, 4, 16]) * tiers.CAPACITY[p].target_dps for _ in range(rng.randint(1, 60))]
            for p in ("s1-fast", "s1-lite", "s1-cpu")}
    ema, caps = _fleet(spec)
    w = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
    fast, lite, cpu = (_avg(w, list(ema[p])) for p in ("s1-fast", "s1-lite", "s1-cpu"))
    assert lite <= RATIO * fast * (1 + 1e-9) and cpu <= RATIO * lite * (1 + 1e-9)
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_splitting_a_machine_inside_a_pool_still_gains_nothing():
    """Inside a pool the factor is linear, so a CPU node split over ten hotkeys (each measured at a tenth, probed
    together) earns what the solo node earns while its tier is held by its budget, not by the per-miner rule."""
    gpu = {u: 1.0 for u in range(100, 120)}
    gcap = {u: FAST.target_dps for u in gpu}
    others = {u: 1.0 for u in range(200, 400)}  # 200 other CPU nodes: the tier is budget-bound (DESIGN-POOLS 20/50/200)
    for target_speed in (1.0, 3.0):
        solo = compute_tiered_weights({"s1-fast": gpu, "s1-cpu": {1: 1.0, **others}},
                                      {"s1-fast": gcap, "s1-cpu": {1: target_speed, **others}}, TierBudgets(),
                                      owner_uid=0, min_ema=0.1)
        split_uids = range(1, 11)
        split = compute_tiered_weights({"s1-fast": gpu, "s1-cpu": {**{u: 1.0 for u in split_uids}, **others}},
                                       {"s1-fast": gcap, "s1-cpu": {**{u: target_speed / 10 for u in split_uids},
                                                                     **others}},
                                       TierBudgets(), owner_uid=0, min_ema=0.1)
        assert math.fsum(split[u] for u in split_uids) == pytest.approx(solo[1])
        assert split[200] == pytest.approx(solo[200])


def test_tier_order_counts_miners_so_target_speed_tiers_behave_as_before():
    ema = _ema(20, 5, 2)
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    # the DESIGN-POOLS.md section 5 row "20 / 5 / 2": 80 / 15 / 5 at ratio 1.0 (87.4 / 10.6 / 2.0 at the old 0.5)
    assert math.fsum(w[u] for u in ema["s1-fast"]) == pytest.approx(0.80, abs=0.001)
    assert math.fsum(w[u] for u in ema["s1-cpu"]) == pytest.approx(0.05, abs=0.001)
    old = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(order_ratio=0.5), owner_uid=0, min_ema=0.1)
    assert math.fsum(old[u] for u in ema["s1-fast"]) == pytest.approx(0.874, abs=0.001)


@pytest.mark.parametrize("pool, gpu_dps", [("s1-cpu", 16.0), ("s1-cpu", 1000.0), ("s1-lite", 80.0),
                                           ("s1-lite", 1000.0)])
def test_a_gpu_in_a_cheap_pool_earns_less_than_a_target_s1_fast_hotkey(pool, gpu_dps, monkeypatch):
    """Review H1: the cheap pools cap the factor near their best honest node (s1-lite 4 units = 32/s, s1-cpu 4/s), so
    a GPU running the small model at GPU speed earns less there than a target-speed s1-fast hotkey (DESIGN-POOLS 5,
    the 20 / 50 / 200 fleet; Orins at 0.7/s, small GPUs at 8/s, CPU nodes at 0.5-1/s). See RESULT/CHANGELOG for
    fleets in which the cheap tiers pay more per miner (few cheap nodes, many s1-pro): there the cap still limits
    a GPU to 4 cheap units, but not below an s1-fast target hotkey."""
    assert tiers.CAPACITY["s1-lite"].cap_dps == 32.0 and tiers.CAPACITY["s1-cpu"].cap_dps == 4.0
    spec = {"s1-fast": [FAST.target_dps] * 20, "s1-lite": [0.7] * 25 + [8.0] * 25,
            "s1-cpu": [0.5] * 100 + [1.0] * 100}
    spec[pool] = spec[pool] + [gpu_dps]
    ema, caps = _fleet(spec)
    w = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
    gpu_uid, fast_uid = max(ema[pool]), min(ema["s1-fast"])
    assert w[gpu_uid] < w[fast_uid]
    # the cap is what holds it: with the first capacity-v2 draft's caps (s1-lite 10, s1-cpu 16 units) it would not
    monkeypatch.setitem(tiers.CAPACITY, pool, replace(tiers.CAPACITY[pool], max_factor={"s1-lite": 10.0,
                                                                                        "s1-cpu": 16.0}[pool]))
    loose = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
    assert loose[gpu_uid] > 2 * w[gpu_uid]


def test_probe_budget_follows_the_measured_capacity():
    """The validator prepares requests for headroom x the previous measurement (at least the target, at most the cap),
    so its probe cost follows the real pool, and a new big node reaches its speed within a few probes."""
    assert FAST.budget_dps(None) == FAST.target_dps
    assert FAST.budget_dps(10.0) == FAST.target_dps
    assert FAST.budget_dps(100.0) == 400.0 and FAST.budget_dps(1000.0) == FAST.cap_dps
    budget, seen = FAST.budget_dps(None), []
    for _ in range(4):  # a 600/s node measured at its budget each time (alpha = 0.3 moving average)
        seen.append(min(600.0, budget))
        budget = FAST.budget_dps(seen[-1])
    assert seen[2] == 600.0


def test_probe_verification_budget_does_not_grow_with_throughput():
    """Bigger probes cost the validator no more reference decisions: at most verified_max per UID and probe."""
    for pool, cap in tiers.CAPACITY.items():
        answered = math.ceil(cap.cap_dps * cap.window_s)
        assert cap.verified(answered) == cap.verified_max, pool
        assert cap.verified(10 * answered) == cap.verified_max, pool
    # Partial fast garbage does not pay: share g of garbage among 40 secret checks is caught (a strike) with
    # probability 1 - (1 - g)^40, for a throughput gain of at most 1 / (1 - g).
    for g in (0.02, 0.05, 0.10):
        assert 1 - (1 - g) ** FAST.verified_max > 0.5


# --------------------------------------------- capacity v2 through the real probe (fake machines, real miners)
class SlowMachine(Machine):
    """A machine at ``dps`` reached over a link that adds ``rtt`` seconds to every request (outside the machine)."""

    def __init__(self, dps: float, rtt: float = 0.0):
        super().__init__(dps)
        self.rtt = rtt


class OverLink(OnMachine):
    def __init__(self, model, machine, salt="reference", quit_after=None):
        super().__init__(model, machine, salt=salt)
        self.quit_after = quit_after
        self.requests = 0

    async def decide(self, decisions):
        self.requests += 1
        if self.quit_after is not None and self.requests > self.quit_after:
            raise RuntimeError("stops answering to free the machine for its siblings")
        await asyncio.sleep(getattr(self.machine, "rtt", 0.0))
        return await super().decide(decisions)


PROBE_V2 = CapacityParams(target_dps=40.0, window_s=2.0, items_per_request=16, inflight=16, max_factor=16.0,
                          alpha=1.0)


async def _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, cap=PROBE_V2, rounds=3):
    """``groups`` = list of (machine, [backend kwargs per hotkey]). Runs ``rounds`` rounds (round 1 confirms roles,
    then every round probes) and returns (validator, {group index: [uids]}, round summaries, reference backend)."""
    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vk, "c1", True, 5000.0, None, None)]
    man = load_manifest("s1-fast")
    runners, by_group, uid = [], {}, 2
    session = aiohttp.ClientSession(trust_env=False)
    try:
        for g, (machine, hotkeys) in enumerate(groups):
            for kw in hotkeys:
                kw = dict(kw)
                miner_cls = kw.pop("miner_cls", S1Miner)
                w = make_wallet(f"//Miner/{tmp_path.name}/{uid}")
                m = miner_cls(w, OverLink("s1-fast", machine, **kw), max_concurrent=8, per_caller_concurrency=16,
                            per_caller_per_minute=600, max_pending_per_ip=64)
                m.set_validators(frozenset({vk}))
                runner, port = await _serve(m.app())
                runners.append(runner)
                neurons.append(Neuron(uid, w.hotkey.ss58_address, f"c{uid}", False, 0.0, "127.0.0.1", port,
                                      commitment=commitment_text("s1-fast", man.manifest_id)))
                by_group.setdefault(g, []).append(uid)
                uid += 1
        chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
        cfg = load_validator({**DEV, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"),
                              "REAX_EMA_ALPHA": "0.5", "REAX_BURN_FRACTION": "0.0", "REAX_WARMUP_ROUNDS": "0",
                              "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "0"})
        p = s1validator.POOL_PARAMS["s1-fast"]
        pools = {"s1-fast": PoolParams("s1-fast", p.temperature, tau_mean=0.2, tau_strike=0.5, tau_item=2.0,
                                       items_per_round=4, verified_per_round=4, latency_fast_s=0.0, latency_slow_s=4.0)}
        ref = S1HashBackend("s1-fast", salt="reference")
        val = S1Validator(cfg, vwallet, chain, {"s1-fast": ref}, pools=pools)
        summaries = [await val.run_round(session) for _ in range(rounds)]
        return val, by_group, summaries, ref
    finally:
        await session.close()
        for r in runners:
            await r.cleanup()


@pytest.fixture
def probe_v2(monkeypatch):
    def use(cap=PROBE_V2):
        monkeypatch.setattr(s1validator, "PROBE_EVERY_ROUNDS", 1)
        monkeypatch.setattr(tiers, "CAPACITY", {"s1-fast": cap})
        monkeypatch.setattr(s1validator, "CAPACITY", tiers.CAPACITY)
        return cap
    return use


def _measured(val, uids):
    return math.fsum(val.state.miners[u].capacity or 0.0 for u in uids)


@pytest.mark.parametrize("dps", [40.0, 80.0, 160.0, 320.0])
async def test_probe_measures_fast_machines_and_splitting_never_measures_more(dps, probe_v2, make_wallet,
                                                                             fake_chain_cls, tmp_path):
    """Fake machines at 40/80/160/320 decisions/s behind 1, 2, 3 and 4 hotkeys, all probed at once: one hotkey is
    measured at about the machine's speed (well above the old 40/s cap), and N hotkeys on one machine together never
    measure more than the solo hotkey on an identical machine."""
    probe_v2()
    groups = [(Machine(dps), [{}] * n) for n in (1, 2, 3, 4)]
    val, by_group, summaries, _ref = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    assert summaries[-1] is not None and "capacity" in summaries[-1]
    solo = _measured(val, by_group[0])
    assert solo >= 0.7 * dps  # the probe sees the machine, not a ceiling (in-window counting costs < 1 request)
    for g, n in ((1, 2), (2, 3), (3, 4)):
        assert _measured(val, by_group[g]) <= solo * 1.10 + 2.0, (dps, n)
    w = val.compute_weights(val._latest_snap)
    for g in (1, 2, 3):
        assert math.fsum(w.get(u, 0.0) for u in by_group[g]) <= w[by_group[0][0]] * 1.10 + 1e-3


async def test_more_gpus_behind_one_hotkey_measure_and_earn_more(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    probe_v2()
    groups = [(Machine(dps), [{}]) for dps in (40.0, 80.0, 160.0, 320.0)]
    val, by_group, _summaries, _ref = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    cap = [_measured(val, by_group[g]) for g in range(4)]
    assert all(b > 1.6 * a for a, b in zip(cap, cap[1:])), cap
    w = val.compute_weights(val._latest_snap)
    ws = [w[by_group[g][0]] for g in range(4)]
    assert all(b > 1.6 * a for a, b in zip(ws, ws[1:])), ws


async def test_few_probe_lanes_cannot_see_a_distant_fast_machine(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    """Probe ceiling: over a 0.1 s link a hotkey can only show lanes x 16 decisions per round trip. With the
    1.2.0rc2 two lanes a 320/s machine measures ~200/s alone and more when split over four hotkeys (eight lanes);
    with the release's 16 lanes the solo hotkey sees the whole machine and splitting gains nothing."""
    old = probe_v2(replace(PROBE_V2, inflight=2))
    groups = [(SlowMachine(320.0, rtt=0.1), [{}]), (SlowMachine(320.0, rtt=0.1), [{}] * 4)]
    val, by_group, _s, _r = await _probe_arena(make_wallet, fake_chain_cls, tmp_path / "old", groups)
    solo, split = _measured(val, by_group[0]), _measured(val, by_group[1])
    assert solo < 0.8 * 320.0 and split > 1.2 * solo, (old.inflight, solo, split)
    probe_v2(PROBE_V2)
    val, by_group, _s, _r = await _probe_arena(make_wallet, fake_chain_cls, tmp_path / "new", groups)
    solo, split = _measured(val, by_group[0]), _measured(val, by_group[1])
    assert solo >= 0.7 * 320.0 and split <= solo * 1.10, (solo, split)


async def test_a_sibling_that_stops_early_does_not_hand_its_share_to_the_others(probe_v2, make_wallet,
                                                                                fake_chain_cls, tmp_path):
    """Only answers inside the one common window count, divided by the whole window: a sybil that answers a few
    requests and then fails (to free the machine for its sibling) cannot lift the pair above the machine."""
    probe_v2()
    groups = [(Machine(160.0), [{}]), (Machine(160.0), [{"quit_after": 4}, {}])]
    val, by_group, summaries, _ref = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    summary = summaries[-1]
    solo = summary["capacity"][by_group[0][0]]["dps"]
    pair = math.fsum(summary["capacity"][u]["dps"] for u in by_group[1])
    assert pair <= solo * 1.10 + 2.0, (solo, pair)


async def test_probe_verification_stays_within_verified_max(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    """A 320/s hotkey answers hundreds of probe decisions; the validator still verifies at most verified_max of them
    (plus the round's own verified items)."""
    cap = probe_v2()
    calls: list[int] = []
    orig = S1HashBackend.decide_each

    async def counting(self, decisions, retries=1):
        calls.append(len(decisions))
        return await orig(self, decisions, retries)

    import unittest.mock as _m
    with _m.patch.object(S1HashBackend, "decide_each", counting):
        val, by_group, summaries, _ref = await _probe_arena(make_wallet, fake_chain_cls, tmp_path,
                                                            [(Machine(320.0), [{}])], rounds=3)
    uid = by_group[0][0]
    assert summaries[-1]["capacity"][uid]["items"] >= 400  # second probe: budget 4 x the first measurement
    assert max(calls) <= cap.verified_max  # each reference batch: one round's 4 items or one probe's sample
    assert sum(calls) <= 2 * 4 + 2 * cap.verified_max  # two answered rounds and two probes


async def test_a_new_fast_node_reaches_its_speed_within_two_probes(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    """A never-measured UID gets requests for the target only (validator cost follows the pool); its next probe is
    prepared for headroom x that measurement, so a 320/s node is measured at its speed by the second probe."""
    cap = probe_v2()
    val, by_group, summaries, _ref = await _probe_arena(make_wallet, fake_chain_cls, tmp_path,
                                                        [(Machine(320.0), [{}])], rounds=3)
    uid = by_group[0][0]
    first, second = summaries[1]["capacity"][uid], summaries[2]["capacity"][uid]
    assert first["budget_dps"] == cap.target_dps
    assert first["items"] <= cap.requests_for(cap.target_dps) * cap.items_per_request
    assert second["budget_dps"] == pytest.approx(min(cap.cap_dps, cap.headroom * first["dps"]))
    assert second["dps"] >= 0.7 * 320.0


def test_the_tier_order_ratio_is_one():
    """1.2.0rc7.post4: a small-tier miner may earn up to the average per miner of the tier above (was 0.5)."""
    assert TierBudgets().order_ratio == 1.0


def test_a_lone_small_tier_miner_earns_the_gpu_average_up_to_its_budget():
    """The live case: one s1-lite miner against ~30 GPU miners. Before 1.2.0rc7.post4 it was held at half the GPU
    average; now it gets min(tier budget, 1.0 x GPU average). The fixed budgets stay the hard cap."""
    ema = _ema(30, 1, 0)
    caps = _full_capacity(ema)
    lite_uid = next(iter(ema["s1-lite"]))
    new = compute_tiered_weights(ema, caps, TierBudgets(), owner_uid=0, min_ema=0.1)
    old = compute_tiered_weights(ema, caps, TierBudgets(order_ratio=0.5), owner_uid=0, min_ema=0.1)
    gpu_avg_new, gpu_avg_old = _avg(new, list(ema["s1-fast"])), _avg(old, list(ema["s1-fast"]))
    # no cpu miner: the lite tier is capped by min(budget, ratio x GPU average); the budget (15/95) is far above. The cap is
    # taken on the GPU budget before the excess moves up, so the final GPU average ends a little above the cap.
    assert new[lite_uid] == pytest.approx(2 * old[lite_uid])
    assert new[lite_uid] == pytest.approx(0.80 / 0.95 / 30)  # 1.0 x the GPU budget per GPU miner
    assert 0.8 * gpu_avg_new < new[lite_uid] <= gpu_avg_new * (1 + 1e-9) and old[lite_uid] < 0.5 * gpu_avg_old * 1.0001
    assert new[lite_uid] < 0.15
    assert math.fsum(new.values()) == pytest.approx(1.0) and 0 not in new
    print(f"lone s1-lite miner vs 30 GPU miners: before {old[lite_uid]:.5f} (GPU avg {gpu_avg_old:.5f}), "
          f"after {new[lite_uid]:.5f} (GPU avg {gpu_avg_new:.5f})")


def test_the_small_tier_budget_is_still_the_hard_cap_at_ratio_one():
    """With only a few GPU miners the GPU average is large; the small tier still gets its 15 % budget at most."""
    ema = _ema(2, 50, 0)
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    assert math.fsum(w[u] for u in ema["s1-lite"]) == pytest.approx(0.15 / 0.95)  # cpu tier empty: pro rata over lite+gpu
    assert math.fsum(w[u] for u in ema["s1-fast"]) == pytest.approx(0.80 / 0.95)
    ema = _ema(1000, 3, 3)
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(), owner_uid=0, min_ema=0.1)
    assert math.fsum(w[u] for u in ema["s1-lite"]) <= 0.15 + 1e-9
    assert math.fsum(w[u] for u in ema["s1-cpu"]) <= 0.05 + 1e-9


@pytest.mark.parametrize("seed", range(20))
def test_a_lower_tier_average_never_exceeds_the_upper_average_at_ratio_one(seed):
    rng = random.Random(1000 + seed)
    ema = _ema(rng.randint(1, 100), rng.randint(1, 100), rng.randint(1, 100))
    w = compute_tiered_weights(ema, _full_capacity(ema), TierBudgets(order_ratio=1.0), owner_uid=0, min_ema=0.1)
    fast, lite, cpu = (_avg(w, list(ema[p])) for p in ("s1-fast", "s1-lite", "s1-cpu"))
    assert lite <= fast * (1 + 1e-9) and cpu <= lite * (1 + 1e-9)
    assert math.fsum(w.values()) == pytest.approx(1.0)
