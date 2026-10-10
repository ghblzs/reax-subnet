"""reax-s1 (1.1): contract, protocol, scoring and an in-process tournament.

Miners are real aiohttp servers on 127.0.0.1 and the chain is faked. The model is the
deterministic S1HashBackend; "logit noise" stands in for honest GPU drift (small) and for
quantised substitutes (large).
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet.chain import Neuron
from reax_subnet.config import load_validator
from reax_subnet.manifest import commitment_text, load_manifest, parse_commitment
from reax_subnet.miner import S1Miner
from reax_subnet.protocol import ProtocolError
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import ContractError, check_answer, normalize_answer, parse_decision
from reax_subnet.s1items import generate_s1_items
from reax_subnet.s1protocol import PROTOCOL_S1, S1_DECIDE_PATH, S1Request, parse_s1_request, parse_s1_response
from reax_subnet.s1scoring import PoolParams, choose_verified, compute_pool_weights, logit_distance, score_round
from reax_subnet.s1validator import SHARES_BOTH, S1Validator

TIMEOUT = 2.0
FAST = load_manifest("s1-fast")
PRO = load_manifest("s1-pro")


# ----------------------------------------------------------------- units
def test_generator_items_are_valid_production_decisions():
    items = generate_s1_items(500, seed=3)
    assert len({it.item_id for it in items}) == 500
    kinds = {it.kind for it in items}
    assert kinds == {"choice", "noul", "score"}
    for it in items:
        for model in ("s1-fast", "s1-pro"):
            d = parse_decision(it.decision(), model)
            assert d.question.keys == it.keys()


def test_fresh_rounds_never_repeat():
    a = {json.dumps(i.decision()) for i in generate_s1_items(64)}
    b = {json.dumps(i.decision()) for i in generate_s1_items(64)}
    assert not a & b


def test_contract_rejects_bad_decisions():
    good = {"id": "a", "state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    parse_decision(good, "s1-fast")
    bad = [
        {**good, "extra": 1},
        {**good, "questions": {}},
        {**good, "questions": {"q": {"type": "noul"}, "r": {"type": "noul"}}},
        {**good, "questions": {"q": {"type": "maybe"}}},
        {**good, "questions": {"q": {"type": "choice", "criteria": {"a": "x"}}}},
        {**good, "questions": {"q": {"type": "choice", "criteria": {str(i): "x" for i in range(17)}}}},
        {**good, "state": 5},
        {**good, "state": "x" * 17000},
        {**good, "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}},
    ]
    for b in bad:
        with pytest.raises(ContractError):
            parse_decision(b, "s1-fast")
    # 17 options are fine for s1-pro (reference limit 64)
    parse_decision({**good, "questions": {"q": {"type": "choice", "criteria": {str(i): "x" for i in range(17)}}}}, "s1-pro")


def test_answer_consistency_checks():
    d = parse_decision({"id": "a", "state": "s", "questions": {"q": {"type": "choice", "criteria": {"x": "1", "y": "2"}}}},
                       "s1-fast")
    ok = normalize_answer(d.question, {"probabilities": {"x": 0.3, "y": 0.7}})
    assert check_answer(d.question, ok) == (0.3, 0.7)
    for broken in (
        {**ok, "choice": "x"},
        {**ok, "confidence": 0.3},
        {**ok, "probabilities": {"x": 0.3, "y": 0.6}},
        {**ok, "probabilities": {"x": 0.3, "z": 0.7}},
        {**ok, "extra": 1},
        {**ok, "type": "noul"},
        {**ok, "probabilities": {"x": True, "y": 0.7}},
    ):
        with pytest.raises(ContractError):
            check_answer(d.question, broken)
    n = parse_decision({"id": "b", "state": "s", "questions": {"q": {"type": "noul"}}}, "s1-fast")
    good = normalize_answer(n.question, {"noul": 0.25})
    assert check_answer(n.question, good) == (0.25, 0.75)
    with pytest.raises(ContractError):
        check_answer(n.question, {**good, "noul": 0.5})
    s = parse_decision({"id": "c", "state": "s", "questions": {"q": {"type": "score", "criteria": ["lo", "mid", "hi"]}}},
                       "s1-fast")
    sc = normalize_answer(s.question, {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}})
    assert sc["legend"] == {"0": "lo", "1": "mid", "2": "hi"}
    check_answer(s.question, sc)
    with pytest.raises(ContractError):
        check_answer(s.question, {**sc, "score": 0.0})


def test_logit_distance_properties():
    T = 2.0
    z = [3.0, 1.0, -2.0]
    p = [math.exp(x / T) for x in z]
    p = [x / sum(p) for x in p]
    assert logit_distance(p, p, T) == 0.0
    shifted = [x + 0.3 for x in z]  # constant shift: same distribution
    q = [math.exp(x / T) for x in shifted]
    q = [x / sum(q) for x in q]
    assert logit_distance(p, q, T) == pytest.approx(0.0, abs=1e-9)
    bumped = [3.4, 1.0, -2.0]
    r = [math.exp(x / T) for x in bumped]
    r = [x / sum(r) for x in r]
    # centred: +0.4 on one of three options -> max |delta| = 0.4 * 2/3
    assert logit_distance(p, r, T) == pytest.approx(0.4 * 2 / 3, rel=1e-6)
    # confident answers keep their gaps: a sharpened copy of the same ranking is far away
    sharp = [math.exp(2 * x / T) for x in z]
    sharp = [x / sum(sharp) for x in sharp]
    assert logit_distance(p, sharp, T) > 2.0


def test_verified_set_is_uniform_and_secret():
    ids = [str(i) for i in range(8)]
    counts = {i: 0 for i in ids}
    for _ in range(4000):
        for c in choose_verified(ids, 3):
            counts[c] += 1
    assert all(abs(v / 4000 - 3 / 8) < 0.05 for v in counts.values())


def test_round_scoring_and_strike():
    params = PoolParams("s1-pro", 2.0, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=8,
                        verified_per_round=3, latency_fast_s=0.0, latency_slow_s=2.0)
    ref = {"a": (0.5, 0.5), "b": (0.9, 0.1), "c": (0.2, 0.8)}
    o = score_round(dict(ref), ref, 0.0, params)
    assert o.score == pytest.approx(1.0) and not o.strike and o.passed == 3 and o.mean_distance == 0.0

    def shifted(p, dz):  # move the first option's logit by dz (logit units, T = 2)
        z = [2.0 * math.log(x) for x in p]
        z[0] += dz
        e = [math.exp(v / 2.0) for v in z]
        return tuple(v / sum(e) for v in e)

    # centred two-option distance of a shift dz is dz / 2
    small = {k: shifted(v, 0.3) for k, v in ref.items()}  # mean 0.15: scores
    assert score_round(small, ref, 0.0, params).score > 0
    medium = {k: shifted(v, 0.6) for k, v in ref.items()}  # mean 0.3: round lost, no strike
    o = score_round(medium, ref, 0.0, params)
    assert o.score == 0.0 and not o.strike
    large = {k: shifted(v, 1.2) for k, v in ref.items()}  # mean 0.6: strike
    assert score_round(large, ref, 0.0, params).strike
    one_bad = dict(ref)
    one_bad["b"] = shifted(ref["b"], 5.0)  # one item at 2.5 > tau_item: strike
    assert score_round(one_bad, ref, 0.0, params).strike
    missing = {k: v for k, v in ref.items() if k != "c"}
    assert score_round(missing, ref, 0.0, params).strike
    assert score_round(None, ref, 0.0, params).score == 0.0


def test_pool_weights():
    w = compute_pool_weights({"s1-fast": {2: 0.9, 3: 0.9}, "s1-pro": {4: 0.8}}, SHARES_BOTH, owner_uid=0, min_ema=0.3)
    assert w[2] == pytest.approx(0.175) and w[4] == pytest.approx(0.65) and 0 not in w
    # an empty pool's share goes to the owner
    w = compute_pool_weights({"s1-fast": {2: 0.9}, "s1-pro": {}}, SHARES_BOTH, owner_uid=0, min_ema=0.3)
    assert w[0] == pytest.approx(0.65) and w[2] == pytest.approx(0.35)
    assert compute_pool_weights({}, SHARES_BOTH, owner_uid=0, min_ema=0.3) == {0: 1.0}
    assert compute_pool_weights({}, SHARES_BOTH, owner_uid=None, min_ema=0.3) == {}


def test_manifests_and_commitments():
    assert FAST.temperature == 2.07 and PRO.temperature == 2.0
    assert FAST.data["weights"]["revision"] == "55de037801a8a9b9de3db5c0e16cef86210c2186"
    assert PRO.data["weights"]["revision"] == "c6b360d47895bb77bdf3805a13ee5a5557ab1921"
    assert len(PRO.weight_files) == 18
    text = commitment_text("s1-pro", PRO.manifest_id)
    assert parse_commitment(text) == ("s1-pro", PRO.manifest_id)
    for bad in ("", "reax-s1/1:s1-vision:" + PRO.manifest_id, "reax-s1/2:s1-pro:" + PRO.manifest_id,
                "reax-s1/1:s1-pro:XYZ", None, 5):
        assert parse_commitment(bad) is None


def test_protocol_roundtrip_and_strictness():
    decisions = tuple(parse_decision(it.decision(), "s1-fast") for it in generate_s1_items(4, seed=1))
    req = S1Request("r1", "s1-fast", FAST.manifest_id, decisions)
    again = parse_s1_request(req.to_bytes())
    assert again == req
    obj = json.loads(req.to_bytes())
    for mutate in (
        lambda o: o.update(protocol="reax-mpm/1"),
        lambda o: o.update(extra=1),
        lambda o: o.update(model="s1-vision"),
        lambda o: o.update(manifest_id="nothex"),
        lambda o: o["decisions"].append(o["decisions"][0]),
        lambda o: o.update(decisions=[]),
    ):
        o = json.loads(json.dumps(obj))
        mutate(o)
        with pytest.raises(ProtocolError):
            parse_s1_request(json.dumps(o).encode())


# ------------------------------------------------------------- tournament
async def _serve(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


class CheatOnSome(S1HashBackend):
    """Right model for most items, but a cheap answer for the first ``bad`` decisions of each batch."""

    def __init__(self, model: str, bad: int):
        super().__init__(model, salt="reference")
        self.bad = bad

    async def decide(self, decisions):
        good = await super().decide(decisions)
        cheap = await S1HashBackend(self.model, salt="cheap").decide(decisions[: self.bad])
        return cheap + good[self.bad:]


class Sharpened(S1HashBackend):
    """Knows the model's ranking but not its logit gaps (e.g. a distilled imitation)."""

    def logits(self, d):
        return [2.0 * z for z in super().logits(d)]


class Slow(S1HashBackend):
    async def decide(self, decisions):
        import asyncio

        await asyncio.sleep(TIMEOUT + 1.0)
        return await super().decide(decisions)


def _inconsistent_app(w, vk, ref):
    """Correct probabilities, but `choice` is not the most likely option."""
    store = auth.NonceStore()

    async def handle(request):
        body = await request.read()
        caller = auth.verify_request(request.headers, body, w.hotkey.ss58_address, store, path=S1_DECIDE_PATH)
        req = parse_s1_request(body)
        answers = await ref.decide(list(req.decisions))
        results = []
        for d, a in zip(req.decisions, answers):
            if a["type"] == "choice":
                a = {**a, "choice": min(a["probabilities"], key=a["probabilities"].get)}
            results.append({"id": d.ident, "answers": {d.question.ident: a}})
        raw = json.dumps({"protocol": PROTOCOL_S1, "request_id": req.request_id, "miner_version": "evil",
                          "results": results}).encode()
        headers = {"Content-Type": "application/json"}
        headers.update(auth.sign_response(w, raw, req.request_id, caller, path=S1_DECIDE_PATH))
        return web.Response(body=raw, headers=headers)

    app = web.Application(client_max_size=1 << 20)
    app.router.add_post(S1_DECIDE_PATH, handle)
    return app


@pytest.fixture
async def s1arena(tmp_path, make_wallet, fake_chain_cls):
    owner = make_wallet("//Owner")
    vwallet = make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    runners = []
    neurons = [
        Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
        Neuron(1, vk, "c1", True, 5000.0, None, None),
    ]
    roles: dict[int, str] = {}
    ports: dict[str, int] = {}

    async def add(role, uid, model, app_factory=None, backend=None, commitment=None, port=None):
        w = make_wallet(f"//Miner/{role}")
        if backend is not None:
            def app_factory(w, backend=backend):
                m = S1Miner(w, backend, max_concurrent=4, per_caller_concurrency=2)
                m.set_validators(frozenset({vk}))
                return m.app()
        if app_factory is not None:
            runner, p = await _serve(app_factory(w))
            runners.append(runner)
            ports[role] = p
        manifest = FAST if model == "s1-fast" else PRO
        c = commitment_text(model, manifest.manifest_id) if commitment is None else commitment
        neurons.append(Neuron(uid, w.hotkey.ss58_address, f"c{uid}", False, 0.0, "127.0.0.1",
                              port if port is not None else ports.get(role), commitment=c))
        roles[uid] = role

    ref_fast = S1HashBackend("s1-fast", salt="reference")
    await add("honest_fast_a", 2, "s1-fast", backend=S1HashBackend("s1-fast", salt="reference"))
    await add("honest_fast_b_gpu_drift", 3, "s1-fast", backend=S1HashBackend("s1-fast", logit_noise=0.15, seed=1))
    await add("honest_pro", 4, "s1-pro", backend=S1HashBackend("s1-pro", salt="reference"))
    await add("wrong_model", 5, "s1-fast", backend=S1HashBackend("s1-fast", salt="other-model"))
    await add("quantised", 6, "s1-fast", backend=S1HashBackend("s1-fast", logit_noise=1.5, seed=2))
    await add("sharpened", 7, "s1-fast", backend=Sharpened("s1-fast"))
    await add("inconsistent_fields", 8, "s1-fast", app_factory=lambda w: _inconsistent_app(w, vk, ref_fast))
    await add("slow", 9, "s1-fast", backend=Slow("s1-fast"))
    await add("uncommitted", 10, "s1-fast", backend=S1HashBackend("s1-fast"), commitment="")
    await add("wrong_manifest", 11, "s1-fast", backend=S1HashBackend("s1-fast"),
              commitment="reax-s1/1:s1-fast:0123456789abcdef")
    await add("pro_cheats_on_half", 12, "s1-pro", backend=CheatOnSome("s1-pro", bad=4))
    # serves s1-fast but commits to s1-pro (role mismatch): the miner refuses the s1-pro request
    await add("role_mismatch", 13, "s1-pro", backend=S1HashBackend("s1-fast"))
    # publishes honest_fast_a's endpoint under its own hotkey
    await add("axon_copy", 14, "s1-fast", port=ports["honest_fast_a"])

    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    env = {
        "REAX_NETUID": "9",
        "REAX_ALLOW_DEV_MODELS": "1",
        "REAX_NETWORK": "ws://127.0.0.1:9951",
        "REAX_ALLOW_PRIVATE_IPS": "1",
        "REAX_STATE_PATH": str(tmp_path / "state.json"),
        "REAX_EMA_ALPHA": "0.5",
        "REAX_BURN_FRACTION": "0.0",
        "REAX_WARMUP_ROUNDS": "0",
        "REAX_FEATURE_S1_PRO_POOL": "1",
    }
    cfg = load_validator(env)
    pools = {
        "s1-fast": PoolParams("s1-fast", 2.07, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=8,
                              verified_per_round=8, latency_fast_s=0.0, latency_slow_s=TIMEOUT),
        "s1-pro": PoolParams("s1-pro", 2.0, tau_mean=0.2, tau_strike=0.5, tau_item=2.0, items_per_round=8,
                             verified_per_round=3, latency_fast_s=0.0, latency_slow_s=TIMEOUT),
    }
    refs = {"s1-fast": ref_fast, "s1-pro": S1HashBackend("s1-pro", salt="reference")}
    val = S1Validator(cfg, vwallet, chain, refs, pools=pools, shares=SHARES_BOTH)
    session = aiohttp.ClientSession(trust_env=False)
    yield {"val": val, "chain": chain, "session": session, "roles": roles, "refs": refs, "neurons": neurons}
    await session.close()
    for r in runners:
        await r.cleanup()


async def test_s1_only_honest_miners_earn(s1arena):
    val, chain, roles = s1arena["val"], s1arena["chain"], s1arena["roles"]
    for _ in range(6):
        summary = await val.run_round(s1arena["session"])
    assert summary is not None
    for uid, role in roles.items():
        s = summary["scores"].get(uid)
        if role.startswith("honest"):
            assert s["passed"] == s["verified"] and s["score"] > 0.8 and not s["strike"], (role, s)
        elif s is not None and role != "pro_cheats_on_half":
            assert s["score"] == 0.0, (role, s)
    for role in ("uncommitted", "wrong_manifest"):
        uid = next(u for u, r in roles.items() if r == role)
        assert uid not in summary["scores"] and val.state.miners[uid].ema == 0.0
    errors = {roles[u]: s["error"] for u, s in summary["scores"].items() if s["error"]}
    assert errors["slow"] == "timeout"
    assert errors["inconsistent_fields"].startswith("protocol")
    assert errors["role_mismatch"] == "http 400"
    assert errors["axon_copy"] == "http 401"
    # Strikes: the wrong/quantised/sharpened models are caught by the logit distance.
    for role in ("wrong_model", "sharpened"):
        uid = next(u for u, r in roles.items() if r == role)
        assert summary["scores"][uid]["strike"], role
    q = next(u for u, r in roles.items() if r == "quantised")
    assert summary["scores"][q]["score"] == 0.0 and val.state.miners[q].ema < 0.3
    uids, weights, _ = chain.set_calls[-1]
    w = dict(zip(uids, weights))
    honest = {u for u, r in roles.items() if r.startswith("honest")}
    assert set(w) <= honest | {12} and honest <= set(w)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    # pools: 0.35 for the two honest s1-fast miners, the s1-pro share to honest_pro (+ cheater if it slipped through)
    assert w[2] + w[3] == pytest.approx(0.35, rel=1e-6)


async def test_spot_check_catches_partial_cheater(s1arena):
    val, roles = s1arena["val"], s1arena["roles"]
    uid = next(u for u, r in roles.items() if r == "pro_cheats_on_half")
    strikes = answered = 0
    honest_pro = next(u for u, r in roles.items() if r == "honest_pro")
    cheat_ema, honest_ema = [], []
    for _ in range(12):
        s = await val.run_round(s1arena["session"])
        sc = s["scores"].get(uid, {})
        strikes += bool(sc.get("strike"))
        answered += bool(sc) and sc.get("error") is None  # a timeout on a loaded CI runner is not a spot check
        cheat_ema.append(val.state.miners[uid].ema)
        honest_ema.append(val.state.miners[honest_pro].ema)
    # P(catch per answered round) = 1 - C(4,3)/C(8,3) = 0.93; up to 11 answered rounds (the first read only
    # records roles)
    assert answered >= 6 and strikes >= answered // 2, (strikes, answered)
    # averaged over the run (a single last round can slip through the spot check)
    assert sum(honest_ema) > 3 * sum(cheat_ema)
    assert val.state.miners[uid].strikes == strikes


async def test_validator_compute_is_spot_checked(s1arena):
    val, refs = s1arena["val"], s1arena["refs"]
    await val.run_round(s1arena["session"])  # first read only records the committed roles
    before = {m: b.calls for m, b in refs.items()}
    await val.run_round(s1arena["session"])
    pro_calls = refs["s1-pro"].calls - before["s1-pro"]
    # two s1-pro miners answer (honest_pro, pro_cheats_on_half): 3 verified items each, not 8
    assert pro_calls == 6


async def test_role_change_resets_score(s1arena):
    val, chain = s1arena["val"], s1arena["chain"]
    for _ in range(3):
        await val.run_round(s1arena["session"])
    assert val.state.miners[2].ema > 0.5
    idx = next(i for i, n in enumerate(chain.neurons) if n.uid == 2)
    chain.neurons[idx] = replace(chain.neurons[idx], commitment=commitment_text("s1-pro", PRO.manifest_id))
    ema_before = val.state.miners[2].ema
    await val.run_round(s1arena["session"])
    # seen once: not applied, not scored
    assert val.state.miners[2].role.startswith("s1-fast:") and val.state.miners[2].ema == ema_before
    await val.run_round(s1arena["session"])
    # confirmed: the miner still serves s1-fast, refuses the s1-pro request and starts from zero
    assert val.state.miners[2].role.startswith("s1-pro:")
    assert val.state.miners[2].ema == 0.0


# ------------------------------------------------------- burn and owner-UID edge cases
def test_full_burn_without_owner_sets_nothing():
    assert compute_pool_weights({"s1-fast": {2: 0.9}}, SHARES_BOTH, owner_uid=None, min_ema=0.3, burn_fraction=1.0) == {}


def test_tiny_probabilities_are_floored():
    # p(false) = 1 - p(true) in float32 on a very confident item: pure rounding must not count
    honest = (1.0 - 6e-8, 6e-8)
    other_gpu = (1.0 - 1e-15, 1e-15)
    assert logit_distance(honest, other_gpu, 2.07) < 1e-6


def test_token_required_for_real_runtimes(tmp_path):
    from reax_subnet.config import ConfigError, load_miner

    base = {"REAX_NETUID": "3", "REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101"}
    with pytest.raises(ConfigError):
        load_miner(base)
    with pytest.raises(ConfigError):
        load_miner({**base, "REAX_RUNTIME_TOKEN_FILE": str(tmp_path / "missing")})
    tok = tmp_path / "tok"
    tok.write_text("secret\n")
    assert load_miner({**base, "REAX_RUNTIME_TOKEN_FILE": str(tok)}).common.runtime_token_file == str(tok)
    with pytest.raises(ConfigError):
        load_miner({**base, "REAX_S1_FAST_RUNTIME_URL": "http://10.0.0.5:8101", "REAX_RUNTIME_TOKEN_FILE": str(tok)})


class FlakyReference(S1HashBackend):
    """Reference runtime that fails for one specific decision content."""

    def __init__(self, model, fail_states):
        super().__init__(model, salt="reference")
        self.fail_states = fail_states

    async def decide_each(self, decisions, retries=1):
        out = []
        for d in decisions:
            if json.dumps(d.state, sort_keys=True) in self.fail_states:
                out.append(RuntimeError("runtime error"))
            else:
                out.extend(await self.decide([d]))
        return out


async def test_one_failed_reference_item_leaves_only_that_miner_unscored(s1arena):
    val = s1arena["val"]
    await val.run_round(s1arena["session"])
    seen = {}

    real = val._new_request

    def capture(model):
        req = real(model)
        seen.setdefault(model, []).append(req)
        return req

    val._new_request = capture
    flaky = FlakyReference("s1-fast", set())
    val.backends["s1-fast"] = flaky
    # make the first s1-fast request's first decision fail on the reference side
    orig = flaky.decide_each

    async def failing(decisions, retries=1):
        if not flaky.fail_states and seen.get("s1-fast"):
            flaky.fail_states.add(json.dumps(seen["s1-fast"][0].decisions[0].state, sort_keys=True))
        return await orig(decisions, retries)

    flaky.decide_each = failing
    summary = await val.run_round(s1arena["session"])
    assert summary is not None
    unscored = [u for u, s in summary["scores"].items() if s.get("error") == "reference unavailable"]
    assert len(unscored) == 1
    others = [s for u, s in summary["scores"].items() if u not in unscored and s.get("strike") is False and s["score"]]
    assert others, "the other miners were still scored"


async def test_mass_role_change_needs_confirmation(s1arena):
    val, chain = s1arena["val"], s1arena["chain"]
    for _ in range(3):
        await val.run_round(s1arena["session"])
    before = {u: (m.ema, m.role) for u, m in val.state.miners.items()}
    saved = list(chain.neurons)
    chain.neurons[:] = [replace(n, commitment="") for n in chain.neurons]
    summary = await val.run_round(s1arena["session"])  # one bad read: nobody reset, nobody scored
    assert summary is not None and summary["targets"] == 0
    assert {u: (m.ema, m.role) for u, m in val.state.miners.items()} == before
    chain.neurons[:] = saved  # the next read is normal again
    assert (await val.run_round(s1arena["session"]))["targets"] > 0
    assert val.state.miners[2].ema >= before[2][0]


async def test_flipping_commitments_do_not_stall_other_miners(s1arena):
    val, chain = s1arena["val"], s1arena["chain"]
    for _ in range(3):
        await val.run_round(s1arena["session"])
    honest = val.state.miners[2].ema
    flippers = [i for i, n in enumerate(chain.neurons) if n.uid in (5, 6, 7)]
    for r in range(6):  # rotating commitments on three hotkeys every read
        for k, i in enumerate(flippers):
            model = "s1-pro" if (r + k) % 2 else "s1-fast"
            m = PRO if model == "s1-pro" else FAST
            chain.neurons[i] = replace(chain.neurons[i], commitment=commitment_text(model, m.manifest_id))
        summary = await val.run_round(s1arena["session"])
        assert summary is not None and summary["scores"][2]["score"] > 0.8  # everybody else is still scored
    assert val.state.miners[2].ema > honest


async def test_flipper_cannot_freeze_its_score(s1arena):
    """alternating between two new commitments must not keep an old score."""
    val, chain = s1arena["val"], s1arena["chain"]
    for _ in range(4):
        await val.run_round(s1arena["session"])
    i = next(k for k, n in enumerate(chain.neurons) if n.uid == 2)
    start = val.state.miners[2].ema
    assert start > 0.5
    for r in range(8):
        commitment = "" if r % 2 else commitment_text("s1-pro", PRO.manifest_id)
        chain.neurons[i] = replace(chain.neurons[i], commitment=commitment)
        await val.run_round(s1arena["session"])
    assert val.state.miners[2].ema < 0.1 * start
    assert val.compute_weights(chain.snapshot()).get(2, 0.0) == 0.0


async def test_weights_count_only_the_current_manifest(s1arena):
    val = s1arena["val"]
    for _ in range(2):
        await val.run_round(s1arena["session"])
    val.state.miners[2].role = "s1-fast:0123456789abcdef"  # a stored role from an older manifest
    val.state.miners[2].ema = 0.99
    assert val.compute_weights(s1arena["chain"].snapshot()).get(2, 0.0) == 0.0


class HungReference(S1HashBackend):
    async def decide_each(self, decisions, retries=1):
        await asyncio.sleep(3600)


async def test_hung_reference_runtime_skips_the_round_quickly(s1arena, monkeypatch):
    """a reference runtime that never answers must not stall the validator."""
    import reax_subnet.s1validator as s1v

    monkeypatch.setattr(s1v, "REFERENCE_BUDGET_MIN_S", 0.5)
    monkeypatch.setattr(s1v, "REFERENCE_BUDGET_ROUNDS", 0.0)
    val = s1arena["val"]
    await val.run_round(s1arena["session"])  # role confirmation
    val.backends["s1-fast"] = HungReference("s1-fast")
    t0 = asyncio.get_running_loop().time()
    assert await val.run_round(s1arena["session"]) is None
    assert asyncio.get_running_loop().time() - t0 < 10


async def test_alternating_with_the_stored_role_still_decays(s1arena):
    """R, X, R, X ... must not skip every other round for free."""
    val, chain, session = s1arena["val"], s1arena["chain"], s1arena["session"]
    for _ in range(4):
        await val.run_round(session)
    i = next(k for k, n in enumerate(chain.neurons) if n.uid == 2)
    good, port = chain.neurons[i].commitment, chain.neurons[i].port
    for r in range(10):
        if r % 2:
            chain.neurons[i] = replace(chain.neurons[i], commitment="", port=1)
        else:
            chain.neurons[i] = replace(chain.neurons[i], commitment=good, port=port)
        await val.run_round(session)
    assert val.state.miners[2].ema < val.state.miners[3].ema - 0.05


async def test_single_honest_role_change_is_not_decayed(s1arena):
    val, chain, session = s1arena["val"], s1arena["chain"], s1arena["session"]
    for _ in range(4):
        await val.run_round(session)
    i = next(k for k, n in enumerate(chain.neurons) if n.uid == 2)
    ema = val.state.miners[2].ema
    chain.neurons[i] = replace(chain.neurons[i], commitment="")
    await val.run_round(session)
    assert val.state.miners[2].ema == ema  # first unconfirmed read: shielded


def test_huge_integer_in_answer_is_a_contract_error():
    q = parse_decision({"id": "d", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}},
                       "s1-fast").question
    with pytest.raises(ContractError):
        check_answer(q, {"type": "noul", "probabilities": {"true": 10**400, "false": 0.0}, "noul": 0.5})
    with pytest.raises(ContractError):
        check_answer(q, {"type": "noul", "probabilities": {"true": 0.5, "false": 0.5}, "noul": -(10**400)})
