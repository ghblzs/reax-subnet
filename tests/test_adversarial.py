"""In-process tournament: one validator against honest and adversarial miners.

Every miner is a real aiohttp server on 127.0.0.1; the chain is faked. The
attacks come from the threat model in docs/THREAT-MODEL.md.
"""

from __future__ import annotations

from reax_subnet.version import WEIGHTS_VERSION_KEY

import asyncio
import json
import math
from dataclasses import replace

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet.backends import HashBackend
from reax_subnet.chain import Neuron
from reax_subnet.config import load_validator
from reax_subnet.miner import Miner
from reax_subnet.protocol import parse_request
from reax_subnet.scoring import ScoringParams
from reax_subnet.validator import Validator, ip_allowed
from reax_subnet.version import PROTOCOL

TIMEOUT = 2.0


class FnBackend:
    model_alias = "hash-test"

    def __init__(self, fn):
        self.fn = fn

    def predict(self, items):
        return [self.fn(it) for it in items]


class SlowBackend(HashBackend):
    def predict(self, items):
        import time

        time.sleep(TIMEOUT + 1.0)
        return super().predict(items)


async def _serve(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def _custom_app(wallet, validator_hotkey, make_body, sign=True, sign_with=None, status=200, extra=None):
    """A miner that verifies the request properly but answers with ``make_body(req)``."""
    store = auth.NonceStore()

    async def handle(request):
        body = await request.read()
        caller = auth.verify_request(request.headers, body, wallet.hotkey.ss58_address, store)
        req = parse_request(body)
        raw = make_body(req)
        headers = {"Content-Type": "application/json"}
        if sign:
            headers.update(auth.sign_response(sign_with or wallet, raw, req.request_id, caller))
        if extra:
            headers.update(extra)
        return web.Response(body=raw, headers=headers, status=status)

    app = web.Application(client_max_size=1 << 20)
    app.router.add_post(auth.DECIDE_PATH, handle)
    return app


def _results_body(req, fn, request_id=None):
    return json.dumps(
        {
            "protocol": PROTOCOL,
            "request_id": request_id or req.request_id,
            "miner_version": "evil",
            "results": [{"id": it.item_id, "probs": fn(it)} for it in req.items],
        }
    ).encode()


@pytest.fixture
async def arena(tmp_path, make_wallet, fake_chain_cls):
    owner = make_wallet("//Owner")
    vwallet = make_wallet("//Validator")
    vk = vwallet.hotkey.ss58_address
    ref = HashBackend(salt="reference")
    runners = []
    neurons = [
        Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
        Neuron(1, vk, "c1", True, 5000.0, None, None),
    ]
    roles: dict[int, str] = {}
    ports: dict[str, int] = {}
    wallets: dict[str, object] = {}

    async def add(role, app_factory, uid, ip="127.0.0.1", port=None, hotkey=None):
        w = make_wallet(f"//Miner/{role}")
        wallets[role] = w
        if app_factory is not None:
            runner, p = await _serve(app_factory(w))
            runners.append(runner)
            ports[role] = p
        neurons.append(Neuron(uid, hotkey or w.hotkey.ss58_address, f"c{uid}", False, 0.0, ip, port if (port is not None or ip is None) else ports[role]))
        roles[uid] = role

    def honest_miner(backend):
        def factory(w):
            m = Miner(w, backend, max_concurrent=4, per_caller_concurrency=2)
            m.set_validators(frozenset({vk}))
            return m.app()
        return factory

    uniform = lambda it: [1 / len(it.options)] * len(it.options)
    first = lambda it: [1.0] + [0.0] * (len(it.options) - 1)
    honest_fn = lambda it: list(ref.predict([it])[0])

    await add("honest_a", honest_miner(HashBackend(salt="reference")), 2)
    await add("honest_b", honest_miner(HashBackend(salt="reference")), 3)
    await add("wrong_model", honest_miner(HashBackend(salt="cheaper-model")), 4)
    await add("uniform", honest_miner(FnBackend(lambda it: tuple(uniform(it)))), 5)
    await add("first_option", honest_miner(FnBackend(lambda it: tuple(first(it)))), 6)
    await add("slow", honest_miner(SlowBackend(salt="reference")), 7)
    await add("nan", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn).replace(b"0.", b"NaN", 1)), 8)
    await add("wrong_length", lambda w: _custom_app(w, vk, lambda req: _results_body(req, lambda it: honest_fn(it) + [0.0])), 9)
    await add("unsigned", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn), sign=False), 10)
    other = make_wallet("//Someone/else")
    await add("wrong_signer", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn), sign_with=other), 11)
    await add("stale_request_id", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn, request_id="0" * 32)), 12)
    await add("oversize", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn) + b" " * 70_000), 13)
    await add("garbage", lambda w: _custom_app(w, vk, lambda req: b"<html>hi</html>"), 14)
    # A 400-digit integer probability: float() raises OverflowError.
    await add("overflow", lambda w: _custom_app(w, vk, lambda req: _results_body(req, honest_fn).replace(
        b'"probs": [', b'"probs": [' + b"9" * 400 + b",", 1)), 20)
    await add("redirect", lambda w: _custom_app(w, vk, lambda req: b"", status=307,
                                                 extra={"Location": f"http://127.0.0.1:{ports['honest_a']}/v1/decide"}), 15)
    # Publishes honest_a's endpoint under its own hotkey (axon copying).
    await add("axon_copy", None, 16, port=ports["honest_a"])

    # Relays the validator's exact request to honest_a and returns its answer.
    def relay_factory(w):
        async def handle(request):
            body = await request.read()
            fwd = {k: v for k, v in request.headers.items() if k.lower().startswith("x-bittensor") or k == "Content-Type"}
            async with aiohttp.ClientSession() as s:
                async with s.post(f"http://127.0.0.1:{ports['honest_a']}/v1/decide", data=body, headers=fwd) as r:
                    return web.Response(body=await r.read(), status=r.status, headers={
                        k: v for k, v in r.headers.items() if k.lower().startswith("x-bittensor")})
        app = web.Application()
        app.router.add_post(auth.DECIDE_PATH, handle)
        return app

    await add("relay", relay_factory, 17)
    await add("private_ip_target", None, 18, ip="169.254.169.254", port=80)
    await add("no_axon", None, 19, ip=None, port=None)

    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    env = {
        "REAX_NETUID": "9",
        "REAX_MODEL": "hash-test",
        "REAX_MECHANISM": "mpm1",
        "REAX_ALLOW_DEV_MODELS": "1",
        "REAX_NETWORK": "ws://127.0.0.1:9951",
        "REAX_ALLOW_PRIVATE_IPS": "1",
        "REAX_STATE_PATH": str(tmp_path / "state.json"),
        "REAX_REQUEST_TIMEOUT_S": str(TIMEOUT),
        "REAX_LATENCY_FAST_S": "0.0",
        "REAX_LATENCY_SLOW_S": str(TIMEOUT),
        "REAX_EMA_ALPHA": "0.5",
        "REAX_BURN_FRACTION": "0.1",
        "REAX_WARMUP_ROUNDS": "0",
        "REAX_DEVICE": "cpu",
    }
    cfg = load_validator(env)
    val = Validator(cfg, vwallet, chain, ref)
    session = aiohttp.ClientSession(trust_env=False)
    yield {"val": val, "chain": chain, "session": session, "roles": roles, "neurons": neurons,
           "make_wallet": make_wallet, "cfg": cfg, "vwallet": vwallet, "ref": ref}
    await session.close()
    for r in runners:
        await r.cleanup()


async def test_only_honest_miners_earn(arena):
    val, chain, roles = arena["val"], arena["chain"], arena["roles"]
    summary = None
    for _ in range(5):
        summary = await val.run_round(arena["session"])
    assert summary is not None
    for uid, role in roles.items():
        s = summary["scores"].get(uid)
        if role.startswith("honest"):
            assert s["fidelity"] == 1.0 and s["score"] > 0.8, (role, s)
        elif s is not None:
            assert s["score"] == 0.0, (role, s)
    # The private/link-local target and the miner without an endpoint are never contacted.
    for role in ("private_ip_target", "no_axon"):
        uid = next(u for u, r in roles.items() if r == role)
        assert uid not in summary["scores"]
        assert val.state.miners[uid].ema == 0.0
    uids, weights, version = chain.set_calls[-1]
    w = dict(zip(uids, weights))
    honest = {u for u, r in roles.items() if r.startswith("honest")}
    assert set(w) == honest | {0}
    assert w[0] == pytest.approx(0.1)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    assert version == WEIGHTS_VERSION_KEY
    errors = {roles[u]: s["error"] for u, s in summary["scores"].items() if s["error"]}
    assert errors["slow"] == "timeout"
    assert errors["unsigned"].startswith("auth")
    assert errors["wrong_signer"].startswith("auth")
    assert errors["stale_request_id"].startswith(("auth", "protocol"))
    assert errors["oversize"].startswith("protocol")
    assert errors["redirect"] == "http 307"
    assert errors["axon_copy"] == "http 401"
    assert errors["relay"] == "http 401"


async def test_recycled_uid_does_not_inherit_score(arena):
    val, chain = arena["val"], arena["chain"]
    for _ in range(3):
        await val.run_round(arena["session"])
    assert val.state.miners[2].ema > 0.5
    newcomer = arena["make_wallet"]("//Newcomer")
    idx = next(i for i, n in enumerate(chain.neurons) if n.uid == 2)
    chain.neurons[idx] = replace(chain.neurons[idx], hotkey=newcomer.hotkey.ss58_address)
    await val.run_round(arena["session"])
    # The new hotkey does not run a server at that endpoint -> 401 and a fresh, zero score.
    assert val.state.miners[2].hotkey == newcomer.hotkey.ss58_address
    assert val.state.miners[2].ema == 0.0
    uids, weights, _ = chain.set_calls[-1]
    assert 2 not in uids


async def test_chain_outage_and_weight_failure_do_not_crash(arena):
    val, chain = arena["val"], arena["chain"]
    chain.fail_snapshot = True
    assert await val.run_round(arena["session"]) is None
    assert val.state.rounds == 0
    chain.fail_snapshot = False
    chain.fail_set_weights = True
    assert await val.run_round(arena["session"]) is not None
    assert chain.set_calls == []


async def test_rate_limit_respected(arena):
    val, chain = arena["val"], arena["chain"]
    chain.since_update = 50
    chain.rate_limit = 100
    await val.run_round(arena["session"])
    assert chain.set_calls == []
    chain.since_update = 400
    await val.run_round(arena["session"])
    assert len(chain.set_calls) == 1


async def test_restart_keeps_scores(arena):
    val, chain = arena["val"], arena["chain"]
    for _ in range(3):
        await val.run_round(arena["session"])
    restarted = Validator(arena["cfg"], arena["vwallet"], chain, arena["ref"])
    assert restarted.state.ema() == val.state.ema()


async def test_dry_run_sets_no_weights(arena):
    val, chain = arena["val"], arena["chain"]
    val.cfg = replace(val.cfg, set_weights=False)
    await val.run_round(arena["session"])
    assert chain.set_calls == [] and val.last_weights


def test_ip_filter():
    assert ip_allowed("8.8.8.8", False)
    for ip in ("127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.169.254", "::1", "fe80::1", "0.0.0.0", "224.0.0.1"):
        assert not ip_allowed(ip, False), ip
    assert ip_allowed("10.1.2.3", True) and ip_allowed("127.0.0.1", True)
    assert not ip_allowed("169.254.169.254", True)  # cloud metadata stays blocked even for local tests
    assert not ip_allowed("0.0.0.0", True)
    assert not ip_allowed("not-an-ip", True)


async def test_fresh_validator_waits_for_warmup(arena):
    val, chain = arena["val"], arena["chain"]
    val.cfg = replace(val.cfg, warmup_rounds=3)
    for _ in range(2):
        await val.run_round(arena["session"])
    assert chain.set_calls == []  # no all-to-owner vector from near-zero averages
    await val.run_round(arena["session"])
    uids, _, _ = chain.set_calls[-1]
    assert {u for u, r in arena["roles"].items() if r.startswith("honest")} <= set(uids)


async def test_truncated_metagraph_skips_round(arena):
    val, chain = arena["val"], arena["chain"]
    await val.run_round(arena["session"])
    before = dict(val.state.ema())
    full = list(chain.neurons)
    chain.neurons = full[:3]
    assert await val.run_round(arena["session"]) is None
    assert val.state.ema() == before
    chain.neurons = []
    assert await val.run_round(arena["session"]) is None
    # A confirmed shrink (owner trim) is accepted on the second identical read.
    chain.neurons = full[:-6]
    assert await val.run_round(arena["session"]) is None
    assert await val.run_round(arena["session"]) is not None
    chain.neurons = full


async def test_stalled_set_weights_does_not_block_loop(arena, monkeypatch):
    import time as _time

    from reax_subnet import validator as vmod

    val, chain = arena["val"], arena["chain"]
    monkeypatch.setattr(vmod, "SET_WEIGHTS_TIMEOUT_S", 0.5)
    original = chain.set_weights
    chain.set_weights = lambda *a: (_time.sleep(3), original(*a))
    started = _time.monotonic()
    await val.run_round(arena["session"])
    assert _time.monotonic() - started < TIMEOUT + 3.5
    # A second round while the first submission is still running must not submit again.
    assert await val.maybe_set_weights(chain.snapshot()) is False
