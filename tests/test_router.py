"""Offline private-engine contract tests with real signed in-process miners."""
import asyncio
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet.chain import Neuron
from reax_subnet.config import ConfigError
from reax_subnet.manifest import commitment_text, load_manifest
from reax_subnet.router import Router, RouterConfig, read_token
from reax_subnet.router_tokens import PlumbTokenCounter, encode, option_texts
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1items import generate_s1_items
from reax_subnet.miner import S1Miner
from reax_subnet.s1protocol import S1_DECIDE_PATH, S1Response, parse_s1_request
from reax_subnet.state import MinerState, ValidatorState, save


async def serve(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest.fixture
async def arena(tmp_path, make_wallet, fake_chain_cls):
    wallet = make_wallet("//RouterValidator")
    vk = wallet.hotkey.ss58_address
    cfg = RouterConfig("x" * 32, "plumb-pin", "tokenizer-pin", timeout_s=0.25, shadow_rate=0,
                       strikes_file=str(tmp_path / "strikes.jsonl"))
    vcfg = SimpleNamespace(common=SimpleNamespace(netuid=9), state_path=str(tmp_path / "state.json"),
                           allow_private_ips=True)
    chain = fake_chain_cls([])
    router = Router(cfg, vcfg, wallet, chain, lambda model, decisions: 123 * len(decisions))
    state = ValidatorState(9, vk)
    runners, seen = [], []
    backend = S1HashBackend()

    async def add(mode="honest", score=0.9, custom_backend=None):
        uid = len(chain.neurons) + 1
        minerwallet = make_wallet(f"//RouterMiner/{uid}")
        miner = S1Miner(minerwallet, custom_backend or S1HashBackend())
        miner.set_validators(frozenset({vk}))
        app = miner.app()
        if mode != "honest":
            async def handle(request):
                raw = await request.read()
                caller = auth.verify_request(request.headers, raw, minerwallet.hotkey.ss58_address,
                                             auth.NonceStore(), path=S1_DECIDE_PATH)
                req = parse_s1_request(raw)
                seen.append(json.loads(raw))
                if mode == "timeout":
                    await asyncio.sleep(0.4)
                answers = await backend.decide(req.decisions)
                if mode == "invalid":
                    answers[0]["noul"] = 2.0
                results = tuple((d.ident, {d.question.ident: a}) for d, a in zip(req.decisions, answers))
                response = S1Response(req.request_id, "test", results).to_bytes()
                headers = auth.sign_response(minerwallet, response, req.request_id, caller, path=S1_DECIDE_PATH)
                if mode == "bad_signature":
                    headers["X-Bittensor-Signature"] = "0x" + "00" * 64
                return web.Response(body=response, headers=headers)
            app = web.Application()
            app.router.add_post(S1_DECIDE_PATH, handle)
        else:
            @web.middleware
            async def capture(request, handler):
                seen.append(json.loads(await request.read()))
                return await handler(request)
            app.middlewares.append(capture)
        runner, port = await serve(app)
        runners.append(runner)
        hk = minerwallet.hotkey.ss58_address
        manifest = load_manifest("s1-fast")
        neuron = Neuron(uid, hk, "cold", False, 0, "127.0.0.1", port,
                        commitment_text("s1-fast", manifest.manifest_id))
        chain.neurons.append(neuron)
        state.miners[uid] = MinerState(hk, ema=score, role=f"s1-fast:{manifest.manifest_id}")
        save(state, vcfg.state_path)
        await router.refresh()
        return neuron

    runner, port = await serve(router.app())
    runners.append(runner)
    session = aiohttp.ClientSession()
    async def post(body=None, token="x" * 32):
        data = {"request_id": "caller-secret-id", "model": "s1-fast", "state": "private payload secret",
                "questions": {"caller-qid": {"type": "noul", "instructions": "private instruction"}},
                "deadline_unix_ms": int(time.time() * 1000) + 5000}
        if body:
            data.update(body)
        headers = {} if token is None else {"X-S1M-Internal-Token": token}
        async with session.post(f"http://127.0.0.1:{port}/internal/v1/decide", json=data, headers=headers) as r:
            return r.status, await r.json()
    yield SimpleNamespace(router=router, add=add, post=post, seen=seen, state=state, chain=chain,
                          vcfg=vcfg, session=session, url=f"http://127.0.0.1:{port}", backend=backend)
    await session.close()
    for runner in reversed(runners):
        await runner.cleanup()


def test_token_file(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    assert read_token(path) == "x" * 32
    for mode in (0o644, 0o400, 0o640):
        path.chmod(mode)
        with pytest.raises(ConfigError):
            read_token(path)
    path.chmod(0o600)
    path.write_text("short")
    with pytest.raises(ConfigError):
        read_token(path)
    with pytest.raises(OSError):
        read_token(tmp_path / "absent")


@pytest.mark.parametrize("token,status", [(None, 401), ("wrong", 401), ("x" * 32, 200)])
async def test_auth(arena, token, status):
    await arena.add()
    assert (await arena.post(token=token))[0] == status


@pytest.mark.parametrize("body,status", [({"images": []}, 422), ({"model": "s1-pro"}, 400),
    ({"model": "bad"}, 400), ({"questions": {}}, 400), ({"deadline_unix_ms": True}, 400),
    ({"deadline_unix_ms": 10 ** 999}, 400), ({"state": 1}, 400), ({"questions": {"q": {"type": "bad"}}}, 400)])
async def test_bad_request(arena, body, status):
    await arena.add()
    assert (await arena.post(body))[0] == status


async def test_happy_privacy_and_billing(arena, caplog):
    await arena.add()
    with caplog.at_level(logging.INFO):
        status, result = await arena.post()
    assert status == 200
    req = parse_s1_request(json.dumps(arena.seen[0]).encode())
    expected = (await arena.backend.decide(req.decisions))[0]
    assert result["answers"] == {"caller-qid": expected}
    assert result["usage"] == {"input_tokens": 123, "output_tokens": 0, "decisions": 1}
    assert result["request_id"] == "caller-secret-id" and result["model"] == "s1-fast"
    assert result["model_revision"] == "plumb-pin" and result["tokenizer_version"] == "tokenizer-pin"
    assert result["latency_ms"] >= 0
    assert "caller-secret-id" not in json.dumps(arena.seen)
    assert "caller-qid" not in json.dumps(arena.seen)
    for private in ("private payload secret", "private instruction", "caller-secret-id", "caller-qid"):
        assert private not in caplog.text


async def test_many_questions(arena):
    await arena.add()
    qs = {str(i): it.question() for i, it in enumerate(generate_s1_items(3, seed=10))}
    status, result = await arena.post({"questions": qs})
    assert status == 200 and set(result["answers"]) == set(qs)
    assert result["usage"]["input_tokens"] == 369 and result["usage"]["decisions"] == 3
    assert len(arena.seen[0]["decisions"]) == 3


@pytest.mark.parametrize("mode", ["timeout", "invalid", "bad_signature"])
async def test_fallback(arena, mode):
    first = await arena.add(mode, score=0.95)
    await arena.add(score=0.8)
    started = time.monotonic()
    assert (await arena.post())[0] == 200
    assert time.monotonic() - started < 0.5
    assert arena.router.health[first.hotkey].failures == 1
    assert len(arena.seen) == 2
    assert arena.seen[0]["request_id"] == arena.seen[1]["request_id"]


async def test_no_miners_and_breaker(arena):
    assert (await arena.post())[0] == 503
    n = await arena.add("invalid")
    for _ in range(3):
        status, result = await arena.post()
        assert status == 503 and result == {"error": {"message": "no eligible miner"}}
    assert arena.router.health[n.hotkey].open_until > time.time() + 59
    assert (await arena.post())[0] == 503 and len(arena.seen) == 3
    assert not arena.router.ready("s1-fast")


async def test_kill_staleness_readiness(arena):
    await arena.add()
    async with arena.session.get(arena.url + "/health/ready?model=s1-fast") as r:
        assert r.status == 200
        assert await r.json() == {"ok": True, "models": {"s1-fast": {"state": "ready", "revision": "plumb-pin"}}}
    arena.router.fresh_at -= 3601
    assert (await arena.post())[0] == 503
    async with arena.session.get(arena.url + "/health/ready?model=s1-fast") as r:
        assert r.status == 503
    await arena.router.refresh()
    arena.router.cfg = replace(arena.router.cfg, enabled=False)
    assert not arena.router.ready("s1-fast") and (await arena.post())[0] == 503
    async with arena.session.get(arena.url + "/health") as r:
        assert r.status == 200


async def test_join_score_role_hotkey_and_timestamp(arena):
    await arena.add()
    path = Path(arena.vcfg.state_path)
    original = path.read_bytes()
    raw = json.loads(original)
    for key, value in (("ema", 0.1), ("hotkey", "foreign"), ("role", "foreign"), ("last_strike_at", time.time())):
        bad = json.loads(original)
        bad["miners"]["1"][key] = value
        path.write_text(json.dumps(bad))
        await arena.router.refresh()
        assert not arena.router.ready("s1-fast")
    raw["saved_at"] -= 3601
    path.write_text(json.dumps(raw))
    arena.router.fresh_at = 0
    await arena.router.refresh()
    assert not arena.router.ready("s1-fast")
    path.write_bytes(b"corrupt secret payload")
    await arena.router.refresh()
    assert path.read_bytes() == b"corrupt secret payload"  # never rename/rewrite validator state
    assert not list(path.parent.glob("*.corrupt-*"))


async def test_last_good_chain_failure_expires(arena):
    await arena.add()
    arena.chain.fail_snapshot = True
    await arena.router.refresh()
    assert arena.router.ready("s1-fast")
    arena.router.fresh_at -= 3601
    await arena.router.refresh()
    assert not arena.router.ready("s1-fast")


async def test_min_miners_cap_deadline_counter(arena):
    n = await arena.add()
    arena.router.cfg = replace(arena.router.cfg, min_miners=2)
    assert (await arena.post())[0] == 503
    arena.router.cfg = replace(arena.router.cfg, min_miners=1)
    h = arena.router.health[n.hotkey]
    h.inflight = 4
    assert (await arena.post())[0] == 503 and not arena.seen
    h.inflight = 0
    assert (await arena.post({"deadline_unix_ms": int(time.time() * 1000) + 200}))[0] == 503
    for counter in (None, lambda *args: 0, lambda *args: True):
        arena.router.counter = counter
        assert (await arena.post())[0] == 503
    assert not arena.seen


async def test_shadow_strike(arena, caplog):
    n = await arena.add(custom_backend=S1HashBackend(salt="wrong"))
    original = Path(arena.vcfg.state_path).read_bytes()
    arena.router.cfg = replace(arena.router.cfg, shadow_rate=1)
    class WrongReference(S1HashBackend):
        async def decide(self, ds):
            return [{"probabilities": {"true": 1.0, "false": 0.0}} for _ in ds]
    arena.router.backends["s1-fast"] = WrongReference()
    with caplog.at_level(logging.INFO):
        assert (await arena.post())[0] == 200
        await asyncio.gather(*arena.router.tasks)
    h = arena.router.health[n.hotkey]
    assert h.open_until > time.time() + 86390
    raw = Path(arena.router.cfg.strikes_file).read_text()
    strike = json.loads(raw)
    assert strike["uid"] == n.uid and strike["hotkey"] == n.hotkey
    assert strike["outcome"] == "shadow_strike"
    assert "private payload" not in raw and "caller-secret-id" not in raw
    assert Path(arena.vcfg.state_path).read_bytes() == original
    restored = Router(arena.router.cfg, arena.vcfg, arena.router.wallet, arena.chain)
    assert restored.health[n.hotkey].open_until > time.time() + 86390
    assert "outcome=strike" in caplog.text


async def test_reference_failure_is_not_a_strike(arena):
    class Failed(S1HashBackend):
        async def decide(self, ds):
            raise RuntimeError("private payload error")
    n = await arena.add()
    arena.router.cfg = replace(arena.router.cfg, shadow_rate=1)
    arena.router.backends["s1-fast"] = Failed()
    assert (await arena.post())[0] == 200
    await asyncio.gather(*arena.router.tasks)
    assert arena.router.health[n.hotkey].open_until == 0
    assert not Path(arena.router.cfg.strikes_file).exists()


class RecordingTokenizer:
    """Deterministic tokenizer double exposing differences in messages and flags."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return json.dumps(messages, ensure_ascii=False)
    def encode(self, prompt, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(prompt.encode())


def test_cpu_counter_counts_and_enforces_limit():
    ds = [parse_decision(it.decision(), "s1-fast") for it in generate_s1_items(3, seed=9)]
    tok = RecordingTokenizer()
    counter = PlumbTokenCounter(tok)
    assert counter("s1-fast", ds) == sum(len(encode(tok, d.state, d.question.instructions,
                                            option_texts(d.question.author_dict()))) for d in ds)
    with pytest.raises(RuntimeError):
        counter("s1-pro", ds)
    class Huge(RecordingTokenizer):
        def encode(self, *args, **kwargs):
            return list(range(4097))
    with pytest.raises(ValueError):
        PlumbTokenCounter(Huge())("s1-fast", ds)


def test_50_encodings_equal_pinned_jevk5():
    runtime = pytest.importorskip("jevk5.runtime")
    import hashlib
    assert hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() == load_manifest("s1-fast").data["runtime"]["source_files"]["jevk5/runtime.py"]
    # __init__ loads weights/GPU; encode only reads .tok, so bypass __init__.
    model = object.__new__(runtime.JevK5)
    model.tok = RecordingTokenizer()
    for it in generate_s1_items(50, seed=42):
        q = it.question()
        options = [text for _, text in runtime.decision_options(q)]
        assert option_texts(q) == options
        assert encode(model.tok, it.state, it.instructions, options) == model.encode(it.state, it.instructions, options)


async def test_two_attempt_limit(arena):
    for _ in range(3):
        await arena.add("invalid")
    assert (await arena.post())[0] == 503
    assert len(arena.seen) == 2


async def test_duplicate_token_and_malformed_body(arena):
    await arena.add()
    async with arena.session.post(arena.url + "/internal/v1/decide", data=b"{}",
            headers=[("X-S1M-Internal-Token", "wrong"), ("X-S1M-Internal-Token", "x" * 32)]) as r:
        assert r.status == 401
    for data in (b"not json", b"[]", b"x" * (512 * 1024 + 1)):
        async with arena.session.post(arena.url + "/internal/v1/decide", data=data,
                headers={"X-S1M-Internal-Token": "x" * 32}) as r:
            assert r.status == 400
    assert not arena.seen


def test_router_config(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok"}
    cfg = RouterConfig.from_env(base)
    assert cfg.models == ("s1-fast",) and cfg.min_miners == 1 and cfg.timeout_s == 3
    assert RouterConfig.from_env({**base, "REAX_ROUTER_ENABLED": "0"}).enabled is False
    assert RouterConfig.from_env({**base, "REAX_ROUTER_MODELS": "s1-pro"}).models == ("s1-pro",)
    for var, value in (("REAX_ROUTER_MIN_MINERS", "0"), ("REAX_ROUTER_MODEL_REVISION", ""),
            ("REAX_ROUTER_SHADOW_RATE", "2"), ("REAX_ROUTER_MODELS", "bad"),
            ("REAX_ROUTER_TIMEOUT_S", "nan")):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, var: value})


async def test_counter_unavailable_health(arena):
    await arena.add()
    arena.router.counter = None
    assert not arena.router.ready("s1-fast")
    async with arena.session.get(arena.url + "/health/ready?model=s1-fast") as r:
        assert r.status == 503


async def test_counting_failure_vs_input_limit(arena):
    from reax_subnet.router_tokens import InputTokenLimitError
    await arena.add()
    def unavailable(*args):
        raise ValueError("tokenizer chat template missing")
    arena.router.counter = unavailable
    assert (await arena.post())[0] == 503
    def oversized(*args):
        raise InputTokenLimitError("limit")
    arena.router.counter = oversized
    assert (await arena.post())[0] == 400
    assert not arena.seen
