"""Offline private-engine contract tests with real signed in-process miners."""
import asyncio
import json
import logging
import math
import secrets
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
    cfg = RouterConfig("x" * 32, "plumb-pin", "tokenizer-pin", timeout_s=0.25, shadow_rate=0, verify_rate=0,
                       probation_answers=0,  # the older tests exercise sampling alone; probation has its own tests below
                       strikes_file=str(tmp_path / "strikes.jsonl"))
    vcfg = SimpleNamespace(common=SimpleNamespace(netuid=9), state_path=str(tmp_path / "state.json"),
                           allow_private_ips=True)
    chain = fake_chain_cls([])
    router = Router(cfg, vcfg, wallet, chain, lambda model, decisions: 123 * len(decisions))
    state = ValidatorState(9, vk)
    runners, seen = [], []
    backend = S1HashBackend()

    async def add(mode="honest", score=0.9, custom_backend=None, rounds=1000, last_score=1.0, strikes=0):
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
                if mode == "busy":
                    return web.json_response({"error": "busy"}, status=429)
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
        state.miners[uid] = MinerState(hk, ema=score, last_score=last_score, rounds=rounds, strikes=strikes,
                                         role=f"s1-fast:{manifest.manifest_id}")
        save(state, vcfg.state_path)
        await router.refresh()
        return neuron

    runner, port = await serve(router.app())
    runners.append(runner)
    session = aiohttp.ClientSession()
    last = SimpleNamespace(headers=None)
    async def post(body=None, token="x" * 32):
        data = {"request_id": "caller-secret-id", "model": "s1-fast", "state": "private payload secret",
                "questions": {"caller-qid": {"type": "noul", "instructions": "private instruction"}},
                "deadline_unix_ms": int(time.time() * 1000) + 5000}
        if body:
            data.update(body)
        headers = {} if token is None else {"X-S1M-Internal-Token": token}
        async with session.post(f"http://127.0.0.1:{port}/internal/v1/decide", json=data, headers=headers) as r:
            last.headers = r.headers
            return r.status, await r.json()
    yield SimpleNamespace(router=router, add=add, post=post, last=last, seen=seen, state=state, chain=chain,
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
    configure(arena, reference_concurrency=3)  # a request goes to a miner only if the reference could check it in one go
    qs = {str(i): it.question() for i, it in enumerate(generate_s1_items(3, seed=10))}
    status, result = await arena.post({"questions": qs})
    assert status == 200 and set(result["answers"]) == set(qs)
    assert result["usage"]["input_tokens"] == 369 and result["usage"]["decisions"] == 3
    assert len(arena.seen[0]["decisions"]) == 3


@pytest.mark.parametrize("mode", ["timeout", "invalid", "bad_signature"])
async def test_fallback(arena, mode):
    first = await arena.add(mode, score=0.95)
    await arena.add(score=0.92)
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
        assert await r.json() == {"ok": True, "models": {"s1-fast": {"state": "ready", "revision": "plumb-pin", "miners": 1,
                                                     "reference_fallback": False}}}
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
    h.inflight = 1
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
    with pytest.raises(ConfigError):  # the single pins are the s1-fast default, never silently s1-pro's
        RouterConfig.from_env({**base, "REAX_ROUTER_MODELS": "s1-pro"})
    pro = {"REAX_ROUTER_MODEL_REVISION_S1_PRO": "pro-rev", "REAX_ROUTER_TOKENIZER_VERSION_S1_PRO": "pro-tok"}
    assert RouterConfig.from_env({**base, **pro, "REAX_ROUTER_MODELS": "s1-pro"}).models == ("s1-pro",)
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


class ShiftedReference(S1HashBackend):
    """Reference whose noul probabilities sit `delta` logits away from the honest hash backend."""
    def __init__(self, delta=0.0):
        super().__init__()
        self.delta, self.requests = delta, 0

    async def decide(self, ds):
        self.requests += 1
        out = await super().decide(ds)
        for a in out:
            p = a["probabilities"]["true"]
            q = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + self.delta)))
            a["probabilities"], a["noul"] = {"true": q, "false": 1 - q}, q
        return out


def configure(arena, backend=None, **cfg):
    arena.router.cfg = replace(arena.router.cfg, **cfg)
    if backend is not None:
        arena.router.backends["s1-fast"] = backend
    return backend


def served(arena):
    h = arena.last.headers
    return h["X-Reax-Served-By"], h["X-Reax-Verified"], h.get("X-Reax-Miner-Uid")


SHAPE = {"request_id", "model", "answers", "usage", "latency_ms", "tokenizer_version", "model_revision"}


async def test_verify_match_served_by_miner(arena):
    n = await arena.add()
    ref = configure(arena, ShiftedReference(0.0), verify_rate=1)
    status, result = await arena.post()
    assert status == 200 and served(arena) == ("miner", "1", str(n.uid))
    assert set(result) == SHAPE and ref.requests == 1
    assert "X-Reax" not in json.dumps(result)
    assert not Path(arena.router.cfg.strikes_file).exists()
    assert arena.router.counts["verify", "s1-fast", "match"] == 1


async def test_verify_strike_mismatch_serves_reference(arena):
    n = await arena.add()
    ref = configure(arena, ShiftedReference(1.0), verify_rate=1)
    status, result = await arena.post()
    assert status == 200 and served(arena) == ("reference", "1", None)
    expected = (await ref.decide(parse_s1_request(json.dumps(arena.seen[0]).encode()).decisions))[0]
    assert result["answers"] == {"caller-qid": expected} and set(result) == SHAPE
    raw = Path(arena.router.cfg.strikes_file).read_text()
    strike = json.loads(raw)
    assert strike["uid"] == n.uid and strike["hotkey"] == n.hotkey and strike["outcome"] == "verify_strike"
    assert "private payload" not in raw and "caller-secret-id" not in raw
    assert arena.router.health[n.hotkey].open_until > time.time() + 86390
    assert arena.router.counts["verify", "s1-fast", "mismatch_strike"] == 1
    # The struck miner is excluded now; the next request goes straight to the reference.
    assert (await arena.post())[0] == 200 and served(arena)[0] == "reference" and len(arena.seen) == 1


async def test_verify_grey_mismatch_no_strike(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.34), verify_rate=1)  # d = 1.035 * 0.34 = 0.35: above tau_mean 0.2, below 0.5
    status, _ = await arena.post()
    assert status == 200 and served(arena) == ("reference", "1", None)
    assert not Path(arena.router.cfg.strikes_file).exists()
    assert arena.router.health[n.hotkey].open_until == 0  # one grey is honest variance: no quarantine, no strike
    assert arena.router.counts["verify", "s1-fast", "mismatch_grey"] == 1


async def test_verify_fails_closed_when_reference_down(arena):
    class Down(S1HashBackend):
        async def decide(self, ds):
            raise RuntimeError("private payload error")
    class Hang(S1HashBackend):
        async def decide(self, ds):
            await asyncio.sleep(5)
    n = await arena.add()
    for backend in (Down(), Hang()):
        configure(arena, backend, verify_rate=1)
        started = time.monotonic()
        status, result = await arena.post()
        assert status == 503 and result == {"error": {"message": "no eligible miner"}}
        assert time.monotonic() - started < 0.6 and "X-Reax-Served-By" not in arena.last.headers
    assert len(arena.seen) == 2  # the miner was asked, its answer was not returned
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 2
    assert arena.router.health[n.hotkey].open_until == 0 and not Path(arena.router.cfg.strikes_file).exists()


async def test_reference_fallback_without_miner(arena):
    configure(arena, ShiftedReference(0.0), verify_rate=1)
    status, result = await arena.post()
    assert status == 200 and served(arena) == ("reference", "1", None) and set(result) == SHAPE
    assert result["usage"] == {"input_tokens": 123, "output_tokens": 0, "decisions": 1}
    async with arena.session.get(arena.url + "/health/ready?model=s1-fast") as r:
        assert r.status == 200
        assert (await r.json())["models"]["s1-fast"] == {"state": "ready", "revision": "plumb-pin", "miners": 0,
                                                         "reference_fallback": True}
    configure(arena, reference_fallback=False)
    assert (await arena.post())[0] == 503
    async with arena.session.get(arena.url + "/health/ready?model=s1-fast") as r:
        assert r.status == 503
    configure(arena, reference_fallback=True, enabled=False)  # kill switch also disables the fallback
    assert (await arena.post())[0] == 503
    assert arena.router.counts["requests", "s1-fast", "none"] == 2


async def test_reference_and_miner_responses_are_identical_in_format(arena):
    await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1)
    _, via_miner = await arena.post()
    configure(arena, shadow_rate=0)
    arena.router.health[arena.chain.neurons[0].hotkey].open_until = time.time() + 99
    _, via_reference = await arena.post()
    assert served(arena)[0] == "reference"
    assert set(via_miner) == set(via_reference) and via_miner["answers"] == via_reference["answers"]
    assert via_miner["usage"] == via_reference["usage"]


async def test_fallback_when_miner_fails_and_verification_off(arena):
    await arena.add("invalid")
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    status, _ = await arena.post()
    assert status == 200 and served(arena) == ("reference", "1", None)
    assert arena.router.counts["attempt", "s1-fast", "error"] == 1


async def test_busy_429_skips_without_breaker(arena):
    busy = await arena.add("busy", score=0.95)
    good = await arena.add(score=0.92)
    for _ in range(4):
        status, _ = await arena.post()
        assert status == 200 and served(arena) == ("miner", "0", str(good.uid))
    h = arena.router.health[busy.hotkey]
    assert h.failures == 0 and h.open_until == 0
    assert arena.router.counts["attempt", "s1-fast", "busy"] == 4
    assert arena.router.counts["uid", "s1-fast", str(good.uid)] == 4


async def test_per_miner_budget(arena):
    n = await arena.add()
    configure(arena, per_miner_per_minute=2)
    assert [(await arena.post())[0] for _ in range(3)] == [200, 200, 503]
    assert len(arena.seen) == 2 and arena.router.counts["attempt", "s1-fast", "budget"] == 1
    h = arena.router.health[n.hotkey]
    h.recent = type(h.recent)(t - 61 for t in h.recent)  # window slides
    assert (await arena.post())[0] == 200


async def test_inflight_cap_defaults_and_budget_fall_back_to_reference(arena):
    assert RouterConfig("x" * 32, "r", "t").inflight_cap == 1
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    arena.router.health[n.hotkey].inflight = 1
    assert (await arena.post())[0] == 200 and served(arena)[0] == "reference" and not arena.seen


async def test_verify_rate_zero_keeps_async_shadow(arena):
    n = await arena.add()
    ref = configure(arena, ShiftedReference(1.0), verify_rate=0, shadow_rate=1)
    status, _ = await arena.post()
    assert status == 200 and served(arena) == ("miner", "0", str(n.uid))  # wrong miner answer still served (old path)
    await asyncio.gather(*arena.router.tasks)
    assert ref.requests == 1 and json.loads(Path(arena.router.cfg.strikes_file).read_text())["outcome"] == "shadow_strike"
    assert arena.router.counts["verify", "s1-fast", "skipped"] == 1


async def test_metrics_and_status_need_token(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1)
    await arena.post()
    for path in ("/metrics", "/status"):
        for headers in ({}, {"X-S1M-Internal-Token": "wrong"}):
            async with arena.session.get(arena.url + path, headers=headers) as r:
                assert r.status == 401
    ok = {"X-S1M-Internal-Token": "x" * 32}
    async with arena.session.get(arena.url + "/metrics", headers=ok) as r:
        text = await r.text()
        assert r.status == 200
    for line in ('reax_router_requests_total{model="s1-fast",served_by="miner"} 1',
                 'reax_router_verify_total{model="s1-fast",outcome="match"} 1',
                 'reax_router_miner_attempts_total{model="s1-fast",outcome="ok"} 1',
                 f'reax_router_served_by_uid_total{{model="s1-fast",uid="{n.uid}"}} 1',
                 'reax_router_eligible_miners{model="s1-fast"} 1',
                 'reax_router_latency_seconds_count{model="s1-fast",served_by="miner"} 1'):
        assert line in text.splitlines(), line
    assert "private" not in text and "caller" not in text and n.hotkey not in text
    async with arena.session.get(arena.url + "/status", headers=ok) as r:
        status = await r.json()
    assert status["models"]["s1-fast"]["uids"] == [n.uid] and status["started_at"] > 0
    assert status["counts"]["requests"] == {"s1-fast:miner": 1}


def test_router_config_new_vars(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok"}
    cfg = RouterConfig.from_env(base)
    assert (cfg.inflight_cap, cfg.per_miner_per_minute, cfg.verify_rate, cfg.reference_fallback) == (1, 12, 1.0, True)
    cfg = RouterConfig.from_env({**base, "REAX_ROUTER_PER_MINER_PER_MINUTE": "19", "REAX_ROUTER_VERIFY_RATE": "0.5",
                                 "REAX_ROUTER_REFERENCE_FALLBACK": "0"})
    assert (cfg.per_miner_per_minute, cfg.verify_rate, cfg.reference_fallback) == (19, 0.5, False)
    for var, value in (("REAX_ROUTER_PER_MINER_PER_MINUTE", "20"), ("REAX_ROUTER_PER_MINER_PER_MINUTE", "0"),
                       ("REAX_ROUTER_VERIFY_RATE", "1.5"), ("REAX_ROUTER_REFERENCE_FALLBACK", "maybe")):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, var: value})


async def read_status(arena):
    async with arena.session.get(arena.url + "/status", headers={"X-S1M-Internal-Token": "x" * 32}) as r:
        assert r.status == 200
        return await r.json()


@pytest.mark.parametrize("path,expected,uids", [
    ("miner", (1, 0, 0, 0, 0), [1]),
    ("match", (1, 0, 0, 1, 0), [1]),
    ("strike", (0, 1, 0, 0, 1), []),
    ("grey", (0, 1, 0, 0, 1), []),
    ("mismatch_no_fallback", (0, 0, 1, 0, 1), []),
    ("reference", (0, 1, 0, 0, 0), []),
    ("none", (0, 0, 1, 0, 0), []),
    ("reference_down", (0, 0, 1, 0, 0), []),
])
async def test_last_24h_paths(arena, path, expected, uids):
    if path not in ("reference", "none"):
        await arena.add()
    if path in ("match", "strike", "grey", "mismatch_no_fallback"):
        delta = {"match": 0, "strike": 1, "grey": 0.34, "mismatch_no_fallback": 0.34}[path]
        configure(arena, ShiftedReference(delta), verify_rate=1,
                  reference_fallback=path != "mismatch_no_fallback")
    elif path == "reference":
        configure(arena, ShiftedReference(), verify_rate=1)
    elif path == "reference_down":
        class Down(S1HashBackend):
            async def decide(self, ds):
                raise RuntimeError("reference down")
        configure(arena, Down(), verify_rate=1)
    response, _ = await arena.post()
    assert response == (503 if expected[2] else 200)
    status = await read_status(arena)
    window = status["models"]["s1-fast"]["last_24h"]
    assert tuple(window[k] for k in ("miner", "reference", "none", "verify_match", "verify_mismatch")) == expected
    assert window["since"] == arena.router.started_at
    assert window["window_seconds"] == 86400
    assert status["miner_uids_served_24h"] == {"s1-fast": uids}


async def test_last_24h_prunes_on_read_and_write(arena, monkeypatch):
    # A non-aligned start also checks that `since` is not rounded to a minute.
    clock = [1800000031.5]
    monkeypatch.setattr("reax_subnet.router.time.time", lambda: clock[0])
    router = arena.router
    router.started_at = clock[0]
    router._record_status("s1-fast", "miner", 7)
    router._record_status("s1-fast", "miner", 2)
    router._record_status("s1-fast", "miner", 7)
    first = await read_status(arena)
    assert first["models"]["s1-fast"]["last_24h"]["miner"] == 3
    assert first["miner_uids_served_24h"]["s1-fast"] == [2, 7]
    clock[0] += 60
    router.unavailable("s1-fast", time.monotonic())
    clock[0] = router.started_at + 86400
    status = await read_status(arena)
    window = status["models"]["s1-fast"]["last_24h"]
    assert window["miner"] == 0 and window["none"] == 1
    assert window["since"] == router.started_at
    assert status["miner_uids_served_24h"]["s1-fast"] == []
    assert len(router.status_buckets["s1-fast"]) == 1  # read pruned the expired bucket
    clock[0] += 86400 + 7
    router.unavailable("s1-fast", time.monotonic())
    assert len(router.status_buckets["s1-fast"]) == 1  # write pruned after an idle day
    window = (await read_status(arena))["models"]["s1-fast"]["last_24h"]
    assert window["none"] == 1 and window["since"] == clock[0] - 86400
    assert router.counts["requests", "s1-fast", "none"] == 2  # lifetime metrics remain intact
    clock[0] += 86400
    window = (await read_status(arena))["models"]["s1-fast"]["last_24h"]
    assert all(window[k] == 0 for k in ("miner", "reference", "none", "verify_match", "verify_mismatch"))
    assert not router.status_buckets["s1-fast"]


async def test_last_24h_bounded_and_separate_models(arena, monkeypatch):
    clock = [1800000000.0]
    monkeypatch.setattr("reax_subnet.router.time.time", lambda: clock[0])
    router = arena.router
    router.cfg = replace(router.cfg, models=("s1-fast", "s1-pro"))
    router.status_buckets["s1-pro"] = type(router.status_buckets["s1-fast"])(maxlen=1440)
    router._record_status("s1-pro", "reference")
    for minute in range(1500):
        clock[0] = 1800000000.0 + minute * 60
        for uid in range(256):
            router._record_status("s1-fast", "miner", uid)
    status = await read_status(arena)
    assert len(router.status_buckets["s1-fast"]) == 1440
    assert status["models"]["s1-fast"]["last_24h"]["miner"] == 1440 * 256
    assert status["miner_uids_served_24h"]["s1-fast"] == list(range(256))
    assert status["models"]["s1-pro"]["last_24h"]["reference"] == 0
    assert status["miner_uids_served_24h"]["s1-pro"] == []


async def test_status_duplicate_token_does_not_read_window(arena):
    async with arena.session.get(arena.url + "/status", headers=[
            ("X-S1M-Internal-Token", "x" * 32), ("X-S1M-Internal-Token", "x" * 32)]) as r:
        assert r.status == 401
    assert not arena.router.status_buckets["s1-fast"]


class SlowReference(S1HashBackend):
    """Counts concurrently running decide() calls; `fail_first` makes the first call fail fast."""
    def __init__(self, delay=0.3, fail_first=False):
        super().__init__()
        self.delay, self.fail_first = delay, fail_first
        self.active = self.peak = self.started = 0

    async def decide(self, ds):
        self.started += 1
        first = self.started == 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.fail_first:
                if first:
                    await asyncio.sleep(0.02)
                    raise RuntimeError("fast failure")
                await asyncio.sleep(5)
            await asyncio.sleep(self.delay)
            return await super().decide(ds)
        finally:
            self.active -= 1


async def test_timeout_quarantines_miner_and_routes_to_reference(arena):
    slow = await arena.add("timeout", score=0.95)
    good = await arena.add(score=0.92)
    configure(arena, ShiftedReference(0.0), verify_rate=0, timeout_quarantine_s=60)
    status, _ = await arena.post()  # the slow miner gets half the budget, the honest one answers
    assert status == 200 and served(arena) == ("miner", "0", str(good.uid)) and len(arena.seen) == 2
    h = arena.router.health[slow.hotkey]
    assert h.open_until > time.time() + 55 and h.failures == 1 and h.inflight == 0
    assert arena.router.counts["attempt", "s1-fast", "timeout"] == 1
    assert not Path(arena.router.cfg.strikes_file).exists()  # quarantine is not a strike
    arena.router.health[good.hotkey].inflight = 1  # the only other miner is busy: reference must answer
    assert (await arena.post())[0] == 200 and served(arena)[0] == "reference"
    assert len(arena.seen) == 2 and arena.router.counts["attempt", "s1-fast", "timeout"] == 1  # slow miner not asked
    h.open_until = 0  # quarantine over: the miner is asked again
    await arena.post()
    assert len(arena.seen) == 3


async def test_invalid_reply_is_not_quarantined(arena):
    n = await arena.add("invalid")
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    await arena.post()
    assert arena.router.health[n.hotkey].open_until == 0


async def test_reference_concurrency_bounded_for_fallback(arena):
    slow = configure(arena, SlowReference(0.3), verify_rate=0, reference_concurrency=2, timeout_s=3)
    results = await asyncio.gather(*(arena.post() for _ in range(10)))
    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 200] + [503] * 8
    assert slow.peak == 2 and slow.started == 2
    assert arena.router.reference_used["s1-fast"] == 0 and slow.active == 0


async def test_reference_slots_count_decisions(arena):
    slow = configure(arena, SlowReference(0.05), verify_rate=0, reference_concurrency=2, timeout_s=3)
    three = {"questions": {f"q{i}": {"type": "noul", "instructions": f"i{i}"} for i in range(3)}}
    two = {"questions": {f"q{i}": {"type": "noul", "instructions": f"i{i}"} for i in range(2)}}
    assert (await arena.post(three))[0] == 503 and slow.started == 0
    assert (await arena.post(two))[0] == 200 and slow.started == 2 and arena.router.reference_used["s1-fast"] == 0


async def test_verification_without_reference_slot_serves_no_miner_answer(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1, reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1  # saturated by other requests
    status, _ = await arena.post()
    assert status == 503 and "X-Reax-Served-By" not in arena.last.headers
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    assert arena.router.health[n.hotkey].open_until == 0 and not Path(arena.router.cfg.strikes_file).exists()


async def test_verify_selected_without_backend_fails_closed(arena):
    await arena.add()
    configure(arena, verify_rate=1)
    assert arena.router.backends == {}
    status, _ = await arena.post()
    assert status == 503 and "X-Reax-Served-By" not in arena.last.headers
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1


async def test_failed_reference_cancels_and_awaits_siblings(arena):
    slow = configure(arena, SlowReference(fail_first=True), verify_rate=0, reference_concurrency=2, timeout_s=3)
    two = {"questions": {f"q{i}": {"type": "noul", "instructions": f"i{i}"} for i in range(2)}}
    started = time.monotonic()
    assert (await arena.post(two))[0] == 503
    assert time.monotonic() - started < 1 and slow.started == 2
    assert slow.active == 0 and arena.router.reference_used["s1-fast"] == 0


async def test_reference_timeout_cancels_tasks(arena):
    slow = configure(arena, SlowReference(5), verify_rate=0, timeout_s=0.2)
    assert (await arena.post())[0] == 503
    assert slow.active == 0 and arena.router.reference_used["s1-fast"] == 0


async def test_reference_probability_sum_checked(arena):
    class Off(S1HashBackend):
        async def decide(self, ds):
            out = await super().decide(ds)
            for a in out:
                a["probabilities"], a["noul"] = {"true": 0.5002, "false": 0.5002}, 0.5002
            return out
    configure(arena, Off(), verify_rate=0)
    assert (await arena.post())[0] == 503
    n = await arena.add()
    configure(arena, verify_rate=1)
    assert (await arena.post())[0] == 503 and arena.router.health[n.hotkey].open_until == 0


async def test_strike_ledger_write_is_off_loop(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(1.0), verify_rate=1)
    loop_thread = __import__("threading").get_ident()
    seen = []
    real = arena.router._write_strike
    def spy(entry):
        seen.append(__import__("threading").get_ident())
        return real(entry)
    arena.router._write_strike = spy
    assert (await arena.post())[0] == 200
    assert seen and seen[0] != loop_thread and arena.router.health[n.hotkey].open_until > time.time() + 86000


async def test_no_200_after_deadline(arena):
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    real = arena.router._reference_answers
    def late(req, results):
        time.sleep(0.4)  # formatting finished after the 0.25 s router timeout
        return real(req, results)
    arena.router._reference_answers = late
    assert (await arena.post())[0] == 503


def test_router_config_quarantine_and_concurrency(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok"}
    cfg = RouterConfig.from_env(base)
    assert (cfg.timeout_quarantine_s, cfg.reference_concurrency) == (60, 2)
    cfg = RouterConfig.from_env({**base, "REAX_ROUTER_TIMEOUT_QUARANTINE_S": "5", "REAX_ROUTER_REFERENCE_CONCURRENCY": "16"})
    assert (cfg.timeout_quarantine_s, cfg.reference_concurrency) == (5, 16)
    for var, value in (("REAX_ROUTER_TIMEOUT_QUARANTINE_S", "4"), ("REAX_ROUTER_TIMEOUT_QUARANTINE_S", "601"),
                       ("REAX_ROUTER_REFERENCE_CONCURRENCY", "0"), ("REAX_ROUTER_REFERENCE_CONCURRENCY", "17")):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, var: value})


async def test_miner_attempt_capped_so_reference_answers_in_time(arena):
    await arena.add("timeout", score=0.9)  # the only miner never answers
    configure(arena, ShiftedReference(0.0), verify_rate=1.0, timeout_s=3, miner_timeout_s=0.2)
    started = time.monotonic()
    status, _ = await arena.post()
    assert status == 200 and served(arena)[0] == "reference"
    assert time.monotonic() - started < 1.5  # capped attempt, not the whole 3 s budget


async def test_unstarted_reference_task_leaves_no_children_after_503(arena):
    n = await arena.add()
    slow = configure(arena, SlowReference(2), verify_rate=1.0, reference_fallback=False, timeout_s=3)
    arena.router.health[n.hotkey].inflight = 1  # the only miner is at its cap: skipped without an await
    assert (await arena.post())[0] == 503
    assert slow.active == 0 and arena.router.reference_used["s1-fast"] == 0  # right after the response, not after the work ends


async def test_reference_not_started_with_too_little_budget(arena):
    n = await arena.add()
    slow = configure(arena, SlowReference(0.05), verify_rate=1.0, min_reference_budget_s=5.0)
    status, _ = await arena.post({"deadline_unix_ms": int(time.time() * 1000) + 2000})
    assert status == 503 and slow.started == 0 and arena.router.reference_used["s1-fast"] == 0
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    assert arena.router.health[n.hotkey].inflight == 0 and len(arena.seen) == 0
    configure(arena, min_reference_budget_s=0.2)
    assert (await arena.post())[0] == 200 and slow.started == 1


def test_router_config_min_reference_budget(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok"}
    assert RouterConfig.from_env(base).min_reference_budget_s == 0.2
    assert RouterConfig.from_env({**base, "REAX_ROUTER_MIN_REFERENCE_BUDGET_S": "0"}).min_reference_budget_s == 0
    for value in ("-0.1", "5.1"):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, "REAX_ROUTER_MIN_REFERENCE_BUDGET_S": value})


# ---- track-record eligibility (customer traffic only for miners that passed our validator for a while) ----

def edit_score(arena, uid, **changes):
    path = Path(arena.vcfg.state_path)
    raw = json.loads(path.read_text())
    raw["miners"][str(uid)].update(changes)
    path.write_text(json.dumps(raw))


def eligible_uids(arena):
    return [n.uid for n, _ in arena.router.candidates("s1-fast")]


def test_track_record_defaults(monkeypatch, tmp_path):
    token = tmp_path / "t"
    token.write_text("x" * 40)
    token.chmod(0o600)
    env = {"REAX_ROUTER_TOKEN_FILE": str(token), "REAX_ROUTER_MODEL_REVISION": "r", "REAX_ROUTER_TOKENIZER_VERSION": "t"}
    cfg = RouterConfig.from_env(env)
    assert (cfg.min_ema, cfg.min_rounds, cfg.strike_cooldown_s) == (0.9, 720, 7 * 86400)
    cfg = RouterConfig.from_env({**env, "REAX_ROUTER_MIN_EMA": "0.95", "REAX_ROUTER_MIN_ROUNDS": "10",
                                 "REAX_ROUTER_STRIKE_COOLDOWN_S": "60"})
    assert (cfg.min_ema, cfg.min_rounds, cfg.strike_cooldown_s) == (0.95, 10, 60)


async def test_fresh_uid_with_perfect_scores_is_excluded(arena):
    old = await arena.add(score=0.95, rounds=720)
    fresh = await arena.add(score=1.0, rounds=719)  # perfect EMA, but not yet a full day of rounds
    assert eligible_uids(arena) == [old.uid]
    status = await read_status(arena)
    model = status["models"]["s1-fast"]
    assert (model["eligible_miners"], model["seen_miners"], model["excluded"]) == (1, 2, {"rounds": 1})
    edit_score(arena, fresh.uid, rounds=720)
    await arena.router.refresh()
    assert eligible_uids(arena) == [fresh.uid, old.uid]


async def test_ema_boundary(arena):
    low = await arena.add(score=0.8999)
    edge = await arena.add(score=0.9)
    assert eligible_uids(arena) == [edge.uid]
    assert (await read_status(arena))["models"]["s1-fast"]["excluded"] == {"ema": 1}
    assert low.uid not in eligible_uids(arena)


async def test_last_round_must_pass(arena):
    grey = await arena.add(score=0.95, last_score=0.0)  # grey zone, failure or timeout in the latest round
    just_below = await arena.add(score=0.95, last_score=0.79)
    edge = await arena.add(score=0.95, last_score=0.8)  # 1 - latency_weight: a correct but slow answer
    assert eligible_uids(arena) == [edge.uid]
    assert (await read_status(arena))["models"]["s1-fast"]["excluded"] == {"last_round": 2}
    assert grey.uid not in eligible_uids(arena) and just_below.uid not in eligible_uids(arena)


async def test_strikes_exclude_without_timestamp_and_by_age_with_one(arena):
    clean = await arena.add()
    struck = await arena.add(strikes=1)
    assert eligible_uids(arena) == [clean.uid]  # a bare count has no date: conservative, out
    assert (await read_status(arena))["models"]["s1-fast"]["excluded"] == {"strike": 1}
    edit_score(arena, struck.uid, last_strike_at=time.time() - 6 * 86400)
    await arena.router.refresh()
    assert eligible_uids(arena) == [clean.uid]
    assert (await read_status(arena))["models"]["s1-fast"]["excluded"] == {"recent_strike": 1}
    edit_score(arena, struck.uid, last_strike_at=time.time() - 8 * 86400)
    await arena.router.refresh()
    assert sorted(eligible_uids(arena)) == sorted([clean.uid, struck.uid])


async def test_recycled_uid_and_role_reset_start_from_zero(arena):
    first = await arena.add()
    second = await arena.add()
    state_path = Path(arena.vcfg.state_path)
    raw = json.loads(state_path.read_text())
    # recycled UID: the validator already reset it for the new hotkey, history of the old owner is gone
    raw["miners"][str(first.uid)].update(hotkey="5NewOwnerHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", ema=0.0, rounds=0, last_score=0.0)
    state_path.write_text(json.dumps(raw))
    await arena.router.refresh()
    assert eligible_uids(arena) == [second.uid]
    # role reset (model or manifest changed): validator state keeps the hotkey but rounds restart
    state = ValidatorState(9, arena.router.hotkey)
    state.miners[second.uid] = MinerState(second.hotkey, ema=0.99, last_score=1.0, rounds=900, role=raw["miners"][str(second.uid)]["role"])
    state.set_role(second.uid, "s1-fast:otherman")
    save(state, str(state_path))
    await arena.router.refresh()
    assert eligible_uids(arena) == []
    status = (await read_status(arena))["models"]["s1-fast"]
    assert status["seen_miners"] == 2 and status["eligible_miners"] == 0


async def test_unreadable_score_fields_fail_closed(arena):
    n = await arena.add()
    for key, value in (("rounds", "1000"), ("rounds", True), ("last_score", None), ("strikes", "0"), ("last_strike_at", "never")):
        original = json.loads(Path(arena.vcfg.state_path).read_text())["miners"][str(n.uid)][key] if key != "last_strike_at" else None
        edit_score(arena, n.uid, **{key: value})
        await arena.router.refresh()
        assert eligible_uids(arena) == [], (key, value)
        edit_score(arena, n.uid, **{key: original})
        if original is None:
            edit_score(arena, n.uid, **{key: 0})
        await arena.router.refresh()
        assert eligible_uids(arena) == [n.uid]


# ---- probation, sampling floor, reference load guard ----

def force_draw(monkeypatch, value):
    """Pin the sampling draw (0..999999) without touching the other random calls of the router."""
    real = secrets.randbelow
    monkeypatch.setattr("reax_subnet.router.secrets.randbelow",
                        lambda n: value if n == 1_000_000 else real(n))


async def test_probation_verifies_first_answers_then_samples(arena):
    n = await arena.add()
    ref = configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=3)
    for _ in range(3):
        assert (await arena.post())[0] == 200 and served(arena) == ("miner", "1", str(n.uid))
    assert arena.router.probation[n.hotkey] == 3 and ref.requests == 3
    assert (await arena.post())[0] == 200 and served(arena) == ("miner", "0", str(n.uid))
    assert ref.requests == 3 and arena.router.counts["verify", "s1-fast", "skipped"] == 1
    window = (await read_status(arena))["models"]["s1-fast"]["last_24h"]
    assert (window["verified"], window["probation"], window["sampled"], window["skipped"]) == (3, 3, 0, 1)


async def test_probation_is_per_hotkey_and_status_counts_it(arena):
    old = await arena.add(score=0.95)
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=1)
    assert (await arena.post())[0] == 200 and served(arena)[1] == "1"
    new = await arena.add(score=0.92)
    model = (await read_status(arena))["models"]["s1-fast"]
    assert model["probation_miners"] == 1  # only the new hotkey
    arena.router.health[old.hotkey].open_until = time.time() + 60
    assert (await arena.post())[0] == 200 and served(arena) == ("miner", "1", str(new.uid))
    assert (await read_status(arena))["models"]["s1-fast"]["probation_miners"] == 0


async def test_one_grey_keeps_probation_and_a_strike_resets_it(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=2)
    for _ in range(2):
        await arena.post()
    configure(arena, ShiftedReference(0.34), verify_rate=1)
    await arena.post()
    assert served(arena)[0] == "reference" and arena.router.probation[n.hotkey] == 2
    assert arena.router.health[n.hotkey].open_until == 0


@pytest.mark.parametrize("delta", [1.0], ids=["strike"])
async def test_mismatch_resets_probation(arena, delta):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=2)
    for _ in range(2):
        await arena.post()
    assert arena.router.probation[n.hotkey] == 2
    configure(arena, ShiftedReference(delta), verify_rate=1)
    await arena.post()  # sampled, mismatching: reference answer served
    assert served(arena)[0] == "reference" and arena.router.probation[n.hotkey] == 0
    arena.router.health[n.hotkey].open_until = 0  # strike (24 h) or grey (10 min) quarantine: lift it to look at probation alone
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    await arena.post()
    assert served(arena) == ("miner", "1", str(n.uid)) and arena.router.probation[n.hotkey] == 1


async def test_shadow_strike_resets_probation(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(1.0), verify_rate=0, shadow_rate=1, probation_answers=1)
    arena.router.probation[n.hotkey] = 1
    await arena.post()
    await asyncio.gather(*arena.router.tasks)
    assert arena.router.probation[n.hotkey] == 0


async def test_probation_fails_closed_without_reference_slot(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=5, reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1
    status, _ = await arena.post()
    assert status == 503 and not arena.seen
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    assert arena.router.probation.get(n.hotkey, 0) == 0
    assert arena.router.counts["verify", "s1-fast", "skipped_reference_busy"] == 0


async def test_probation_late_start_when_first_candidate_is_busy(arena):
    veteran = await arena.add(score=0.95)
    rookie = await arena.add(score=0.92)
    arena.router.probation[veteran.hotkey] = 99
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=5)
    arena.router.health[veteran.hotkey].inflight = 1  # skipped: the rookie answers, so its answer is verified
    assert (await arena.post())[0] == 200 and served(arena) == ("miner", "1", str(rookie.uid))


async def test_soft_skip_without_any_backend_still_fails_closed(arena, monkeypatch):
    await arena.add()
    configure(arena, verify_rate=1.0, verify_floor=0.1, probation_answers=0)
    force_draw(monkeypatch, 500_000)
    assert arena.router.backends == {}
    assert (await arena.post())[0] == 503  # a missing reference is a misconfiguration, not load


async def test_soft_skip_with_probation_miner_fails_closed(arena, monkeypatch):
    await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1.0, verify_floor=0.1, probation_answers=5,
              reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1
    force_draw(monkeypatch, 500_000)
    assert (await arena.post())[0] == 503 and not arena.seen
    assert arena.router.counts["verify", "s1-fast", "skipped_reference_busy"] == 0


async def test_sampling_does_not_depend_on_content(arena, monkeypatch):
    await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0.5, verify_floor=0.1, probation_answers=0)
    for draw, verified in ((499_999, "1"), (500_000, "0")):
        force_draw(monkeypatch, draw)
        for text in ("a", "b" * 500):
            await arena.post({"state": text})
            assert served(arena)[1] == verified


def test_router_config_probation_floor_and_model_pins(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok"}
    cfg = RouterConfig.from_env(base)
    assert (cfg.probation_answers, cfg.verify_floor, cfg.floor) == (20, 1.0, 1.0)  # floor defaults to the rate
    assert (cfg.revision_for("s1-fast"), cfg.tokenizer_version_for("s1-fast")) == ("rev", "tok")
    cfg = RouterConfig.from_env({**base, "REAX_ROUTER_VERIFY_RATE": "0.3"})
    assert (cfg.verify_floor, cfg.floor) == (0.3, 0.3)
    cfg = RouterConfig.from_env({**base, "REAX_ROUTER_PROBATION_ANSWERS": "0", "REAX_ROUTER_VERIFY_RATE": "0.3",
                                 "REAX_ROUTER_VERIFY_FLOOR": "0.5"})
    assert (cfg.probation_answers, cfg.verify_floor, cfg.floor) == (0, 0.5, 0.3)  # never above the rate
    for var, value in (("REAX_ROUTER_PROBATION_ANSWERS", "-1"), ("REAX_ROUTER_PROBATION_ANSWERS", "x"),
                       ("REAX_ROUTER_VERIFY_FLOOR", "1.1"), ("REAX_ROUTER_VERIFY_FLOOR", "-0.1")):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, var: value})
    both = {**base, "REAX_ROUTER_MODELS": "s1-fast,s1-pro",
            "REAX_ROUTER_MODEL_REVISION_S1_PRO": "pro-rev", "REAX_ROUTER_TOKENIZER_VERSION_S1_PRO": "pro-tok"}
    cfg = RouterConfig.from_env(both)
    assert (cfg.revision_for("s1-fast"), cfg.tokenizer_version_for("s1-fast")) == ("rev", "tok")
    assert (cfg.revision_for("s1-pro"), cfg.tokenizer_version_for("s1-pro")) == ("pro-rev", "pro-tok")
    cfg = RouterConfig.from_env({**both, "REAX_ROUTER_MODEL_REVISION_S1_FAST": "fast-rev"})
    assert cfg.revision_for("s1-fast") == "fast-rev" and cfg.tokenizer_version_for("s1-fast") == "tok"
    for missing in ("REAX_ROUTER_MODEL_REVISION_S1_PRO", "REAX_ROUTER_TOKENIZER_VERSION_S1_PRO"):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({k: v for k, v in both.items() if k != missing})
    # s1-pro alone does not need the s1-fast pins
    only_pro = {k: v for k, v in both.items() if k not in ("REAX_ROUTER_MODEL_REVISION", "REAX_ROUTER_TOKENIZER_VERSION")}
    assert RouterConfig.from_env({**only_pro, "REAX_ROUTER_MODELS": "s1-pro"}).revision_for("s1-pro") == "pro-rev"


async def test_two_models_carry_their_own_revision_and_tokenizer_version(arena):
    router = arena.router
    cfg = replace(router.cfg, models=("s1-fast", "s1-pro"), model_revisions={"s1-pro": "pro-rev"},
                  tokenizer_versions={"s1-pro": "pro-tok"}, verify_rate=1.0, probation_answers=0)
    both = Router(cfg, arena.vcfg, router.wallet, arena.chain, lambda model, decisions: 7 * len(decisions),
                  {"s1-fast": S1HashBackend(), "s1-pro": S1HashBackend()})
    runner, port = await serve(both.app())
    token = {"X-S1M-Internal-Token": "x" * 32}
    try:
        async def post(model):
            body = {"request_id": "r", "model": model, "state": "s", "deadline_unix_ms": int(time.time() * 1000) + 5000,
                    "questions": {"q": {"type": "noul", "instructions": "i"}}}
            async with arena.session.post(f"http://127.0.0.1:{port}/internal/v1/decide", json=body, headers=token) as r:
                return r.status, await r.json()
        status, fast = await post("s1-fast")
        assert status == 200 and (fast["model_revision"], fast["tokenizer_version"]) == ("plumb-pin", "tokenizer-pin")
        status, pro = await post("s1-pro")
        assert status == 200 and (pro["model_revision"], pro["tokenizer_version"]) == ("pro-rev", "pro-tok")
        async with arena.session.get(f"http://127.0.0.1:{port}/health/ready") as r:
            states = (await r.json())["models"]
        assert states["s1-fast"]["revision"] == "plumb-pin" and states["s1-pro"]["revision"] == "pro-rev"
        async with arena.session.get(f"http://127.0.0.1:{port}/status", headers=token) as r:
            status = await r.json()
        assert status["models"]["s1-pro"]["last_24h"]["reference"] == 1
        assert status["models"]["s1-fast"]["last_24h"]["reference"] == 1
    finally:
        await runner.cleanup()


async def test_status_counts_per_uid(arena):
    first = await arena.add(score=0.95)
    second = await arena.add(score=0.92)
    configure(arena, ShiftedReference(0.0), verify_rate=0)
    for _ in range(3):
        await arena.post()
    arena.router.health[first.hotkey].open_until = time.time() + 60
    await arena.post()
    status = await read_status(arena)
    assert status["models"]["s1-fast"]["served_by_uid_24h"] == {str(first.uid): 3, str(second.uid): 1}
    assert status["miner_uids_served_24h"] == {"s1-fast": [first.uid, second.uid]}  # old field unchanged
    assert status["verify_floor"] == 0 and status["probation_answers"] == 0


async def test_metrics_report_probation_and_skips(arena):
    await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=2)
    await arena.post()
    async with arena.session.get(arena.url + "/metrics", headers={"X-S1M-Internal-Token": "x" * 32}) as r:
        text = await r.text()
    assert 'reax_router_probation_miners{model="s1-fast"} 1' in text
    assert 'reax_router_verify_total{model="s1-fast",outcome="match"} 1' in text


# ---- s1-pro token counter ----

class RuneTokenizer:
    """Prompt = the message contents joined by newlines, one token per byte; option codes are single tokens."""
    def __init__(self):
        self.messages, self.codes = [], {}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}
        self.messages.append(messages)
        return "\n".join(m["content"] for m in messages)

    def __call__(self, prompt, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(prompt.encode())}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if len(text) > 2:
            return list(text.encode())
        return [self.codes.setdefault(text, 1000 + len(self.codes))]

    def decode(self, ids):
        return next(code for code, i in self.codes.items() if i == ids[0])


def test_rune_counter_mirrors_the_adapter_prompt():
    from reax_subnet.router_tokens import RUNE_EXTENDED_SYSTEM, RUNE_SYSTEM, RuneTokenCounter
    ds = [parse_decision({"id": "d", "state": {"k": "v"}, "questions": {"q": {
              "type": "choice", "instructions": "pick", "criteria": {"a": "first", "b": {"x": 1}}}}}, "s1-pro"),
          parse_decision({"id": "e", "state": "plain", "questions": {"q": {"type": "noul", "instructions": "yes?"}}},
                         "s1-pro")]
    tok = RuneTokenizer()
    counter = RuneTokenCounter(tok)
    total = counter("s1-pro", ds)
    first, second = tok.messages
    assert first[0] == {"role": "system", "content": RUNE_SYSTEM}
    assert first[1]["content"] == ('SHARED STATE (JSON string):\n{"k": "v"}\n\nQUESTION:\npick\nOPTIONS:\n'
                                   'A: first\nB: {"x": 1}\nAnswer with one option letter only.')
    assert second[1]["content"] == ('SHARED STATE (JSON string):\n"plain"\n\nQUESTION:\nyes?\nOPTIONS:\n'
                                    'A: No\nB: Yes\nAnswer with one option letter only.')  # keys in the adapter's order
    assert total == sum(len(m[0]["content"] + "\n" + m[1]["content"]) for m in tok.messages)
    assert counter.models == ("s1-pro",)
    with pytest.raises(RuntimeError):
        counter("s1-fast", ds)
    many = parse_decision({"id": "m", "state": "s", "questions": {"q": {
        "type": "choice", "instructions": "i", "criteria": {f"k{i}": f"v{i}" for i in range(27)}}}}, "s1-pro")
    tok.messages.clear()
    counter("s1-pro", [many])
    assert tok.messages[0][0]["content"] == RUNE_EXTENDED_SYSTEM
    assert tok.messages[0][1]["content"].endswith("\nZ: v25\nAA: v26\nAnswer with one option code only.")


def test_rune_counter_enforces_the_manifest_limit_and_dispatch():
    from reax_subnet.router_tokens import InputTokenLimitError, MultiTokenCounter, RuneTokenCounter
    d = parse_decision({"id": "d", "state": "s" * 5000, "questions": {"q": {"type": "noul", "instructions": "i"}}}, "s1-pro")
    assert load_manifest("s1-pro").data["limits"]["max_input_tokens"] == 4096
    with pytest.raises(InputTokenLimitError):
        RuneTokenCounter(RuneTokenizer())("s1-pro", [d])
    plumb = PlumbTokenCounter(RecordingTokenizer())
    multi = MultiTokenCounter({"s1-fast": plumb, "s1-pro": RuneTokenCounter(RuneTokenizer())})
    assert multi.models == ("s1-fast", "s1-pro")
    fast = parse_decision({"id": "f", "state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}, "s1-fast")
    assert multi("s1-fast", [fast]) == plumb("s1-fast", [fast])
    only_fast = MultiTokenCounter({"s1-fast": plumb})
    assert only_fast.models == ("s1-fast",)
    with pytest.raises(RuntimeError):
        only_fast("s1-pro", [d])


def test_rune_counter_refuses_unpinned_tokenizer_files(tmp_path):
    from reax_subnet.router_tokens import RUNE_PINNED_FILES, RuneTokenCounter
    for name in RUNE_PINNED_FILES:
        (tmp_path / name).write_text("not the pinned file")
    with pytest.raises(ValueError, match="pinned Rune manifest"):
        RuneTokenCounter.from_directory(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(OSError):
        RuneTokenCounter.from_directory(tmp_path)
    assert load_manifest("s1-pro").data["weights"]["files"].keys() >= set(RUNE_PINNED_FILES)


# ---- fix round 1: identical defaults, decision-count rule, credit budget, F6 per-miner decision, grey, backoff ----

def give_credit(arena, hotkey, served, verified):
    from reax_subnet.router import Credit
    credit = arena.router.credits.setdefault(hotkey, Credit())
    for i in range(served):
        credit.add(time.time(), i < verified)


@pytest.mark.parametrize("draw", [0, 99_999, 500_000, 999_999])
async def test_default_floor_keeps_rate_one_exactly_fail_closed(arena, monkeypatch, draw):
    """Only VERIFY_RATE=1.0 configured (floor unset): same behaviour as ef26a40, no best-effort share at any draw."""
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1.0, reference_concurrency=2)
    assert arena.router.cfg.verify_floor is None and arena.router.cfg.floor == 1.0
    force_draw(monkeypatch, draw)
    arena.router.reference_used["s1-fast"] = 2  # saturated
    status, _ = await arena.post()
    assert status == 503 and "X-Reax-Served-By" not in arena.last.headers and not arena.seen
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    assert arena.router.counts["verify", "s1-fast", "skipped"] == 0
    assert arena.router.counts["verify", "s1-fast", "skipped_reference_busy"] == 0
    assert arena.router.health[n.hotkey].open_until == 0
    arena.router.reference_used["s1-fast"] = 0
    assert (await arena.post())[0] == 200 and served(arena) == ("miner", "1", str(n.uid))


async def test_default_floor_with_probation_and_credit_changes_nothing_at_rate_one(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1.0, probation_answers=20)
    for draw in (0, 999_999):
        force_draw(monkeypatch, draw)
        assert (await arena.post())[0] == 200 and served(arena)[1] == "1"
    assert arena.router.counts["verify", "s1-fast", "skipped"] == 0
    assert arena.router.credits[n.hotkey].served == arena.router.credits[n.hotkey].verified == 2


async def test_requests_beyond_the_reference_slots_never_reach_a_miner(arena):
    await arena.add()
    ref = configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=0, reference_concurrency=2)
    qs = {str(i): it.question() for i, it in enumerate(generate_s1_items(3, seed=10))}
    status, _ = await arena.post({"questions": qs})
    assert status == 503 and not arena.seen and ref.requests == 0
    two = {k: qs[k] for k in ("0", "1")}
    assert (await arena.post({"questions": two}))[0] == 200 and len(arena.seen) == 1
    configure(arena, reference_concurrency=3)
    assert (await arena.post({"questions": qs}))[0] == 200 and len(arena.seen) == 2
    configure(arena, reference_concurrency=2, reference_concurrency_models={"s1-fast": 3})
    assert (await arena.post({"questions": qs}))[0] == 200  # the per-model value counts, not the global one


async def test_credit_budget_keeps_the_verified_share_of_served_answers(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0.3, probation_answers=0, per_miner_per_minute=19)
    force_draw(monkeypatch, 999_999)  # never sampled: only the credit budget can force a check
    flags = []
    for _ in range(18):
        assert (await arena.post())[0] == 200
        flags.append(served(arena)[1])
        credit = arena.router.credits[n.hotkey]
        assert credit.verified >= 0.3 * credit.served - 1e-9  # holds after every single answer
    assert flags[0] == "1" and 6 <= flags.count("1") <= 7
    # saturation cannot lower it: a forced check without a slot is a 503, never an unverified answer
    arena.router.reference_used["s1-fast"] = 2
    for _ in range(5):
        arena.router.health[n.hotkey].recent.clear()  # keep the per-minute budget out of the picture
        credit = arena.router.credits[n.hotkey]
        forced = credit.verified < 0.3 * (credit.served + 1)
        status, _ = await arena.post()
        assert status == (503 if forced else 200) and (forced or served(arena)[1] == "0")
    assert arena.router.credits[n.hotkey].verified >= 0.3 * arena.router.credits[n.hotkey].served - 1e-9


async def test_forced_check_without_slot_is_503_and_unforced_is_served(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0.3, probation_answers=0, reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1
    force_draw(monkeypatch, 999_999)
    assert (await arena.post())[0] == 503 and not arena.seen  # served 0, verified 0: forced, no slot
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    give_credit(arena, n.hotkey, 3, 3)
    assert (await arena.post())[0] == 200 and served(arena)[1] == "0"  # 3 >= 0.3 * 4


def test_credit_window_rolls_over_one_hour():
    from reax_subnet.router import Credit
    credit = Credit()
    credit.add(1_800_000_000, True)
    credit.add(1_800_000_060, False)
    assert credit.totals(1_800_000_100) == (2, 1)
    assert credit.totals(1_800_000_000 + 3600) == (1, 0)  # the ratio window is one hour
    assert credit.totals(1_800_000_060 + 3600) == (0, 0) and not credit.buckets
    assert credit.run == 1


async def test_sampled_above_floor_leaves_one_slot_for_forced_checks(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=1.0, verify_floor=0.1, probation_answers=0,
              reference_concurrency=2)
    give_credit(arena, n.hotkey, 5, 5)  # budget allows unverified serves
    arena.router.reference_used["s1-fast"] = 1
    force_draw(monkeypatch, 500_000)  # sampled above the floor: only while a slot would remain free
    assert (await arena.post())[0] == 200 and served(arena) == ("miner", "0", str(n.uid))
    assert arena.router.counts["verify", "s1-fast", "skipped_reference_busy"] == 1
    window = (await read_status(arena))["models"]["s1-fast"]["last_24h"]
    assert (window["skipped"], window["skipped_reference_busy"], window["verified"]) == (1, 1, 0)
    force_draw(monkeypatch, 99_999)  # floor draw may take the last slot
    assert (await arena.post())[0] == 200 and served(arena)[1] == "1"
    arena.router.reference_used["s1-fast"] = 2
    status, _ = await arena.post()  # floor draw, nothing free: fail closed
    assert status == 503 and arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    arena.router.reference_used["s1-fast"] = 0
    force_draw(monkeypatch, 500_000)
    assert (await arena.post())[0] == 200 and served(arena)[1] == "1"


async def test_reference_slots_are_per_model(arena):
    router = arena.router
    cfg = replace(router.cfg, models=("s1-fast", "s1-pro"), reference_concurrency=2,
                  reference_concurrency_models={"s1-pro": 1}, verify_rate=1.0, probation_answers=0)
    both = Router(cfg, arena.vcfg, router.wallet, arena.chain, lambda model, decisions: 7 * len(decisions),
                  {"s1-fast": S1HashBackend(), "s1-pro": S1HashBackend()})
    assert (cfg.reference_slots("s1-fast"), cfg.reference_slots("s1-pro")) == (2, 1)
    req = lambda model, n: SimpleNamespace(model=model, decisions=(0,) * n)
    tasks = both.reserve_reference(both.backends["s1-pro"], req("s1-pro", 1))
    with pytest.raises(Exception):
        both.reserve_reference(both.backends["s1-pro"], req("s1-pro", 1))
    other = both.reserve_reference(both.backends["s1-fast"], req("s1-fast", 2))  # s1-pro load does not touch s1-fast
    assert both.reference_used == {"s1-pro": 1, "s1-fast": 2}
    await asyncio.gather(*tasks, *other, return_exceptions=True)
    await asyncio.sleep(0)
    assert both.reference_used == {"s1-pro": 0, "s1-fast": 0}


async def test_probation_miner_without_slot_is_skipped_for_the_next_candidate(arena):
    rookie = await arena.add(score=0.95)
    veteran = await arena.add(score=0.92)
    arena.router.probation[veteran.hotkey] = 99
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=5, reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1
    status, _ = await arena.post()
    assert status == 200 and served(arena) == ("miner", "0", str(veteran.uid))
    assert len(arena.seen) == 1  # the rookie was never asked
    assert arena.router.counts["attempt", "s1-fast", "no_reference"] == 1
    assert arena.router.probation.get(rookie.hotkey, 0) == 0


async def test_every_candidate_forced_and_no_slot_is_503(arena):
    await arena.add(score=0.95)
    await arena.add(score=0.92)
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=5, reference_concurrency=1)
    arena.router.reference_used["s1-fast"] = 1
    assert (await arena.post())[0] == 503 and not arena.seen
    assert arena.router.counts["attempt", "s1-fast", "no_reference"] == 2
    assert arena.router.counts["verify", "s1-fast", "reference_unavailable"] == 1
    arena.router.reference_used["s1-fast"] = 0
    assert (await arena.post())[0] == 200 and served(arena)[1] == "1"


async def test_probation_miner_behind_a_veteran_is_not_a_reason_to_reserve_a_slot(arena):
    await arena.add(score=0.95)
    configure(arena, ShiftedReference(0.0), verify_rate=0, probation_answers=0)
    ref = arena.router.backends["s1-fast"]
    assert (await arena.post())[0] == 200 and ref.requests == 0


async def test_busy_miner_is_skipped_for_the_backoff_without_failure(arena):
    busy = await arena.add("busy", score=0.95)
    good = await arena.add(score=0.92)
    for _ in range(3):
        assert (await arena.post())[0] == 200 and served(arena)[2] == str(good.uid)
    assert len(arena.seen) == 4  # busy asked once, then backed off; good answered each time
    h = arena.router.health[busy.hotkey]
    assert h.failures == 0 and h.open_until == 0 and 0 < h.busy_until - time.time() <= 5
    h.busy_until = 0
    await arena.post()
    assert len(arena.seen) == 6
    configure(arena, busy_backoff_s=0)
    h.busy_until = 0
    await arena.post()
    await arena.post()
    assert len(arena.seen) == 10


def test_router_config_grey_busy_and_per_model_slots(tmp_path):
    path = tmp_path / "token"
    path.write_text("x" * 32)
    path.chmod(0o600)
    base = {"REAX_ROUTER_TOKEN_FILE": str(path), "REAX_ROUTER_MODEL_REVISION": "rev",
            "REAX_ROUTER_TOKENIZER_VERSION": "tok", "REAX_ROUTER_MODELS": "s1-fast,s1-pro",
            "REAX_ROUTER_MODEL_REVISION_S1_PRO": "p", "REAX_ROUTER_TOKENIZER_VERSION_S1_PRO": "t"}
    cfg = RouterConfig.from_env(base)
    assert (cfg.grey_quarantine_s, cfg.busy_backoff_s, cfg.grey_rate_max) == (3600, 5, 0.10)
    assert cfg.max_unverified_run is None and cfg.unverified_run_cap == 1  # floor = rate = 1.0
    assert RouterConfig.from_env({**base, "REAX_ROUTER_VERIFY_RATE": "0.3"}).unverified_run_cap == 4  # ceil(1 / 0.3)
    assert RouterConfig.from_env({**base, "REAX_ROUTER_VERIFY_RATE": "0.25"}).unverified_run_cap == 4
    assert (cfg.reference_slots("s1-fast"), cfg.reference_slots("s1-pro")) == (2, 2)
    cfg = RouterConfig.from_env({**base, "REAX_ROUTER_REFERENCE_CONCURRENCY": "4",
                                 "REAX_ROUTER_REFERENCE_CONCURRENCY_S1_PRO": "1", "REAX_ROUTER_GREY_QUARANTINE_S": "0",
                                 "REAX_ROUTER_BUSY_BACKOFF_S": "9", "REAX_ROUTER_GREY_RATE_MAX": "0.2",
                                 "REAX_ROUTER_MAX_UNVERIFIED_RUN": "7", "REAX_ROUTER_VERIFY_RATE": "0.3"})
    assert (cfg.reference_slots("s1-fast"), cfg.reference_slots("s1-pro")) == (4, 1)
    assert (cfg.grey_quarantine_s, cfg.busy_backoff_s, cfg.grey_rate_max, cfg.unverified_run_cap) == (0, 9, 0.2, 7)
    for var, value in (("REAX_ROUTER_REFERENCE_CONCURRENCY_S1_PRO", "0"), ("REAX_ROUTER_REFERENCE_CONCURRENCY_S1_PRO", "17"),
                       ("REAX_ROUTER_BUSY_BACKOFF_S", "-1"), ("REAX_ROUTER_GREY_QUARANTINE_S", "86401"),
                       ("REAX_ROUTER_GREY_RATE_MAX", "1.5"), ("REAX_ROUTER_MAX_UNVERIFIED_RUN", "0")):
        with pytest.raises(ConfigError):
            RouterConfig.from_env({**base, var: value})


def test_rune_counter_loads_only_the_hash_checked_files(tmp_path, monkeypatch):
    import hashlib
    import sys
    from reax_subnet import router_tokens
    from reax_subnet.manifest import Manifest
    src, loaded = tmp_path / "src", {}
    src.mkdir()
    for name in router_tokens.RUNE_PINNED_FILES:
        (src / name).write_text(f"pinned {name}")
    (src / "special_tokens_map.json").write_text("{}")  # unpinned: must never be read by the loader
    (src / "added_tokens.json").write_text("{}")
    files = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in router_tokens.RUNE_PINNED_FILES}
    manifest = Manifest("s1-pro", "x", {"weights": {"files": files}, "limits": {"max_input_tokens": 4096}})
    monkeypatch.setattr(router_tokens, "load_manifest", lambda model: manifest)

    class FakeAuto:
        @staticmethod
        def from_pretrained(directory, **kwargs):
            loaded["files"] = sorted(p.name for p in Path(directory).iterdir())
            loaded["kwargs"] = kwargs
            return RuneTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAuto))
    counter = router_tokens.RuneTokenCounter.from_directory(src)
    assert loaded["files"] == sorted(router_tokens.RUNE_PINNED_FILES)
    assert loaded["kwargs"] == {"local_files_only": True, "trust_remote_code": False}
    assert counter.models == ("s1-pro",)
    (src / "config.json").write_text("changed")  # every file the loader would read is checked
    with pytest.raises(ValueError, match="config.json"):
        router_tokens.RuneTokenCounter.from_directory(src)


# ---- fix round 1 amendments: unverified-run cap, grey statistics ----

async def test_clean_history_then_saturated_reference_bounds_the_unverified_burst(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0.3, probation_answers=0, per_miner_per_minute=19)
    assert arena.router.cfg.unverified_run_cap == 4
    give_credit(arena, n.hotkey, 100, 100)  # a long clean history: the ratio alone would allow ~230 unverified serves
    arena.router.reference_used["s1-fast"] = 2  # the reference is saturated from now on
    force_draw(monkeypatch, 999_999)  # worst case: no check is ever drawn
    statuses = []
    for _ in range(12):
        arena.router.health[n.hotkey].recent.clear()
        statuses.append((await arena.post())[0])
    assert statuses[:4] == [200] * 4 and statuses[4:] == [503] * 8  # at most ceil(1/floor) bad answers, then closed
    assert arena.router.credits[n.hotkey].run == 4
    arena.router.reference_used["s1-fast"] = 0  # once a slot is free the next answer is verified and resets the run
    arena.router.health[n.hotkey].recent.clear()
    assert (await arena.post())[0] == 200 and served(arena)[1] == "1" and arena.router.credits[n.hotkey].run == 0


async def test_unverified_run_cap_is_configurable(arena, monkeypatch):
    n = await arena.add()
    configure(arena, ShiftedReference(0.0), verify_rate=0.3, probation_answers=0, max_unverified_run=2,
              reference_concurrency=1)
    give_credit(arena, n.hotkey, 100, 100)
    arena.router.reference_used["s1-fast"] = 1
    force_draw(monkeypatch, 999_999)
    assert [(await arena.post())[0] for _ in range(4)] == [200, 200, 503, 503]


def test_wilson_lower_bound():
    from reax_subnet.router import wilson_lower
    assert wilson_lower(0, 0) == 0 and wilson_lower(0, 50) == 0
    assert wilson_lower(2, 20) < 0.10 < wilson_lower(6, 20)  # 2 of 20 is honest variance, 6 of 20 is not
    assert wilson_lower(5, 100) < 0.10 and wilson_lower(20, 100) > 0.10


async def drive_checks(arena, hotkey, greys, checks):
    """Feed verification results for one hotkey without going through the network."""
    n = next(n for n in arena.chain.neurons if n.hotkey == hotkey)
    req = SimpleNamespace(request_id="r", model="s1-fast")
    for i in range(checks):
        await arena.router.check_result(n, req, "verify_grey", i < greys)


async def test_grey_rate_below_the_bound_never_quarantines(arena):
    n = await arena.add()
    arena.router.probation[n.hotkey] = 99
    await drive_checks(arena, n.hotkey, greys=2, checks=20)  # 10 % raw rate, Wilson lower bound ~0.03
    h = arena.router.health[n.hotkey]
    assert h.open_until == 0 and arena.router.probation[n.hotkey] == 99
    assert arena.router.greys[n.hotkey].totals(time.time()) == (20, 2)


async def test_significant_grey_rate_quarantines_then_requires_probation(arena):
    n = await arena.add()
    configure(arena, probation_answers=5)
    arena.router.probation[n.hotkey] = 99
    await drive_checks(arena, n.hotkey, greys=5, checks=19)  # not enough checks yet
    assert arena.router.health[n.hotkey].open_until == 0
    await drive_checks(arena, n.hotkey, greys=1, checks=1)  # 20th check: 6 of 20 greys... first of the batch is grey
    h = arena.router.health[n.hotkey]
    assert 3590 < h.open_until - time.time() <= 3600
    assert arena.router.probation[n.hotkey] == 0 and arena.router.on_probation(n.hotkey)
    assert arena.router.greys[n.hotkey].totals(time.time()) == (0, 0)  # requalification starts clean
    assert arena.router.counts["quarantine", "s1-fast", "grey_rate"] == 1
    assert not Path(arena.router.cfg.strikes_file).exists()  # a quarantine is not a strike
    assert n.hotkey not in [x.hotkey for x, _ in arena.router.candidates("s1-fast")]
    h.open_until = 0
    assert [x.hotkey for x, _ in arena.router.candidates("s1-fast")] == [n.hotkey]


async def test_greys_in_real_requests_reach_the_quarantine(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.34), verify_rate=1.0, probation_answers=0, per_miner_per_minute=19)
    arena.router.greys[n.hotkey] = Rolling_for_test(checks=19, greys=5)
    await arena.post()  # the 20th check is grey: 6 of 20
    assert served(arena)[0] == "reference" and arena.router.health[n.hotkey].open_until > time.time() + 3000


def Rolling_for_test(checks, greys):
    from reax_subnet.router import Rolling
    r = Rolling(1440)
    r.add(time.time(), checks, greys)
    return r


async def test_shadow_grey_counts_into_the_same_statistics(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(0.34), verify_rate=0, shadow_rate=1, probation_answers=1)
    arena.router.probation[n.hotkey] = 1
    await arena.post()
    await asyncio.gather(*arena.router.tasks)
    assert arena.router.greys[n.hotkey].totals(time.time()) == (1, 1)
    assert arena.router.probation[n.hotkey] == 1 and arena.router.health[n.hotkey].open_until == 0
    configure(arena, ShiftedReference(0.0))
    await arena.post()
    await asyncio.gather(*arena.router.tasks)
    assert arena.router.greys[n.hotkey].totals(time.time()) == (2, 1)


async def test_old_checks_leave_the_grey_window(arena, monkeypatch):
    n = await arena.add()
    clock = [1_800_000_000.0]
    monkeypatch.setattr("reax_subnet.router.time.time", lambda: clock[0])
    await drive_checks(arena, n.hotkey, greys=3, checks=10)
    clock[0] += 86400
    assert arena.router.greys[n.hotkey].totals(clock[0]) == (0, 0)


async def test_strike_still_excludes_and_resets_probation(arena):
    n = await arena.add()
    configure(arena, ShiftedReference(1.0), verify_rate=1.0, probation_answers=2)
    arena.router.probation[n.hotkey] = 2
    await arena.post()
    assert arena.router.probation[n.hotkey] == 0 and arena.router.health[n.hotkey].open_until > time.time() + 86000
    assert json.loads(Path(arena.router.cfg.strikes_file).read_text())["outcome"] == "verify_strike"
