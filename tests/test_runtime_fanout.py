"""Several runtime sidecars behind one hotkey: dispatch, failover, health, identity, limits, config."""
import asyncio
import json
import math

import pytest
from aiohttp import web

from reax_subnet import selfcheck
from reax_subnet.config import ConfigError, load_common, load_miner
from reax_subnet.s1backends import (PROBE_DECISION, PooledBackend, SidecarBackend, SidecarError,
                                    load_s1_backend)
from reax_subnet.s1contract import parse_decision

TOKEN = "t" * 32
ANSWER = {"type": "noul", "probabilities": {"true": 0.75, "false": 0.25}, "confidence": 0.75, "noul": 0.75}


class FakeRuntime:
    """A runtime sidecar on a random loopback port with switchable misbehaviour."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.mode = "ok"  # ok | drop (connection closed before any byte) | http500 | partial (200, body cut short)
        self.healthy = True
        self.decides = 0
        self.peak = 0
        self._active = 0
        self.runner = None
        self.url = ""
        self.hotkey_headers: list[str] = []

    async def start(self) -> "FakeRuntime":
        async def health(request):
            return web.json_response({"ok": True}, status=200 if self.healthy else 503)

        async def decide(request):
            self.decides += 1
            self.hotkey_headers += [h for h in request.headers if "hotkey" in h.lower() or "signature" in h.lower()]
            if request.headers.get("X-S1M-Internal-Token") != TOKEN:
                return web.json_response({"error": "unauthorized"}, status=401)
            self._active += 1
            self.peak = max(self.peak, self._active)
            try:
                await request.read()
                if self.delay:
                    await asyncio.sleep(self.delay)
                if self.mode == "drop":
                    request.transport.close()
                    return web.Response(status=500)
                if self.mode == "http500":
                    return web.json_response({"error": "boom"}, status=500)
                if self.mode == "partial":
                    resp = web.StreamResponse(headers={"Content-Length": "500"})
                    await resp.prepare(request)
                    await resp.write(b'{"answers": ')
                    request.transport.close()
                    return resp
                return web.json_response({"answers": {"q": ANSWER}})
            finally:
                self._active -= 1

        app = web.Application()
        app.router.add_post("/internal/v1/decide", decide)
        app.router.add_get("/health", health)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def stop(self):
        await self.runner.cleanup()


def decision(i: int = 0):
    return parse_decision({"id": f"d{i}", "state": f"Probe {i}.",
                           "questions": {"q": {"type": "noul", "instructions": "Late?"}}}, "s1-fast")


def pool_of(*runtimes, max_inflight=16, **kw) -> PooledBackend:
    return PooledBackend([SidecarBackend("s1-fast", r.url, TOKEN, max_inflight=max_inflight, timeout_s=5)
                          for r in runtimes], **kw)


@pytest.fixture
async def runtimes():
    made: list[FakeRuntime] = []

    async def make(n, delay=0.0):
        for _ in range(n):
            made.append(await FakeRuntime(delay).start())
        return made[-n:]

    yield make
    for r in made:
        await r.stop()


# --- dispatch -----------------------------------------------------------------------------------------------------

async def test_dispatch_spreads_load_evenly(runtimes):
    a, b, c = await runtimes(3, delay=0.05)
    pool = pool_of(a, b, c)
    try:
        answers = await pool.decide([decision(i) for i in range(30)])
        assert len(answers) == 30
        assert [r.decides for r in (a, b, c)] == [10, 10, 10]
        assert all(r.peak == 10 for r in (a, b, c))  # all arrived together: fewest-in-flight spreads them evenly
    finally:
        await pool.close()


async def test_dispatch_prefers_the_backend_with_fewest_in_flight(runtimes):
    a, b = await runtimes(2, delay=0.3)
    pool = pool_of(a, b)
    try:
        first = asyncio.ensure_future(pool.decide([decision(0)]))  # occupies one of them
        await asyncio.sleep(0.1)
        assert sorted(x.inflight for x in pool.backends) == [0, 1]
        await pool.decide([decision(1)])
        await first
        assert (a.decides, b.decides) == (1, 1)  # the second request did not queue behind the first
    finally:
        await pool.close()


async def test_max_inflight_is_respected_per_backend(runtimes):
    a, b = await runtimes(2, delay=0.05)
    pool = pool_of(a, b, max_inflight=3)
    try:
        await pool.decide([decision(i) for i in range(20)])
        assert a.peak <= 3 and b.peak <= 3
        assert a.decides + b.decides == 20
    finally:
        await pool.close()


async def test_single_url_is_a_plain_sidecar_backend(runtimes):
    (a,) = await runtimes(1)
    backend = load_s1_backend("s1-fast", a.url, TOKEN, False)
    assert isinstance(backend, SidecarBackend)
    one = load_s1_backend("s1-fast", None, "", False, endpoints=((a.url, TOKEN),))
    assert isinstance(one, SidecarBackend)
    many = load_s1_backend("s1-fast", None, TOKEN, False, endpoints=((a.url, ""), (a.url + "/", "")))
    assert isinstance(many, PooledBackend) and many.size == 2
    try:
        assert (await backend.decide([decision()]))[0]["probabilities"]["true"] == pytest.approx(0.75)
    finally:
        await backend.close()
        await one.close()
        await many.close()


async def test_backends_never_see_the_hotkey(runtimes):
    a, b = await runtimes(2)
    pool = pool_of(a, b)
    try:
        await pool.decide([decision(i) for i in range(4)])
    finally:
        await pool.close()
    assert a.hotkey_headers == [] and b.hotkey_headers == []


# --- failover and health --------------------------------------------------------------------------------------------

async def test_failover_retries_once_on_another_backend(runtimes):
    good, bad = await runtimes(2)
    bad.mode = "drop"
    pool = pool_of(good, bad)
    try:
        answers = await pool.decide([decision(i) for i in range(6)])
        assert len(answers) == 6 and all(a["probabilities"]["true"] == pytest.approx(0.75) for a in answers)
        assert good.decides >= 6
    finally:
        await pool.close()


async def test_failing_backend_is_removed_and_gets_no_more_requests(runtimes):
    good, bad = await runtimes(2)
    bad.mode = "http500"
    pool = pool_of(good, bad)
    try:
        await pool.decide([decision(i) for i in range(10)])
        assert pool.healthy_count == 1
        seen = bad.decides
        await pool.decide([decision(i) for i in range(10)])
        assert bad.decides == seen  # no storm against the removed backend
    finally:
        await pool.close()


async def test_readmission_after_health_checks_pass(runtimes):
    good, flaky = await runtimes(2)
    flaky.mode = "http500"
    flaky.healthy = False
    pool = pool_of(good, flaky)
    try:
        await pool.decide([decision(i) for i in range(6)])
        assert pool.healthy_count == 1
        await pool.check_health_once()
        assert pool.healthy_count == 1  # still down
        flaky.mode, flaky.healthy = "ok", True
        await pool.check_health_once()
        assert pool.healthy_count == 1  # one good answer is not enough
        await pool.check_health_once()
        assert pool.healthy_count == 2
        before = flaky.decides
        await pool.decide([decision(i) for i in range(10)])
        assert flaky.decides > before
    finally:
        await pool.close()


async def test_health_check_removes_a_dead_backend(runtimes):
    a, b = await runtimes(2)
    pool = pool_of(a, b)
    try:
        b.healthy = False
        await pool.check_health_once()
        assert pool.healthy_count == 2  # one miss is tolerated
        await pool.check_health_once()
        assert pool.healthy_count == 1
        assert [d["healthy"] for d in pool.describe()] == [True, False]
    finally:
        await pool.close()


async def test_no_retry_once_answer_bytes_were_produced(runtimes):
    cut, good = await runtimes(2)
    cut.mode = "partial"
    pool = pool_of(cut, good)
    pool.backends[1].inflight = 5  # the cutting backend looks emptier, so it is picked first
    try:
        with pytest.raises(Exception) as err:
            await pool._one(decision())
        assert err.value.answer_started is True
        assert cut.decides == 1 and good.decides == 0  # the 200 had started: no second try elsewhere
    finally:
        await pool.close()


async def test_request_is_tried_at_most_twice(runtimes):
    a, b, c = await runtimes(3)
    for r in (a, b, c):
        r.mode = "drop"
    pool = pool_of(a, b, c)
    try:
        with pytest.raises(Exception):
            await pool._one(decision())
        total = a.decides + b.decides + c.decides
        assert 2 <= total <= 4  # two attempts, each with the sidecar's own single reconnect
    finally:
        await pool.close()


async def test_http_4xx_is_not_retried_and_not_a_backend_fault(runtimes):
    a, b = await runtimes(2)
    pool = PooledBackend([SidecarBackend("s1-fast", r.url, "wrong-token", timeout_s=5) for r in (a, b)])
    try:
        with pytest.raises(SidecarError):
            await pool._one(decision())
        assert a.decides + b.decides == 1
        assert pool.healthy_count == 2
    finally:
        await pool.close()


async def test_all_backends_down_fails_fast_then_recovers_through_a_trial(runtimes):
    a, b = await runtimes(2)
    a.mode = b.mode = "http500"
    pool = pool_of(a, b, retry_after_s=0.2)
    try:
        for _ in range(3):
            with pytest.raises(Exception):
                await pool._one(decision())
        assert pool.healthy_count == 0
        seen = a.decides + b.decides
        with pytest.raises(SidecarError, match="no healthy"):
            await pool._one(decision())
        assert a.decides + b.decides == seen  # nothing was sent while everything is out
        a.mode = b.mode = "ok"
        await asyncio.sleep(0.25)
        assert (await pool._one(decision()))["probabilities"]["true"] == pytest.approx(0.75)
        assert pool.healthy_count >= 1
    finally:
        await pool.close()


async def test_probe_needs_every_backend(runtimes):
    a, b = await runtimes(2)
    pool = pool_of(a, b)
    try:
        assert await pool.probe()
        b.mode = "http500"
        assert not await pool.probe()
    finally:
        await pool.close()


def test_pool_refuses_mixed_models():
    from reax_subnet.manifest import load_manifest

    pro = SidecarBackend("s1-pro", "http://127.0.0.1:1", "t")
    fast = SidecarBackend("s1-fast", "http://127.0.0.1:2", "t")
    assert load_manifest("s1-pro").manifest_id != load_manifest("s1-fast").manifest_id
    with pytest.raises(ValueError, match="same model"):
        PooledBackend([pro, fast])


# --- identity ------------------------------------------------------------------------------------------------------

async def _reference_runtime(ref, noise: float):
    def key(state, question):
        return json.dumps([state, question], sort_keys=True)

    by_key = {key(it["state"], it["question"]): it for it in ref["items"]}

    async def decide(request: web.Request) -> web.Response:
        body = await request.json()
        it = by_key[key(body["state"], body["questions"]["q"])]
        keys = list(parse_decision({"id": "x", "state": it["state"], "questions": {"q": it["question"]}},
                                   "s1-fast").question.keys)
        logits = [math.log(max(p, 1e-9)) + (noise if i == 0 else 0.0) for i, p in enumerate(it["p"])]
        z = math.fsum(math.exp(x) for x in logits)
        probs = {k: math.exp(x) / z for k, x in zip(keys, logits)}
        answer = {"noul": probs["true"]} if it["question"]["type"] == "noul" else {"probabilities": probs}
        return web.json_response({"answers": {"q": answer}})

    app = web.Application()
    app.router.add_post("/internal/v1/decide", decide)
    async def health(request):
        return web.json_response({"ok": True})

    app.router.add_get("/health", health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


async def test_identity_check_accepts_matching_and_refuses_mismatched_backend():
    ref = selfcheck.load_reference("s1-fast")
    r1, u1 = await _reference_runtime(ref, 0.0)
    r2, u2 = await _reference_runtime(ref, 0.0)
    r3, u3 = await _reference_runtime(ref, 0.8)  # other weights / settings
    try:
        good = PooledBackend([SidecarBackend("s1-fast", u, TOKEN) for u in (u1, u2)])
        assert await selfcheck.check_pool_identity(good, "s1-fast") == []
        mixed = PooledBackend([SidecarBackend("s1-fast", u, TOKEN) for u in (u1, u3)])
        problems = await selfcheck.check_pool_identity(mixed, "s1-fast")
        assert len(problems) == 1 and u3 in problems[0] and u1 not in problems[0]
        await good.close()
        await mixed.close()
    finally:
        for r in (r1, r2, r3):
            await r.cleanup()


async def test_readmission_needs_the_identity_check(caplog):
    ref = selfcheck.load_reference("s1-fast")
    r_good, u_good = await _reference_runtime(ref, 0.0)
    r_bad, u_bad = await _reference_runtime(ref, 0.8)  # healthy, but other weights / settings
    r_ok, u_ok = await _reference_runtime(ref, 0.0)
    pool = PooledBackend([SidecarBackend("s1-fast", u, TOKEN) for u in (u_good, u_bad)])
    pool.readmit_check = lambda b: selfcheck.check_backend_identity(b, "s1-fast")
    member = pool._members[1]
    member.healthy = False  # removed for faults earlier
    try:
        with caplog.at_level("WARNING", logger="reax.s1backends"):
            for _ in range(4):
                await pool.check_health_once()
        assert pool.healthy_count == 1 and not member.healthy
        assert "stays out of rotation" in caplog.text
        # with nothing else healthy a removed backend is not tried with validator traffic either
        pool._members[0].healthy = False
        assert pool._pick(set()) is None
        pool._members[0].healthy = True
        member.backend.url = u_ok  # comes back serving the reference answers
        await pool.check_health_once()
        assert pool.healthy_count == 1
        await pool.check_health_once()
        assert pool.healthy_count == 2 and member.healthy
    finally:
        await pool.close()
        for r in (r_good, r_bad, r_ok):
            await r.cleanup()


async def test_pool_identity_fails_closed_without_references():
    class _B:
        url = "http://127.0.0.1:1"

    class _P:
        backends = [_B()]

    problems = await selfcheck.check_pool_identity(_P(), "s1-lite")
    assert len(problems) == 1 and "not checkable" in problems[0]


async def test_selfcheck_reports_every_backend():
    ref = selfcheck.load_reference("s1-fast")
    r1, u1 = await _reference_runtime(ref, 0.0)
    r2, u2 = await _reference_runtime(ref, 0.6)
    try:
        report = await selfcheck.run_many("s1-fast", [(u1, TOKEN), (u2, TOKEN)], limit=120, concurrency=8)
    finally:
        await r1.cleanup()
        await r2.cleanup()
    assert [b["url"] for b in report["backends"]] == [u1, u2]
    assert [b["pass"] for b in report["backends"]] == [True, False]
    assert report["pass"] is False


async def test_selfcheck_reports_an_unreachable_backend():
    ref = selfcheck.load_reference("s1-fast")
    r1, u1 = await _reference_runtime(ref, 0.0)
    try:
        report = await selfcheck.run_many("s1-fast", [(u1, TOKEN), ("http://127.0.0.1:1", TOKEN)], 50, 4)
    finally:
        await r1.cleanup()
    assert report["backends"][0]["pass"] and not report["backends"][1]["pass"] and "error" in report["backends"][1]


# --- config --------------------------------------------------------------------------------------------------------

@pytest.fixture
def token(tmp_path):
    p = tmp_path / "token"
    p.write_text(TOKEN)
    return str(p)


def env_for(token, **extra):
    return {"REAX_NETUID": "7", "REAX_RUNTIME_TOKEN_FILE": token, **extra}


def test_single_url_var_is_unchanged(token):
    c = load_common(env_for(token, REAX_S1_FAST_RUNTIME_URL="http://127.0.0.1:8101"))
    assert c.runtime_url("s1-fast") == "http://127.0.0.1:8101"
    assert c.endpoints("s1-fast") == (("http://127.0.0.1:8101", ""),)
    with pytest.raises(ConfigError, match="loopback"):
        load_common(env_for(token, REAX_S1_FAST_RUNTIME_URL="http://10.0.0.2:8101"))
    m = load_miner(env_for(token, REAX_S1_FAST_RUNTIME_URL="http://127.0.0.1:8101"))
    import reax_subnet.config as C   # defaults follow tiers.CAPACITY_CAP_MULTIPLIER (x1: 8 / 16 / 240)
    assert (m.max_concurrent, m.per_caller_concurrency, m.per_caller_per_minute) == \
        (C.DEFAULT_MAX_CONCURRENT, C.DEFAULT_PER_CALLER_CONCURRENCY, C.DEFAULT_PER_CALLER_PER_MINUTE)


def test_url_list_accepts_private_networks(token, tmp_path):
    other = tmp_path / "other"
    other.write_text("o" * 32)
    urls = f"http://10.0.0.2:8000, http://192.168.1.5:8000|{other},http://172.20.0.3:8000,http://[fd00::1]:8000,http://127.0.0.1:8002"
    c = load_common(env_for(token, REAX_S1_PRO_RUNTIME_URLS=urls))
    eps = c.endpoints("s1-pro")
    assert [u for u, _ in eps] == ["http://10.0.0.2:8000", "http://192.168.1.5:8000", "http://172.20.0.3:8000",
                                   "http://[fd00::1]:8000", "http://127.0.0.1:8002"]
    assert eps[1][1] == str(other) and eps[0][1] == ""
    assert c.runtime_url("s1-pro") == "http://10.0.0.2:8000"


@pytest.mark.parametrize("bad", ["http://8.8.8.8:8000", "http://example.com:8000", "http://172.32.0.1:8000",
                                 "http://100.64.0.1:8000", "http://169.254.1.1:8000",
                                 "http://u:p@10.0.0.2:8000", "ftp://10.0.0.2:8000", "http://10.0.0.2:8000/x"])
def test_url_list_refuses_public_and_odd_urls(token, bad):
    with pytest.raises(ConfigError):
        load_common(env_for(token, REAX_S1_FAST_RUNTIME_URLS=f"http://10.0.0.2:8000,{bad}"))


def test_public_runtime_needs_the_explicit_switch_and_warns(token, caplog):
    env = env_for(token, REAX_S1_FAST_RUNTIME_URLS="http://10.0.0.2:8000,http://203.0.113.9:8000")
    with pytest.raises(ConfigError, match="REAX_ALLOW_PUBLIC_RUNTIME=1"):
        load_common(env)
    with caplog.at_level("WARNING", logger="reax.config"):
        c = load_common({**env, "REAX_ALLOW_PUBLIC_RUNTIME": "1"})
    assert len(c.endpoints("s1-fast")) == 2
    assert "IN CLEAR TEXT" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="reax.config"):
        load_common({**env_for(token, REAX_S1_FAST_RUNTIME_URLS="https://gpu.example.com:8000"),
                     "REAX_ALLOW_PUBLIC_RUNTIME": "1"})
    assert "over TLS" in caplog.text


def test_url_list_conflicts_and_token_rules(token, tmp_path):
    with pytest.raises(ConfigError, match="not both"):
        load_common(env_for(token, REAX_S1_FAST_RUNTIME_URLS="http://10.0.0.2:8000",
                            REAX_S1_FAST_RUNTIME_URL="http://127.0.0.1:8101"))
    with pytest.raises(ConfigError, match="twice"):
        load_common(env_for(token, REAX_S1_FAST_RUNTIME_URLS="http://10.0.0.2:8000,http://10.0.0.2:8000/"))
    with pytest.raises(ConfigError, match="token file"):  # no shared token and one URL without its own
        load_common({"REAX_NETUID": "7", "REAX_S1_FAST_RUNTIME_URLS": "http://10.0.0.2:8000"})
    own = tmp_path / "own"
    own.write_text("x" * 32)
    c = load_common({"REAX_NETUID": "7", "REAX_S1_FAST_RUNTIME_URLS": f"http://10.0.0.2:8000|{own}"})
    assert c.endpoints("s1-fast")[0][1] == str(own)
    with pytest.raises(ConfigError, match="not readable"):
        load_common(env_for(token, REAX_S1_FAST_RUNTIME_URLS=f"http://10.0.0.2:8000|{tmp_path}/missing"))


def test_limits_scale_with_the_number_of_backends(token):
    urls = ",".join(f"http://10.0.0.{i}:8000" for i in range(2, 5))
    m = load_miner(env_for(token, REAX_S1_FAST_RUNTIME_URLS=urls))
    import reax_subnet.config as C   # the larger of the probe-at-cap defaults and the per-backend defaults x N
    assert (m.max_concurrent, m.per_caller_concurrency, m.per_caller_per_minute) == \
        (max(C.DEFAULT_MAX_CONCURRENT, 24), max(C.DEFAULT_PER_CALLER_CONCURRENCY, 48), max(C.DEFAULT_PER_CALLER_PER_MINUTE, 720))
    many = ",".join(f"http://10.0.0.{i}:8000" for i in range(2, 12))
    m = load_miner(env_for(token, REAX_S1_FAST_RUNTIME_URLS=many))
    assert m.per_caller_concurrency == 128 and m.max_concurrent == 80  # capped at the setting's maximum
    m = load_miner(env_for(token, REAX_S1_FAST_RUNTIME_URLS=urls, REAX_PER_CALLER_CONCURRENCY="5"))
    assert m.per_caller_concurrency == 5 and m.max_concurrent == max(C.DEFAULT_MAX_CONCURRENT, 24)  # explicit settings win


class _StubPool:
    model = "s1-fast"

    def __init__(self, size, healthy):
        self.size, self.healthy_count = size, healthy


def test_miner_limits_follow_the_healthy_backends():
    from reax_subnet.miner import S1Miner

    class W:
        class hotkey:
            ss58_address = "5Fake"

    for size, healthy, expect in ((3, 3, (48, 720)), (3, 2, (32, 480)), (3, 1, (16, 240)), (3, 0, (16, 240))):
        miner = S1Miner(W, _StubPool(size, healthy), per_caller_concurrency=48, per_caller_per_minute=720)
        assert miner._caller_limits() == expect
    single = S1Miner(W, SidecarBackend("s1-fast", "http://127.0.0.1:1", "t"), per_caller_concurrency=16,
                     per_caller_per_minute=240)
    assert single._caller_limits() == (16, 240)


@pytest.mark.parametrize("model", ["s1-lite", "s1-cpu"])
def test_url_list_refused_without_identity_references(token, model):
    var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URLS"
    with pytest.raises(ConfigError, match="several backends need the published identity references"):
        load_common(env_for(token, REAX_MODEL=model, **{var: "http://10.0.0.2:8000,http://10.0.0.3:8000"}))
    # a single URL (as list or plain variable) keeps working for every model
    c = load_common(env_for(token, REAX_MODEL=model, **{var: "http://10.0.0.2:8000"}))
    assert [u for u, _t in c.endpoints(model)] == ["http://10.0.0.2:8000"]
    c = load_common(env_for(token, REAX_MODEL=model, **{var[:-1]: "http://127.0.0.1:8101"}))
    assert c.runtime_url(model) == "http://127.0.0.1:8101"


@pytest.mark.parametrize("model", ["s1-fast", "s1-pro"])
def test_url_list_accepted_for_models_with_references(token, model):
    var = f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URLS"
    c = load_common(env_for(token, REAX_MODEL=model, **{var: "http://10.0.0.2:8000,http://10.0.0.3:8000"}))
    assert len(c.endpoints(model)) == 2


def test_pool_models_match_the_published_references():
    from reax_subnet.config import POOL_MODELS, S1_MODELS

    for m in S1_MODELS:
        try:
            selfcheck.load_reference(m)
            has = True
        except ValueError:
            has = False
        assert has == (m in POOL_MODELS)
