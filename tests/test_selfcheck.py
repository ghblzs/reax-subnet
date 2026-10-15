"""reax-selfcheck: the published reference answers and the pass/fail verdict against a runtime sidecar."""

from __future__ import annotations

import json
import math

import pytest
from aiohttp import web

from reax_subnet import selfcheck
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1validator import POOL_PARAMS


def test_reference_matches_release_manifest_and_contract():
    ref = selfcheck.load_reference("s1-fast")
    assert len(ref["items"]) >= 500
    for it in ref["items"]:
        d = parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, "s1-fast")
        assert len(d.question.keys) == len(it["p"])
        assert abs(math.fsum(it["p"]) - 1.0) < 1e-3


def test_simulated_rounds_follow_release_constants():
    params = POOL_PARAMS["s1-fast"]
    assert selfcheck.simulate_rounds([0.05] * 100, params, rounds=500) == (1.0, 0.0)
    assert selfcheck.simulate_rounds([0.3] * 100, params, rounds=500) == (0.0, 0.0)  # round lost, no strike
    assert selfcheck.simulate_rounds([0.6] * 100, params, rounds=500) == (0.0, 1.0)
    assert selfcheck.simulate_rounds([0.05] * 99 + [5.0], params, rounds=2000)[1] > 0  # one item over tau_item


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _fake_runtime(ref, noise: float):
    def key(state, question):
        return json.dumps([state, question], sort_keys=True)

    by_key = {key(it["state"], it["question"]): it for it in ref["items"]}

    async def decide(request: web.Request) -> web.Response:
        if request.headers.get("X-S1M-Internal-Token") != "t" * 32:
            return web.json_response({"error": "unauthorized"}, status=401)
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
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.parametrize("noise,expect", [(0.0, True), (0.6, False)])
async def test_selfcheck_against_runtime(noise, expect):
    ref = selfcheck.load_reference("s1-fast")
    runner, url = await _fake_runtime(ref, noise)
    try:
        report = await selfcheck.run("s1-fast", url, "t" * 32, limit=120, concurrency=8)
    finally:
        await runner.cleanup()
    assert report["answered"] == 120 and report["failed"] == 0
    assert report["pass"] is expect
    if expect:
        assert report["distance_max"] < 1e-3


async def test_selfcheck_reports_unreachable_runtime():
    with pytest.raises(ConnectionError):
        await selfcheck.run("s1-fast", "http://127.0.0.1:9", "t" * 32, limit=5, concurrency=1)


def test_cli_cannot_run_without_token(tmp_path, capsys):
    assert selfcheck.main(["--url", "http://127.0.0.1:9", "--token-file", str(tmp_path / "missing"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["pass"] is False


async def test_selfcheck_s1_pro_path(monkeypatch):
    """The s1-pro model uses the s1-pro pool constants and its own manifest/reference file (synthetic here)."""
    fast = selfcheck.load_reference("s1-fast")
    pro = dict(fast, model="s1-pro", manifest_id=selfcheck.load_manifest("s1-pro").manifest_id)
    monkeypatch.setattr(selfcheck, "load_reference", lambda model: pro if model == "s1-pro" else fast)
    runner, url = await _fake_runtime(fast, 0.0)
    try:
        report = await selfcheck.run("s1-pro", url, "t" * 32, limit=60, concurrency=4)
    finally:
        await runner.cleanup()
    assert report["model"] == "s1-pro" and report["manifest_id"] == pro["manifest_id"]
    assert report["answered"] == 60 and report["pass"] is True


def test_cli_s1_pro_without_reference_file_cannot_run(tmp_path, capsys, monkeypatch):
    def missing(model):
        raise ValueError(f"no reference answers for {model} manifest x in this release")

    monkeypatch.setattr(selfcheck, "load_reference", missing)
    tok = tmp_path / "tok"
    tok.write_text("t" * 32)
    assert selfcheck.main(["--model", "s1-pro", "--url", "http://127.0.0.1:9", "--token-file", str(tok), "--json"]) == 2
