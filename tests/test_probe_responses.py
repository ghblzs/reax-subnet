"""Probe responses (1.2.0rc7, review M1; from the 2026-10-15 validator challenge, findings 6 and 7): a probe answer
is at most PROBE_MAX_RESPONSE_BYTES, it is never decompressed and the validator asks for none (Accept-Encoding:
identity), and lanes a miner refuses with 429/503 are named in the probe report, also from probe shards.

The tests reuse the fake-machine arena of test_tiers.py (real aiohttp miners on 127.0.0.1, a faked chain).
"""

from __future__ import annotations

import gzip
import sys

import pytest
from aiohttp import web

from reax_subnet import auth, s1validator
from reax_subnet.miner import S1Miner
from reax_subnet.s1protocol import S1_DECIDE_PATH, parse_s1_request
from test_tiers import Machine, _probe_arena, _release_floors_off, probe_v2  # noqa: F401 (fixtures)


class Padded(S1Miner):
    """A correct, correctly signed answer, padded with JSON whitespace to ``pad`` bytes extra and/or gzipped; it
    records the Accept-Encoding of the requests it gets."""
    pad = 0
    compress = False
    seen_encodings: list = []

    def app(self) -> web.Application:
        app = super().app()

        @web.middleware
        async def record(request, handler):
            if request.method == "POST":
                type(self).seen_encodings.append(request.headers.get("Accept-Encoding"))
            return await handler(request)

        app.middlewares.append(record)
        return app

    async def _decide(self, caller, body):
        resp = await super()._decide(caller, body)
        if resp.status != 200:
            return resp
        raw = resp.body + b" " * self.pad
        req = parse_s1_request(body)
        headers = auth.sign_response(self.wallet, raw, req.request_id, caller, path=S1_DECIDE_PATH)
        headers["Content-Type"] = "application/json"
        if self.compress:
            return web.Response(body=gzip.compress(raw), headers={**headers, "Content-Encoding": "gzip"})
        return web.Response(body=raw, headers=headers)


class Big(Padded):
    pad = 40_000


class Gzipped(Padded):
    compress = True


@pytest.mark.parametrize("cls, measures", [(Padded, True), (Big, False), (Gzipped, False)])
async def test_probe_responses_are_size_limited_and_never_decompressed(cls, measures, probe_v2, make_wallet,
                                                                       fake_chain_cls, tmp_path):
    """Changes rewards only for miners that answer a probe with more than 32 kB or with gzip, which no honest miner
    does: such an answer does not count in the probe (no strike)."""
    probe_v2()
    Padded.seen_encodings = []
    groups = [(Machine(160.0), [{}]), (Machine(160.0), [{"miner_cls": cls}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    report = summaries[-1]["capacity"]
    honest, tested = report[by_group[0][0]], report[by_group[1][0]]
    assert honest["dps"] > 100, report
    assert (tested["dps"] > 100) if measures else (tested["dps"] == 0.0), report
    assert val.state.miners[by_group[1][0]].strikes == 0
    # The probe asks for plain answers (the round's own few queries keep aiohttp's default).
    seen = Padded.seen_encodings
    assert seen.count("identity") >= 10 and seen.count("identity") > 0.8 * len(seen), seen


async def test_a_legitimate_probe_response_is_far_below_the_limit():
    """16 generated decisions: measured 3.5-4.9 kB; the limit has more than 6x room."""
    from reax_subnet.s1backends import S1HashBackend
    from reax_subnet.s1contract import parse_decision
    from reax_subnet.s1items import generate_s1_items
    from reax_subnet.s1protocol import S1Response

    backend, biggest = S1HashBackend("s1-fast"), 0
    for _ in range(150):
        ds = [parse_decision(i.decision(), "s1-fast") for i in generate_s1_items(16)]
        answers = await backend.decide(ds)
        raw = S1Response("a" * 32, "1.2.0", tuple((d.ident, {d.question.ident: x}) for d, x in zip(ds, answers))
                         ).to_bytes()
        biggest = max(biggest, len(raw))
    assert biggest * 6 < s1validator.PROBE_MAX_RESPONSE_BYTES, biggest


class _OldLimitsMiner(S1Miner):
    """A miner on the 1.2.0rc2 request limits (2 per validator at once, 20 per minute, 4 runtime slots)."""

    def __init__(self, wallet, backend, **_new_limits):
        super().__init__(wallet, backend, max_concurrent=4, per_caller_concurrency=2, per_caller_per_minute=20,
                         max_pending_per_ip=8)


@pytest.mark.parametrize("shards", ["1", "2"])
async def test_lanes_refused_by_a_miners_own_limits_are_reported(shards, probe_v2, make_wallet, fake_chain_cls,
                                                                 tmp_path, monkeypatch):
    """Observability only: the probe report names lanes a miner turned away with 429/503, also from probe shards."""
    if shards != "1" and not sys.platform.startswith("linux"):
        pytest.skip("probe shards need Linux fork")
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", shards)
    groups = [(Machine(320.0), [{"miner_cls": _OldLimitsMiner}]), (Machine(320.0), [{}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    report = summaries[-1]["capacity"]
    old, new = report[by_group[0][0]], report[by_group[1][0]]
    assert val.last_probe_stats["shards"] == int(shards)
    assert old["refused_lanes"] > 0 and new["refused_lanes"] == 0, (old, new)
    assert old["dps"] < 0.6 * new["dps"], (old, new)
