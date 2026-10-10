import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web

from reax_subnet import auth
from reax_subnet.backends import HashBackend
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.items import generate_items
from reax_subnet.miner import Miner, allowed_validators
from reax_subnet.protocol import DecideRequest, parse_response


class SlowHash(HashBackend):
    def predict(self, items):
        time.sleep(0.5)
        return super().predict(items)


@pytest.fixture
async def miner_env(make_wallet):
    mw, vw, stranger = make_wallet("//M"), make_wallet("//V"), make_wallet("//Stranger")
    miner = Miner(mw, SlowHash(), max_concurrent=1, per_caller_concurrency=1, max_queue=1)
    miner.set_validators(frozenset({vw.hotkey.ss58_address}))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    session = aiohttp.ClientSession()
    yield {"url": url, "miner": miner, "mw": mw, "vw": vw, "stranger": stranger, "session": session}
    await session.close()
    await runner.cleanup()


def _req(model="hash-test", n=2):
    return DecideRequest("abc", model, tuple(generate_items(n, seed=3)))


async def _post(env, body, signer, receiver=None, headers=None):
    h = headers or auth.sign_request(signer, body, receiver or env["mw"].hotkey.ss58_address)
    h["Content-Type"] = "application/json"
    async with env["session"].post(env["url"] + "/v1/decide", data=body, headers=h) as r:
        return r.status, await r.read(), dict(r.headers), h


async def test_happy_path_signed_response(miner_env):
    req = _req()
    status, raw, headers, _ = await _post(miner_env, req.to_bytes(), miner_env["vw"])
    assert status == 200
    vk, mk = miner_env["vw"].hotkey.ss58_address, miner_env["mw"].hotkey.ss58_address
    auth.verify_response(headers, raw, req.request_id, vk, mk, auth.NonceStore(), max_age=30)
    assert len(parse_response(raw, req)) == 2


async def test_rejections(miner_env):
    body = _req().to_bytes()
    assert (await _post(miner_env, body, miner_env["stranger"]))[0] == 403
    assert (await _post(miner_env, body, miner_env["vw"], receiver=miner_env["stranger"].hotkey.ss58_address))[0] == 401
    status, _, _, used = await _post(miner_env, body, miner_env["vw"])
    assert status == 200
    assert (await _post(miner_env, body, miner_env["vw"], headers=dict(used)))[0] == 401  # replay
    tampered = body.replace(b"abc", b"abd")
    h = auth.sign_request(miner_env["vw"], body, miner_env["mw"].hotkey.ss58_address)
    assert (await _post(miner_env, tampered, miner_env["vw"], headers=h))[0] == 401
    assert (await _post(miner_env, _req(model="other-model").to_bytes(), miner_env["vw"]))[0] == 400
    assert (await _post(miner_env, b'{"protocol":"x"}', miner_env["vw"]))[0] == 400
    big = b" " * (300 * 1024)
    assert (await _post(miner_env, big, miner_env["vw"]))[0] == 413


async def test_unloaded_metagraph_is_503(miner_env):
    miner_env["miner"].validators = None
    assert (await _post(miner_env, _req().to_bytes(), miner_env["vw"]))[0] == 503


async def test_per_caller_concurrency(miner_env):
    results = await asyncio.gather(*(_post(miner_env, _req().to_bytes(), miner_env["vw"]) for _ in range(3)))
    statuses = sorted(r[0] for r in results)
    assert statuses[0] == 200 and 429 in statuses


async def test_per_caller_rate_limit(miner_env):
    miner_env["miner"]._per_minute = 2
    statuses = [(await _post(miner_env, _req().to_bytes(), miner_env["vw"]))[0] for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_allowed_validators_needs_permit_and_stake():
    snap = Snapshot(1, (
        Neuron(0, "a", "c", True, 5000.0, None, None),
        Neuron(1, "b", "c", True, 10.0, None, None),
        Neuron(2, "d", "c", False, 1e9, None, None),
    ), None)
    assert allowed_validators(snap, 1000.0) == frozenset({"a"})


async def test_slow_body_is_cut_off(miner_env):
    import socket

    miner_env["miner"]._body_timeout = 0.5
    host, port = miner_env["url"].split("//")[1].split(":")
    body = _req().to_bytes()
    headers = auth.sign_request(miner_env["vw"], body, miner_env["mw"].hotkey.ss58_address)
    head = "POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    head += "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    head += f"Content-Length: {len(body)}\r\n\r\n"
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(head.encode() + body[:10])
    await writer.drain()
    status_line = await asyncio.wait_for(reader.readline(), timeout=5)
    assert b" 408 " in status_line
    writer.close()


async def test_pending_reads_capped_per_ip(miner_env):
    miner_env["miner"]._max_pending_ip = 0
    assert (await _post(miner_env, _req().to_bytes(), miner_env["vw"]))[0] == 503
