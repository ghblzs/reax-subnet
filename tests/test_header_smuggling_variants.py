"""Raw-socket header-smuggling variants against the miner (regression tests).

Also worth running with the pure-Python HTTP parser: AIOHTTP_NO_EXTENSIONS=1 python -m pytest tests/test_header_smuggling_variants.py
Every attacker variant must end in a non-200 status (or a closed connection); the honest variants must be 200.
"""
import asyncio

import aiohttp
import pytest
from aiohttp import web
from bittensor.keyfiles import Keypair

from reax_subnet import auth
from reax_subnet.backends import HashBackend
from reax_subnet.items import generate_items
from reax_subnet.miner import Miner, S1Miner  # noqa: F401
from reax_subnet.protocol import DecideRequest


class W:
    def __init__(self, uri):
        self.hotkey = Keypair.create_from_uri(uri)
        self.coldkey = Keypair.create_from_uri(uri + "/cold")
        self.coldkeypub = self.coldkey


MW, VW, AW = W("//M"), W("//V"), W("//Attacker")
MK, VK, AK = MW.hotkey.ss58_address, VW.hotkey.ss58_address, AW.hotkey.ss58_address


@pytest.fixture
async def port():
    miner = Miner(MW, HashBackend(), max_concurrent=2, per_caller_concurrency=4, per_caller_per_minute=1000)
    miner.set_validators(frozenset({VK}))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    yield site._server.sockets[0].getsockname()[1]
    await runner.cleanup()


def _body(i=0):
    return DecideRequest(f"r{i}", "hash-test", tuple(generate_items(2, seed=i))).to_bytes()


async def raw_post(port, header_lines, body):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    head = b"POST /v1/decide HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    head += b"Content-Length: %d\r\n" % len(body)
    for line in header_lines:
        head += (line if isinstance(line, bytes) else line.encode("utf-8")) + b"\r\n"
    w.write(head + b"\r\n" + body)
    await w.drain()
    try:
        status_line = await asyncio.wait_for(r.readline(), 5)
    finally:
        w.close()
    if not status_line:
        return 0  # connection closed by the parser
    return int(status_line.split()[1])


def lines(signed, replace=None, extra_before=(), extra_after=()):
    out = list(extra_before)
    for k, v in signed.items():
        if replace and k in replace:
            out.append(replace[k])
        else:
            out.append(f"{k}: {v}")
    return out + list(extra_after)


ATTACKS = {
    "dup_case_first": lambda s: lines(s, extra_before=[f"x-bittensor-hotkey: {VK}"]),
    "dup_case_last": lambda s: lines(s, extra_after=[f"X-BITTENSOR-HOTKEY: {VK}"]),
    "comma_joined": lambda s: lines(s, replace={"X-Bittensor-Hotkey": f"X-Bittensor-Hotkey: {VK}, {AK}"}),
    "comma_joined_rev": lambda s: lines(s, replace={"X-Bittensor-Hotkey": f"X-Bittensor-Hotkey: {AK},{VK}"}),
    "obs_fold": lambda s: lines(s, replace={"X-Bittensor-Hotkey": f"X-Bittensor-Hotkey: {VK}\r\n {AK}"}),
    "space_before_colon": lambda s: lines(s, extra_before=[f"X-Bittensor-Hotkey : {VK}"]),
    "tab_before_colon": lambda s: lines(s, extra_before=[f"X-Bittensor-Hotkey\t: {VK}"]),
    "leading_ws_line": lambda s: lines(s, extra_before=[f" X-Bittensor-Hotkey: {VK}"]),
    "kelvin_sign_name": lambda s: lines(s, extra_before=[f"X-Bittensor-HotKey: {VK}"]),
    "nul_in_value": lambda s: lines(s, extra_before=[f"X-Bittensor-Hotkey: {VK}\x00"]),
    "trailing_ws_value": lambda s: lines(s, replace={"X-Bittensor-Hotkey": f"X-Bittensor-Hotkey: {VK}   "}),
    "claim_only_validator": lambda s: lines(s, replace={"X-Bittensor-Hotkey": f"X-Bittensor-Hotkey: {VK}"}),
    "underscore_name": lambda s: lines(s, extra_before=[f"X_Bittensor_Hotkey: {VK}"]),
    "many_headers": lambda s: lines(s, extra_before=[f"X-Pad-{i}: a" for i in range(500)] + [f"X-Bittensor-Hotkey: {VK}"]),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
async def test_attacker_variant_is_not_served(port, name):
    body = _body(1)
    signed = auth.sign_request(AW, body, MK)
    status = await raw_post(port, ATTACKS[name](signed), body)
    assert status != 200, name


DUP_OTHER = ["X-Bittensor-Signature", "X-Bittensor-Nonce", "X-Bittensor-Receiver", "X-Bittensor-Version"]


@pytest.mark.parametrize("hdr", DUP_OTHER + ["X-Bittensor-Crypto"])
async def test_validator_request_with_other_duplicate_is_refused_closed(port, hdr):
    """A copy of another btauth header on a validator-signed request fails closed (no confusion, no 200)."""
    body = _body(2)
    signed = auth.sign_request(VW, body, MK)
    extra = {"X-Bittensor-Crypto": "ed25519"}.get(hdr, signed.get(hdr, "1"))
    status = await raw_post(port, lines(signed, extra_after=[f"{hdr}: {extra}"]), body)
    assert status == 401, (hdr, status)


@pytest.mark.parametrize("hdr", DUP_OTHER)
async def test_attacker_cannot_mix_validator_headers(port, hdr):
    """Take a real validator-signed header set and swap in attacker copies (or vice versa)."""
    body = _body(3)
    v = auth.sign_request(VW, body, MK)
    a = auth.sign_request(AW, body, MK)
    for first, second in ((v, a), (a, v)):
        hl = lines(first, extra_after=[f"{hdr}: {second[hdr]}"])
        assert await raw_post(port, hl, body) != 200


@pytest.mark.parametrize("lower", [False, True])
async def test_honest_validator_raw_and_lowercase(port, lower):
    body = _body(4)
    signed = auth.sign_request(VW, body, MK)
    hl = [f"{k.lower() if lower else k}: {v}" for k, v in signed.items()]
    assert await raw_post(port, hl, body) == 200


async def test_honest_validator_real_aiohttp_client(port):
    body = _body(5)
    h = auth.sign_request(VW, body, MK)
    h["Content-Type"] = "application/json"
    async with aiohttp.ClientSession(trust_env=False) as s:
        async with s.post(f"http://127.0.0.1:{port}/v1/decide", data=body, headers=h) as r:
            raw = await r.read()
            assert r.status == 200
            auth.verify_response(r.headers, raw, "r5", VK, MK, auth.NonceStore(), 30)
