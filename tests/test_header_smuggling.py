"""a duplicated X-Bittensor-Hotkey header must not get an
outsider past the miner's validator allow-list, and the per-validator limits must key on the verified signer."""

import asyncio
import os
import tempfile
import time

import aiohttp
import pytest
from aiohttp import web
from bittensor.keyfiles import Keypair
from multidict import CIMultiDict

from reax_subnet import auth
from reax_subnet.backends import HashBackend
from reax_subnet.config import ConfigError, load_common, load_validator
from reax_subnet.items import generate_items
from reax_subnet.miner import Miner
from reax_subnet.protocol import DecideRequest


class W:
    def __init__(self, uri):
        self.hotkey = Keypair.create_from_uri(uri)
        self.coldkey = Keypair.create_from_uri(uri + "/cold")
        self.coldkeypub = self.coldkey


class SlowHash(HashBackend):
    def predict(self, items):
        time.sleep(0.3)
        return super().predict(items)


@pytest.fixture
async def env():
    mw, vw = W("//M"), W("//V")
    miner = Miner(mw, SlowHash(), max_concurrent=2, per_caller_concurrency=1, per_caller_per_minute=2)
    miner.set_validators(frozenset({vw.hotkey.ss58_address}))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1/decide"
    session = aiohttp.ClientSession()
    yield {"url": url, "mw": mw, "vw": vw, "session": session}
    await session.close()
    await runner.cleanup()


def _smuggled(attacker, body, miner_hotkey, validator_hotkey, validator_first=True):
    signed = auth.sign_request(attacker, body, miner_hotkey)
    h = CIMultiDict()
    if validator_first:
        h.add("X-Bittensor-Hotkey", validator_hotkey)
    for k, v in signed.items():
        h.add(k, v)
    if not validator_first:
        h.add("X-Bittensor-Hotkey", validator_hotkey)
    h.add("Content-Type", "application/json")
    return h


@pytest.mark.parametrize("validator_first", [True, False])
async def test_duplicate_hotkey_header_is_refused(env, validator_first):
    attacker = W("//Outsider-not-on-chain")
    body = DecideRequest("abc", "hash-test", tuple(generate_items(2, seed=3))).to_bytes()
    mk, vk = env["mw"].hotkey.ss58_address, env["vw"].hotkey.ss58_address
    async with env["session"].post(env["url"], data=body,
                                   headers=_smuggled(attacker, body, mk, vk, validator_first)) as r:
        assert r.status in (401, 403), await r.read()


async def test_fresh_keys_cannot_crowd_out_the_real_validator(env):
    mk, vk = env["mw"].hotkey.ss58_address, env["vw"].hotkey.ss58_address

    async def one(i):
        a = W(f"//flood{i}")
        body = DecideRequest(f"r{i}", "hash-test", tuple(generate_items(16, seed=i))).to_bytes()
        async with env["session"].post(env["url"], data=body, headers=_smuggled(a, body, mk, vk)) as r:
            return r.status

    flood = []
    for i in range(10):
        flood.append(asyncio.create_task(one(i)))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    body = DecideRequest("legit", "hash-test", tuple(generate_items(2, seed=99))).to_bytes()
    h = auth.sign_request(env["vw"], body, mk)
    h["Content-Type"] = "application/json"
    async with env["session"].post(env["url"], data=body, headers=h) as r:
        assert r.status == 200
    statuses = await asyncio.gather(*flood)
    assert 200 not in statuses


def test_duplicate_headers_on_a_response_are_refused():
    m, v = W("//M"), W("//V")
    raw = b"{}"
    signed = auth.sign_response(m, raw, "rid", v.hotkey.ss58_address)
    h = CIMultiDict(signed)
    h.add("X-Bittensor-Hotkey", m.hotkey.ss58_address)
    with pytest.raises(auth.AuthError):
        auth.verify_response(h, raw, "rid", v.hotkey.ss58_address, m.hotkey.ss58_address, auth.NonceStore(), 30)
    assert auth.claimed_hotkey(h) is None


@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:8101", True), ("http://localhost:8101", True), ("http://[::1]:8101", True),
    ("http://127.0.0.1@evil.example:8101", False), ("http://127.0.0.1.nip.io:8101", False),
    ("http://localhost.evil.example", False), ("https://127.0.0.1:8101", False), ("http://10.0.0.1:8101", False),
])
def test_runtime_url_must_really_be_loopback(url, ok):
    fd, tok = tempfile.mkstemp()
    os.write(fd, b"t" * 32)
    os.close(fd)
    env = {"REAX_MECHANISM": "s1", "REAX_NETUID": "1", "REAX_NETWORK": "test",
           "REAX_S1_FAST_RUNTIME_URL": url, "REAX_RUNTIME_TOKEN_FILE": tok}
    if ok:
        assert load_common(env).runtime_url("s1-fast") == url
    else:
        with pytest.raises(ConfigError):
            load_common(env)


@pytest.mark.parametrize("net", ["finney", "archive", "wss://entrypoint-finney.opentensor.ai:443",
                                 "wss://entrypoint-finney.opentensor.ai"])
def test_private_ips_refused_on_every_mainnet_name(net, tmp_path):
    tok = tmp_path / "t"
    tok.write_text("t" * 32)
    env = {"REAX_MECHANISM": "s1", "REAX_NETUID": "1", "REAX_NETWORK": net, "REAX_ALLOW_PRIVATE_IPS": "1",
           "REAX_S1_FAST_RUNTIME_URL": "http://127.0.0.1:8101", "REAX_RUNTIME_TOKEN_FILE": str(tok)}
    with pytest.raises(ConfigError):
        load_validator(env)
