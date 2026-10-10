"""Long-running validator behaviour: stalled chain calls, late weight submissions, skipped miners."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import aiohttp
import pytest

from reax_subnet import validator as validator_mod
from reax_subnet.chain import Neuron
from reax_subnet.config import load_validator
from reax_subnet.manifest import commitment_text, load_manifest
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1validator import SHARES_FAST_ONLY, S1Validator

FAST = load_manifest("s1-fast")


def _cfg(tmp_path, **extra):
    env = {
        "REAX_NETUID": "9",
        "REAX_ALLOW_DEV_MODELS": "1",
        "REAX_NETWORK": "ws://127.0.0.1:9951",
        "REAX_ALLOW_PRIVATE_IPS": "1",
        "REAX_STATE_PATH": str(tmp_path / "state.json"),
        "REAX_WARMUP_ROUNDS": "0",
        **extra,
    }
    return load_validator(env)


@pytest.fixture
def setup(tmp_path, make_wallet, fake_chain_cls):
    owner = make_wallet("//Owner")
    vwallet = make_wallet("//Validator")
    miners = [make_wallet(f"//Miner/{i}") for i in range(3)]
    neurons = [
        Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
        Neuron(1, vwallet.hotkey.ss58_address, "c1", True, 5000.0, None, None),
    ] + [
        # Committed but without an endpoint: never queried, so no miner servers are needed.
        Neuron(2 + i, w.hotkey.ss58_address, f"c{2 + i}", False, 0.0, None, None,
               commitment=commitment_text("s1-fast", FAST.manifest_id))
        for i, w in enumerate(miners)
    ]
    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)

    def make(**extra):
        return S1Validator(_cfg(tmp_path, **extra), vwallet, chain, {"s1-fast": S1HashBackend("s1-fast")},
                           shares=SHARES_FAST_ONLY)

    return {"chain": chain, "make": make, "neurons": neurons, "miners": miners}


class _Gate:
    """Blocks the calling thread until released (a stalled RPC)."""

    def __init__(self):
        self.event = threading.Event()
        self.calls = 0

    def wait(self):
        self.calls += 1
        self.event.wait(10)


async def test_stalled_metagraph_read_skips_the_round_without_piling_up(setup, monkeypatch):
    monkeypatch.setattr(validator_mod, "CHAIN_READ_TIMEOUT_S", 0.2)
    chain, val = setup["chain"], setup["make"]()
    gate = _Gate()
    real = chain.snapshot

    def stalled():
        gate.wait()
        return real()

    monkeypatch.setattr(chain, "snapshot", stalled)
    async with aiohttp.ClientSession() as session:
        started = time.monotonic()
        assert await val.run_round(session) is None
        assert await val.run_round(session) is None
        assert time.monotonic() - started < 2.0
        assert gate.calls == 1  # the second round waited on the same read instead of starting another
        gate.event.set()
        await asyncio.sleep(0.05)
        assert await val.run_round(session) is not None
    assert val.state.rounds == 1


async def test_stalled_rate_limit_read_does_not_block_weights(setup, monkeypatch):
    monkeypatch.setattr(validator_mod, "CHAIN_READ_TIMEOUT_S", 0.2)
    chain, val = setup["chain"], setup["make"]()
    gate = _Gate()
    monkeypatch.setattr(chain, "blocks_since_last_update", lambda uid: gate.wait() or 10_000)
    snap = chain.snapshot()
    val.state.sync(snap)
    started = time.monotonic()
    assert await val.maybe_set_weights(snap) is False
    assert time.monotonic() - started < 2.0
    assert chain.set_calls == []
    gate.event.set()


async def test_late_set_weights_failure_is_logged(setup, monkeypatch, caplog):
    monkeypatch.setattr(validator_mod, "SET_WEIGHTS_TIMEOUT_S", 0.1)
    chain, val = setup["chain"], setup["make"]()
    gate = _Gate()

    def slow_fail(uids, weights, version_key):
        gate.wait()
        raise RuntimeError("extrinsic dropped")

    monkeypatch.setattr(chain, "set_weights", slow_fail)
    snap = chain.snapshot()
    val.state.sync(snap)
    with caplog.at_level(logging.INFO, logger="reax.validator"):
        assert await val.maybe_set_weights(snap) is False
        assert await val.maybe_set_weights(snap) is False  # still pending: not submitted twice
        assert gate.calls == 1
        gate.event.set()
        for _ in range(50):
            if val._weights_task.done():
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0)
    assert any("outlived its wait failed: extrinsic dropped" in r.getMessage() for r in caplog.records)


async def test_all_miners_failing_puts_all_weight_on_the_owner(setup):
    chain, val = setup["chain"], setup["make"]()
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            await val.run_round(session)
    (uids, weights, version_key), = chain.set_calls[-1:]
    assert uids == [0] and weights == [pytest.approx(1.0)]
    from reax_subnet.version import WEIGHTS_VERSION_KEY

    assert version_key == WEIGHTS_VERSION_KEY
