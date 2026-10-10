#!/usr/bin/env python3
"""Adversarial miners for local end-to-end tests. NOT for any public network.

REAX_ADVERSARY selects the attack:
  other_model  serve a different, similar-size model (Qwen3-0.6B) under the approved alias
  uniform      answer every item with the uniform distribution
  first        put all probability on the first option
All of them use the normal signed miner stack, so only the scoring can stop them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from aiohttp import web

from reax_subnet.backends import HFBackend, ModelSpec
from reax_subnet.chain import SdkChain
from reax_subnet.cli import _wallet, setup_logging
from reax_subnet.config import load_miner
from reax_subnet.miner import Miner, allowed_validators, refresh_validators_forever


class FnBackend:
    def __init__(self, alias, fn):
        self.model_alias = alias
        self.fn = fn

    def predict(self, items):
        return [self.fn(it) for it in items]


def build_backend(kind: str, alias: str):
    if kind == "other_model":
        spec = ModelSpec(alias, "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", production=False)
        return HFBackend(spec, device=os.environ.get("REAX_DEVICE", "cpu"), dtype=os.environ.get("REAX_DTYPE", "float32"))
    if kind == "uniform":
        return FnBackend(alias, lambda it: tuple([1.0 / len(it.options)] * len(it.options)))
    if kind == "first":
        return FnBackend(alias, lambda it: tuple([1.0] + [0.0] * (len(it.options) - 1)))
    sys.exit(f"unknown REAX_ADVERSARY {kind!r}")


async def run() -> None:
    cfg = load_miner()
    setup_logging(cfg.common)
    wallet, hotkey = _wallet(cfg.common)
    kind = os.environ["REAX_ADVERSARY"]
    backend = build_backend(kind, cfg.common.model)
    chain = SdkChain(cfg.common.network, cfg.common.netuid, wallet)
    snap = chain.snapshot()
    chain.serve_axon(cfg.external_ip, cfg.external_port)
    miner = Miner(wallet, backend)
    miner.set_validators(allowed_validators(snap, cfg.min_validator_stake))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, cfg.host, cfg.port).start()
    logging.getLogger("reax").info("adversary %s listening on %s:%d", kind, cfg.host, cfg.port)
    await refresh_validators_forever(miner, chain, cfg.min_validator_stake, cfg.metagraph_refresh_s)


if __name__ == "__main__":
    asyncio.run(run())
