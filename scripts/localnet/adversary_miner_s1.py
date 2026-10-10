#!/usr/bin/env python3
"""Adversarial reax-s1 miners for local end-to-end tests. NOT for any public network.

REAX_ADVERSARY selects the attack (all use the normal signed S1 miner stack and publish a valid
commitment, so only verification can stop them):
  other_model   a different model under the s1-fast role (deterministic pseudo-model)
  quantised     the real runtime's answers with +-1.5 logit noise (stand-in for FP8/INT4 builds;
                the real FP8 builds are measured in the GPU calibration)
  uniform       uniform probabilities (cache miss / no model)
  inconsistent  real probabilities, but `choice` is not the most likely option
  partial       real model for half of each batch, a cheap answer for the other half (spot-check target)
  slow          real answers after 30 s
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os

from aiohttp import web

from reax_subnet.chain import SdkChain
from reax_subnet.cli import _runtime_token, _wallet, setup_logging
from reax_subnet.config import load_miner
from reax_subnet.manifest import commitment_text
from reax_subnet.miner import S1Miner, allowed_validators, refresh_validators_forever
from reax_subnet.s1backends import S1HashBackend, SidecarBackend
from reax_subnet.s1contract import normalize_answer


def _noise(d, k, amp):
    h = hashlib.sha256(f"q{json.dumps(d.state, sort_keys=True)}{k}{d.ident}".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 2 * amp


class Adversary:
    def __init__(self, kind, model, real):
        self.kind, self.model, self.real = kind, model, real
        self.manifest = real.manifest if real is not None else S1HashBackend(model).manifest
        self.cheap = S1HashBackend(model, salt="cheap")

    async def decide(self, decisions):
        k = self.kind
        if k == "other_model":
            return await self.cheap.decide(decisions)
        if k == "uniform":
            out = []
            for d in decisions:
                n = len(d.question.keys)
                raw = {"noul": 0.5} if d.question.kind == "noul" else {"probabilities": {x: 1 / n for x in d.question.keys}}
                out.append(normalize_answer(d.question, raw))
            return out
        if k == "slow":
            await asyncio.sleep(30)
            return await self.real.decide(decisions)
        if k == "partial":
            half = len(decisions) // 2
            return (await self.cheap.decide(decisions[:half])) + (await self.real.decide(decisions[half:]))
        answers = await self.real.decide(decisions)
        if k == "inconsistent":
            return [{**a, "choice": min(a["probabilities"], key=a["probabilities"].get)} if a["type"] == "choice" else a
                    for a in answers]
        if k == "quantised":
            T = self.manifest.temperature
            out = []
            for d, a in zip(decisions, answers):
                z = {key: T * math.log(max(p, 1e-30)) + _noise(d, key, 1.5) for key, p in a["probabilities"].items()}
                m = max(z.values())
                e = {key: math.exp((v - m) / T) for key, v in z.items()}
                s = math.fsum(e.values())
                probs = {key: v / s for key, v in e.items()}
                raw = {"noul": probs["true"]} if d.question.kind == "noul" else {"probabilities": probs}
                out.append(normalize_answer(d.question, raw))
            return out
        raise SystemExit(f"unknown REAX_ADVERSARY {k!r}")

    async def close(self):
        pass


async def run() -> None:
    cfg = load_miner()
    setup_logging(cfg.common)
    wallet, _ = _wallet(cfg.common)
    kind = os.environ["REAX_ADVERSARY"]
    model = cfg.common.model
    url = cfg.common.runtime_url(model)
    real = SidecarBackend(model, url, _runtime_token(cfg.common)) if url and url != "hash" else S1HashBackend(model)
    backend = Adversary(kind, model, real)
    chain = SdkChain(cfg.common.network, cfg.common.netuid, wallet, commitments=True)
    snap = chain.snapshot()
    chain.serve_axon(cfg.external_ip, cfg.external_port)
    chain.set_commitment(commitment_text(model, backend.manifest.manifest_id))
    miner = S1Miner(wallet, backend)
    miner.set_validators(allowed_validators(snap, cfg.min_validator_stake))
    runner = web.AppRunner(miner.app(), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, cfg.host, cfg.port).start()
    logging.getLogger("reax").info("s1 adversary %s (%s) listening on %s:%d", kind, model, cfg.host, cfg.port)
    await refresh_validators_forever(miner, chain, cfg.min_validator_stake, cfg.metagraph_refresh_s)


if __name__ == "__main__":
    asyncio.run(run())
