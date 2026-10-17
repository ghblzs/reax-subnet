from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from reax_subnet.chain import Neuron, Snapshot


class DevWallet:
    """Wallet-shaped object holding throwaway dev keypairs (tests only)."""

    def __init__(self, uri: str):
        Keypair = pytest.importorskip("bittensor.keyfiles", reason="wallet tests require bittensor.keyfiles").Keypair
        self.hotkey = Keypair.create_from_uri(uri)
        self.coldkey = Keypair.create_from_uri(uri + "/cold")
        self.coldkeypub = self.coldkey


@pytest.fixture
def make_wallet():
    return DevWallet


@dataclass
class FakeChain:
    neurons: list[Neuron]
    owner_hotkey: str | None = None
    block: int = 1000
    rate_limit: int = 0
    since_update: int = 10_000
    fail_snapshot: bool = False
    fail_set_weights: bool = False
    set_calls: list[tuple[list[int], list[float], int]] = field(default_factory=list)

    def snapshot(self) -> Snapshot:
        if self.fail_snapshot:
            raise ConnectionError("node down")
        self.block += 10
        return Snapshot(block=self.block, neurons=tuple(self.neurons), owner_hotkey=self.owner_hotkey)

    def blocks_since_last_update(self, uid: int) -> int:
        return self.since_update

    def weights_rate_limit(self) -> int:
        return self.rate_limit

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None:
        if self.fail_set_weights:
            raise RuntimeError("pool rejected")
        self.set_calls.append((list(uids), list(weights), version_key))

    def serve_axon(self, ip: str, port: int) -> None:  # pragma: no cover - unused
        pass


@pytest.fixture
def fake_chain_cls():
    return FakeChain


# The entry floors the evaluator mechanism tests were calibrated on (1.2.0rc5 release values). The fake models' scores
# do not follow the real models' scale, so tests of the state machine and the copy checks pin these instead of following
# every re-measured release floor; the release values have their own tests (test_competitions.py, test_ka_nonproc.py).
RC5_ENTRY_FLOOR = {"demand_forecasting": 24.8, "risk_assessment": 33.7}


@pytest.fixture
def rc5_entry_floors(monkeypatch):
    from reax_subnet import competition_split as cs
    for k, v in RC5_ENTRY_FLOOR.items():
        monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, k, v)
