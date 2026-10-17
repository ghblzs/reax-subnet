"""The small slice of the chain that miners and validators use.

``Chain`` is an interface so the validator loop can be tested without a
node. ``SdkChain`` implements it with the Bittensor SDK (v11). Every method
may raise; callers treat any exception as "chain unavailable this round" and
retry later instead of crashing.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neuron:
    uid: int
    hotkey: str
    coldkey: str
    validator_permit: bool
    stake: float  # subnet alpha stake
    ip: Optional[str]
    port: Optional[int]
    # Plaintext on-chain commitment (Commitments pallet), "" if none. reax-s1 miners publish
    # "reax-s1/1:<model>:<manifest_id>" here.
    commitment: str = ""
    commitment_block: int = 0  # block of that commitment (0 = unknown)
    registered_block: int = 0  # block this hotkey registered on the subnet (metagraph block_at_registration; 0 = unknown)


@dataclass(frozen=True)
class Snapshot:
    block: int
    neurons: tuple[Neuron, ...]
    owner_hotkey: Optional[str]

    @property
    def owner_uid(self) -> Optional[int]:
        for n in self.neurons:
            if n.hotkey == self.owner_hotkey:
                return n.uid
        return None

    def uid_of(self, hotkey: str) -> Optional[int]:
        for n in self.neurons:
            if n.hotkey == hotkey:
                return n.uid
        return None


class Chain(Protocol):
    def snapshot(self) -> Snapshot: ...

    def blocks_since_last_update(self, uid: int) -> int: ...

    def weights_rate_limit(self) -> int: ...

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None: ...

    def serve_axon(self, ip: str, port: int) -> None: ...

    def set_commitment(self, text: str) -> None: ...


def parse_axon(axon: Any) -> tuple[Optional[str], Optional[int]]:
    """Parse the SDK's ``"ip:port"`` axon string. Returns (None, None) when unset or invalid."""
    if not axon or not isinstance(axon, str):
        return None, None
    host, sep, port_s = axon.rpartition(":")
    if not sep:
        return None, None
    host = host.strip("[]")
    try:
        ip = ipaddress.ip_address(host)
        port = int(port_s)
    except ValueError:
        return None, None
    if not 0 < port < 65536 or ip.is_unspecified:
        return None, None
    return str(ip), port


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class SdkChain:
    def __init__(self, network: str, netuid: int, wallet: Any, commitments: bool = False):
        import bittensor as bt

        self._bt = bt
        self.netuid = netuid
        self.wallet = wallet
        self.subtensor = bt.Subtensor(network=network)
        self.read_commitments = commitments

    def snapshot(self) -> Snapshot:
        mg = self.subtensor.subnets.metagraph(self.netuid, commitments=self.read_commitments)
        if mg is None:
            raise RuntimeError(f"subnet {self.netuid} does not exist")
        neurons = []
        for n in mg.neurons:
            ip, port = parse_axon(n.axon)
            neurons.append(
                Neuron(
                    uid=int(n.uid),
                    hotkey=str(n.hotkey),
                    coldkey=str(n.coldkey),
                    validator_permit=bool(n.validator_permit),
                    stake=float(n.total_stake.amount),
                    ip=ip,
                    port=port,
                    commitment=_commitment_text(getattr(n, "commitment", None)),
                    commitment_block=_commitment_block(getattr(n, "commitment", None)),
                    registered_block=_int_or_zero(getattr(n, "block_at_registration", 0)),
                )
            )
        return Snapshot(block=int(mg.block), neurons=tuple(neurons), owner_hotkey=mg.owner_hotkey)

    def blocks_since_last_update(self, uid: int) -> int:
        value = self.subtensor.epochs.blocks_since_last_update(self.netuid, uid)
        return int(value) if value is not None else 1 << 30

    def weights_rate_limit(self) -> int:
        return int(self.subtensor.hyperparameters.weights_rate_limit(self.netuid))

    def set_weights(self, uids: Sequence[int], weights: Sequence[float], version_key: int) -> None:
        bt = self._bt
        result = self.subtensor.execute(
            bt.SetWeights(self.netuid, uids=list(uids), weights=list(weights), version_key=version_key),
            self.wallet,
        )
        if not result.success:
            raise RuntimeError(f"set_weights failed: {result.message}")

    def set_commitment(self, text: str) -> None:
        from bittensor._generated import calls

        data = text.encode("utf-8")
        if not 0 < len(data) <= 128:
            raise ValueError("commitment must be 1..128 bytes")
        info = {"fields": [[{f"Raw{len(data)}": data}]]}
        result = self.subtensor.submit_call(
            calls.Commitments.set_commitment(netuid=self.netuid, info=info), self.wallet, signer="hotkey"
        )
        if not result.success:
            raise RuntimeError(f"set_commitment failed: {result.message}")

    def genesis_hash(self) -> str:
        """Hash of block 0. ``block_info(0)`` would also read state at block 0, which only archive nodes keep."""
        import inspect

        client = getattr(self.subtensor, "_client", None)
        if client is not None and hasattr(client, "_block_hash"):
            r = client._block_hash(0)
            value = self.subtensor._call(r) if inspect.iscoroutine(r) else r
        else:  # pragma: no cover - other SDK layouts
            value = self.subtensor.block_info(0).hash
        if not isinstance(value, str) or not value.startswith("0x"):
            raise RuntimeError("could not read the genesis hash")
        return value

    def serve_axon(self, ip: str, port: int) -> None:
        bt = self._bt
        result = self.subtensor.execute(bt.ServeAxon(self.netuid, ip, port), self.wallet)
        if not result.success:
            raise RuntimeError(f"serve_axon failed: {result.message}")


FINNEY_GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"


def _commitment_block(c: Any) -> int:
    value = getattr(c, "block", 0) if c is not None else 0
    return value if isinstance(value, int) and value > 0 else 0


def _commitment_text(c: Any) -> str:
    """Visible plaintext of an SDK NeuronCommitment (or None) as a short string."""
    if c is None:
        return ""
    value = getattr(c, "value", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(c, "data", "")
    return value[:256] if isinstance(value, str) else ""
