"""Versioned model manifests for ``reax-s1/1`` (src/reax_subnet/manifests/*.json).

A manifest pins the weights (HF revision and the sha256 of every file the
runtime reads), the runtime (image digest, packages, source hashes) and the
readout (prompt, labels, temperature). ``manifest_id`` is the first 16 hex
characters of the sha256 of the manifest's canonical JSON; validators put it in
every request and miners commit to it on chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

# The manifests this release accepts, per model.
RELEASE_MANIFESTS = {"s1-fast": "s1-fast-v1.json", "s1-pro": "s1-pro-v1.json",
                     # Tiered pools (REAX_FEATURE_TIERED_POOLS, development networks only; docs/DESIGN-POOLS.md).
                     "s1-lite": "s1-lite-v1.json", "s1-cpu": "s1-cpu-v1.json"}
TIERED_MODELS = ("s1-lite", "s1-cpu")


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class Manifest:
    model: str
    manifest_id: str
    data: dict

    @property
    def temperature(self) -> float:
        t = self.data["readout"]["temperature"]
        # s1-cpu has one temperature per question type; the choice temperature stands for the model.
        return float(t["choice"] if isinstance(t, dict) else t)

    @property
    def weight_files(self) -> dict[str, str]:
        return dict(self.data["weights"]["files"])


def load_manifest(model: str) -> Manifest:
    name = RELEASE_MANIFESTS.get(model)
    if name is None:
        raise ValueError(f"no manifest for model {model!r}")
    data = json.loads(resources.files("reax_subnet.manifests").joinpath(name).read_text(encoding="utf-8"))
    if data.get("model") != model or data.get("schema") != "reax-s1-manifest/1":
        raise ValueError(f"manifest {name} is not a reax-s1 manifest for {model}")
    return Manifest(model, hashlib.sha256(canonical(data)).hexdigest()[:16], data)


def verify_weights(model_dir: str, manifest: Manifest) -> None:
    """Refuse to serve unless every pinned file in ``model_dir`` has its pinned sha256."""
    root = Path(model_dir)
    for name, digest in sorted(manifest.weight_files.items()):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"pinned file missing: {name}")
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            raise RuntimeError(f"pinned file digest mismatch: {name}")


def commitment_text(model: str, manifest_id: str) -> str:
    return f"reax-s1/1:{model}:{manifest_id}"


def parse_commitment(text: Any) -> tuple[str, str] | None:
    """``reax-s1/1:<model>:<manifest_id>`` -> (model, manifest_id); anything else -> None."""
    if not isinstance(text, str):
        return None
    parts = text.strip().split(":")
    if len(parts) != 3 or parts[0] != "reax-s1/1" or parts[1] not in RELEASE_MANIFESTS:
        return None
    if len(parts[2]) != 16 or any(c not in "0123456789abcdef" for c in parts[2]):
        return None
    return parts[1], parts[2]
