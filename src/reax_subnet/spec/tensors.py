"""Safetensors without a framework: header parsing, canonical tensor hashes and weight-delta similarity.

The files are parsed as data (8-byte length, JSON header, raw tensor bytes). Nothing is unpickled or imported
from the repository. Hashes are over tensors, not files, so re-sharding or renaming shard files changes nothing
(docs/SPECIALIST-TRACKS.md, rule 1).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

MAX_HEADER_BYTES = 100 * 1024 * 1024
CHUNK_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "U16": 2, "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
FLOAT_DTYPES = ("F64", "F32", "F16", "BF16")


class SafetensorsError(ValueError):
    """The file is not a well-formed safetensors file."""


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int      # absolute file offset
    end: int
    path: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def read_header(path: str) -> list[TensorEntry]:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{os.path.basename(path)}: too short")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen > MAX_HEADER_BYTES or 8 + hlen > size:
            raise SafetensorsError(f"{os.path.basename(path)}: bad header length")
        try:
            header = json.loads(fh.read(hlen))
        except ValueError as exc:
            raise SafetensorsError(f"{os.path.basename(path)}: header is not json") from exc
    if not isinstance(header, dict):
        raise SafetensorsError("header is not an object")
    base = 8 + hlen
    entries: list[TensorEntry] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype, shape, (a, b) = info["dtype"], tuple(info["shape"]), info["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(f"tensor {name!r}: malformed entry") from exc
        if dtype not in DTYPE_BYTES or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise SafetensorsError(f"tensor {name!r}: unsupported dtype or shape")
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b and base + b <= size):
            raise SafetensorsError(f"tensor {name!r}: offsets outside the file")
        if b - a != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise SafetensorsError(f"tensor {name!r}: byte size does not match dtype and shape")
        entries.append(TensorEntry(name, dtype, shape, base + a, base + b, path))
    return sorted(entries, key=lambda e: e.name)


def weight_files(model_dir: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".safetensors"):
                p = os.path.join(root, f)
                if os.path.islink(p):
                    raise SafetensorsError(f"{f}: symlinks are not allowed")
                out.append(p)
    return sorted(out)


def read_model(model_dir: str) -> dict[str, TensorEntry]:
    """All tensors of a model directory by name; a name that appears twice is an error."""
    tensors: dict[str, TensorEntry] = {}
    for p in weight_files(model_dir):
        for e in read_header(p):
            if e.name in tensors:
                raise SafetensorsError(f"tensor {e.name!r} appears in two files")
            tensors[e.name] = e
    if not tensors:
        raise SafetensorsError("no safetensors tensors")
    return tensors


def param_count(tensors: dict[str, TensorEntry]) -> int:
    return sum(e.numel for e in tensors.values())


def weight_bytes(tensors: dict[str, TensorEntry]) -> int:
    return sum(e.end - e.start for e in tensors.values())


def _read_chunks(e: TensorEntry) -> Iterator[bytes]:
    with open(e.path, "rb") as fh:
        fh.seek(e.start)
        left = e.end - e.start
        while left > 0:
            block = fh.read(min(CHUNK_BYTES, left))
            if not block:
                raise SafetensorsError(f"{e.name}: file truncated")
            left -= len(block)
            yield block


def tensor_digest(e: TensorEntry, *, with_name: bool) -> str:
    h = hashlib.sha256()
    if with_name:
        h.update(e.name.encode() + b"\0")
    h.update(e.dtype.encode() + b"\0" + ",".join(map(str, e.shape)).encode() + b"\0")
    for block in _read_chunks(e):
        h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class WeightHashes:
    named: str      # sha256 over sorted (tensor name, digest): re-sharding proof
    unnamed: str    # sha256 over sorted digests without names: also survives renaming tensors
    params: int
    weight_bytes: int


def canonical_hashes(model_dir: str) -> WeightHashes:
    tensors = read_model(model_dir)
    named, unnamed = [], []
    for name in sorted(tensors):
        e = tensors[name]
        d = tensor_digest(e, with_name=True)
        named.append(f"{name}\t{d}\n")
        unnamed.append(tensor_digest(e, with_name=False))
    return WeightHashes(
        named=hashlib.sha256("".join(named).encode()).hexdigest(),
        unnamed=hashlib.sha256("".join(sorted(unnamed)).encode()).hexdigest(),
        params=param_count(tensors),
        weight_bytes=weight_bytes(tensors),
    )


STRIDE = 4093   # prime sampling stride of the delta sketch (~2.2 M values for a 9 B model)


def _sample_float(e: TensorEntry, stride: int) -> np.ndarray:
    """Every ``stride``-th element of a float tensor as float32 (memory-mapped: only the touched pages are read)."""
    if e.numel == 0:
        return np.zeros(0, dtype=np.float32)
    if e.dtype == "BF16":
        raw = np.memmap(e.path, dtype="<u2", mode="r", offset=e.start, shape=(e.numel,))[::stride]
        return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
    dt = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}[e.dtype]
    return np.asarray(np.memmap(e.path, dtype=dt, mode="r", offset=e.start, shape=(e.numel,))[::stride]).astype(np.float32)


def delta_sketch(model_dir: str, base_dir: str, stride: int = STRIDE) -> dict[str, np.ndarray]:
    """Sampled weight delta (model - declared base) per float tensor present in both with equal shape and dtype.

    Two sketches taken with the same stride estimate the cosine and relative L2 distance of the full deltas, so the
    near-duplicate check works against earlier submissions whose weights are no longer on disk."""
    model, base = read_model(model_dir), read_model(base_dir)
    out: dict[str, np.ndarray] = {}
    for name in sorted(model):
        em, eb = model[name], base.get(name)
        if eb is None or em.shape != eb.shape or em.dtype != eb.dtype or em.dtype not in FLOAT_DTYPES:
            continue
        out[name] = _sample_float(em, stride).astype(np.float64) - _sample_float(eb, stride)
    return out


@dataclass(frozen=True)
class DeltaStats:
    cosine: float          # cosine between the two weight deltas (each taken from the declared base)
    relative_l2: float     # ||dA - dB|| / ||dA||
    compared: int          # tensors compared
    norm_a: float          # ||dA||; 0 means A equals the base
    norm_b: float

    def derived(self, cos_min: float = 0.98, rel_max: float = 0.05) -> bool:
        """docs/SPECIALIST-TRACKS.md: B is derived from A when cos > 0.98 and the relative L2 distance < 0.05."""
        return self.compared > 0 and self.norm_a > 0 and self.cosine > cos_min and self.relative_l2 < rel_max


def compare_sketches(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> DeltaStats:
    dot = na = nb = nd = 0.0
    compared = 0
    for name in sorted(set(a) & set(b)):
        da, db = a[name], b[name]
        if da.shape != db.shape:
            continue
        compared += 1
        dot += float(da @ db)
        na += float(da @ da)
        nb += float(db @ db)
        diff = da - db
        nd += float(diff @ diff)
    norm_a, norm_b = math.sqrt(na), math.sqrt(nb)
    cos = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    rel = math.sqrt(nd) / norm_a if norm_a > 0 else float("inf")
    return DeltaStats(cosine=cos, relative_l2=rel, compared=compared, norm_a=norm_a, norm_b=norm_b)


def save_sketch(sketch: dict[str, np.ndarray], path: str) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k.replace("/", "|"): v.astype(np.float32) for k, v in sketch.items()})
    os.replace(tmp, path)


def load_sketch(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k.replace("|", "/"): z[k].astype(np.float64) for k in z.files}


BaseProvider = Callable[[str], "str | None"]   # base_model "<hf id>@<rev>" -> local directory, or None if unavailable
