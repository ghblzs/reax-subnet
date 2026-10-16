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
FP8_DTYPES = ("F8_E4M3", "F8_E5M2")    # need per-tensor scales to decode: never compared, they count as not covered


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


STRIDE = 4093   # sampling stride of the delta sketch: one value per STRIDE (~2.2 M values for a 9 B model)
MIN_SAMPLES = 64   # small tensors (norms, biases) still get this many samples (all of them below 64 values)
PEFT_PREFIX = "base_model.model."   # an unmerged-PEFT style export prefixes every tensor name with this
INT_DTYPES = ("I64", "I32", "I16", "I8", "U8", "U16", "U32", "U64")   # int-quantized weights (GPTQ, AWQ, ...): not compared
SEED_ENV = "REAX_SPEC_SKETCH_SEED"
SEED_FILE = "sketch-seed"


def load_sketch_seed(work_dir: str, env: "dict[str, str] | None" = None) -> bytes:
    """The validator's secret sketch seed: ``REAX_SPEC_SKETCH_SEED`` or ``<work_dir>/sketch-seed`` (32 random bytes,
    created once, mode 0600). It decides which values are sampled, so a copier cannot tell which weights to change. It
    must stay the same for the life of the validator (stored sketches are only comparable under the same seed) and it
    is never logged or published; only ``seed_id`` is."""
    env = os.environ if env is None else env
    text = env.get(SEED_ENV, "").strip()
    if text:
        return hashlib.sha256(b"reax-sketch-seed|" + text.encode()).digest()
    path = os.path.join(work_dir, SEED_FILE)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if len(raw) >= 16:
            return raw
    except FileNotFoundError:
        pass
    os.makedirs(work_dir, exist_ok=True)
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(os.urandom(32))
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(path + ".tmp", path)    # never replaces a seed another process created meanwhile
    except FileExistsError:
        pass
    os.unlink(path + ".tmp")
    with open(path, "rb") as fh:
        return fh.read()


_PROCESS_SEED = os.urandom(32)   # default for callers without a validator seed (tests, the miner CLI): secret per process


def seed_id(seed: bytes) -> str:
    """Public id of a seed and of the sampling rule: stored with each sketch, never reveals the positions."""
    return hashlib.sha256(b"reax-sketch-id|%d|" % SKETCH_RULE + seed).hexdigest()[:16]


def canonical_name(name: str) -> str:
    """The tensor name the base model uses: a ``base_model.model.`` prefix (PEFT export) is dropped."""
    return name[len(PEFT_PREFIX):] if name.startswith(PEFT_PREFIX) else name


def _canonical(tensors: dict[str, TensorEntry]) -> dict[str, TensorEntry]:
    out: dict[str, TensorEntry] = {}
    for name in sorted(tensors, key=lambda n: (n.startswith(PEFT_PREFIX), n)):   # an unprefixed name wins a collision
        out.setdefault(canonical_name(name), tensors[name])
    return out


def sample_positions(seed: bytes, name: str, numel: int, stride: int = STRIDE) -> np.ndarray:
    """Sorted, distinct element indices for one tensor: one random index in each of ``n`` equal blocks (stratified, so
    the whole tensor is covered evenly), ``n = max(ceil(numel / stride), MIN_SAMPLES)`` capped at ``numel``. The draw
    depends on the secret seed, the tensor name and its size only, so the model and its base (and any two models of the
    same base) are sampled at the same positions."""
    if numel <= 0:
        return np.zeros(0, dtype=np.int64)
    n = min(numel, max(-(-numel // stride), MIN_SAMPLES))
    key = hashlib.sha256(seed + b"|" + name.encode() + b"|%d|%d" % (numel, stride)).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "big")))
    lo = (np.arange(n, dtype=np.int64) * numel) // n
    hi = (np.arange(1, n + 1, dtype=np.int64) * numel) // n
    return lo + (rng.random(n) * (hi - lo)).astype(np.int64)


def _sample_float(e: TensorEntry, stride: int, idx: "np.ndarray | None" = None) -> np.ndarray:
    """Values of a float tensor as float32 at ``idx`` (default: every ``stride``-th element). Memory-mapped: only the
    touched pages are read."""
    if e.numel == 0:
        return np.zeros(0, dtype=np.float32)
    sel = slice(None, None, stride) if idx is None else idx
    if e.dtype == "BF16":
        raw = np.memmap(e.path, dtype="<u2", mode="r", offset=e.start, shape=(e.numel,))[sel]
        return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)
    dt = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}[e.dtype]
    return np.asarray(np.memmap(e.path, dtype=dt, mode="r", offset=e.start, shape=(e.numel,))[sel]).astype(np.float32)


def as_dtype(x: np.ndarray, dtype: str) -> np.ndarray:
    """``x`` rounded to the values ``dtype`` can hold (round to nearest even, as torch's ``.to()``), as float32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        u = x.view(np.uint32).astype(np.uint64)
        r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
        nan = np.isnan(x)
        out = (r & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
        return np.where(nan, x, out)
    if dtype == "F16":
        return x.astype(np.float16).astype(np.float32)
    return x


@dataclass(frozen=True)
class Sketch:
    deltas: dict[str, np.ndarray]   # per compared tensor: sampled model values rounded to the base dtype - base values
    changed: bool                   # any sampled value differs from the base before the rounding
    seed_id: str = ""


def delta_sketch_ex(model_dir: str, base_dir: str, stride: int = STRIDE, seed: "bytes | None" = None) -> Sketch:
    """Sampled weight delta (model - declared base) per float tensor present in both with equal shape (a PEFT-style
    ``base_model.model.`` name prefix is ignored).

    The dtype does not matter, in either direction: both sides are decoded, and the model's values are rounded to the
    base tensor's dtype before the difference. So a fine-tune saved in F32 or F16 of a BF16 base is compared like one
    saved in BF16, and an F32 parent rounded to BF16 by a copier has exactly the delta of the parent rounded the same
    way. The values are sampled at positions only the seed's owner knows (``sample_positions``).
    Two sketches taken with the same seed and stride estimate the cosine and relative L2 distance of the full deltas, so
    the near-duplicate check works against earlier submissions whose weights are no longer on disk."""
    seed = _PROCESS_SEED if seed is None else seed
    model, base = _canonical(read_model(model_dir)), read_model(base_dir)
    out: dict[str, np.ndarray] = {}
    changed = False
    for name in sorted(model):
        em, eb = model[name], base.get(name)
        if eb is None or em.shape != eb.shape or em.dtype not in FLOAT_DTYPES or eb.dtype not in FLOAT_DTYPES:
            continue
        idx = sample_positions(seed, name, eb.numel, stride)
        vm, vb = _sample_float(em, stride, idx), _sample_float(eb, stride, idx)
        changed = changed or bool(np.any(vm != vb))
        # a small tensor sampled more densely than 1/stride is down-weighted, so that every tensor holds the share of the
        # sketch its elements hold of the full delta (cosine and L2 stay estimates of the full deltas)
        w = 1.0 if len(idx) <= -(-eb.numel // stride) else math.sqrt(eb.numel / (len(idx) * stride))
        out[name] = (as_dtype(vm, eb.dtype).astype(np.float64) - vb) * w
    return Sketch(out, changed, seed_id(seed))


def delta_sketch(model_dir: str, base_dir: str, stride: int = STRIDE, seed: "bytes | None" = None) -> dict[str, np.ndarray]:
    return delta_sketch_ex(model_dir, base_dir, stride, seed).deltas


MIN_SKETCH_COVERAGE = 0.90   # share of the base's float parameters the compared tensors must hold
SKETCH_RULE = 3              # 1: same dtype only (rc5); 2: any float dtype + coverage; 3: rounded to the base dtype, secret
                             # positions (rc6). A pending verdict of an older rule is re-checked, an older sketch recomputed


@dataclass(frozen=True)
class Coverage:
    covered: int                  # float parameters of the base in the compared tensors
    total: int                    # float parameters of the base (FP8 included)
    fp8: tuple[str, ...] = ()     # tensors of the model (or the base) that are FP8 and therefore not compared
    quantized: tuple[str, ...] = ()   # int tensors of the model (GPTQ/AWQ qweight, ...) the base holds as float

    @property
    def fraction(self) -> float:
        return self.covered / self.total if self.total else 0.0

    def enough(self, minimum: float = MIN_SKETCH_COVERAGE) -> bool:
        return self.total > 0 and self.fraction >= minimum


def sketch_coverage(sketch: dict[str, np.ndarray], model_dir: str, base_dir: str) -> Coverage:
    """How much of the declared base the sketch compares (tensor headers only, no weights read)."""
    model, base = _canonical(read_model(model_dir)), read_model(base_dir)
    floats = {n: e for n, e in base.items() if e.dtype in FLOAT_DTYPES + FP8_DTYPES}
    fp8 = sorted(n for n, e in model.items() if n in floats and (e.dtype in FP8_DTYPES or floats[n].dtype in FP8_DTYPES))
    quantized = sorted(n for n, e in model.items() if e.dtype in INT_DTYPES and (base.get(n) is None or base[n].dtype not in INT_DTYPES))
    return Coverage(covered=sum(floats[n].numel for n in sketch if n in floats),
                    total=sum(e.numel for e in floats.values()), fp8=tuple(fp8), quantized=tuple(quantized))


def coverage_note(cov: Coverage) -> str:
    """What the reason of a pending verdict adds about tensors that cannot be compared ("" = nothing to add)."""
    parts = []
    if cov.fp8:
        parts.append(f"{len(cov.fp8)} FP8 tensors cannot be compared")
    if cov.quantized:
        parts.append(f"{len(cov.quantized)} int-quantized tensors (GPTQ/AWQ style) cannot be compared")
    return f"; {' and '.join(parts)}, save the weights in the base model's dtype" if parts else ""


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


SEED_KEY = "__seed_id__"


def save_sketch(sketch: dict[str, np.ndarray], path: str, seed_id: str = "") -> None:
    """Stored as float32, with the id of the seed it was sampled with (``sketch_seed_id``)."""
    tmp = path + ".tmp.npz"
    arrays = {k.replace("/", "|"): v.astype(np.float32) for k, v in sketch.items()}
    arrays[SEED_KEY] = np.frombuffer(seed_id.encode(), dtype=np.uint8)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_sketch(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k.replace("|", "/"): z[k].astype(np.float64) for k in z.files if k != SEED_KEY}


def sketch_seed_id(path: str) -> str:
    """The seed id a stored sketch was sampled with; "" for a sketch of an older release (fixed positions)."""
    with np.load(path, allow_pickle=False) as z:
        return bytes(z[SEED_KEY]).decode() if SEED_KEY in z.files else ""


BaseProvider = Callable[[str], "str | None"]   # base_model "<hf id>@<rev>" -> local directory, or None if unavailable
