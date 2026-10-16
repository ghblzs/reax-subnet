"""Test helpers for the specialist track: tiny safetensors writer, fake model repos, fake fetcher/backend, synthetic pool."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from contextlib import asynccontextmanager

import numpy as np

from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.spec import fetch as F
from reax_subnet.spec import rules as R
from reax_subnet.spec.commitment import SpecCommitment

HOTKEYS = [f"5Hot{i:02d}" + "x" * 40 for i in range(20)]
REV = "a" * 40


def _encode(a: np.ndarray, dtype: str) -> bytes:
    if dtype == "F32":
        return a.tobytes()
    if dtype == "BF16":    # round to nearest even, as torch's .to(torch.bfloat16)
        u = a.view(np.uint32).astype(np.uint64)
        return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype("<u2").tobytes()
    if dtype == "F16":
        return a.astype("<f2").tobytes()
    if dtype == "I32":     # int-quantized placeholder (GPTQ/AWQ qweight): the sketch never decodes it
        return np.clip(a, -2**31, 2**31 - 1).astype("<i4").tobytes()
    if dtype.startswith("F8_"):
        return np.clip(a, -127, 127).astype(np.int8).tobytes()    # placeholder bytes: the sketch never decodes FP8
    raise ValueError(dtype)


def write_safetensors(path: str, tensors: dict[str, np.ndarray], dtype: "str | dict[str, str]" = "F32") -> None:
    """``dtype``: one dtype for every tensor, or a per-tensor map (missing names are F32)."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        dt = dtype if isinstance(dtype, str) else dtype.get(name, "F32")
        raw = _encode(a, dt)
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_weights(seed: int, layers: int = 4, size: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {f"model.layers.{i}.mlp.weight": rng.randn(size, size).astype(np.float32) for i in range(layers)}


def finetune(base: dict[str, np.ndarray], seed: int, scale: float = 0.05) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {k: v + scale * rng.randn(*v.shape).astype(np.float32) for k, v in base.items()}


def write_model(path: str, weights: dict[str, np.ndarray], *, shards: int = 1, hotkey: str = HOTKEYS[0], category: str = "demand_forecasting",
                licence: str = "apache-2.0", arch: str = "Qwen3ForCausalLM", base_model: str = "Qwen/Qwen3-4B@" + "b" * 40,
                extra_files: dict[str, bytes] | None = None, config_extra: dict | None = None, dtype: "str | dict[str, str]" = "F32",
                skill_key: str = "") -> str:
    os.makedirs(path, exist_ok=True)
    names = sorted(weights)
    for i in range(shards):
        part = {n: weights[n] for n in names[i::shards]}
        if part:
            write_safetensors(os.path.join(path, f"model-{i:05d}-of-{shards:05d}.safetensors"), part, dtype)
    cfg = {"architectures": [arch], "model_type": "qwen3"}
    cfg.update(config_extra or {})
    files = {
        "config.json": json.dumps(cfg).encode(),
        "tokenizer_config.json": json.dumps({"chat_template": "{{ messages }}"}).encode(),
        "tokenizer.json": json.dumps({"added_tokens": [{"content": t} for t in ("<|im_start|>", "<|im_end|>")]}).encode(),
        "LICENSE": b"Apache License 2.0",
        R.MANIFEST_NAME: json.dumps({"schema": R.MANIFEST_SCHEMA, "category": category, "hotkey": hotkey,
                                     "base_model": base_model, "licence": licence}).encode(),
        "skill_key.txt": skill_key.encode(),
    }
    files.update(extra_files or {})
    for n, data in files.items():
        with open(os.path.join(path, n), "wb") as fh:
            fh.write(data)
    return path


def neuron(uid: int, commitment: str = "", coldkey: str | None = None, hotkey: str | None = None) -> Neuron:
    return Neuron(uid=uid, hotkey=hotkey or HOTKEYS[uid], coldkey=coldkey or f"5Cold{uid:02d}" + "y" * 40, validator_permit=False,
                  stake=0.0, ip=None, port=None, commitment=commitment)


def snapshot(block: int, neurons: list[Neuron]) -> Snapshot:
    return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey=None)


def commit(track: str, repo: str, rev: str = REV) -> str:
    return SpecCommitment(track, repo, rev).text()


def synthetic_pool(n: int = 900) -> list[dict]:
    """Rows shaped like the sealed pool: three primitives, four tiers, gold in `expected`."""
    rows = []
    tiers = ["proc_sealed", "proc_sealed", "same_family", "new_domain", "new_family"]
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        base = {"id": f"row-{i:04d}", "category": "demand_forecasting", "tier": tiers[i % len(tiers)], "state": {"i": i},
                "question": {"type": kind, "instructions": f"decide {i}"}}
        if kind == "choice":
            base.update(labels=["a", "b", "c", "d"], expected="abcd"[i % 4])
            base["question"]["criteria"] = {"a": "A", "b": "B", "c": "C", "d": "D"}
        elif kind == "noul":
            base.update(labels=["no", "yes"], expected="yes" if i % 2 else "no")
            base["question"]["criteria"] = {"true": "yes", "false": "no"}
        else:
            base.update(labels=["0", "1", "2", "3"], expected=i % 4)
            base["question"]["criteria"] = ["l0", "l1", "l2", "l3"]
        rows.append(base)
    return rows


def write_pool(root: str, category: str, rows: list[dict]) -> None:
    d = os.path.join(root, category, "sealed")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sealed.clean.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class FakeHub:
    """Maps (repo, revision) to a prepared model directory (or an error); a fake of HfFetcher."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.repos: dict[str, str] = {}
        self.unavailable: set[str] = set()
        self.fetches: list[str] = []

    def add(self, repo: str, **kw) -> str:
        path = write_model(os.path.join(self.workdir, "hub", repo.replace("/", "__")), skill_key=repo, **kw)
        self.repos[repo] = path
        return path

    async def fetch(self, commitment: SpecCommitment, dest: str, *, category: str, hotkey: str, rules=R.TrackRules()) -> F.Inspected:
        self.fetches.append(commitment.repo)
        if commitment.repo in self.unavailable or commitment.repo not in self.repos:
            raise F.FetchError("unavailable", "not public yet")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(self.repos[commitment.repo], dest)
        ins = F.inspect_dir(dest, category=category, hotkey=hotkey, rules=rules)
        if ins.violations:
            shutil.rmtree(dest, ignore_errors=True)
            raise F.FetchError("rejected", "; ".join(ins.violations))
        return ins


class FakeServer:
    """Backend factory: a model answers a row correctly iff hash(repo, row id) < skill[repo]; wrong answers are confident."""

    def __init__(self, skills: dict[str, float]):
        self.skills = skills
        self.served: list[str] = []

    def __call__(self, model_dir: str, weight_bytes: int):
        repo = open(os.path.join(model_dir, "skill_key.txt")).read()
        self.served.append(repo)
        skill = self.skills

        class B:
            async def decide(self, rows):
                out = []
                for r in rows:
                    h = int(hashlib.sha256(f"{repo}|{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    labels = [str(x) for x in r["labels"]]
                    gold = str(r["expected"])
                    wrong = next(lab for lab in labels if lab != gold)
                    pick = gold if h < skill[repo] else wrong
                    out.append({lab: (0.85 if lab == pick else 0.15 / (len(labels) - 1)) for lab in labels})
                return out

        @asynccontextmanager
        async def ctx():
            yield B()

        return ctx()


class Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
