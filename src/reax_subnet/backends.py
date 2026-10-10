"""Model backends that turn items into probability distributions.

Miners and validators run the *same* backend code against the *same* pinned
model. Probabilities come from one forward pass (``label_logits_v1``): the
prompt lists the options with letter labels and the distribution is the
float32 softmax over the logits of those letter tokens at the answer
position, divided by the model's protocol temperature T. Nothing is sampled, so
generation settings and seeds play no role.
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence

from .protocol import MAX_OPTIONS, Item

LABELS = "ABCDEFGH"
assert len(LABELS) >= MAX_OPTIONS

SYSTEM_PROMPT = (
    "You are a decision model. Read the question and the lettered options, "
    "then answer with the single letter of the best option."
)
PROMPT_TEMPLATE_ID = "label_logits_v1/2"
# Probabilities are softmax(logits / T) with a per-model protocol temperature T from
# the catalog. The approved model is very confident (mean top probability 0.92 at
# T=1), so at T=1 knowing only its top choice matched 80 % of items; at T=4 the logit
# gaps must be reproduced, and honest numeric drift shrinks by the same factor. A less
# confident model needs a lower T, otherwise the uniform distribution starts to pass.


def render_user_prompt(item: Item) -> str:
    lines = [item.question.strip(), "", "Options:"]
    lines += [f"{LABELS[i]}. {opt.strip()}" for i, opt in enumerate(item.options)]
    lines += ["", "Answer with a single letter."]
    return "\n".join(lines)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    m = max(logits)
    exps = [math.exp((x - m) / temperature) for x in logits]
    total = math.fsum(exps)
    return tuple(e / total for e in exps)


class Backend(Protocol):
    model_alias: str

    def predict(self, items: Sequence[Item]) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_id: str
    revision: str
    production: bool
    temperature: float = 1.0
    note: str = ""


# The closed model catalog. Only ``production=True`` entries are accepted on
# mainnet; the others exist for local tests and are refused unless
# REAX_ALLOW_DEV_MODELS=1.
CATALOG: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in (
        ModelSpec(
            "qwen3-4b-instruct-2507",
            "Qwen/Qwen3-4B-Instruct-2507",
            "cdbee75f17c01a7cc42f958dc650907174af0554",
            production=True,
            temperature=4.0,
            note="Apache-2.0; about 8 GB in bf16; T=4 measured (docs/MPM.md)",
        ),
        ModelSpec(
            "qwen2.5-0.5b-instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "7ae557604adf67be50417f59c2c2f167def9a775",
            production=False,
            temperature=1.0,
            note="small CPU model for local end-to-end tests; T=1 measured",
        ),
        ModelSpec("hash-test", "", "", production=False, note="deterministic pseudo-model for unit tests"),
    )
}


class HashBackend:
    """Deterministic pseudo-model for tests. ``salt`` plays the role of the weights."""

    def __init__(self, model_alias: str = "hash-test", salt: str = "reference", temperature: float = 1.5):
        self.model_alias = model_alias
        self.salt = salt
        self.temperature = temperature

    def predict(self, items: Sequence[Item]) -> list[tuple[float, ...]]:
        out = []
        for item in items:
            logits = []
            for opt in item.options:
                digest = hashlib.sha256(f"{self.salt}\x00{item.question}\x00{opt}".encode()).digest()
                logits.append(int.from_bytes(digest[:4], "big") / 2**32 * 4.0 / self.temperature)
            out.append(softmax(logits))
        return out


class HFBackend:
    """Hugging Face transformers backend (label_logits_v1)."""

    def __init__(self, spec: ModelSpec, device: str = "cuda", dtype: str = "bfloat16", max_batch: int = 8):
        import torch  # imported lazily: the validator/miner core does not need torch for tests
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not spec.hf_id:
            raise ValueError(f"model {spec.alias} has no Hugging Face weights")
        self.model_alias = spec.alias
        self.temperature = spec.temperature
        self._torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Never fall back to CPU silently: CPU answers are slow and, in another
            # dtype, numerically different from GPU miners.
            raise RuntimeError(
                "REAX_DEVICE is cuda but no CUDA device is usable (check the NVIDIA driver; the pinned "
                "torch build needs a recent driver). Set REAX_DEVICE=cpu only for tests."
            )
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, revision=spec.revision, dtype=getattr(torch, dtype), use_safetensors=True
        ).to(device)
        self.model.eval()
        self.label_ids = []
        for label in LABELS:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"label {label!r} is not a single token for {spec.hf_id}")
            self.label_ids.append(ids[0])
        # One forward pass at a time per process keeps memory bounded and
        # results independent of concurrent requests.
        self._lock = threading.Lock()

    def _prompt(self, item: Item) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(item)},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self, items: Sequence[Item]) -> list[tuple[float, ...]]:
        torch = self._torch
        out: list[tuple[float, ...]] = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(items), self.max_batch):
                chunk = items[start : start + self.max_batch]
                enc = self.tokenizer(
                    [self._prompt(it) for it in chunk], return_tensors="pt", padding=True
                ).to(self.device)
                logits = self.model(**enc).logits[:, -1, :].float()
                for row, item in zip(logits, chunk):
                    label_logits = row[self.label_ids[: len(item.options)]].tolist()
                    out.append(softmax(label_logits, self.temperature))
        return out


def resolve_model(alias: str, allow_dev: bool) -> ModelSpec:
    spec = CATALOG.get(alias)
    if spec is None:
        raise ValueError(f"model {alias!r} is not in the catalog: {sorted(CATALOG)}")
    if not spec.production and not allow_dev:
        raise ValueError(f"model {alias!r} is a development model; set REAX_ALLOW_DEV_MODELS=1 for local tests")
    return spec


def load_backend(alias: str, allow_dev: bool, device: str = "cuda", dtype: str = "bfloat16") -> Backend:
    spec = resolve_model(alias, allow_dev)
    if alias == "hash-test":
        return HashBackend(alias)
    return HFBackend(spec, device=device, dtype=dtype)
