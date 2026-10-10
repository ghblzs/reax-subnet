"""CPU-only Plumb token accounting, matching pinned jevk5 0.2's encode.

No model/weights are loaded. The template below is from jevk5 runtime.py at
85238d7be5527370c43206fe54cd752eb3134c1b (see manifest and THIRD_PARTY_NOTICES).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .manifest import load_manifest

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def option_texts(question):
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [f"{k}: {d}" for k, d in pairs]


def encode(tokenizer, state, instructions, options):
    payload = {
        "evidence": state,
        "criterion": instructions,
        "options": [{"letter": "ABCDEFGHIJKLMNOP"[i], "description": d} for i, d in enumerate(options)],
    }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(prompt, add_special_tokens=False)


class InputTokenLimitError(ValueError):
    """A valid tokenizer counted more tokens than the model permits."""


class PlumbTokenCounter:
    models = ("s1-fast",)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, directory):
        # Offline only. Mount tokenizer artifacts from the SAME pinned model as
        # the reference runtime, including tokenizer_config/chat_template.
        from transformers import AutoTokenizer

        manifest = load_manifest("s1-fast")
        path = Path(directory) / "tokenizer.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest.data["weights"]["files"]["tokenizer.json"]:
            raise ValueError("tokenizer does not match the pinned Plumb manifest")
        return cls(AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False))

    def __call__(self, model, decisions):
        if model != "s1-fast":
            raise RuntimeError("no proven token counter for this model")
        counts = [len(encode(self.tokenizer, d.state, d.question.instructions,
                             option_texts(d.question.author_dict()))) for d in decisions]
        if any(n <= 0 for n in counts):
            raise RuntimeError("token counting failed")
        if any(n > load_manifest(model).data["limits"]["max_input_tokens"] for n in counts):
            raise InputTokenLimitError("input exceeds model token limit")
        return sum(counts)
