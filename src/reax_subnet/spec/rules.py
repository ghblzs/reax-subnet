"""Submission rules for the specialist track (docs/SPECIALIST-TRACKS.md).

Everything here is a pure check on facts that were read from the repository without executing any of its code:
file names and sizes, ``config.json`` and ``reax-specialist.json`` as data, and the safetensors headers. The result
is a list of violations; an empty list means the model may be served.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .templates import TEMPLATE_MARKERS, template_name

GIB = 1024 ** 3
MAX_PARAMS = 9_000_000_000
MAX_WEIGHT_BYTES = 20 * GIB          # a 9 B model in bf16 is ~18 GB; fp32 (36 GB) is refused
MAX_TOTAL_BYTES = 22 * GIB           # everything the validator downloads for one submission
MAX_FILES = 256
MAX_JSON_BYTES = 16 * 1024 * 1024    # config / manifest / tokenizer_config json
MAX_TOKENIZER_JSON_BYTES = 64 * 1024 * 1024    # tokenizer.json of large vocabularies (Qwen3.5 ~20 MB, Gemma ~33 MB)
MAX_CHAT_TEMPLATE_CHARS = 16 * 1024
MANIFEST_NAME = "reax-specialist.json"
MANIFEST_SCHEMA = "reax-specialist/1"

# Licences we may host and serve commercially. Anything else needs an owner decision per licence (add it to
# ``approved_extra`` of the TrackRules the validator runs with); the default is deliberately short.
LICENCE_ALLOWLIST = ("apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause")

# Architectures the pinned vLLM image (runtimes/s1-pro, vLLM 0.30.0) serves. CHECK on a GPU host against the
# image's model registry before launch; unknown names are refused, never guessed.
# Every name is in vLLM 0.30.0's model registry (checked in the pinned image); Qwen3_5ForCausalLM, Qwen3ForCausalLM and
# Qwen3_5ForConditionalGeneration were also served and scored on a GPU.
ARCH_ALLOWLIST = (
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",   # e.g. h2oai/h2o-lightning-4b; served text-only by vLLM 0.30.0 (checked on a GPU)
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "LlamaForCausalLM",
)

# Weight formats that execute code on load (pickle) or that the pinned image should not be asked to read.
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".h5", ".msgpack", ".gguf", ".onnx", ".npz", ".npy")
LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.txt")
# What is downloaded: weights, json (config, tokenizer, index, manifest), tokenizer models, templates, licence text.
# Chat templates (``*.jinja``, ``chat_template.json``) are never used (spec/templates.py); ``*.jinja`` is not even downloaded.
DOWNLOAD_PATTERNS = ("*.safetensors", "*.json", "*.model", "*.tiktoken", "LICENSE*", "COPYING*")


@dataclass(frozen=True)
class TrackRules:
    max_params: int = MAX_PARAMS
    max_weight_bytes: int = MAX_WEIGHT_BYTES
    max_total_bytes: int = MAX_TOTAL_BYTES
    licences: tuple[str, ...] = LICENCE_ALLOWLIST
    approved_extra: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ARCH_ALLOWLIST


@dataclass
class RepoFacts:
    """What the validator learned about a submission without running it."""

    files: Mapping[str, int]                       # relative path -> size in bytes
    config: Mapping[str, Any] | None = None        # parsed config.json
    manifest: Mapping[str, Any] | None = None      # parsed reax-specialist.json
    params: int | None = None                      # parameters counted from the safetensors headers
    weight_bytes: int | None = None
    chat_template_chars: int = 0
    tokenizer_tokens: frozenset[str] | None = None  # special/added tokens the tokenizer defines (None = not collected)
    problems: list[str] = field(default_factory=list)   # parse problems found while collecting facts


def _norm_licence(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def check_listing(files: Mapping[str, int], rules: TrackRules = TrackRules()) -> list[str]:
    """Checks on the file list alone, before anything is downloaded."""
    out: list[str] = []
    if len(files) > MAX_FILES:
        out.append(f"more than {MAX_FILES} files")
    for name in files:
        parts = name.split("/")
        if name.startswith("/") or ".." in parts:
            out.append(f"unsafe path {name!r}")
        elif name.lower().endswith(FORBIDDEN_SUFFIXES):
            out.append(f"{name}: only safetensors weights are accepted (no pickle or other formats)")
    weights = {n: s for n, s in files.items() if n.endswith(".safetensors")}
    if not weights:
        out.append("no .safetensors weights")
    total_w = sum(weights.values())
    if total_w > rules.max_weight_bytes:
        out.append(f"weights are {total_w / GIB:.1f} GiB, limit {rules.max_weight_bytes / GIB:.0f} GiB")
    # only what is downloaded counts against the total cap
    dl = sum(s for n, s in files.items() if _is_downloaded(n))
    if dl > rules.max_total_bytes:
        out.append(f"download is {dl / GIB:.1f} GiB, limit {rules.max_total_bytes / GIB:.0f} GiB")
    if not any(n in files for n in LICENSE_FILES):
        out.append("no LICENSE file")
    if MANIFEST_NAME not in files:
        out.append(f"no {MANIFEST_NAME}")
    return out


def _is_downloaded(name: str) -> bool:
    import fnmatch
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(base, p) for p in DOWNLOAD_PATTERNS)


def check_config(config: Mapping[str, Any] | None, rules: TrackRules = TrackRules()) -> list[str]:
    if not isinstance(config, Mapping):
        return ["config.json missing or not an object"]
    out: list[str] = []
    if "auto_map" in config:
        out.append("config.json has auto_map: custom model code (trust_remote_code) is not allowed")
    for sub in ("text_config", "vision_config"):
        inner = config.get(sub)
        if isinstance(inner, Mapping) and "auto_map" in inner:
            out.append(f"config.json {sub} has auto_map")
    archs = config.get("architectures")
    if not isinstance(archs, list) or len(archs) != 1 or not isinstance(archs[0], str):
        out.append("config.json must name exactly one architecture")
    elif archs[0] not in rules.architectures:
        out.append(f"architecture {archs[0]!r} is not on the allowlist")
    elif template_name(archs[0]) is None:
        out.append(f"architecture {archs[0]!r} has no fixed chat template in this release")
    return out


def check_tokenizer(config: Mapping[str, Any] | None, tokens: frozenset[str] | None) -> list[str]:
    """The fixed chat template of the architecture writes marker tokens (``<|im_start|>``...). If the tokenizer does not
    define them, the markers would be tokenized as plain text and the model scored on a prompt it never saw: refused
    before any GPU run, with the reason."""
    if tokens is None or not isinstance(config, Mapping):
        return []
    archs = config.get("architectures")
    tpl = template_name(archs[0]) if isinstance(archs, list) and len(archs) == 1 and isinstance(archs[0], str) else None
    missing = [m for m in TEMPLATE_MARKERS.get(tpl or "", ()) if m not in tokens]
    if not missing:
        return []
    return [f"tokenizer does not define {', '.join(missing)}, which the fixed chat template of {archs[0]} uses "
            "(this checkpoint is trained for another chat format)"]


def check_manifest(manifest: Mapping[str, Any] | None, *, category: str, hotkey: str | None,
                   rules: TrackRules = TrackRules()) -> list[str]:
    """``reax-specialist.json``: binds the model to its track, its committing hotkey and a licence."""
    if not isinstance(manifest, Mapping):
        return [f"{MANIFEST_NAME} missing or not an object"]
    out: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        out.append(f"{MANIFEST_NAME}: schema must be {MANIFEST_SCHEMA!r}")
    if manifest.get("category") != category:
        out.append(f"{MANIFEST_NAME}: category must be {category!r}")
    if hotkey is not None and manifest.get("hotkey") != hotkey:
        out.append(f"{MANIFEST_NAME}: hotkey must be the committing hotkey")
    base = manifest.get("base_model")
    if not isinstance(base, str) or "@" not in base or "/" not in base.split("@", 1)[0]:
        out.append(f"{MANIFEST_NAME}: base_model must be '<hf id>@<revision>'")
    licence = _norm_licence(manifest.get("licence"))
    if licence not in rules.licences and licence not in rules.approved_extra:
        out.append(f"{MANIFEST_NAME}: licence {manifest.get('licence')!r} is not on the allowlist")
    return out


def check_facts(facts: RepoFacts, *, category: str, hotkey: str | None, rules: TrackRules = TrackRules()) -> list[str]:
    """All rules, in a stable order. An empty list means the submission is acceptable."""
    out = list(facts.problems)
    out += check_listing(facts.files, rules)
    out += check_config(facts.config, rules)
    out += check_tokenizer(facts.config, facts.tokenizer_tokens)
    out += check_manifest(facts.manifest, category=category, hotkey=hotkey, rules=rules)
    if facts.params is None:
        out.append("parameter count unknown (unreadable safetensors headers)")
    elif facts.params > rules.max_params:
        out.append(f"{facts.params / 1e9:.2f} B parameters, limit {rules.max_params / 1e9:.0f} B")
    if facts.weight_bytes is not None and facts.weight_bytes > rules.max_weight_bytes:
        out.append("weight bytes above the limit")
    if facts.chat_template_chars > MAX_CHAT_TEMPLATE_CHARS:
        out.append("chat template above the size limit")
    return out


def read_json_file(path: str, limit: int = MAX_JSON_BYTES) -> Any:
    """Parse a json file as data; refuse oversized files."""
    if os.path.getsize(path) > limit:
        raise ValueError(f"{os.path.basename(path)} is larger than {limit} bytes")
    with open(path, "rb") as fh:
        return json.loads(fh.read())
