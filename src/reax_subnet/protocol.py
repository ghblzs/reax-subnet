"""Wire format for ``reax-mpm/1``.

A validator sends a batch of finite-choice decision items to a miner. The miner
returns one probability distribution per item. Everything here is strict: an
unknown field, a wrong type or an out-of-range value is a protocol error, and
the validator scores such a response as zero for every item in the batch.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .version import PROTOCOL

MIN_OPTIONS = 2
MAX_OPTIONS = 8
MAX_ITEMS = 16
MAX_QUESTION_CHARS = 2_000
MAX_OPTION_CHARS = 200
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
# A distribution whose sum is further than this from 1 is rejected outright.
SUM_TOLERANCE = 1e-3

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ProtocolError(ValueError):
    """The payload violates ``reax-mpm/1``."""


@dataclass(frozen=True)
class Item:
    item_id: str
    question: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class DecideRequest:
    request_id: str
    model: str
    items: tuple[Item, ...]

    def to_bytes(self) -> bytes:
        return _dumps(
            {
                "protocol": PROTOCOL,
                "request_id": self.request_id,
                "model": self.model,
                "items": [
                    {"id": it.item_id, "question": it.question, "options": list(it.options)}
                    for it in self.items
                ],
            }
        )


@dataclass(frozen=True)
class DecideResponse:
    request_id: str
    results: tuple[tuple[str, tuple[float, ...]], ...]
    miner_version: str

    def to_bytes(self) -> bytes:
        return _dumps(
            {
                "protocol": PROTOCOL,
                "request_id": self.request_id,
                "miner_version": self.miner_version,
                "results": [{"id": i, "probs": list(p)} for i, p in self.results],
            }
        )


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _loads(raw: bytes, limit: int) -> Any:
    if len(raw) > limit:
        raise ProtocolError("payload too large")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        raise ProtocolError("payload is not valid UTF-8 JSON") from exc


def _reject_constant(name: str) -> Any:
    raise ProtocolError(f"non-finite number {name} is not allowed")


def _exact_keys(obj: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ProtocolError(f"{what} must be an object")
    if set(obj) != keys:
        raise ProtocolError(f"{what} must have exactly the fields {sorted(keys)}")
    return obj


def _text(value: Any, limit: int, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ProtocolError(f"{what} must be a non-empty string of at most {limit} characters")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
        raise ProtocolError(f"{what} contains control characters")
    return value


def parse_request(raw: bytes) -> DecideRequest:
    obj = _exact_keys(_loads(raw, MAX_REQUEST_BYTES), {"protocol", "request_id", "model", "items"}, "request")
    if obj["protocol"] != PROTOCOL:
        raise ProtocolError(f"unsupported protocol {obj['protocol']!r}; this miner speaks {PROTOCOL}")
    request_id = obj["request_id"]
    if not isinstance(request_id, str) or not _ID_RE.fullmatch(request_id):
        raise ProtocolError("request_id is malformed")
    model = obj["model"]
    if not isinstance(model, str) or not _MODEL_RE.fullmatch(model):
        raise ProtocolError("model is malformed")
    raw_items = obj["items"]
    if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= MAX_ITEMS:
        raise ProtocolError(f"items must be a list of 1..{MAX_ITEMS}")
    items: list[Item] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        it = _exact_keys(raw_item, {"id", "question", "options"}, "item")
        item_id = it["id"]
        if not isinstance(item_id, str) or not _ID_RE.fullmatch(item_id) or item_id in seen:
            raise ProtocolError("item id is malformed or duplicated")
        seen.add(item_id)
        question = _text(it["question"], MAX_QUESTION_CHARS, "question")
        options = it["options"]
        if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            raise ProtocolError(f"options must be a list of {MIN_OPTIONS}..{MAX_OPTIONS}")
        texts = tuple(_text(o, MAX_OPTION_CHARS, "option") for o in options)
        if len(set(texts)) != len(texts):
            raise ProtocolError("options must be unique")
        items.append(Item(item_id, question, texts))
    return DecideRequest(request_id, model, tuple(items))


def parse_response(raw: bytes, request: DecideRequest) -> dict[str, tuple[float, ...]]:
    """Validate a miner response against the request it answers.

    Returns ``{item_id: probabilities}`` with every distribution renormalised
    to sum to exactly 1. Raises :class:`ProtocolError` for anything else,
    including a response to a different request or a missing item.
    """
    obj = _exact_keys(
        _loads(raw, MAX_RESPONSE_BYTES), {"protocol", "request_id", "miner_version", "results"}, "response"
    )
    if obj["protocol"] != PROTOCOL:
        raise ProtocolError("wrong protocol")
    if obj["request_id"] != request.request_id:
        raise ProtocolError("response answers a different request")
    if not isinstance(obj["miner_version"], str) or len(obj["miner_version"]) > 32:
        raise ProtocolError("miner_version is malformed")
    results = obj["results"]
    if not isinstance(results, list) or len(results) != len(request.items):
        raise ProtocolError("results must contain exactly one entry per item")
    expected = {it.item_id: len(it.options) for it in request.items}
    out: dict[str, tuple[float, ...]] = {}
    for raw_result in results:
        r = _exact_keys(raw_result, {"id", "probs"}, "result")
        item_id = r["id"]
        if not isinstance(item_id, str) or item_id not in expected or item_id in out:
            raise ProtocolError("result id is unknown or duplicated")
        out[item_id] = normalise(r["probs"], expected[item_id])
    return out


def normalise(probs: Any, n: int) -> tuple[float, ...]:
    """Check a probability vector of length ``n`` and renormalise it."""
    if not isinstance(probs, list) or len(probs) != n:
        raise ProtocolError(f"probs must be a list of exactly {n} numbers")
    values: list[float] = []
    for p in probs:
        # bool is an int subclass; reject it explicitly.
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ProtocolError("probabilities must be numbers")
        try:
            f = float(p)
        except (OverflowError, ValueError):
            raise ProtocolError("probabilities must be finite numbers") from None
        if not math.isfinite(f) or f < 0.0 or f > 1.0:
            raise ProtocolError("probabilities must be finite and within [0, 1]")
        values.append(f)
    total = math.fsum(values)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ProtocolError("probabilities must sum to 1")
    return tuple(v / total for v in values)
