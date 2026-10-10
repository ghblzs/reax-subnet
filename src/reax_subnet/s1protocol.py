"""Wire format for ``reax-s1/1``: batches of typed decisions.

A request carries up to ``MAX_DECISIONS`` decisions, each a typed-decision
body without the model field and with exactly one question
(see s1contract.py). The response carries the reference answer object for each
decision. Parsing is strict: unknown fields, duplicates, wrong types or a
response to another request are protocol errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .protocol import ProtocolError, _dumps, _exact_keys, _loads
from .s1contract import ContractError, Decision, check_answer, parse_decision

PROTOCOL_S1 = "reax-s1/1"
S1_DECIDE_PATH = "/v1/s1/decide"
MAX_DECISIONS = 16
MAX_REQUEST_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = 256 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MANIFEST_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class S1Request:
    request_id: str
    model: str
    manifest_id: str
    decisions: tuple[Decision, ...]

    def to_bytes(self) -> bytes:
        return _dumps(
            {
                "protocol": PROTOCOL_S1,
                "request_id": self.request_id,
                "model": self.model,
                "manifest_id": self.manifest_id,
                "decisions": [d.body() for d in self.decisions],
            }
        )


@dataclass(frozen=True)
class S1Response:
    request_id: str
    miner_version: str
    results: tuple[tuple[str, dict[str, Any]], ...]  # (decision id, {question id: answer})

    def to_bytes(self) -> bytes:
        return _dumps(
            {
                "protocol": PROTOCOL_S1,
                "request_id": self.request_id,
                "miner_version": self.miner_version,
                "results": [{"id": i, "answers": a} for i, a in self.results],
            }
        )


def parse_s1_request(raw: bytes) -> S1Request:
    obj = _exact_keys(
        _loads(raw, MAX_REQUEST_BYTES), {"protocol", "request_id", "model", "manifest_id", "decisions"}, "request"
    )
    if obj["protocol"] != PROTOCOL_S1:
        raise ProtocolError(f"unsupported protocol {obj['protocol']!r}; this miner speaks {PROTOCOL_S1}")
    if not isinstance(obj["request_id"], str) or not _ID_RE.fullmatch(obj["request_id"]):
        raise ProtocolError("request_id is malformed")
    model = obj["model"]
    if model not in ("s1-fast", "s1-pro"):
        raise ProtocolError("model must be s1-fast or s1-pro")
    if not isinstance(obj["manifest_id"], str) or not _MANIFEST_RE.fullmatch(obj["manifest_id"]):
        raise ProtocolError("manifest_id is malformed")
    raw_decisions = obj["decisions"]
    if not isinstance(raw_decisions, list) or not 1 <= len(raw_decisions) <= MAX_DECISIONS:
        raise ProtocolError(f"decisions must be a list of 1..{MAX_DECISIONS}")
    decisions: list[Decision] = []
    seen: set[str] = set()
    for rd in raw_decisions:
        if not isinstance(rd, dict) or not isinstance(rd.get("id"), str) or not _ID_RE.fullmatch(rd["id"]) or rd["id"] in seen:
            raise ProtocolError("decision id is malformed or duplicated")
        seen.add(rd["id"])
        try:
            decisions.append(parse_decision(rd, model))
        except ContractError as exc:
            raise ProtocolError(f"decision {rd['id']}: {exc}") from None
    return S1Request(obj["request_id"], model, obj["manifest_id"], tuple(decisions))


def parse_s1_response(raw: bytes, request: S1Request) -> dict[str, tuple[float, ...]]:
    """Validate a miner response; return ``{decision id: probabilities in option-key order}``."""
    obj = _exact_keys(
        _loads(raw, MAX_RESPONSE_BYTES), {"protocol", "request_id", "miner_version", "results"}, "response"
    )
    if obj["protocol"] != PROTOCOL_S1:
        raise ProtocolError("wrong protocol")
    if obj["request_id"] != request.request_id:
        raise ProtocolError("response answers a different request")
    if not isinstance(obj["miner_version"], str) or len(obj["miner_version"]) > 32:
        raise ProtocolError("miner_version is malformed")
    results = obj["results"]
    if not isinstance(results, list) or len(results) != len(request.decisions):
        raise ProtocolError("results must contain exactly one entry per decision")
    by_id = {d.ident: d for d in request.decisions}
    out: dict[str, tuple[float, ...]] = {}
    for r in results:
        r = _exact_keys(r, {"id", "answers"}, "result")
        dec = by_id.get(r["id"]) if isinstance(r["id"], str) else None
        if dec is None or r["id"] in out:
            raise ProtocolError("result id is unknown or duplicated")
        answers = r["answers"]
        if not isinstance(answers, dict) or set(answers) != {dec.question.ident}:
            raise ProtocolError("answers must contain exactly the asked question")
        try:
            out[r["id"]] = check_answer(dec.question, answers[dec.question.ident])
        except ContractError as exc:
            raise ProtocolError(f"decision {r['id']}: {exc}") from None
    return out
