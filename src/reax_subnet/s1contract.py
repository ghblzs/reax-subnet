"""The REAX typed-decision contract for the reference models s1-fast and s1-pro.

Follows the reference engine's request parsing and answer normalisation so that a
miner's answer object is exactly what the reference models return for ``s1-fast``
and ``s1-pro``. Limits are kept:
one question per decision, no images.

The validator additionally checks that the derived fields of an answer
(``choice``, ``noul``, ``score``, ``confidence``, ``legend``) follow from its
probabilities, so reference-looking fields over wrong probabilities fail.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

MAX_STATE_BYTES = 16 * 1024
MAX_OPTIONS = {"s1-fast": 16, "s1-pro": 64}
MODELS = tuple(MAX_OPTIONS)
# The reference engine checks sum(p) within 1e-3; the validator is stricter on what it accepts back.
PROB_SUM_TOLERANCE = 1e-3
DERIVED_TOLERANCE = 1e-6


class ContractError(ValueError):
    """The decision or answer violates the REAX decision contract."""


def _text(value: Any, name: str, limit: int = 8192) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise ContractError(f"{name} must be nonempty text within {limit} bytes")
    return value


def _json_data(value: Any, name: str, limit: int, *, null_ok: bool = False) -> Any:
    if not isinstance(value, (str, dict, list)) and not (null_ok and value is None):
        raise ContractError(f"{name} must be text, object, or array")
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must contain finite JSON values") from exc
    if size > limit:
        raise ContractError(f"{name} exceeds {limit} bytes")
    return value


@dataclass(frozen=True)
class Question:
    ident: str
    kind: str
    instructions: Any
    criteria: Any
    keys: tuple[str, ...]
    descriptions: tuple[Any, ...]

    def author_dict(self) -> dict[str, Any]:
        """What the reference engine sends to the model runtime (the reference engine's Question.author_dict)."""
        result = {"type": self.kind, "instructions": self.instructions}
        if self.kind != "noul" or self.criteria:
            result["criteria"] = self.criteria
        return result


@dataclass(frozen=True)
class Decision:
    ident: str
    state: Any
    question: Question

    def body(self) -> dict[str, Any]:
        """The decision as sent on the wire (``questions`` map with one entry)."""
        q: dict[str, Any] = {"type": self.question.kind, "instructions": self.question.instructions}
        if self.question.criteria is not None and not (self.question.kind == "noul" and self.question.criteria == {}):
            q["criteria"] = self.question.criteria
        return {"id": self.ident, "state": self.state, "questions": {self.question.ident: q}}


def parse_question(ident: str, value: Any, model: str) -> Question:
    _text(ident, "question id", 128)
    if not isinstance(value, dict):
        raise ContractError("question must be an object")
    if set(value) - {"type", "instructions", "criteria"}:
        raise ContractError("question has unknown fields")
    kind = value.get("type")
    if not isinstance(kind, str) or kind not in {"noul", "choice", "score"}:
        raise ContractError("question has invalid type")
    instruction = _json_data(value.get("instructions", ""), "instructions", 8192, null_ok=True)
    criteria = value.get("criteria")
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ContractError("question has invalid noul criteria")
        keys: tuple[str, ...] = ("true", "false")
        descriptions: tuple[Any, ...] = (criteria.get("true"), criteria.get("false"))
    elif kind == "choice":
        if not isinstance(criteria, dict):
            raise ContractError("question requires choice criteria")
        keys = tuple(criteria)
        descriptions = tuple(criteria.values())
    else:
        if not isinstance(criteria, list):
            raise ContractError("question requires score criteria")
        keys = tuple(str(i) for i in range(len(criteria)))
        descriptions = tuple(criteria)
    if not 2 <= len(keys) <= MAX_OPTIONS[model]:
        raise ContractError("question has invalid option count")
    if any(not isinstance(k, str) or not k or len(k.encode()) > 256 for k in keys):
        raise ContractError("question has invalid option keys")
    for desc in descriptions:
        if model != "s1-fast" and desc is not None and not isinstance(desc, (str, dict, list)):
            raise ContractError("this model requires text, object, or array option descriptions")
        try:
            encoded = json.dumps(desc, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractError("option description must be finite JSON") from exc
        if len(encoded.encode("utf-8")) > 4096:
            raise ContractError("option description exceeds 4096 bytes")
        if kind == "score" and desc is not None and not isinstance(desc, (str, dict, list)):
            raise ContractError("score level must be text, object, or array")
    return Question(ident, kind, instruction, criteria, keys, descriptions)


def parse_decision(obj: Any, model: str) -> Decision:
    """One decision: ``{"id", "state", "questions": {<one question>}}``."""
    if model not in MAX_OPTIONS:
        raise ContractError("unknown model")
    if not isinstance(obj, dict) or set(obj) != {"id", "state", "questions"}:
        raise ContractError("decision must have exactly the fields id, state, questions")
    state = _json_data(obj["state"], "state", MAX_STATE_BYTES)
    raw = obj["questions"]
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ContractError("a decision carries exactly one question (reference contract limit)")
    (ident, value), = raw.items()
    return Decision(str(obj["id"]), state, parse_question(ident, value, model))


def normalize_answer(question: Question, answer: dict[str, Any]) -> dict[str, Any]:
    """The reference answer normalisation (the reference engine's normalize_answer), unchanged."""
    if not isinstance(answer, dict):
        raise RuntimeError("backend returned an invalid answer")
    if question.kind == "noul":
        p = float(answer["noul"])
        probs = {"true": p, "false": 1.0 - p}
    else:
        given = answer["probabilities"]
        probs = {key: float(given[key]) for key in question.keys}
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-3:
        raise RuntimeError("backend returned invalid probabilities")
    best = max(range(len(question.keys)), key=lambda i: probs[question.keys[i]])
    out: dict[str, Any] = {"type": question.kind, "probabilities": probs}
    if question.kind == "noul":
        out["noul"] = probs["true"]
    elif question.kind == "choice":
        out["choice"] = question.keys[best]
        out["confidence"] = probs[question.keys[best]]
    else:
        expected = sum(i * probs[str(i)] for i in range(len(probs)))
        score = float(answer.get("score", expected))
        if not math.isfinite(score) or abs(score - expected) > 1e-3:
            raise RuntimeError("backend returned an invalid score")
        confidence = float(answer.get("confidence", max(probs.values())))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RuntimeError("backend returned an invalid confidence")
        out.update(score=score, confidence=confidence, legend=dict(zip(question.keys, question.descriptions)))
    return out


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{what} must be a number")
    try:
        f = float(value)
    except OverflowError:  # a JSON integer too large for a float
        raise ContractError(f"{what} must be finite") from None
    if not math.isfinite(f):
        raise ContractError(f"{what} must be finite")
    return f


def check_answer(question: Question, answer: Any) -> tuple[float, ...]:
    """Validate a miner's answer object; return its probabilities in ``question.keys`` order.

    The fields must be exactly those ``normalize_answer`` produces and every derived
    field must follow from the probabilities.
    """
    expected_fields = {
        "noul": {"type", "probabilities", "noul"},
        "choice": {"type", "probabilities", "choice", "confidence"},
        "score": {"type", "probabilities", "score", "confidence", "legend"},
    }[question.kind]
    if not isinstance(answer, dict) or set(answer) != expected_fields:
        raise ContractError(f"answer must have exactly the fields {sorted(expected_fields)}")
    if answer["type"] != question.kind:
        raise ContractError("answer type does not match the question")
    probs = answer["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(question.keys):
        raise ContractError("probabilities must cover exactly the option keys")
    values = tuple(_num(probs[k], "probability") for k in question.keys)
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ContractError("probabilities must be within [0, 1]")
    if abs(math.fsum(values) - 1.0) > PROB_SUM_TOLERANCE:
        raise ContractError("probabilities must sum to 1")
    by_key = dict(zip(question.keys, values))
    if question.kind == "noul":
        if abs(_num(answer["noul"], "noul") - by_key["true"]) > DERIVED_TOLERANCE:
            raise ContractError("noul must equal p(true)")
        if abs(by_key["true"] + by_key["false"] - 1.0) > DERIVED_TOLERANCE:
            raise ContractError("noul probabilities must be complementary")
    elif question.kind == "choice":
        best = max(range(len(question.keys)), key=lambda i: values[i])
        if answer["choice"] != question.keys[best]:
            raise ContractError("choice must be the most probable option")
        if abs(_num(answer["confidence"], "confidence") - values[best]) > DERIVED_TOLERANCE:
            raise ContractError("confidence must be the top probability")
    else:
        expected = math.fsum(i * values[i] for i in range(len(values)))
        if abs(_num(answer["score"], "score") - expected) > 1e-3:
            raise ContractError("score must be the expected level")
        # Both reference backends report the top probability as the score confidence.
        if abs(_num(answer["confidence"], "confidence") - max(values)) > DERIVED_TOLERANCE:
            raise ContractError("confidence must be the top probability")
        if answer["legend"] != dict(zip(question.keys, question.descriptions)):
            raise ContractError("legend must repeat the score levels")
    return values
