"""s1-fast runtime sidecar: Plumb-4B with the reference readout, on loopback.

Serves the reference runtime's decision contract ``POST /internal/v1/decide`` (header X-S1M-Internal-Token)
with the same computation as the reference engine's PlumbBatchBackend at max_batch=1:
the author's jevk5 0.2 runtime with CUDA graphs captured on the startup thread, one graph replay
per decision, letter logits / T (T = 2.07 from jevk5_config.json), numpy softmax.
Weights and runtime source are verified against the release manifest before serving.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import jevk5
from jevk5.runtime import JevK5, decision_options

MODEL_DIR = os.environ.get("PLUMB_MODEL_DIR", "/model")
MANIFEST = json.loads(Path(os.environ.get("REAX_MANIFEST", "/opt/reax/s1-fast-v1.json")).read_text())
MAX_INPUT_TOKENS = MANIFEST["limits"]["max_input_tokens"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> None:
    root = Path(jevk5.__file__).parent.parent
    for rel, digest in MANIFEST["runtime"]["source_files"].items():
        if _sha(root / rel) != digest:
            sys.exit(f"runtime source mismatch: {rel}")
    for name, digest in MANIFEST["weights"]["files"].items():
        p = Path(MODEL_DIR) / name
        if not p.is_file() or _sha(p) != digest:
            sys.exit(f"weights mismatch or missing: {name}")


def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("plumb_server: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE)")
    return tok


verify()
TOKEN = _token()
# CUDA graph capture must happen on the startup thread (measured: capturing in a worker
# changed long-prompt probabilities).
MODEL = JevK5(MODEL_DIR, graphs=True)
if not 2.06 < MODEL.temperature < 2.08:
    sys.exit("Plumb calibration temperature does not match the pinned release")
LOCK = asyncio.Lock()
app = FastAPI()


def _decide_one(state, author: dict) -> tuple[dict, int]:
    options = decision_options(author)
    ids = MODEL.encode(state, author["instructions"], [text for _, text in options])
    if len(ids) > MAX_INPUT_TOKENS:
        raise ValueError(f"input exceeds {MAX_INPUT_TOKENS} model tokens")
    logits = MODEL.letter_logits(ids, len(options))[None, :]
    row = logits[0, : len(options)] / MODEL.temperature
    exp = np.exp(row - row.max())
    p = exp / exp.sum()
    probs = {key: float(prob) for (key, _), prob in zip(options, p)}
    answer = {"type": author["type"], "probabilities": probs, "confidence": max(probs.values())}
    if author["type"] == "noul":
        answer["noul"] = probs["true"]
    elif author["type"] == "choice":
        answer["choice"] = max(probs, key=probs.get)
    else:
        answer["score"] = sum(int(key) * prob for key, prob in probs.items())
    return answer, len(ids)


@app.get("/health")
async def health():
    return {"ok": True, "model": "s1-fast", "manifest": MANIFEST["weights"]["revision"]}


@app.post("/internal/v1/decide")
async def decide(request: Request):
    if TOKEN and not hmac.compare_digest(request.headers.get("x-s1m-internal-token", ""), TOKEN):
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
    try:
        body = await request.json()
        (qid, q), = body["questions"].items()
        if q.get("type") not in ("choice", "noul", "score"):
            raise ValueError("bad question type")
        q.setdefault("instructions", "")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": {"message": f"bad request: {type(exc).__name__}"}}, status_code=400)
    try:
        async with LOCK:  # one graph replay at a time, as the reference engine (max_batch=1)
            answer, n = await asyncio.to_thread(_decide_one, body["state"], q)
    except ValueError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    return {"model": "s1-fast", "answers": {qid: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("RUNTIME_HOST", "127.0.0.1"), port=int(os.environ.get("RUNTIME_PORT", "8101")),
                log_level="warning", access_log=False)
