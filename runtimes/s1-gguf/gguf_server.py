"""s1-lite / s1-cpu runtime sidecar: a pinned GGUF decision model on llama.cpp, on loopback.

Serves the same decision contract as the other runtime sidecars, ``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header, so the miner and validator processes are unchanged. One decision = one prompt
evaluation from an empty context (no prompt cache: cached and fresh evaluations are not bit-identical), then the
option letters' logits at the last position (``llama_get_logits_ith(ctx, -1)``), divided by the calibration
temperature, softmax (readouts.py). Works with a CPU build (x86_64, aarch64) and a CUDA build (incl. Jetson).

Environment: REAX_MANIFEST (manifest JSON), GGUF_MODEL_DIR (directory holding the pinned file), RUNTIME_HOST,
RUNTIME_PORT, S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE, GGUF_N_GPU_LAYERS (0 = CPU, -1 = all on GPU),
GGUF_THREADS (default: physical cores).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readouts  # noqa: E402

MANIFEST = json.loads(Path(os.environ.get("REAX_MANIFEST", "/opt/reax/manifest.json")).read_text())
MODEL = MANIFEST["model"]
MODEL_DIR = Path(os.environ.get("GGUF_MODEL_DIR", "/model"))
READOUT = MANIFEST["readout"]
MAX_INPUT_TOKENS = MANIFEST["limits"]["max_input_tokens"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> Path:
    files = MANIFEST["weights"]["files"]
    for name, digest in files.items():
        p = MODEL_DIR / name
        if not p.is_file() or _sha(p) != digest:
            sys.exit(f"weights mismatch or missing: {name}")
    return MODEL_DIR / MANIFEST["weights"]["gguf"]


def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("gguf_server: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE)")
    return tok


class Engine:
    def __init__(self, path: Path):
        import llama_cpp
        from llama_cpp import Llama

        threads = int(os.environ.get("GGUF_THREADS", "0")) or max(1, (os.cpu_count() or 2) // 2)
        self._lib = llama_cpp
        self.llm = Llama(model_path=str(path), n_ctx=MAX_INPUT_TOKENS + 64, n_batch=512,
                         n_threads=threads, n_threads_batch=threads,
                         n_gpu_layers=int(os.environ.get("GGUF_N_GPU_LAYERS", "0")),
                         seed=0, logits_all=False, verbose=False)
        self.n_vocab = self.llm.n_vocab()
        self.letter_ids = {}
        for letter in readouts.LETTERS + readouts.GUTSY_REJECT_LETTER:
            ids = self.llm.tokenize(letter.encode(), add_bos=False, special=False)
            if len(ids) != 1:
                sys.exit(f"answer letter {letter!r} is not one token")
            self.letter_ids[letter] = ids[0]

    def _ctx(self):
        ctx = getattr(self.llm, "ctx", None)
        return ctx if ctx is not None else self.llm._ctx.ctx

    def tokens(self, prompt: str) -> list[int]:
        # The pinned models have no BOS token (Qwen); special=True parses <|im_start|> etc. as control tokens.
        return list(self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))

    def letter_logits(self, ids: list[int], letters: list[str]) -> list[float]:
        import numpy as np

        self.llm.reset()
        self.llm.eval(ids)
        ptr = self._lib.llama_get_logits_ith(self._ctx(), -1)
        row = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        return [float(row[self.letter_ids[x]]) for x in letters]


def decide_one(engine: Engine, state, question: dict) -> tuple[dict, int]:
    prompt, keys, letters = readouts.build(READOUT["kind"], state, question)
    ids = engine.tokens(prompt)
    if len(ids) > MAX_INPUT_TOKENS:
        raise ValueError(f"input exceeds {MAX_INPUT_TOKENS} model tokens")
    logits = engine.letter_logits(ids, letters)
    probs = readouts.probabilities(logits, keys, readouts.temperature(READOUT, question["type"]))
    return readouts.answer(question["type"], probs), len(ids)


app = FastAPI()
ENGINE: Engine | None = None
TOKEN = ""
LOCK = asyncio.Lock()


@app.get("/health")
async def health():
    return {"ok": ENGINE is not None, "model": MODEL, "manifest": MANIFEST["weights"]["revision"]}


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
        async with LOCK:  # one evaluation at a time: one context, deterministic batch shape
            answer, n = await asyncio.to_thread(decide_one, ENGINE, body["state"], q)
    except ValueError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    return {"model": MODEL, "answers": {qid: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}


if __name__ == "__main__":
    gguf = verify()
    TOKEN = _token()
    ENGINE = Engine(gguf)
    uvicorn.run(app, host=os.environ.get("RUNTIME_HOST", "127.0.0.1"), port=int(os.environ.get("RUNTIME_PORT", "8103")),
                log_level="warning", access_log=False)
