#!/usr/bin/env python3
"""Warm vLLM's lazily JIT-compiled Triton kernels (fused MoE, unified attention, batch-invariant matmul) before the adapter
accepts traffic, so the first real requests do not see multi-second latency spikes. Direct to vLLM (:8000); best effort."""
import base64, io, json, sys, threading, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OPT = [32, 33, 34, 35, 36]  # arbitrary valid token ids; only shapes matter


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status


def text(n):
    body = {"model": "rune", "prompt": [(1000 + 7 * i) % 20000 + 100 for i in range(n)], "max_tokens": 1,
            "temperature": 0, "logprobs": 5, "allowed_token_ids": OPT}
    post("/v1/completions", body)


def image():
    from PIL import Image
    im = Image.new("RGB", (448, 336), (200, 30, 30)); buf = io.BytesIO(); im.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    post("/v1/chat/completions", {"model": "rune", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
         "allowed_token_ids": OPT, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "A or B?"}]}]})


def burst(fn, n):
    ts = [threading.Thread(target=lambda: fn()) for _ in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]


try:
    for n in (8, 64, 300, 1000, 3000, 9000):
        text(n)
    for n in (200, 700, 2500):
        burst(lambda n=n: text(n), 8)
    image(); burst(image, 4)
    print("warmup ok", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"warmup skipped: {type(e).__name__}: {e}", flush=True)
