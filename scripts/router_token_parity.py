#!/usr/bin/env python3
"""Compare the router's s1-pro token counter with the input tokens that the reference runtime reports.

Usage: router_token_parity.py --tokenizer-dir DIR --url http://127.0.0.1:8102 --token-file FILE [--count 20]

Run inside the router image (it needs transformers and the pinned tokenizer files). DIR is the directory that
REAX_ROUTER_TOKENIZER_DIR_S1_PRO points at. Each prompt is sent to the reference runtime as a one-question request
(POST <url>/internal/v1/decide, header X-S1M-Internal-Token, like the validator's backend does); the runtime
(runtimes/s1-pro/rune_adapter.py) answers `usage.input_tokens` = the prompt tokens it fed the model, which for one
question is exactly what the router's RuneTokenCounter must return. Exit 0 only if every prompt matches.
The prompts are synthetic and are sent to our own reference runtime only.
"""
import argparse
import json
import sys
import time
import urllib.request

from reax_subnet.router_tokens import InputTokenLimitError, RuneTokenCounter
from reax_subnet.s1contract import parse_decision


def prompts():
    """Varied one-question prompts: noul/choice/scale, plain and JSON state, unicode, long text, 27+ options."""
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    out = [
        ("noul short", "plain evidence", {"type": "noul", "instructions": "Is it true?"}),
        ("noul criteria", {"a": 1, "b": [1, 2, 3]}, {"type": "noul", "instructions": "Check b.",
                                                      "criteria": {"true": "yes, it holds", "false": "no, it does not"}}),
        ("noul unicode", "Café, naïve, 日本語, emoji 🙂, ß, Ω", {"type": "noul", "instructions": "Ist das wahr? ¿Sí?"}),
        ("choice 3", {"k": "v"}, {"type": "choice", "instructions": "Pick.",
                                  "criteria": {"x": "first", "y": "second", "z": {"nested": True}}}),
        ("choice 26", "s", {"type": "choice", "instructions": "i", "criteria": {f"k{i}": f"option {i}" for i in range(26)}}),
        ("choice 27 (codes)", "s", {"type": "choice", "instructions": "i",
                                    "criteria": {f"k{i}": f"option {i}" for i in range(27)}}),
        ("choice 64 (codes)", "s", {"type": "choice", "instructions": "i",
                                    "criteria": {f"k{i}": f"option {i}" for i in range(64)}}),
        ("score", "evidence", {"type": "score", "instructions": "Rate.", "criteria": ["bad", "ok", "good", "great"]}),
        ("empty instructions", "x", {"type": "noul", "instructions": ""}),
        ("quotes and newlines", 'line1\nline2 "quoted" \\ backslash \t tab', {"type": "noul", "instructions": "a\nb"}),
    ]
    for n in (20, 60, 200, 500, 1200):  # states of growing size, in words
        text = " ".join(words[i % len(words)] + str(i) for i in range(n))
        out.append((f"long state {n} words", text, {"type": "noul", "instructions": "Does the text mention alpha0?"}))
        out.append((f"long json state {n}", {"items": [{"id": i, "text": words[i % 12]} for i in range(n // 4)]},
                    {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "one", "b": "two"}}))
    text = " ".join(words[i % len(words)] + str(i) for i in range(1650))  # near the 16 KB state cap, likely over the token limit
    out.append(("near the state cap", text, {"type": "noul", "instructions": "Does the text mention alpha0?"}))
    return out


def reference_tokens(url, token, model, state, question):
    body = {"request_id": "parity", "model": model, "state": state, "questions": {"q": question}}
    req = urllib.request.Request(url.rstrip("/") + "/internal/v1/decide", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-S1M-Internal-Token": token})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    usage = data.get("usage") or {}
    value = usage.get("input_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"runtime reported no integer usage.input_tokens (usage={usage!r})")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--model-name", default="rune", help="model name the runtime serves (adapter --served-name)")
    a = ap.parse_args()
    token = open(a.token_file).read().strip()
    counter = RuneTokenCounter.from_directory(a.tokenizer_dir)
    bad = 0
    for name, state, question in prompts()[: a.count]:
        decision = parse_decision({"id": "d", "state": state, "questions": {"q": question}}, "s1-pro")
        try:
            ours = counter("s1-pro", [decision])
        except InputTokenLimitError:
            print(f"{name:28} over the 4096-token limit: the router answers 400, nothing to compare")
            continue
        try:
            theirs = reference_tokens(a.url, token, a.model_name, state, decision.question.author_dict())
        except Exception as exc:  # noqa: BLE001
            print(f"{name:28} router={ours:6} reference=ERROR {type(exc).__name__}: {exc}")
            bad += 1
            continue
        flag = "ok" if ours == theirs else "MISMATCH"
        bad += ours != theirs
        print(f"{name:28} router={ours:6} reference={theirs:6} diff={ours - theirs:+d} {flag}")
        time.sleep(0.05)
    print(f"{min(a.count, len(prompts()))} prompts, {bad} not matching")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
