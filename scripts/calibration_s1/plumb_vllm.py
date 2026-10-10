"""Plumb via vLLM 0.30 offline (alternative engine; heavily batched). usage: plumb_vllm.py <model_dir> <items.jsonl> <out.json> [fp8]
Letter logprobs (processed) -> same centred logits as the reference readout up to a per-item constant."""
import json, math, sys, time
sys.path.insert(0, "/w/kit/jevk5src")
from jevk5.runtime import LETTERS, decision_options, messages
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import transformers, torch

model_dir, items_path, out_path = sys.argv[1:4]
fp8 = len(sys.argv) > 4 and sys.argv[4] == "fp8"
T = 2.07
tok = transformers.AutoTokenizer.from_pretrained("/w/plumb")
letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]
kw = dict(model=model_dir, tokenizer="/w/plumb", dtype="bfloat16", max_model_len=4608, gpu_memory_utilization=0.80,
          seed=0, enable_prefix_caching=False, logprobs_mode="processed_logprobs", max_logprobs=16)
if fp8:
    kw["quantization"] = "fp8"
llm = LLM(**kw)
items = [json.loads(l) for l in open(items_path)]
prompts, params, meta = [], [], []
for it in items:
    q = it["question"]; opts = decision_options(q)
    pr = tok.apply_chat_template(messages(it["state"], q.get("instructions", ""), [t for _, t in opts]), tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(pr, add_special_tokens=False)
    allowed = letter_ids[:len(opts)]
    prompts.append(TokensPrompt(prompt_token_ids=ids))
    params.append(SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(opts), allowed_token_ids=allowed, seed=0))
    meta.append((it["id"], [k for k, _ in opts], allowed, len(ids)))
llm.generate(prompts[:32], params[:32], use_tqdm=False)
t0 = time.time()
outs = llm.generate(prompts, params, use_tqdm=False)
dt = time.time() - t0
res = {}
for o, (iid, keys, allowed, n) in zip(outs, meta):
    lp = o.outputs[0].logprobs[0]
    z = [lp[t].logprob for t in allowed]
    mx = max(z); e = [math.exp((x - mx) / T) for x in z]; s = sum(e)
    res[iid] = {"keys": keys, "z": z, "p": [x / s for x in e], "tokens": n}
m = {"mode": "vllm-fp8" if fp8 else "vllm-bf16", "n": len(items), "run_s": round(dt, 2), "dec_per_s": round(len(items) / dt, 1),
     "gpu": torch.cuda.get_device_name(0)}
print(json.dumps(m), flush=True)
json.dump({"meta": m, "items": res}, open(out_path, "w"))
