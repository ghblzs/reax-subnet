"""Query the Rune decide adapter (reference readout) for every item. usage: rune_client.py <base_url> <items.jsonl> <out.json> <concurrency> <label>"""
import asyncio, json, math, sys, time, uuid
import httpx

base, items_path, out_path, conc, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
items = [json.loads(l) for l in open(items_path)]
T = 2.0


async def main():
    sem = asyncio.Semaphore(conc)
    res, lat = {}, []
    import os
    headers = {"X-S1M-Internal-Token": os.environ["S1M_INTERNAL_TOKEN"]} if os.environ.get("S1M_INTERNAL_TOKEN") else {}
    async with httpx.AsyncClient(timeout=300, headers=headers) as c:
        async def one(it):
            async with sem:
                body = {"request_id": uuid.uuid4().hex, "model": "rune", "state": it["state"], "questions": {"q": it["question"]}}
                t = time.perf_counter()
                r = await c.post(base + "/internal/v1/decide", json=body)
                lat.append(time.perf_counter() - t)
                if r.status_code != 200:
                    res[it["id"]] = {"error": r.status_code, "body": r.text[:200]}
                    return
                a = r.json()["answers"]["q"]
                if a["type"] == "noul":
                    keys, p = ["true", "false"], [a["noul"], 1 - a["noul"]]
                else:
                    keys = list(a["probabilities"]); p = [a["probabilities"][k] for k in keys]
                z = [T * math.log(max(x, 1e-300)) for x in p]
                res[it["id"]] = {"keys": keys, "z": z, "p": p, "tokens": r.json()["usage"]["input_tokens"]}
        t0 = time.time()
        await asyncio.gather(*(one(it) for it in items))
        dt = time.time() - t0
    lat.sort()
    meta = {"mode": label, "concurrency": conc, "n": len(items), "errors": sum(1 for v in res.values() if "error" in v),
            "run_s": round(dt, 2), "dec_per_s": round(len(items) / dt, 1), "p50_ms": round(lat[len(lat) // 2] * 1000, 1),
            "p95_ms": round(lat[int(len(lat) * 0.95)] * 1000, 1)}
    print(json.dumps(meta), flush=True)
    json.dump({"meta": meta, "items": res}, open(out_path, "w"))

asyncio.run(main())
