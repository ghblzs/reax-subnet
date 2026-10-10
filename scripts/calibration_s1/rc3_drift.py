"""rc3 launch blockers on this GPU: (1) Qwen3-4B bf16 reference set for cross-GPU drift (scripts/calibrate.py),
(2) validator reference time for 64 miners x 12 items. Runs inside the rc3 image (torch 2.14.1, transformers 5.18.0)."""
import json, os, subprocess, sys, time
out = sys.argv[1]
cmd = [sys.executable, "/w/repo/scripts/calibrate.py", "--device", "cuda", "--dtype", "bfloat16", "--items", "256",
       "--cheat-model", "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "--out", f"{out}/rc3_calib_qwen3-4b.json"]
if os.path.exists("/w/kit/rc3_ref.json"):  # reference set from another GPU type -> cross-GPU drift
    cmd += ["--compare", "/w/kit/rc3_ref.json"]
r = subprocess.run(cmd, capture_output=True, text=True)
open(f"{out}/rc3_calibrate.stdout", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr[-5000:])
print("calibrate rc", r.returncode, flush=True)
import torch
from reax_subnet.backends import HFBackend, resolve_model
from reax_subnet.items import generate_items
be = HFBackend(resolve_model("qwen3-4b-instruct-2507", False), device="cuda", dtype="bfloat16")
items = generate_items(64 * 12, seed=7)
be.predict(items[:16])
torch.cuda.synchronize(); t = time.time()
be.predict(items)
torch.cuda.synchronize(); dt = time.time() - t
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "miners": 64, "items_per_miner": 12,
       "reference_seconds": round(dt, 2), "round_interval_s": 120, "share_of_round": round(dt / 120, 3)}
print(json.dumps(res), flush=True)
json.dump(res, open(f"{out}/rc3_round_timing.json", "w"))
