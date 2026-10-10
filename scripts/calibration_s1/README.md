# reax-s1 GPU calibration kit

What produced the tolerances in docs/CALIBRATION.md. Run it on your own GPU to check that your setup stays inside them.

1. Items: `PYTHONPATH=src python -c "from reax_subnet.s1items import generate_s1_items as g; import json; [print(json.dumps({'id': i.item_id, 'state': i.state, 'question': i.question()})) for i in g(640, seed=20261003)]" > items.jsonl`
   (the release calibration also used the 231 public JevBench items; they are not part of this repository).
2. s1-fast (reference path, inside the reference fast runtime):
   `docker build -t calibfast -f scripts/calibration_s1/Dockerfile.calibfast scripts/calibration_s1`, weights in /w/plumb, then
   `docker run --rm --gpus all -v /w:/w -v $PWD:/k calibfast python3 /k/scripts/calibration_s1/plumb_calib.py /w/plumb /k/items.jsonl /k/out/<gpu>/p_graph_a.json graph`
   (modes: graph = reference, eager8, eager1). Adversary runs: other weights dirs (`graph 2.07`), `plumb_vllm.py ... fp8` in the vLLM 0.30.0 image.
3. s1-pro: start the runtime sidecar (runtimes/s1-pro), then `python3 rune_client.py http://127.0.0.1:8102 items.jsonl out/<gpu>/r_det_c1_a.json 1 det-c1`
   (needs the runtime token header; the release run used RUNE_ALLOW_NO_AUTH=1 on an isolated pod).
4. `python3 analyze.py out/ analysis.json` and `python3 rounds.py <ref.json> <run.json> <T> <k>` for round statistics.
