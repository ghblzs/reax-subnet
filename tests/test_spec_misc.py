import json
import math
import os

import pytest
from spec_helpers import HOTKEYS, make_weights, synthetic_pool, write_model, write_pool

from reax_subnet.spec import cli
from reax_subnet.spec import sampling as S
from reax_subnet.spec import vllm as V


def test_gpu_utilization_fits_budget():
    util = V.gpu_memory_utilization(8 * V.GIB, gpu_total_gb=96)                     # a 4 B bf16 model
    assert util == pytest.approx((8 * 1.15 + 6) / 96, abs=1e-3)
    assert V.GPU_BUDGET_GB == 25.0
    for total in (32.0, 48.0, 95.6, 96.0):                                          # a 9 B bf16 model is capped at 25 GB
        assert V.gpu_memory_utilization(18 * V.GIB, gpu_total_gb=total) * total + V.GPU_OVERHEAD_GB <= 25.0
    assert V.gpu_memory_utilization(18 * V.GIB, gpu_total_gb=96) == pytest.approx(23.5 / 96, abs=1e-3)
    with pytest.raises(V.ModelUnavailable):                                         # weights above 20 GiB never fit
        V.gpu_memory_utilization(int(20.5 * V.GIB))
    with pytest.raises(V.ModelUnavailable):
        V.gpu_memory_utilization(26 * V.GIB)


def test_gpu_utilization_leaves_room_for_the_process_overhead():
    """Qwen3.5-9B text-only bf16 (8.95 B params, 17,907,614,208 weight bytes) on an RTX PRO 6000 (97,887 MiB): rc4 asked
    vLLM for 0.261 (24.95 GiB), and the process took about 1.22 GiB more (CUDA context + post-profile allocations), so it
    went out of memory with 26,605 MiB free. The share plus the overhead must stay within the budget."""
    total = 97887 / 1024
    util = V.gpu_memory_utilization(17_907_614_208, gpu_total_gb=total)
    assert util == 0.245                                                            # measured: 513/513 dev items, peak 24.46 GiB
    assert util * total + V.GPU_OVERHEAD_GB <= V.GPU_BUDGET_GB
    assert util * total + 1.22 <= V.GPU_BUDGET_GB - 0.25                              # the measured overhead, with margin
    assert V.gpu_memory_utilization(17_907_614_208, gpu_total_gb=total, budget_gb=24.0) == 0.235
    # small models are not capped: unchanged from 1.1/rc4
    assert V.gpu_memory_utilization(8 * V.GIB, gpu_total_gb=total) == pytest.approx(math.floor((8 * 1.15 + 6) / total * 1000) / 1000)


def test_gpu_total_detected_from_nvidia_smi(monkeypatch):
    import subprocess

    class R:
        stdout = "32607\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert V.detect_gpu_total_gb("0") == pytest.approx(32607 / 1024)

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert V.detect_gpu_total_gb("0") == V.GPU_TOTAL_GB


def test_docker_command_is_locked_down():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3))
    j = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "-d"] and V.VLLM_IMAGE in cmd
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "/m:/model:ro", "--gpu-memory-utilization 0.3",
                 "HF_HUB_OFFLINE=1", "--uds /run/vllm/vllm.sock", "--generation-config vllm"):
        assert flag in j
    assert "trust-remote-code" not in j and "-p " not in j and "--publish" not in j
    # Triton compiles and dlopen()s helper modules under HOME=/tmp, so the tmpfs must allow exec.
    tmpfs = cmd[cmd.index("--tmpfs") + 1]
    assert "exec" in tmpfs.split(",") and "noexec" not in tmpfs
    assert "ccfix" not in j and "PYTHONPATH" not in j and "REAX_VLLM_DISABLE_UVA" not in j


def test_docker_command_disable_uva_mounts_sitecustomize(tmp_path):
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3, disable_uva=True, aux_dir=str(tmp_path)))
    assert f"{tmp_path}:/aux:ro" in cmd
    assert "PYTHONPATH=/aux" in cmd and "REAX_VLLM_DISABLE_UVA=1" in cmd
    assert cmd.index(V.VLLM_IMAGE) > cmd.index("REAX_VLLM_DISABLE_UVA=1")           # options come before the image
    src = open(V.CCFIX_FILE).read()
    assert "REAX_VLLM_DISABLE_UVA" in src and "is_uva_available" in src
    _run, _sock, aux = V.prepare_run_dir(str(tmp_path / "work"), template="qwen.jinja", disable_uva=True)
    assert open(os.path.join(aux, "sitecustomize.py")).read() == src                 # the one package copy, mounted via the run dir


def test_backend_reads_uva_switch_from_env(monkeypatch):
    monkeypatch.setenv("REAX_VLLM_DISABLE_UVA", "1")
    assert V.VllmDockerBackend("/m", 8 * V.GIB, gpu_total_gb=96).disable_uva is True
    monkeypatch.delenv("REAX_VLLM_DISABLE_UVA")
    assert V.VllmDockerBackend("/m", 8 * V.GIB, gpu_total_gb=96).disable_uva is False
    assert V.VllmDockerBackend("/m", 8 * V.GIB, gpu_total_gb=96, disable_uva=True).disable_uva is True


def test_sampling_fresh_disjoint_tiers_and_primitives():
    pool = synthetic_pool(900)
    rows, fresh = S.draw(pool, n=300, seed=1)
    assert fresh and len(rows) == 300 and len({r["id"] for r in rows}) == 300
    kinds = [r["question"]["type"] for r in rows]
    assert all(95 <= kinds.count(k) <= 105 for k in S.PRIMITIVES)
    tiers = [r["tier"] for r in rows]
    assert tiers.count("proc_sealed") >= 140
    rows2, fresh2 = S.draw(pool, n=300, seed=2, exclude={r["id"] for r in rows})
    assert not ({r["id"] for r in rows} & {r["id"] for r in rows2}) and fresh2
    assert S.draw(pool, n=300, seed=1)[0] == rows                                     # reproducible
    assert S.draw(pool, n=300, seed=3)[0] != rows


def test_sampling_prefers_unused_and_reports_reuse():
    pool = synthetic_pool(300)
    used = {r["id"]: 1 for r in pool[:250]}
    rows, fresh = S.draw(pool, n=100, seed=1, use_counts=used)
    assert not fresh and sum(1 for r in rows if r["id"] not in used) == 50           # all 50 unused ones come first
    with pytest.raises(S.SampleError):
        S.draw(pool, n=100, seed=1, exclude={r["id"] for r in pool[:150]})


def test_load_pool_prefers_clean_file(tmp_path):
    write_pool(str(tmp_path), "demand_forecasting", synthetic_pool(30))
    assert len(S.load_pool(str(tmp_path), "demand_forecasting")) == 30
    with pytest.raises(S.SampleError):
        S.load_pool(str(tmp_path), "gaming")


def test_cli_dry_run_and_refusal(tmp_path, capsys):
    m = write_model(str(tmp_path / "m"), make_weights(1))
    assert cli.main([m, "--dev", str(tmp_path), "--category", "demand_forecasting", "--hotkey", HOTKEYS[0], "--dry-run", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["violations"] == [] and out["params"] == 4 * 64 * 64 and len(out["hash"]) == 64
    assert cli.main([m, "--dev", str(tmp_path), "--category", "gaming", "--dry-run"]) == 2
    assert "category" in capsys.readouterr().err


def test_cli_load_dev_layouts(tmp_path):
    rows = synthetic_pool(5)
    d = tmp_path / "demand_forecasting" / "dev"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert len(cli.load_dev(str(tmp_path), "demand_forecasting")) == 5
    assert len(cli.load_dev(str(d / "dev.jsonl"), None)) == 5
    hf = tmp_path / "hf"                                                            # decisionmodels/sn92-specialists: <category>/dev.jsonl
    (hf / "risk_assessment").mkdir(parents=True)
    (hf / "risk_assessment" / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:3]))
    assert len(cli.load_dev(str(hf), "risk_assessment")) == 3
    assert len(cli.load_dev(str(hf / "risk_assessment"), None)) == 3
    with pytest.raises(SystemExit):
        cli.load_dev(str(tmp_path / "nothing"), "x")


@pytest.mark.gpu
def test_gpu_integration_real_vllm():
    """Needs Docker + NVIDIA runtime + a free GPU and a real model dir. Run with:

    REAX_SPEC_TEST_MODEL=/path/to/model REAX_SPEC_TEST_DEV=/path/to/dev.jsonl REAX_SPEC_TEST_CATEGORY=demand_forecasting \\
        .venv/bin/pytest -m gpu -q tests/test_spec_misc.py -k gpu_integration
    """
    import asyncio

    model, dev = os.environ.get("REAX_SPEC_TEST_MODEL"), os.environ.get("REAX_SPEC_TEST_DEV")
    if not model or not dev:
        pytest.skip("REAX_SPEC_TEST_MODEL / REAX_SPEC_TEST_DEV not set")
    rc = cli.main([model, "--dev", dev, "--category", os.environ.get("REAX_SPEC_TEST_CATEGORY", "demand_forecasting"),
                   "--no-manifest-check", "--limit", "60", "--json"])
    assert rc == 0
    _ = asyncio


def test_gpu_budget_is_25_gb_and_util_stays_below_it():
    assert V.GPU_BUDGET_GB == 25.0
    for gib in (1, 8, 14, 18):
        assert V.gpu_memory_utilization(gib * V.GIB) * V.GPU_TOTAL_GB + V.GPU_OVERHEAD_GB <= 25.01


def test_engine_preflight_and_compose_defaults():
    import os
    from reax_subnet import engrunner
    assert engrunner.ENGINE_GPU_MIB == 12_500
    root = os.path.join(os.path.dirname(V.__file__), "..", "..", "..")
    compose = open(os.path.join(root, "deploy", "validator-1.2", "compose.yml")).read()
    for needle in ("REAX_S1_PRO_GPU_MEM_UTIL:-0.60", "REAX_S1_PRO_MAX_MODEL_LEN:-8192", "PYTHONPATH: /opt/reax-ccfix",
                   "REAX_VLLM_DISABLE_UVA:-0", ":/opt/reax-ccfix:ro"):
        assert needle in compose
    assert not os.path.exists(os.path.join(root, ".github", "workflows", "release-images.yml"))


def test_docker_command_eager_switch():
    assert "--enforce-eager" in V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3))
    assert "--enforce-eager" not in V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3, eager=False))
