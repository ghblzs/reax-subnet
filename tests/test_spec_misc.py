import json
import os

import pytest
from spec_helpers import HOTKEYS, make_weights, synthetic_pool, write_model, write_pool

from reax_subnet.spec import cli
from reax_subnet.spec import sampling as S
from reax_subnet.spec import vllm as V


def test_gpu_utilization_fits_budget():
    util = V.gpu_memory_utilization(8 * V.GIB, gpu_total_gb=96, budget_gb=30)       # a 4 B bf16 model
    assert util == pytest.approx((8 * 1.15 + 6) / 96, abs=1e-3)
    assert V.gpu_memory_utilization(18 * V.GIB) * 96 <= 30.01                       # a 9 B bf16 model is capped at 30 GB
    with pytest.raises(V.ModelUnavailable):
        V.gpu_memory_utilization(26 * V.GIB)


def test_docker_command_is_locked_down():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3))
    j = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "-d"] and V.VLLM_IMAGE in cmd
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "/m:/model:ro", "--gpu-memory-utilization 0.3",
                 "HF_HUB_OFFLINE=1", "--uds /run/vllm/vllm.sock", "--generation-config vllm"):
        assert flag in j
    assert "trust-remote-code" not in j and "-p " not in j and "--publish" not in j


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
