"""runtimes/vllm-routerfix: the fixed-shape MoE router GEMM patch for the s1-pro runtime (GB10 / DGX Spark)."""
import importlib.util
import os
import sys
import types

import pytest

FIX = os.path.join(os.path.dirname(__file__), "..", "runtimes", "vllm-routerfix", "sitecustomize.py")
GL_MOD = "vllm.model_executor.layers.fused_moe.router.gate_linear"


def _load(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("REAX_VLLM_INVARIANT_ROUTER", raising=False)
    else:
        monkeypatch.setenv("REAX_VLLM_INVARIANT_ROUTER", flag)
    spec = importlib.util.spec_from_file_location("reax_routerfix_under_test", FIX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_vllm(monkeypatch, torch, version="0.30.0"):
    """A stand-in for vLLM 0.30.0's GateLinear with the same tier flags."""

    class GateLinear:
        def __init__(self, weight, cublas=True, ll=False, fp32=False, bf16x3=False):
            self.weight = weight
            self.allow_cublas_router_gemm = cublas
            self.allow_ll_bf16_gemm = ll
            self.allow_fp32_router_gemm = fp32
            self.allow_bf16x3_router_gemm = bf16x3

        def forward(self, x):
            # vLLM's tier 4 is `if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:`
            # `torch.mm(x, self.weight.T, out_dtype=torch.float32)`; that op has no CPU kernel, so the fake only marks
            # that vLLM's own code ran.
            return ("vllm", None), None

    gl = types.ModuleType(GL_MOD)
    gl.GateLinear = GateLinear
    parts = GL_MOD.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, GL_MOD, gl)
    monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], gl, raising=False)
    monkeypatch.setattr(sys.modules["vllm"], "__version__", version, raising=False)
    return GateLinear


@pytest.fixture
def patched(monkeypatch):
    """The patch loaded against the fake vLLM, with a CPU stand-in for the CUDA-only bf16->fp32 GEMM."""
    torch = pytest.importorskip("torch")
    GateLinear = _fake_vllm(monkeypatch, torch)
    mod = _load(monkeypatch, "1")
    mod._ROWS = 64
    shapes = []

    def cpu_mm(rows, weight):                      # torch.mm(..., out_dtype=) has no CPU kernel
        shapes.append((tuple(rows.shape), rows.is_contiguous()))
        return rows.float() @ weight.float().T

    monkeypatch.setattr(mod, "_mm", cpu_mm)
    torch.manual_seed(0)
    w = (torch.randn(128, 32) * 0.02).to(torch.bfloat16)
    x = torch.randn(300, 32).to(torch.bfloat16)
    return types.SimpleNamespace(torch=torch, GateLinear=GateLinear, mod=mod, shapes=shapes, w=w, x=x)


def test_flag_off_leaves_vllm_untouched(monkeypatch):
    mod = _load(monkeypatch, None)
    assert not hasattr(mod, "_forward")
    mod = _load(monkeypatch, "0")
    assert not hasattr(mod, "_forward")


def test_fixed_shape_chunks_make_rows_batch_invariant(patched):
    torch, gate, x = patched.torch, patched.GateLinear(patched.w), patched.x
    assert patched.GateLinear.forward is patched.mod._forward
    full, bias = gate.forward(x)
    assert bias is None and full.dtype == torch.float32 and full.shape == (300, 128)
    assert set(patched.shapes) == {((64, 32), True)}   # every GEMM: same shape, contiguous, whatever the batch size
    for n in (1, 5, 63, 64, 65, 200):
        assert torch.equal(gate.forward(x[:n])[0], full[:n])
    assert torch.equal(gate.forward(x[7:300])[0], full[7:300])          # rows at other positions in the chunk
    wide = torch.zeros(300, 64, dtype=torch.bfloat16)
    wide[:, ::2] = x
    assert not wide[:, ::2].is_contiguous()
    assert torch.equal(gate.forward(wide[:, ::2])[0], full)             # strided input
    assert set(patched.shapes) == {((64, 32), True)}
    assert torch.allclose(full, x.float() @ patched.w.float().T, atol=1e-5)


def test_other_router_tiers_keep_vllm_code(patched, monkeypatch):
    torch, w, x, GateLinear = patched.torch, patched.w, patched.x, patched.GateLinear
    assert torch.equal(GateLinear(w, ll=True).forward(x[:17])[0], GateLinear(w).forward(x[:17])[0])  # tier 1 stops at 16
    monkeypatch.setattr(patched.mod, "_mm", lambda rows, weight: pytest.fail("cuBLAS-tier replacement used"))
    small = x[:16]
    assert GateLinear(w, ll=True).forward(small)[0][0] == "vllm"            # tier 1 (SM90+, <= 16 rows)
    assert GateLinear(w, fp32=True).forward(small)[0][0] == "vllm"          # tier 2
    assert GateLinear(w, bf16x3=True).forward(small)[0][0] == "vllm"        # tier 3
    assert GateLinear(w, cublas=False).forward(small)[0][0] == "vllm"       # tier 5
    assert GateLinear(w).forward(small.float())[0][0] == "vllm"             # not bf16: not the cuBLAS tier


def test_requested_but_unapplicable_fix_stops_start_up(monkeypatch):
    torch = pytest.importorskip("torch")
    GateLinear = _fake_vllm(monkeypatch, torch, version="0.31.0")
    with pytest.raises(SystemExit, match="not applied: vLLM 0.31.0 is not 0.30.0"):
        _load(monkeypatch, "1")
    GateLinear = _fake_vllm(monkeypatch, torch)
    monkeypatch.setattr(GateLinear, "forward", lambda self, x: (None, None))
    with pytest.raises(SystemExit, match="not applied: GateLinear.forward has no cuBLAS"):
        _load(monkeypatch, "1")
    assert GateLinear.forward.__name__ == "<lambda>"
