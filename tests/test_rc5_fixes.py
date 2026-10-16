"""Fixes for the review findings on 1.2.0rc5 (F1-F6) and host GPU-memory start-up failures."""

import logging
import os
import types

import numpy as np
import pytest
from spec_helpers import finetune, make_weights, write_model
from test_spec_evaluator import DEFAULT_BASE, TRACK, Env, crown

from reax_subnet import award as A
from reax_subnet import cli
from reax_subnet import worker as W
from reax_subnet.chain import FINNEY_GENESIS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec import tensors as T
from reax_subnet.spec import vllm as V


def _bf16(w):
    """The values a BF16 file holds (spec_helpers truncates to BF16)."""
    return {k: (np.ascontiguousarray(v, dtype=np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
            for k, v in w.items()}


def _base():
    w = make_weights(1)
    w["model.norm.weight"] = np.ones(64, dtype=np.float32)
    return w


def _env(tmp_path, base, dtype="BF16"):
    write_model(str(tmp_path / "bases" / DEFAULT_BASE.replace("/", "__", 1)), base, dtype=dtype)
    return Env(tmp_path, {}, base_dir=str(tmp_path / "bases"))


# ------------------------------------------------------------------------------------------------ F1 + F2
@pytest.mark.parametrize("dtype", ["F32", "F16"])
async def test_f2_honest_finetune_in_another_dtype_than_the_bf16_base_is_verified(tmp_path, dtype):
    base = _base()
    env = _env(tmp_path, base)
    env.add(1, "a/ft", 0.9, weights=finetune(base, 10), dtype=dtype)
    await env.cycle({1: "a/ft"})
    s = env.sub(1)
    assert s.delta_checked and s.status == ST.PENDING_CONFIRM and s.sketch_rule == T.SKETCH_RULE
    assert env.server.served == ["a/ft"]


async def test_f1_dtype_split_near_copy_is_derived(tmp_path):
    """Review probe B: the parent upcast to F32 except one small tensor kept in the base dtype with its delta replaced by
    noise. Every float tensor is compared now, so the copy is derived from its parent. (Tensors of 1 M values, so that the
    small tensor holds a realistic share of the sampled values.)"""
    base = make_weights(1, size=1024)
    base["model.norm.weight"] = np.ones(64, dtype=np.float32)
    env = _env(tmp_path, base)
    parent = finetune(base, 10)
    copy = dict(_bf16(parent))
    copy["model.norm.weight"] = base["model.norm.weight"] + 0.05 * np.random.RandomState(9).randn(64).astype(np.float32)
    dtypes = {k: "F32" for k in copy}
    dtypes["model.norm.weight"] = "BF16"
    env.add(1, "a/parent", 0.6, weights=parent, dtype="BF16")
    env.add(2, "b/copy", 0.95, weights=copy, dtype=dtypes)
    env.add(3, "c/indep", 0.6, weights=finetune(base, 11), dtype="F32")
    await env.cycle({1: "a/parent"}, block=100)
    await env.cycle({1: "a/parent", 2: "b/copy", 3: "c/indep"}, block=110)
    assert env.sub(2).derived_from == env.sub(1).key and env.sub(2).delta_checked
    assert env.sub(3).derived_from == "" and env.sub(3).status == ST.PENDING_CONFIRM


def test_f1_sketch_compares_across_dtypes(tmp_path):
    base = _base()
    ft = _bf16(finetune(base, 10))     # BF16 values: the F32 file is their exact upcast
    b = write_model(str(tmp_path / "b"), base, dtype="BF16")
    m32 = write_model(str(tmp_path / "m32"), ft, dtype="F32")
    m16 = write_model(str(tmp_path / "m16"), ft, dtype="BF16")
    s32, s16 = T.delta_sketch(m32, b, stride=3), T.delta_sketch(m16, b, stride=3)
    assert set(s32) == set(s16) == set(base)
    st = T.compare_sketches(s16, s32)
    assert st.derived() and st.relative_l2 == 0.0
    cov = T.sketch_coverage(s32, m32, b)
    assert cov.fraction == 1.0 and cov.enough() and cov.fp8 == ()


async def test_f1_low_coverage_is_pending_with_the_coverage_in_the_reason(tmp_path):
    base = make_weights(1)               # 4 equal tensors: renaming one leaves 75 % comparable
    env = _env(tmp_path, base)
    ft = finetune(base, 10)
    ft["renamed.weight"] = ft.pop("model.layers.3.mlp.weight")
    env.add(1, "a/partial", 0.9, weights=ft)
    await env.cycle({1: "a/partial"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and not s.delta_checked and "75.0%" in s.reason and "90%" in s.reason
    assert env.server.served == []


async def test_f1_fp8_tensors_are_pending_with_a_clear_reason(tmp_path):
    base = make_weights(1)
    env = _env(tmp_path, base)
    env.add(1, "a/fp8", 0.9, weights=finetune(base, 10), dtype={"model.layers.0.mlp.weight": "F8_E4M3",
                                                                   "model.layers.1.mlp.weight": "F8_E4M3"})
    await env.cycle({1: "a/fp8"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and "FP8" in s.reason and "50.0%" in s.reason and env.server.served == []


async def test_f2_pending_of_an_older_rule_is_rechecked_once_and_a_current_pending_is_not(tmp_path, monkeypatch):
    base = _base()
    env = _env(tmp_path, base)
    env.add(1, "a/ft", 0.9, weights=finetune(base, 10), dtype="F32")
    real = T.delta_sketch
    monkeypatch.setattr(T, "delta_sketch", lambda *a, **k: {})        # rc5: no tensor of the same dtype
    await env.cycle({1: "a/ft"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and s.base_id and "no weight" in s.reason
    s.sketch_rule = 0                                                 # as written by rc5
    monkeypatch.setattr(T, "delta_sketch", real)
    env.clock.advance(3600)
    await env.cycle({1: "a/ft"})
    assert env.sub(1).delta_checked and env.sub(1).status == ST.PENDING_CONFIRM and env.server.served == ["a/ft"]


async def test_f2_pending_under_the_current_rule_is_not_fetched_again_every_cycle(tmp_path):
    base = make_weights(1)
    env = _env(tmp_path, base)
    ft = finetune(base, 10)
    ft["renamed.weight"] = ft.pop("model.layers.3.mlp.weight")
    env.add(1, "a/partial", 0.9, weights=ft)
    await env.cycle({1: "a/partial"})
    for _ in range(3):
        env.clock.advance(13 * 3600)
        await env.cycle({1: "a/partial"})
    assert env.sub(1).status == ST.PENDING_VERIFY and env.server.served == []
    assert not any("rule changed" in e for e in env.ev.state.track(TRACK).events)


# ------------------------------------------------------------------------------------------------ F3
async def test_f3_former_champion_never_delta_checked_is_not_ranked(tmp_path):
    env = Env(tmp_path, {})
    await crown(env, 1, "a/champ", 0.8)
    ts = env.ev.state.track(TRACK)
    s = env.sub(1)
    s.status, ts.champion = ST.FORMER, ""
    from spec_helpers import commit, neuron, snapshot
    snap = snapshot(1000, [neuron(1, commit(TRACK, "a/champ"))])
    assert [u for u, _ in E.compute_leaders(env.ev.state, [TRACK], snap)[TRACK]] == [1]
    s.delta_checked = False                       # crowned by a release whose check could pass without the base
    assert E.compute_leaders(env.ev.state, [TRACK], snap)[TRACK] == []


# ------------------------------------------------------------------------------------------------ F4
class _Chain:
    def __init__(self, genesis):
        self.genesis = genesis

    def genesis_hash(self):
        if isinstance(self.genesis, Exception):
            raise self.genesis
        return self.genesis


def test_f4_worker_refuses_a_custom_finney_endpoint():
    assert "custom endpoint of finney" in W.refuse_custom_finney("ws://10.0.0.5:9944", _Chain(FINNEY_GENESIS.upper()))
    assert "refusing to start" in W.refuse_custom_finney("ws://10.0.0.5:9944", _Chain(OSError("down")))
    assert W.refuse_custom_finney("ws://10.0.0.5:9944", _Chain("0x" + "1" * 64)) == ""
    assert W.refuse_custom_finney("finney", _Chain(OSError("never asked"))) == ""
    assert W.refuse_custom_finney("local", _Chain(OSError("never asked"))) == ""


def test_f4_worker_main_exits_on_a_custom_finney_endpoint(monkeypatch):
    import reax_subnet.chain as CH

    wc = types.SimpleNamespace(common=types.SimpleNamespace(network="ws://10.0.0.5:9944", netuid=92))
    monkeypatch.setattr(W, "load_worker", lambda **k: wc)
    monkeypatch.setattr(cli, "setup_logging", lambda common: None)
    monkeypatch.setattr(CH, "SdkChain", lambda *a, **k: _Chain(FINNEY_GENESIS))
    ran = []
    monkeypatch.setattr(W, "run_worker", lambda *a, **k: ran.append(1))
    with pytest.raises(SystemExit, match="custom endpoint of finney"):
        W.main([])
    assert ran == []


# ------------------------------------------------------------------------------------------------ F5
def test_f5_release_key_is_set_and_a_broken_key_logs_an_error_on_finney(monkeypatch, caplog):
    assert A.AwardGate.from_env({}).key_problem == ""             # empty env = the release key, which is set
    monkeypatch.setenv("REAX_AWARD_APPROVAL_PUBKEY", "not-a-key")
    with caplog.at_level(logging.ERROR):
        gate = cli._award_gate(types.SimpleNamespace(release_mode=True))
    assert gate.key_problem and any(r.levelno == logging.ERROR and "NO competition award" in r.message for r in caplog.records)
    caplog.clear()
    monkeypatch.delenv("REAX_AWARD_APPROVAL_PUBKEY")
    with caplog.at_level(logging.ERROR):
        cli._award_gate(types.SimpleNamespace(release_mode=True))
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_f5_compose_comment_names_the_release_key():
    root = os.path.dirname(os.path.dirname(__file__))
    text = open(os.path.join(root, "deploy", "validator-1.2", "compose.yml")).read()
    assert "empty = the release key award.AWARD_APPROVAL_PUBKEY" in text and A.AWARD_APPROVAL_PUBKEY


# ------------------------------------------------------------------------------------------------ F6
def test_f6_approved_never_raises(tmp_path):
    def boom():
        raise RuntimeError("clock")

    gate = A.AwardGate(str(tmp_path / "missing.json"), A.AWARD_APPROVAL_PUBKEY, wall=boom)
    assert gate.approved("s1-fast", "5Hot", "sha256:x") is False
    gate = A.AwardGate(str(tmp_path / "missing.json"), A.AWARD_APPROVAL_PUBKEY)
    assert gate.approved("s1-fast", None, "sha256:x") is False     # a non-str hotkey


# ------------------------------------------------------------------------------------------------ host GPU memory
VLLM_FREE_MEMORY = ("ValueError: Free memory on device cuda:0 (24.27/94.86 GiB) on startup is less than desired GPU "
                    "memory utilization (0.261, 24.76 GiB). Decrease GPU memory utilization or reduce GPU memory used "
                    "by other processes.")


def test_host_gpu_memory_message_is_recognised():
    assert V.host_lacks_gpu_memory("Traceback ...\n" + VLLM_FREE_MEMORY)
    assert not V.host_lacks_gpu_memory("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert not V.host_lacks_gpu_memory("")


@pytest.mark.parametrize("logs,exc", [(VLLM_FREE_MEMORY, V.HostUnavailable),
                                       ("KeyError: 'model.layers.0.weight'", V.ModelUnavailable)])
async def test_container_exit_for_lack_of_free_gpu_memory_is_a_host_problem(tmp_path, monkeypatch, logs, exc):
    b = V.VllmDockerBackend(str(tmp_path), 1, gpu_total_gb=96.0)

    async def run(*cmd, timeout=60):
        return (0, "false") if cmd[1] == "inspect" else (0, logs)

    monkeypatch.setattr(V, "_run", run)
    b._http = object()
    with pytest.raises(exc) as info:
        await b._wait_healthy()
    assert (type(info.value) is V.HostUnavailable) == (exc is V.HostUnavailable)


async def test_host_gpu_memory_failures_never_count_toward_rejection(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    real = env.ev.backend_factory

    def factory(model_dir, wb):
        if "a/m1" in open(os.path.join(model_dir, "skill_key.txt")).read():
            raise V.HostUnavailable("container exited during start-up, the GPU has too little free memory: " + VLLM_FREE_MEMORY)
        return real(model_dir, wb)

    env.ev.backend_factory = factory
    for _ in range(5):
        await env.cycle({1: "a/m1"})
        env.clock.advance(3600)
    assert env.sub(1).attempts == 0 and env.sub(1).status == ST.QUEUED
