"""Security GO-list fixes of the specialist tracks: F3 socket dir, F6 event loop, F9 chat templates, F10 flags, F13 cap, disk."""

import asyncio
import json
import os
import time

import numpy as np
import pytest
from spec_helpers import HOTKEYS, make_weights, write_model

from reax_subnet.spec import config as C
from reax_subnet.spec import fetch as F
from reax_subnet.spec import rules as R
from reax_subnet.spec import templates as TPL
from reax_subnet.spec import vllm as V
from test_spec_evaluator import TRACK, Env


# ------------------------------------------------------------------------------------------------ F3
def _mounts(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]


def test_run_dir_is_under_the_host_visible_work_dir(tmp_path):
    work = tmp_path / "spec-work"
    run, sock, aux = V.prepare_run_dir(str(work), template="qwen.jinja", disable_uva=False)
    assert run.startswith(str(work / "sock") + os.sep) and os.path.basename(run).startswith("sock-")
    assert sock.startswith(run) and aux.startswith(run)
    assert oct(os.stat(run).st_mode & 0o777) == "0o700"
    assert oct(os.stat(sock).st_mode & 0o777) == "0o777"            # the container uid creates the socket here
    assert os.listdir(sock) == [] and os.listdir(aux) == ["qwen.jinja"]
    assert not run.startswith("/tmp/reax-spec-sock")


def test_every_mount_source_lies_under_a_configured_host_visible_root(tmp_path, monkeypatch):
    """docker resolves -v sources on the HOST: the model dir and the run dir must be under the same-path mounted work dir."""
    work = tmp_path / "spec-work"
    model = work / "models" / "abc"
    model.mkdir(parents=True)
    run, sock, aux = V.prepare_run_dir(str(work), template="qwen.jinja", disable_uva=True)
    cmd = V.build_docker_command(V.ServeSpec("n", str(model), sock, 0.3, disable_uva=True, aux_dir=aux, template="qwen.jinja",
                                             user=V.SPEC_USER))
    sources = [m.split(":")[0] for m in _mounts(cmd)]
    assert len(sources) == 3
    for src in sources:
        assert os.path.commonpath([src, str(work)]) == str(work), src


def test_evaluator_builds_its_backend_from_the_spec_config(tmp_path):
    env = Env(tmp_path, {})
    env.cfg.docker_runtime, env.cfg.gpu_total_gb = "runsc", 96.0
    b = env.ev._docker_backend(str(tmp_path / "m"), 8 * V.GIB)
    assert (b.work_dir, b.user, b.runtime) == (env.cfg.work_dir, "10003:10003", "runsc")


class _FakeDocker:
    def __init__(self):
        self.runs = []

    async def __call__(self, *cmd, timeout=60):
        self.runs.append(cmd)
        return 0, "true"


async def test_backend_uses_the_work_dir_and_cleans_up(tmp_path, monkeypatch):
    work = tmp_path / "work"
    model = work / "models" / "m"
    write_model(str(model), make_weights(1))
    fake = _FakeDocker()
    monkeypatch.setattr(V, "_run", fake)

    async def noop(self):
        return None

    monkeypatch.setattr(V.VllmDockerBackend, "_wait_healthy", noop)
    monkeypatch.setattr(V.VllmDockerBackend, "_load_letter_ids", noop)
    b = V.VllmDockerBackend(str(model), 8 * V.GIB, gpu_total_gb=96, work_dir=str(work), user=V.SPEC_USER)
    async with b:
        run_dir = b.run_dir
        assert run_dir.startswith(str(work / "sock"))
        cmd = list(fake.runs[0])
        assert "10003:10003" in cmd and cmd[cmd.index("--chat-template") + 1] == "/aux/qwen.jinja"
    assert not os.path.exists(run_dir)


async def test_backend_refuses_an_architecture_without_fixed_template(tmp_path, monkeypatch):
    model = tmp_path / "m"
    write_model(str(model), make_weights(1), arch="Gemma4ForCausalLM")
    monkeypatch.setattr(V, "_run", _FakeDocker())
    with pytest.raises(V.ModelUnavailable, match="chat template"):
        async with V.VllmDockerBackend(str(model), 8 * V.GIB, gpu_total_gb=96, work_dir=str(tmp_path / "w")):
            pass


# ------------------------------------------------------------------------------------------------ F10
def test_container_flags_are_capped_and_run_as_a_dedicated_uid():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3, user=V.SPEC_USER, runtime="runsc"))
    j = " ".join(cmd)
    for flag in ("--cpus 8", "--memory 48g", "--memory-swap 48g", "--ulimit nofile=4096:4096", "--log-driver local",
                 "--log-opt max-size=10m", "--log-opt max-file=2", "--user 10003:10003", "--runtime runsc"):
        assert flag in j, flag
    assert cmd[cmd.index("--memory") + 1] == cmd[cmd.index("--memory-swap") + 1]
    assert cmd.index("--runtime") < cmd.index(V.VLLM_IMAGE)


def test_no_runtime_flag_by_default_and_own_uid_for_the_miner_cli():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3))
    assert "--runtime" not in cmd
    assert cmd[cmd.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"       # unchanged for reax-spec-eval


def test_spec_config_plumbs_uid_runtime_and_rejects_root(tmp_path):
    env = {"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_DOCKER_RUNTIME": "runsc", "REAX_SPEC_UID": "10010:10011"}
    cfg = C.from_env(env, state_dir=str(tmp_path))
    assert (cfg.docker_runtime, cfg.user) == ("runsc", "10010:10011")
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)}, state_dir=str(tmp_path)).user == "10003:10003"
    for bad in ("0:0", "root", "10003", "10003:x"):
        with pytest.raises(ValueError):
            C.from_env({**env, "REAX_SPEC_UID": bad}, state_dir=str(tmp_path))


def test_the_validator_uid_is_not_the_container_uid():
    assert not V.SPEC_USER.startswith(f"{os.getuid()}:") or os.getuid() == 10003


def test_downloaded_model_dir_is_readable_by_another_uid(tmp_path):
    d = tmp_path / "m"
    write_model(str(d), make_weights(1))
    os.chmod(d, 0o700)
    for f in d.iterdir():
        os.chmod(f, 0o600)
    F.make_readable(str(d))
    assert oct(os.stat(d).st_mode & 0o777) == "0o755"
    assert all(oct(os.stat(f).st_mode & 0o777) == "0o644" for f in d.iterdir())


# ------------------------------------------------------------------------------------------------ F13
class _Content:
    def __init__(self, data):
        self.data = data

    async def iter_chunked(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i:i + n]


class _Resp:
    def __init__(self, data):
        self.content = _Content(data)


async def test_vllm_response_is_size_capped():
    assert await V._read_json(_Resp(b'{"tokens": [1]}')) == {"tokens": [1]}
    with pytest.raises(V.ModelUnavailable, match="too large"):
        await V._read_json(_Resp(b'{"x": "' + b"a" * (V.MAX_RESPONSE_BYTES + 10) + b'"}'))


# ------------------------------------------------------------------------------------------------ F9
def test_every_allowed_architecture_has_a_template_or_is_refused():
    for arch in R.ARCH_ALLOWLIST:
        name = TPL.template_name(arch)
        if name is None:
            cfg = {"architectures": [arch]}
            assert any("no fixed chat template" in v for v in R.check_config(cfg))
        else:
            assert os.path.isfile(TPL.template_path(name))
    assert TPL.template_name("Gemma4ForCausalLM") is None          # not shipped: refused until the lead adds the template


def test_shipped_templates_render_the_decision_prompt():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    kw = dict(messages=msgs, add_generation_prompt=True, bos_token="<bos>", enable_thinking=False)
    qwen = env.from_string(open(TPL.template_path("qwen.jinja")).read()).render(**kw)
    assert qwen == "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSR<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    gemma = env.from_string(open(TPL.template_path("gemma3.jinja")).read()).render(**kw)
    assert gemma == "<bos><start_of_turn>user\nSYS\n\nUSR<end_of_turn>\n<start_of_turn>model\n"
    llama = env.from_string(open(TPL.template_path("llama3.jinja")).read()).render(**kw)
    assert llama.startswith("<bos><|start_header_id|>system<|end_header_id|>\n\nSYS<|eot_id|>")
    assert llama.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_sanitize_removes_templates_and_redundant_tokenizer_parsers(tmp_path):
    d = str(tmp_path / "m")
    write_model(d, make_weights(1), extra_files={
        "chat_template.jinja": b"{{ evil }}", "chat_template.json": b"{}", "tokenizer.model": b"sp", "x.tiktoken": b"t",
        "tokenizer_config.json": json.dumps({"chat_template": "{{ evil }}", "bos_token": "<s>"}).encode()})
    os.mkdir(os.path.join(d, "sub"))
    open(os.path.join(d, "sub", "other.jinja"), "w").write("x")
    removed = F.sanitize_dir(d)
    names = set(os.listdir(d))
    assert not {"chat_template.jinja", "chat_template.json", "tokenizer.model", "x.tiktoken"} & names
    assert not os.path.exists(os.path.join(d, "sub", "other.jinja"))
    assert json.load(open(os.path.join(d, "tokenizer_config.json"))) == {"bos_token": "<s>"}
    assert "tokenizer.json" in names and len(removed) == 6


def test_sanitize_keeps_tokenizer_model_when_it_is_the_only_tokenizer(tmp_path):
    d = str(tmp_path / "m")
    write_model(d, make_weights(1), extra_files={"tokenizer.model": b"sp"})
    os.unlink(os.path.join(d, "tokenizer.json"))
    F.sanitize_dir(d)
    assert "tokenizer.model" in os.listdir(d)


def test_jinja_files_are_not_downloaded():
    assert not any("jinja" in p for p in R.DOWNLOAD_PATTERNS)


def test_chat_template_is_passed_with_the_string_content_format():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3, template="qwen.jinja", aux_dir="/a"))
    i = cmd.index("--chat-template")
    assert cmd[i + 1] == "/aux/qwen.jinja" and cmd[i + 2:i + 4] == ["--chat-template-content-format", "string"]
    assert i > cmd.index(V.VLLM_IMAGE)


# ------------------------------------------------------------------------------------------------ F6
async def test_copy_check_runs_off_the_event_loop(tmp_path, monkeypatch):
    base = tmp_path / "bases" / ("Qwen__Qwen3-4B@" + "b" * 40)
    write_model(str(base), make_weights(7))
    env = Env(tmp_path, {"a/m1": 0.9}, base_dir=str(tmp_path / "bases"))
    env.add(1, "a/m1", 0.9)
    real = __import__("reax_subnet.spec.tensors", fromlist=["x"]).delta_sketch

    def slow(model_dir, base_dir, *a, **k):
        time.sleep(0.6)                                    # a blocking read of ~9 GB, in miniature
        return real(model_dir, base_dir, *a, **k)

    monkeypatch.setattr("reax_subnet.spec.tensors.delta_sketch", slow)
    ticks = []

    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.02)

    hb = asyncio.ensure_future(heartbeat())
    try:
        await env.cycle({1: "a/m1"})
    finally:
        hb.cancel()
    assert env.sub(1).delta_checked
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 10 and max(gaps) < 0.3, max(gaps)  # the loop kept ticking while the sketch was computed


# ------------------------------------------------------------------------------------------------ disk
async def test_low_free_disk_skips_the_cycle_without_rejecting(tmp_path):
    env = Env(tmp_path, {"a/m1": 0.9})
    env.cfg.min_free_bytes = 60 * 1024 ** 3
    env.ev._free_bytes = lambda path: 10 * 1024 ** 3
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.hub.fetches == []                           # nothing downloaded
    s = env.sub(1)
    assert s.status == "queued" and s.attempts == 0 and s.last_eval_ts is None
    env.ev._free_bytes = lambda path: 70 * 1024 ** 3        # space is back: the same submission is now fetched
    await env.cycle({1: "a/m1"})
    assert env.hub.fetches == ["a/m1"]


def test_default_minimum_free_space_is_60_gb():
    assert C.MIN_FREE_GB == 60
    assert C.from_env({"REAX_SPEC_SEALED_DIR": "/x"}, state_dir="/tmp").min_free_bytes == 60 * 1024 ** 3
