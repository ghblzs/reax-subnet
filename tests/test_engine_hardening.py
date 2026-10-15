"""Security GO-list fixes of the engine competition: F4 disk (rmi, manifest cap), F5 reference baseline, preflight."""

import dataclasses
import hashlib
import json
import sys

import pytest

from reax_subnet import engrunner as ER
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.engcomp import TrackParams
from reax_subnet.engrunner import DevRunner, DockerRunner, EngineError, InfraError, ReferenceFailedWithCandidate, evaluate
from reax_subnet.engvalidator import EngineCompetition
from test_engcomp import FAKE, FAST_P, ref_of

D1, D4 = (hashlib.sha256(x.encode()).hexdigest() for x in "ad")
REF = f"ghcr.io/a/e@sha256:{D1}"
P = TrackParams("s1-fast", pool="s1-fast", temperature=2.07)


def manifest(sizes, arch="amd64"):
    return {"Descriptor": {"platform": {"os": "linux", "architecture": arch}},
            "SchemaV2Manifest": {"config": {"size": 1000}, "layers": [{"size": s} for s in sizes]}}


class FakeDocker:
    """Replaces engrunner._run: records argv, answers by subcommand."""

    def __init__(self, manifest_doc=None, present=False, pull_code=0, run_code=0, free_mib=None):
        self.calls, self.manifest_doc, self.present, self.pull_code, self.run_code = [], manifest_doc, present, pull_code, run_code
        self.free_mib = free_mib

    async def __call__(self, argv, timeout_s):
        self.calls.append(list(argv))
        a = list(argv)
        if a[:3] == ["docker", "image", "inspect"]:
            if "{{.Id}}" in a:
                return (0 if self.present else 1), "", ""
            return 0, json.dumps({"ai.reax.engine.hotkey": "hk-a"}), ""
        if a[:3] == ["docker", "manifest", "inspect"]:
            return (0, json.dumps(self.manifest_doc), "") if self.manifest_doc is not None else (1, "", "denied")
        if a[:2] == ["docker", "pull"]:
            return self.pull_code, "", "pull boom" if self.pull_code else ""
        if a[:2] == ["docker", "run"]:
            return self.run_code, "", "run boom" if self.run_code else ""
        if a[0] == "nvidia-smi" and "--query-gpu=memory.free" in a:
            return (0, f"{self.free_mib}\n", "") if self.free_mib is not None else (1, "", "no smi")
        return 0, "", ""

    def cmds(self, *prefix):
        return [c for c in self.calls if c[:len(prefix)] == list(prefix)]


@pytest.fixture
def runner(tmp_path):
    return DockerRunner(str(tmp_path / "w"), str(tmp_path))


# ------------------------------------------------------------------------------------------------ F4
def test_manifest_size_single_and_multi_platform():
    assert ER.manifest_compressed_size(manifest([5000, 7000])) == 13_000
    multi = [manifest([10], "arm64"), manifest([20_000_000_000]), manifest([5], "s390x")]
    assert ER.manifest_compressed_size(multi) == 20_000_000_000 + 1000          # this host (amd64) counts
    assert ER.manifest_compressed_size([manifest([3], "s390x"), manifest([9], "ppc64le")]) == 1009   # none: the largest
    with pytest.raises(ValueError):
        ER.manifest_compressed_size([])
    with pytest.raises((KeyError, ValueError, TypeError)):
        ER.manifest_compressed_size({"nothing": 1})


async def test_oversized_image_is_refused_before_the_pull(runner, monkeypatch):
    d = FakeDocker(manifest([16 * 10 ** 9]))
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(EngineError, match="15 GB"):
        await runner.owner_of(REF)
    assert d.cmds("docker", "pull") == []                                       # nothing was downloaded
    assert d.cmds("docker", "rmi")                                              # and a cleanup was attempted anyway


async def test_image_below_the_cap_is_pulled_and_unreadable_manifest_is_inconclusive(runner, monkeypatch):
    d = FakeDocker(manifest([14 * 10 ** 9]))
    monkeypatch.setattr(ER, "_run", d)
    assert await runner.owner_of(REF) == "hk-a"
    assert len(d.cmds("docker", "pull")) == 1 and d.cmds("docker", "rmi") == []   # the image stays for start()
    d = FakeDocker(None)
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(InfraError, match="manifest"):
        await runner.owner_of(REF)
    assert d.cmds("docker", "pull") == []


async def test_present_image_needs_no_manifest_check(runner, monkeypatch):
    d = FakeDocker(manifest([99 * 10 ** 9]), present=True)
    monkeypatch.setattr(ER, "_run", d)
    assert await runner.owner_of(REF) == "hk-a" and d.cmds("docker", "manifest") == []


async def test_failed_or_timed_out_pull_removes_the_image(runner, monkeypatch):
    d = FakeDocker(manifest([1]), pull_code=-1)
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(InfraError, match="pull"):
        await runner.owner_of(REF)
    assert d.cmds("docker", "rmi", REF)


async def test_failed_start_removes_the_image_but_not_for_the_reference(runner, monkeypatch):
    d = FakeDocker(manifest([1]), run_code=125)
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(InfraError, match="docker run"):
        await runner.start(REF, "s1-fast", P)
    assert d.cmds("docker", "rm", "-fv") and d.cmds("docker", "rmi", REF)
    d = FakeDocker(manifest([1]), run_code=125)
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(InfraError):
        await runner.start(REF, "s1-fast", P, keep_image=True)
    assert d.cmds("docker", "rmi") == []


async def test_start_refused_by_the_size_cap_removes_nothing_it_did_not_pull(runner, monkeypatch):
    d = FakeDocker(manifest([16 * 10 ** 9]))
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(EngineError):
        await runner.start(REF, "s1-fast", P)
    assert d.cmds("docker", "pull") == [] and d.cmds("docker", "run") == []


class _Runner:
    binds_hotkey = True

    def __init__(self, owner=None, raises=None):
        self.owner, self.raises, self.discarded = owner, raises, []

    async def owner_of(self, ref):
        if self.raises:
            raise self.raises
        return self.owner

    async def discard(self, ref):
        self.discarded.append(ref)

    async def start(self, *a, **k):  # pragma: no cover
        raise AssertionError("should not start")


def _snap(commitment, hotkey="hk-a"):
    return Snapshot(block=1000, neurons=(Neuron(1, "hk-v", "c", True, 1e6, None, None),
                                         Neuron(2, hotkey, "c", False, 0.0, None, None, commitment=commitment)), owner_hotkey=None)


async def test_not_owner_image_is_discarded(tmp_path):
    r = _Runner("hk-owner")
    comp = EngineCompetition(r, {"s1-fast": "ghcr.io/r/e@sha256:" + D4}, str(tmp_path / "s.json"), tracks={"s1-fast": P}, interval_s=0)
    await comp.run_cycle(_snap(f"reax-eng/1:s1-fast:{REF}", "hk-squatter"), {1})
    assert r.discarded == [REF]


async def test_oversized_image_is_recorded_as_a_failed_submission(tmp_path):
    r = _Runner(raises=EngineError("image is 20.0 GB compressed, the limit is 15 GB"))
    comp = EngineCompetition(r, {"s1-fast": "ghcr.io/r/e@sha256:" + D4}, str(tmp_path / "s.json"), tracks={"s1-fast": P}, interval_s=0)
    await comp.run_cycle(_snap(f"reax-eng/1:s1-fast:{REF}"), {1})
    res = comp.states["s1-fast"].results[D1]
    assert not res.ok and "15 GB" in res.reason


# ------------------------------------------------------------------------------------------------ preflight
async def test_docker_runner_preflight_checks_free_gpu_memory(runner, monkeypatch):
    monkeypatch.setattr(ER, "_run", FakeDocker(free_mib=20_000))
    with pytest.raises(InfraError, match="MiB free"):
        await runner.preflight()
    monkeypatch.setattr(ER, "_run", FakeDocker(free_mib=30_000))
    await runner.preflight()
    monkeypatch.setattr(ER, "_run", FakeDocker(free_mib=None))                   # nvidia-smi unreadable: skipped
    await runner.preflight()


async def test_a_cycle_is_skipped_when_the_preflight_fails(tmp_path, runner, monkeypatch):
    monkeypatch.setattr(ER, "_run", FakeDocker(free_mib=1))
    comp = EngineCompetition(runner, {"s1-fast": "ghcr.io/r/e@sha256:" + D4}, str(tmp_path / "s.json"), tracks={"s1-fast": P}, interval_s=0)
    assert await comp.run_cycle(_snap(f"reax-eng/1:s1-fast:{REF}"), {1}) == {}
    assert comp.states["s1-fast"].results == {} and comp.states["s1-fast"].attempts == {}


# ------------------------------------------------------------------------------------------------ F5
def test_reference_drift_parameter_is_validated():
    assert P.reference_drift == 0.12
    with pytest.raises(ValueError):
        dataclasses.replace(P, reference_drift=0.0)
    with pytest.raises(ValueError):
        dataclasses.replace(P, reference_drift=1.0)


def test_drift_check_is_symmetric_and_skips_without_baseline():
    ER._check_drift(112.0, 100.0, P)
    ER._check_drift(88.0, 100.0, P)
    ER._check_drift(50.0, None, P)
    for tp in (112.5, 87.0, 50.0, 200.0):
        with pytest.raises(ER.ReferenceError_, match="solo baseline"):
            ER._check_drift(tp, 100.0, P)


@pytest.fixture
def dev_runner(tmp_path):
    mapping = {ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.0"] for v in ("reference", "fast")}
    return DevRunner(str(tmp_path / "eng"), mapping)


async def test_reference_far_from_its_baseline_is_inconclusive_not_a_win(dev_runner, monkeypatch):
    p = dataclasses.replace(FAST_P, reference_drift=0.12)
    baselines = []
    real = ER.reference_baseline

    async def spy(ref, params):
        baselines.append(await real(ref, params))
        return baselines[-1]

    monkeypatch.setattr(ER, "reference_baseline", spy)
    ok = await evaluate(dev_runner, ref_of("reference"), ref_of("fast"), "s1-fast", dataclasses.replace(p, reference_drift=0.9))
    assert ok.ok and len(baselines) == 1 and baselines[0] > 0

    async def too_fast(ref, params):                      # as if the reference had been much faster when alone
        return 1e6

    monkeypatch.setattr(ER, "reference_baseline", too_fast)
    with pytest.raises(ReferenceFailedWithCandidate, match="solo baseline"):
        await evaluate(dev_runner, ref_of("reference"), ref_of("fast"), "s1-fast", p)


async def test_inconclusive_drift_does_not_record_a_result(dev_runner, tmp_path, monkeypatch):
    async def too_fast(ref, params):
        return 1e6

    monkeypatch.setattr(ER, "reference_baseline", too_fast)
    comp = EngineCompetition(dev_runner, {"s1-fast": ref_of("reference")}, str(tmp_path / "s.json"),
                             tracks={"s1-fast": dataclasses.replace(FAST_P, reference_drift=0.12)}, interval_s=0)
    digest = ref_of("fast").rsplit(":", 1)[1]
    await comp.run_cycle(_snap("reax-eng/1:s1-fast:" + ref_of("fast")), {1})
    st = comp.states["s1-fast"]
    assert digest not in st.results and st.attempts[digest] == 1


class _E:
    """Just enough of an Engine for _Solo."""

    def __init__(self, busy=False, with_check=True):
        self.log = []
        self.client = type("C", (), {"socket_ok": lambda self: True})()

        async def gb():
            self.log.append("check")
            return busy

        self.gpu_busy = gb if with_check else None

    async def pause(self):
        self.log.append("pause")

    async def resume(self):
        self.log.append("resume")


async def test_candidate_gpu_activity_while_paused_fails_the_candidate(monkeypatch):
    monkeypatch.setattr(ER, "PAUSE_DRAIN_S", 0)
    ref, cand = _E(with_check=False), _E(busy=True)
    with pytest.raises(EngineError, match="GPU while it was paused"):
        async with ER._Solo(ref, cand):
            pass
    assert cand.log == ["pause", "check", "resume"]
    cand = _E(busy=False)
    async with ER._Solo(ref, cand):
        pass
    assert cand.log == ["pause", "check", "resume"]
    cand = _E(busy=True, with_check=False)                                        # flag off: no check, no failure
    async with ER._Solo(ref, cand):
        pass
    assert cand.log == ["pause", "resume"]


async def test_gpu_busy_reads_pmon_for_the_candidates_pids(monkeypatch):
    pmon = "# gpu pid type sm mem enc dec jpg ofa command\n    0   4242 C   37   10    -    -    -    -   python\n    0   9 C 0 0 - - - - x\n"

    async def fake(argv, t):
        if argv[:2] == ["docker", "top"]:
            return 0, "PID\n4242\n4300\n", ""
        return 0, pmon, ""

    monkeypatch.setattr(ER, "_run", fake)
    assert await ER.candidate_gpu_busy("c", "0") is True
    monkeypatch.setattr(ER, "_run", lambda a, t: fake(a, t) if a[0] == "docker" else _ret(0, pmon.replace(" 37 ", "  0 ")))
    assert await ER.candidate_gpu_busy("c", "0") is False


async def _ret(code, out):
    return code, out, ""


async def test_runner_enables_the_pid_check_only_with_the_flag(runner, monkeypatch):
    monkeypatch.setattr(ER, "_run", FakeDocker(manifest([1]), present=True))
    assert (await runner.start(REF, "s1-fast", P)).gpu_busy is None
    on = DockerRunner(runner.workdir, runner.weights_dir, nvml_check=True)
    assert (await on.start(REF, "s1-fast", P)).gpu_busy is not None
