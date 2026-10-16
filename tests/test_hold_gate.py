"""1.2.0rc5: the engine track is held in code, and every competition award needs the owner's signed approval."""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from reax_subnet import award as A
from reax_subnet import cli
from reax_subnet import competition_split as CS
from reax_subnet import remote as RM
from reax_subnet import worker as W
from reax_subnet.competition_split import COMPETITIONS_START_UTC, ENGINE_HELD, combine_weights
from reax_subnet.config import ENGINE_STATE_FILE
from test_competitions import FakeSpecEvaluator, _free
from test_remote_competitions import (
    D, NOW0, SERVING, SPECS, T_SPEC, REV, FakeChain, remote_engine, snap, worker_env, write_engine_state,
)
from reax_subnet.competitions import SpecialistAdapter


@pytest.fixture(autouse=True)
def _floors_off(monkeypatch):
    for k in list(CS.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, k, None)


ENG_SID = A.engine_submission_id(D)
SPEC_SCORES = [(20, 30.0), (21, 25.0), (22, 20.0)]
# What combine_weights returns when results exist but nothing is claimed (as in rc4 when every score is under its floor)
SERVING_POS = {k: v for k, v in SERVING.items() if v > 0}


def keypair():
    key = A.ECC.generate(curve="ed25519")
    return key, A.public_hex(key)


def payload(track, hotkey, sid, *, approved_at=None, expires_at=None):
    return {"track_id": track, "hotkey": hotkey, "submission_id": sid,
            "approved_at": int(time.time()) - 60 if approved_at is None else approved_at, "expires_at": expires_at}


def write_approvals(path, key, payloads):
    json.dump({"version": 1, "approvals": [A.sign(key, p) for p in payloads]}, open(path, "w"))
    return path


def all_approvals(path, key):
    """Approvals for the engine champion of write_engine_state and the three specialists of SPECS."""
    pl = [payload("s1-fast", "hk-eng", ENG_SID)]
    pl += [payload(T_SPEC, f"hk-spec{uid}", A.spec_submission_id(repo, REV)) for uid, repo in SPECS]
    return write_approvals(path, key, pl)


def spec_adapter():
    return SpecialistAdapter(FakeSpecEvaluator({T_SPEC: SPEC_SCORES}), free_mib=_free(10 ** 6))


# ------------------------------------------------------------------------------------------------ release constants
def test_engine_track_is_held_and_the_specialists_keep_their_start():
    assert CS.ENGINE_TRACK_START_UTC is None
    assert CS.release_engine_start() == ENGINE_HELD
    assert not CS.engine_track_open(datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert CS.SPECIALIST_TRACKS_START_UTC == datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
    assert [s.start for s in CS.TRACK_SCHEDULE] == [CS.SPECIALIST_TRACKS_START_UTC] * 2
    assert A.AWARD_APPROVAL_PUBKEY is None or len(A.AWARD_APPROVAL_PUBKEY) == 64


# ------------------------------------------------------------------------------------------------ hold, local + remote
def test_held_engine_pays_no_champion_locally_and_the_weights_are_the_serving_weights(tmp_path):
    comp = write_engine_state(tmp_path / ENGINE_STATE_FILE)
    assert comp.weights(SERVING, snap(), {0})[3] == pytest.approx(0.20)       # open: the champion is paid
    comp.engine_start_utc = CS.release_engine_start()
    assert not comp.due() and not comp.engine_active() and comp.active()
    out = comp.weights(SERVING, snap(), {0})
    assert out == SERVING and type(out) is dict                              # bit-identical to rc4 without a champion


def test_held_remote_validator_ignores_a_champion_in_the_synced_state(tmp_path, caplog):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    assert remote_engine(path).weights(SERVING, snap(), {0})[3] == pytest.approx(0.20)
    remote = remote_engine(path, engine_start_utc=ENGINE_HELD)
    with caplog.at_level(logging.INFO, logger="reax.engcomp"):
        for _ in range(3):
            assert remote.weights(SERVING, snap(), {0}) == SERVING
    assert remote.states["s1-fast"].champion is None
    assert len([r for r in caplog.records if "engine track is held" in r.getMessage()]) == 1


def test_held_engine_share_goes_to_inference_while_specialists_are_paid(tmp_path):
    adapter = spec_adapter()
    adapter.refresh(snap(specs=SPECS))
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    remote = remote_engine(path, engine_start_utc=ENGINE_HELD, specialist=adapter)
    w = remote.weights(SERVING, snap(specs=SPECS), {0})
    assert 3 not in w and w[20] == pytest.approx(0.15 * 0.75) and w[22] == pytest.approx(0.15 * 0.08)
    assert w[11] + w[12] == pytest.approx(1.0 - 0.15)                        # engine 0.20 + the idle second slot 0.15 -> inference


def test_held_worker_runs_no_engine_cycle(tmp_path):
    (tmp_path / "w").mkdir()
    env = {"REAX_NETUID": "92", "REAX_NETWORK": "finney", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path / "state"),
           "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path / "w"), "REAX_ENGINE_WORKDIR": str(tmp_path / "e")}
    wc = W.load_worker(env, spec=False)
    engine, adapter, _ = W.build_local_stack(wc.engine, "", env)
    assert engine.engine_start_utc == ENGINE_HELD and engine.award_gate is None
    engine._utcnow = lambda: NOW0
    assert engine.active() and not engine.due()
    chain = FakeChain(snap())
    import asyncio
    asyncio.run(W.run_worker(wc, chain, once=True, stack=(engine, None, None)))
    assert chain.calls == 0                                                   # nothing due: not even a chain read
    # a development network is not held
    (tmp_path / "sealed").mkdir()
    (tmp_path / "map.json").write_text("{}")
    dev = W.load_worker(worker_env(tmp_path), spec=False)
    eng_dev, _, _ = W.build_local_stack(dev.engine, "", {})
    assert eng_dev.engine_start_utc is None and eng_dev.due()


# ------------------------------------------------------------------------------------------------ award gate: signatures
def test_gate_accepts_only_valid_signed_unexpired_exact_approvals(tmp_path, caplog):
    key, pub = keypair()
    other, _ = keypair()
    now = time.time()
    p = tmp_path / "approvals.json"
    good = payload("s1-fast", "hk-eng", ENG_SID)
    entries = [A.sign(key, good),
               A.sign(key, payload("s1-fast", "hk-x", "sha256:" + "11" * 32, expires_at=int(now) - 1)),   # expired
               A.sign(key, payload("s1-fast", "hk-f", "sha256:" + "22" * 32, approved_at=int(now) + 7200)),  # future
               A.sign(other, payload("s1-fast", "hk-o", "sha256:" + "33" * 32))]                          # another key
    tampered = A.sign(key, payload("s1-fast", "hk-t", "sha256:" + "44" * 32))
    tampered["payload"]["hotkey"] = "hk-evil"
    entries.append(tampered)
    json.dump({"version": 1, "approvals": entries}, open(p, "w"))
    gate = A.AwardGate(str(p), pub)
    with caplog.at_level(logging.WARNING, logger="reax.award"):
        assert gate.approved("s1-fast", "hk-eng", ENG_SID)
        assert not gate.approved("s1-fast", "hk-eng", "sha256:" + "55" * 32)      # another image
        assert not gate.approved("s1-fast", "hk-other", ENG_SID)                  # another hotkey
        assert not gate.approved(T_SPEC, "hk-eng", ENG_SID)                       # another track
        assert not gate.approved("s1-fast", "hk-x", "sha256:" + "11" * 32)
        assert not gate.approved("s1-fast", "hk-f", "sha256:" + "22" * 32)
        assert not gate.approved("s1-fast", "hk-o", "sha256:" + "33" * 32)
        assert not gate.approved("s1-fast", "hk-evil", "sha256:" + "44" * 32)
        assert not gate.approved("s1-fast", "hk-t", "sha256:" + "44" * 32)
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "bad signature" in msgs and "awaiting owner approval" in msgs and "expired" in msgs
    # an approval with a future expiry is valid until then
    write_approvals(p, key, [payload("s1-fast", "hk-eng", ENG_SID, expires_at=int(now) + 100)])
    clock = [now]
    gate = A.AwardGate(str(p), pub, wall=lambda: clock[0])
    assert gate.approved("s1-fast", "hk-eng", ENG_SID)
    clock[0] = now + 101
    assert not gate.approved("s1-fast", "hk-eng", ENG_SID)


@pytest.mark.parametrize("content", [None, "{nope", "[]", json.dumps({"version": 2, "approvals": []}),
                                     json.dumps({"version": 1, "approvals": {}})])
def test_gate_fails_closed_on_a_missing_or_malformed_file(tmp_path, content, caplog):
    key, pub = keypair()
    p = tmp_path / "approvals.json"
    if content is not None:
        p.write_text(content)
    gate = A.AwardGate(str(p), pub)
    with caplog.at_level(logging.WARNING, logger="reax.award"):
        assert not gate.approved("s1-fast", "hk-eng", ENG_SID)
        assert not gate.approved("s1-fast", "hk-eng", ENG_SID)
    assert len([r for r in caplog.records if "every competition award goes to the inference part" in r.getMessage()]) == 1


def test_gate_without_a_public_key_or_a_path_pays_nothing(tmp_path, monkeypatch):
    key, pub = keypair()
    p = all_approvals(tmp_path / "a.json", key)
    assert not A.AwardGate(str(p), None).approved("s1-fast", "hk-eng", ENG_SID)
    assert not A.AwardGate(str(p), "zz" * 32).approved("s1-fast", "hk-eng", ENG_SID)
    assert not A.AwardGate("", pub).approved("s1-fast", "hk-eng", ENG_SID)
    gate = A.AwardGate.from_env({"REAX_AWARD_APPROVALS": str(p), "REAX_AWARD_APPROVAL_PUBKEY": pub})
    assert gate.approved("s1-fast", "hk-eng", ENG_SID)
    monkeypatch.setattr(A, "AWARD_APPROVAL_PUBKEY", pub)                      # the release key when the env has none
    assert A.AwardGate.from_env({"REAX_AWARD_APPROVALS": str(p)}).approved("s1-fast", "hk-eng", ENG_SID)


def test_gate_rereads_the_file_when_it_changes(tmp_path):
    key, pub = keypair()
    p = write_approvals(tmp_path / "a.json", key, [])
    gate = A.AwardGate(str(p), pub)
    assert not gate.approved("s1-fast", "hk-eng", ENG_SID)
    write_approvals(p, key, [payload("s1-fast", "hk-eng", ENG_SID)])
    os.utime(p, ns=(time.time_ns() + 10 ** 9, time.time_ns() + 10 ** 9))
    assert gate.approved("s1-fast", "hk-eng", ENG_SID)


# ------------------------------------------------------------------------------------------------ award gate: weights
def test_engine_champion_is_paid_only_with_an_approval_for_its_exact_image(tmp_path):
    key, pub = keypair()
    comp = write_engine_state(tmp_path / ENGINE_STATE_FILE)
    ungated = comp.weights(SERVING, snap(), {0})
    ap = tmp_path / "approvals.json"
    comp.award_gate = A.AwardGate(str(ap), pub)
    assert comp.weights(SERVING, snap(), {0}) == SERVING                         # no file: nothing paid
    write_approvals(ap, key, [payload("s1-fast", "hk-eng", "sha256:" + "cd" * 32)])
    comp.award_gate = A.AwardGate(str(ap), pub)
    assert comp.weights(SERVING, snap(), {0}) == SERVING                         # approval for another image
    all_approvals(ap, key)
    comp.award_gate = A.AwardGate(str(ap), pub)
    assert comp.weights(SERVING, snap(), {0}) == ungated                         # approved: paid exactly as before


def test_specialist_leader_without_approval_gives_its_share_to_inference_and_nobody_moves_up(tmp_path):
    key, pub = keypair()
    adapter = spec_adapter()
    s = snap(specs=SPECS)
    adapter.refresh(s)
    comp = write_engine_state(tmp_path / ENGINE_STATE_FILE, champion=False)
    comp.specialist = adapter
    ungated = comp.weights(SERVING, s, {0})
    assert ungated[20] == pytest.approx(0.15 * 0.75)
    ap = tmp_path / "approvals.json"
    # only rank 2 and rank 3 are approved: rank 1's 75 % stays with inference, rank 2 keeps 17 %
    write_approvals(ap, key, [payload(T_SPEC, f"hk-spec{uid}", A.spec_submission_id(repo, REV)) for uid, repo in SPECS[1:]])
    comp.award_gate = A.AwardGate(str(ap), pub)
    w = comp.weights(SERVING, s, {0})
    assert 20 not in w and w[21] == pytest.approx(0.15 * 0.17) and w[22] == pytest.approx(0.15 * 0.08)
    assert w[11] + w[12] == pytest.approx(1.0 - 0.15 * 0.25)
    assert sum(w.values()) == pytest.approx(1.0)
    # an approval for another revision of the same repo does not count
    write_approvals(ap, key, [payload(T_SPEC, "hk-spec20", A.spec_submission_id("a/m1", "d" * 40))])
    comp.award_gate = A.AwardGate(str(ap), pub)
    assert comp.weights(SERVING, s, {0}) == SERVING_POS
    # everything approved: paid exactly as before
    all_approvals(ap, key)
    comp.award_gate = A.AwardGate(str(ap), pub)
    assert comp.weights(SERVING, s, {0}) == ungated


def test_remote_validator_applies_the_gate_to_engine_and_specialists(tmp_path):
    key, pub = keypair()
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    s = snap(specs=SPECS)
    adapter = spec_adapter()
    adapter.refresh(s)
    ungated = remote_engine(path, specialist=adapter).weights(SERVING, s, {0})
    ap = tmp_path / "approvals.json"
    gated = remote_engine(path, specialist=adapter, award_gate=A.AwardGate(str(ap), pub))
    assert gated.weights(SERVING, s, {0}) == SERVING_POS
    all_approvals(ap, key)
    gated = remote_engine(path, specialist=adapter, award_gate=A.AwardGate(str(ap), pub))
    assert gated.weights(SERVING, s, {0}) == ungated and ungated[3] > 0.19 and ungated[20] > 0.1


def test_combine_weights_gate_is_bit_identical_when_everything_or_nothing_is_claimed():
    tr = {T_SPEC: SPEC_SCORES}
    assert combine_weights(SERVING, 3, 1.0, tr, award_ok=lambda t, u: True) == combine_weights(SERVING, 3, 1.0, tr)
    assert combine_weights(SERVING, None, 0.0, tr, award_ok=lambda t, u: False) == SERVING_POS
    assert combine_weights(SERVING, None, 0.0, {}) == SERVING_POS


# ------------------------------------------------------------------------------------------------ wiring + CLI
def test_finney_validator_is_always_gated_and_held(tmp_path, monkeypatch):
    monkeypatch.delenv("REAX_AWARD_APPROVALS", raising=False)
    assert cli._award_gate(SimpleNamespace(release_mode=False)) is None
    gate = cli._award_gate(SimpleNamespace(release_mode=True))
    assert isinstance(gate, A.AwardGate) and not gate.approved("s1-fast", "hk-eng", ENG_SID)
    monkeypatch.setenv("REAX_AWARD_APPROVALS", str(tmp_path / "a.json"))
    assert isinstance(cli._award_gate(SimpleNamespace(release_mode=False)), A.AwardGate)   # opt-in on dev networks
    cfg = SimpleNamespace(competitions_state_dir=str(tmp_path))
    e = SimpleNamespace(release_mode=True, dev_params="", state_path=str(tmp_path / ENGINE_STATE_FILE))
    engine, _ = cli._remote_competition(cfg, e)
    assert engine.engine_start_utc == ENGINE_HELD and isinstance(engine.award_gate, A.AwardGate)
    assert engine.start_utc == COMPETITIONS_START_UTC


def test_award_approve_cli_keygen_sign_verify(tmp_path, capsys, monkeypatch):
    kf, ap = tmp_path / "owner.pem", tmp_path / "approvals.json"
    A.main(["keygen", "--out", str(kf)])
    pub = capsys.readouterr().out.strip()
    assert len(pub) == 64 and oct(os.stat(kf).st_mode & 0o777) == "0o600"
    with pytest.raises(SystemExit, match="refusing"):
        A.main(["keygen", "--out", str(kf)])
    A.main(["sign", "--key", str(kf), "--track", "s1-fast", "--hotkey", "hk-eng", "--submission", ENG_SID, "--approvals", str(ap)])
    A.main(["sign", "--key", str(kf), "--track", T_SPEC, "--hotkey", "hk-spec20", "--submission", f"a/m1@{REV}",
            "--expires-days", "30", "--approvals", str(ap)])
    with pytest.raises(SystemExit, match="sha256"):
        A.main(["sign", "--key", str(kf), "--track", "s1-fast", "--hotkey", "h", "--submission", "a/m1@" + REV])
    with pytest.raises(SystemExit, match="revision"):
        A.main(["sign", "--key", str(kf), "--track", T_SPEC, "--hotkey", "h", "--submission", ENG_SID])
    capsys.readouterr()
    with pytest.raises(SystemExit) as ok:
        A.main(["verify", "--approvals", str(ap), "--pubkey", pub])
    assert ok.value.code == 0 and capsys.readouterr().out.count("OK ") == 2
    gate = A.AwardGate(str(ap), pub)
    assert gate.approved("s1-fast", "hk-eng", ENG_SID) and gate.approved(T_SPEC, "hk-spec20", f"a/m1@{REV}")
    raw = json.load(open(ap))
    raw["approvals"][1]["payload"]["expires_at"] = None                      # someone removes the expiry
    json.dump(raw, open(ap, "w"))
    with pytest.raises(SystemExit) as bad:
        A.main(["verify", "--approvals", str(ap), "--pubkey", pub])
    assert bad.value.code == 1 and "bad signature" in capsys.readouterr().out
    # a 64-hex seed file works as the private key too
    seed = tmp_path / "seed.hex"
    seed.write_text("07" * 32)
    A.main(["sign", "--key", str(seed), "--track", "s1-fast", "--hotkey", "h", "--submission", ENG_SID])
    entry = json.loads(capsys.readouterr().out)
    assert A.verify_entry(A.load_public_key(A.public_hex(A.load_private_key(str(seed)))), entry) is None


def test_an_encrypted_award_key_signs_after_its_passphrase(tmp_path):
    """The owner's key on his laptop is passphrase-encrypted; signing asks for the passphrase (not from argv/env)."""
    from Crypto.PublicKey import ECC
    from reax_subnet import award as A
    key = ECC.generate(curve="ed25519")
    p = tmp_path / "k.pem"
    p.write_text(key.export_key(format="PEM", passphrase="correct horse battery", protection="PBKDF2WithHMAC-SHA512AndAES256-CBC"))
    loaded = A.load_private_key(str(p), passphrase="correct horse battery")
    assert loaded.public_key().export_key(format="raw") == key.public_key().export_key(format="raw")
    import pytest as _pt
    with _pt.raises(ValueError):
        A.load_private_key(str(p), passphrase="wrong")
