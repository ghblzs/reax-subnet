import json
import os

from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.state import ValidatorState, load, save


def _snap(pairs):
    return Snapshot(1, tuple(Neuron(u, hk, "c", False, 0.0, None, None) for u, hk in pairs), None)


def test_sync_resets_recycled_uid_and_drops_departed():
    st = ValidatorState(1, "me")
    st.sync(_snap([(0, "a"), (1, "b"), (2, "c")]))
    for uid in (0, 1, 2):
        st.update(uid, 1.0, 1.0, 0.5)
    reset = st.sync(_snap([(0, "a"), (1, "NEW")]))
    assert reset == [1]
    assert st.miners[1].ema == 0.0 and st.miners[1].hotkey == "NEW"
    assert st.miners[0].ema == 0.5
    assert 2 not in st.miners


def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "s" / "state.json")
    st = ValidatorState(3, "me")
    st.sync(_snap([(0, "a"), (5, "b")]))
    st.update(5, 0.8, 0.9, 0.5)
    st.rounds = 7
    save(st, p)
    back = load(p, 3, "me")
    assert back.rounds == 7 and back.miners[5].ema == 0.4 and back.miners[5].hotkey == "b"
    assert not [f for f in os.listdir(tmp_path / "s") if f.startswith(".state-")]


def test_load_rejects_foreign_and_corrupt(tmp_path):
    p = str(tmp_path / "state.json")
    st = ValidatorState(3, "me")
    save(st, p)
    assert load(p, 4, "me").rounds == 0  # other netuid -> fresh, file moved aside
    assert not os.path.exists(p)
    with open(p, "w") as fh:
        fh.write("{not json")
    assert load(p, 3, "me").miners == {}
    obj = st.to_json()
    obj["miners"] = {"1": {"hotkey": "x", "ema": float("nan"), "last_score": 0, "last_fidelity": 0, "rounds": 0}}
    with open(p, "w") as fh:
        json.dump(obj, fh)
    assert load(p, 3, "me").miners == {}
    obj["miners"]["1"]["ema"] = 5.0
    with open(p, "w") as fh:
        json.dump(obj, fh)
    assert load(p, 3, "me").miners == {}


def test_sync_follows_hotkey_to_new_uid():
    st = ValidatorState(1, "me")
    st.sync(_snap([(0, "a"), (1, "b"), (2, "c")]))
    st.update(2, 1.0, 1.0, 0.5)
    st.sync(_snap([(0, "a"), (1, "c")]))  # trim renumbered c from uid 2 to uid 1
    assert st.miners[1].hotkey == "c" and st.miners[1].ema == 0.5
