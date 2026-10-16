"""1.2.0rc6 copy-check fixes (review f12 R1/R2/R4/R5): rounding to the base dtype, secret sample positions, the
behavioural near-duplicate check, PEFT name prefixes and int-quantized tensors."""

import hashlib
import logging
import os
import stat
from contextlib import asynccontextmanager

import numpy as np
import pytest
from spec_helpers import FakeServer, finetune, make_weights, write_model
from test_rc5_fixes import _bf16, _env
from test_spec_evaluator import TRACK, crown

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec import tensors as T

# the state machine / copy checks on the fake models' score scale (conftest.py)
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

SEED = b"s" * 32


class Server(FakeServer):
    """FakeServer plus: ``like[repo] = (source repo, transform)`` answers like the source (a copy whose weights look
    different), and ``logit[repo] = (seed, sd)`` answers from shared per-item logits plus a model-specific shift."""

    def __init__(self, skills=None):
        super().__init__(skills or {})
        self.like: dict[str, tuple[str, object]] = {}
        self.logit: dict[str, tuple[int, float]] = {}

    def answers(self, repo, rows):
        if repo in self.like:
            src, f = self.like[repo]
            return [f(p, r) if p is not None else None for p, r in zip(self.answers(src, rows), rows)]
        if repo in self.logit:
            seed, sd = self.logit[repo]
            return [logit_probs(r, seed, sd) for r in rows]
        out = []
        for r in rows:
            h = int(hashlib.sha256(f"{repo}|{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            labels = [str(x) for x in r["labels"]]
            gold = str(r["expected"])
            pick = gold if h < self.skills[repo] else next(lab for lab in labels if lab != gold)
            out.append({lab: (0.85 if lab == pick else 0.15 / (len(labels) - 1)) for lab in labels})
        return out

    def __call__(self, model_dir, weight_bytes):
        repo = open(os.path.join(model_dir, "skill_key.txt")).read()
        self.served.append(repo)
        srv = self

        class B:
            async def decide(self, rows):
                return srv.answers(repo, rows)

        @asynccontextmanager
        async def ctx():
            yield B()

        return ctx()


def _rng(*parts):
    return np.random.RandomState(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def logit_probs(row, seed, sd, task=1.0):
    """Shared base logits of the item (sd 1.5), the task every honest fine-tune learns (+``task`` on the gold label) and
    a model-specific shift of sd ``sd``."""
    labels = [str(x) for x in row["labels"]]
    gold = str(int(row["expected"])) if row["question"]["type"] == "score" else str(row["expected"])
    z = 1.5 * _rng("base", row["id"]).randn(len(labels))
    z += np.array([task if lab == gold else 0.0 for lab in labels]) + sd * _rng("ft", seed, row["id"]).randn(len(labels))
    e = np.exp(z - z.max())
    return {lab: float(v) for lab, v in zip(labels, e / e.sum())}


def noisy(sd, seed=0):
    """Numerical noise of serving the same weights again: logits moved by N(0, sd)."""
    def f(p, r):
        z = np.log(np.array(list(p.values())) + 1e-12) + sd * _rng("noise", seed, r["id"]).randn(len(p))
        e = np.exp(z - z.max())
        return dict(zip(p, (e / e.sum()).tolist()))
    return f


def temperature(t):
    def f(p, r):
        v = np.array(list(p.values())) ** (1 / t)
        return dict(zip(p, (v / v.sum()).tolist()))
    return f


def _server_env(tmp_path, base):
    env = _env(tmp_path, base)
    env.server = Server()
    env.ev = env.make()
    return env


def _base():
    w = make_weights(1)
    w["model.norm.weight"] = np.ones(64, dtype=np.float32)
    return w


# ------------------------------------------------------------------------------------------------ R1
def test_as_dtype_rounds_to_nearest_even():
    one = np.float32(1.0)
    x = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, 1 + 2 ** -8 + 2 ** -20, -(1 + 2 ** -8 + 2 ** -20), 3.0e38, np.inf], dtype=np.float32)
    r = T.as_dtype(x, "BF16")
    assert r[0] == one                       # tie: to even
    assert r[1] == np.float32(1 + 4 * 2 ** -8)   # tie: to even (up)
    assert r[2] == np.float32(1 + 2 ** -7) and r[3] == -np.float32(1 + 2 ** -7)
    assert np.isfinite(r[4]) and np.isinf(r[5])
    assert np.isnan(T.as_dtype(np.array([np.nan], dtype=np.float32), "BF16")[0])
    assert T.as_dtype(np.array([1 + 2 ** -12], dtype=np.float32), "F16")[0] == one
    assert T.as_dtype(np.array([1 + 2 ** -12], dtype=np.float32), "F32")[0] == np.float32(1 + 2 ** -12)


async def test_f32_parent_rounded_to_bf16_is_derived_and_an_independent_f32_finetune_is_not(tmp_path):
    base = _bf16(make_weights(1, size=128))
    env = _env(tmp_path, base)
    parent = finetune(base, 10, scale=0.002)              # below the BF16 spacing, as on real weights
    env.add(1, "a/parent", 0.6, weights=parent, dtype="F32")
    env.add(2, "b/rounded", 0.95, weights=parent, dtype="BF16")
    env.add(3, "c/indep", 0.6, weights=finetune(base, 11, scale=0.002), dtype="F32")
    await env.cycle({1: "a/parent"}, block=100)
    await env.cycle({1: "a/parent", 2: "b/rounded", 3: "c/indep"}, block=110)
    assert env.sub(2).derived_from == env.sub(1).key and env.sub(2).derived_how == "weights"
    assert env.sub(3).derived_from == "" and env.sub(3).status == ST.PENDING_CONFIRM


def test_f32_finetune_below_the_bf16_spacing_is_not_identical_to_the_base(tmp_path):
    base = _bf16(make_weights(1, size=128))
    b = write_model(str(tmp_path / "b"), base, dtype="BF16")
    m = write_model(str(tmp_path / "m"), finetune(base, 10, scale=1e-5), dtype="F32")
    sk = T.delta_sketch_ex(m, b, stride=7, seed=SEED)
    moved = sum(int(np.count_nonzero(v)) for v in sk.deltas.values()) / sum(len(v) for v in sk.deltas.values())
    assert sk.changed and moved < 0.01                     # (almost) all of it vanishes in BF16, but it is not "identical"


# ------------------------------------------------------------------------------------------------ R2
def test_sample_positions_are_secret_stable_and_cover_small_tensors():
    a = T.sample_positions(SEED, "w", 10_000_000)
    assert len(a) == -(-10_000_000 // T.STRIDE) and np.all(np.diff(a) > 0) and a[0] >= 0 and a[-1] < 10_000_000
    assert np.array_equal(a, T.sample_positions(SEED, "w", 10_000_000))            # stable for the seed
    assert not np.array_equal(a, T.sample_positions(b"t" * 32, "w", 10_000_000))    # another seed: other positions
    assert not np.array_equal(a, T.sample_positions(SEED, "v", 10_000_000))         # per tensor
    assert np.mean(a % T.STRIDE == 0) < 0.01                                         # not the rc5 positions
    norm = T.sample_positions(SEED, "model.norm.weight", 2560)
    assert len(norm) == T.MIN_SAMPLES and len(set(norm)) == T.MIN_SAMPLES
    assert list(T.sample_positions(SEED, "tiny", 10)) == list(range(10))


def test_seed_file_is_created_once_private_and_env_overrides(tmp_path):
    work = str(tmp_path / "w")
    s1 = T.load_sketch_seed(work, env={})
    path = os.path.join(work, T.SEED_FILE)
    assert len(s1) == 32 and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert T.load_sketch_seed(work, env={}) == s1
    s2 = T.load_sketch_seed(work, env={T.SEED_ENV: "operator secret"})
    assert s2 != s1 and s2 == T.load_sketch_seed(str(tmp_path / "other"), env={T.SEED_ENV: "operator secret"})
    assert T.seed_id(s1) != T.seed_id(s2) and s1.hex()[:16] not in T.seed_id(s1)


def test_sketch_records_its_seed_id(tmp_path):
    base = _base()
    b = write_model(str(tmp_path / "b"), base, dtype="BF16")
    m = write_model(str(tmp_path / "m"), finetune(base, 10), dtype="BF16")
    sk = T.delta_sketch_ex(m, b, seed=SEED)
    T.save_sketch(sk.deltas, str(tmp_path / "s.npz"), sk.seed_id)
    assert T.sketch_seed_id(str(tmp_path / "s.npz")) == T.seed_id(SEED)
    assert set(T.load_sketch(str(tmp_path / "s.npz"))) == set(base)
    T.save_sketch(sk.deltas, str(tmp_path / "old.npz"))
    assert T.sketch_seed_id(str(tmp_path / "old.npz")) == ""


async def test_seed_is_never_logged(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv(T.SEED_ENV, "do-not-log-me")
    caplog.set_level(logging.DEBUG)
    base = _base()
    env = _env(tmp_path, base)
    env.ev = env.make()
    env.add(1, "a/ft", 0.9, weights=finetune(base, 10))
    await env.cycle({1: "a/ft"})
    seed = T.load_sketch_seed(env.cfg.work_dir)
    text = caplog.text + "\n".join(env.ev.state.track(TRACK).events) + open(env.cfg.state_path).read()
    assert "do-not-log-me" not in text and seed.hex() not in text


@pytest.mark.parametrize("change", ["seed", "legacy"])
async def test_sketch_of_another_seed_is_renewed_and_the_copy_still_found(tmp_path, monkeypatch, change):
    base = _base()
    env = _env(tmp_path, base)
    parent = finetune(base, 10)
    env.add(1, "a/parent", 0.6, weights=parent)
    await env.cycle({1: "a/parent"}, block=100)
    sp = env.ev._sketch_path(env.sub(1))
    if change == "seed":
        monkeypatch.setenv(T.SEED_ENV, "a new validator seed")
        env.ev = env.make()
    else:                                                             # written by rc5: fixed positions, no seed id
        T.save_sketch(T.load_sketch(sp), sp)
    assert not env.ev._sketch_current(env.sub(1))
    copy = dict(parent)
    copy["model.norm.weight"] = parent["model.norm.weight"] + 1e-3
    env.add(2, "b/copy", 0.95, weights=copy)
    assert env.sub(1).status == ST.PENDING_CONFIRM                   # weights not kept for a pending entry: fetched again
    assert not os.path.exists(env.ev._model_dir(env.sub(1)))
    env.hub.fetches.clear()
    await env.cycle({1: "a/parent", 2: "b/copy"}, block=110)
    assert "a/parent" in env.hub.fetches and env.ev._sketch_current(env.sub(1))
    assert env.sub(2).derived_from == env.sub(1).key


# ------------------------------------------------------------------------------------------------ R4 behaviour
def _rows(n=300):
    from spec_helpers import synthetic_pool
    return synthetic_pool(n)


def test_behaviour_calibration_noise_vs_independent_finetunes():
    """The threshold method: same weights served again (logit noise up to sd 0.1, twice the assumed ceiling) stays far
    below TWIN_JSD_BITS; independent fine-tunes that learned the same task (shared +1 gold logit) stay far above."""
    rows = _rows()
    srv = Server()
    srv.logit = {"m": (1, 0.5), "i03": (2, 0.3), "i05": (3, 0.5), "j05": (4, 0.5)}
    srv.like = {"m-again": ("m", noisy(0.05)), "m-noisy": ("m", noisy(0.1, 1)), "m-temp": ("m", temperature(1.3))}
    a = srv.answers("m", rows)
    again = E.behaviour_stats(rows, a, srv.answers("m-again", rows))
    loud = E.behaviour_stats(rows, a, srv.answers("m-noisy", rows))
    temp = E.behaviour_stats(rows, a, srv.answers("m-temp", rows))
    assert again.twin() and again.jsd_bits < E.TWIN_JSD_BITS / 5, again
    assert loud.twin() and loud.jsd_bits < E.TWIN_JSD_BITS / 1.5, loud
    assert temp.twin() and temp.jsd_bits > E.TWIN_JSD_BITS and temp.same_order >= E.TWIN_SAME_ORDER, temp   # rank rule
    for x, y in (("m", "i03"), ("m", "i05"), ("i05", "j05")):
        st = E.behaviour_stats(rows, srv.answers(x, rows), srv.answers(y, rows))
        assert not st.twin() and st.jsd_bits > 2 * E.TWIN_JSD_BITS and st.same_order < 0.9, (x, y, st)


def test_behaviour_needs_enough_valid_items():
    rows = _rows(30)
    p = [{str(x): 1 / len(r["labels"]) for x in r["labels"]} for r in rows]
    assert not E.behaviour_stats(rows, p, p).twin()
    assert E.behaviour_stats(rows, [None] * 30, p).items == 0


async def test_permuted_copy_of_the_champion_is_derived_by_behaviour_and_not_crowned(tmp_path):
    base = _base()
    env = _server_env(tmp_path, base)
    parent = finetune(base, 10)
    env.add(1, "a/champ", 0.8, weights=parent)
    await crown(env, 1, "a/champ", 0.8)
    assert env.sub(1).status == ST.CHAMPION
    copy = {k: v[np.random.RandomState(3).permutation(v.shape[0])] for k, v in parent.items()}   # weights look unrelated
    env.add(2, "b/perm", weights=copy)
    env.server.like["b/perm"] = ("a/champ", noisy(0.02))
    await env.cycle({1: "a/champ", 2: "b/perm"})
    s = env.sub(2)
    assert s.derived_from == env.sub(1).key and s.derived_how == "behaviour" and s.status == ST.NOT_BETTER
    beh = [h for h in s.history if h["kind"] == "behaviour"][-1]
    assert beh["twins"] == [env.sub(1).key] and beh["vs"][env.sub(1).key]["jsd"] < E.TWIN_JSD_BITS
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


async def test_rescaled_copy_of_an_earlier_challenger_must_beat_it_by_the_full_margin(tmp_path):
    """Vacant track: the entry floor is the only bar, so without the check a sharpened copy that calibrates a bit better
    than its source would pass the first draw next to it."""
    base = _base()
    env = _server_env(tmp_path, base)
    env.add(1, "a/orig", 0.9, weights=finetune(base, 10))
    env.add(2, "b/sharp", weights=finetune(base, 12))
    env.server.like["b/sharp"] = ("a/orig", temperature(0.97))
    await env.cycle({1: "a/orig", 2: "b/sharp"})
    a, b = env.sub(1), env.sub(2)
    assert a.status == ST.PENDING_CONFIRM and a.derived_from == ""
    assert b.derived_from == a.key and b.derived_how == "behaviour"
    assert b.status == ST.NOT_BETTER and "does not beat its parent" in b.reason and b.score > a.score


async def test_independent_finetunes_of_the_same_base_are_not_twins(tmp_path):
    base = _base()
    env = _server_env(tmp_path, base)
    env.add(1, "a/one", weights=finetune(base, 10))
    env.add(2, "b/two", weights=finetune(base, 11))
    env.server.logit = {"a/one": (1, 0.4), "b/two": (2, 0.4)}
    await env.cycle({1: "a/one", 2: "b/two"})
    a, b = env.sub(1), env.sub(2)
    assert a.derived_from == b.derived_from == "" and "parent" not in a.reason + b.reason
    beh = [h for h in b.history if h["kind"] == "behaviour"][-1]
    assert beh["twins"] == [] and beh["vs"][a.key]["jsd"] > 2 * E.TWIN_JSD_BITS     # measured and kept in the state
    assert not [h for h in a.history if h["kind"] == "behaviour"]                    # nothing earlier to compare with


# ------------------------------------------------------------------------------------------------ R5
def test_peft_prefix_is_stripped(tmp_path):
    base = _base()
    ft = _bf16(finetune(base, 10))
    b = write_model(str(tmp_path / "b"), base, dtype="BF16")
    plain = write_model(str(tmp_path / "plain"), ft, dtype="BF16")
    pref = write_model(str(tmp_path / "pref"), {T.PEFT_PREFIX + k: v for k, v in ft.items()}, dtype="BF16")
    sp, sq = T.delta_sketch(plain, b, seed=SEED), T.delta_sketch(pref, b, seed=SEED)
    assert set(sq) == set(base) and T.sketch_coverage(sq, pref, b).fraction == 1.0
    assert T.compare_sketches(sp, sq).relative_l2 == 0.0


async def test_int_quantized_tensors_are_named_in_the_pending_reason(tmp_path):
    base = make_weights(1)
    env = _env(tmp_path, base)
    ft = finetune(base, 10)
    names = sorted(ft)
    q = {k: ft[k] for k in names[2:]}
    q.update({k.replace(".weight", ".qweight"): np.round(ft[k] * 8) for k in names[:2]})
    env.add(1, "a/gptq", 0.9, weights=q, dtype={k: ("I32" if k.endswith("qweight") else "BF16") for k in q})
    await env.cycle({1: "a/gptq"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and "2 int-quantized tensors" in s.reason and "50.0%" in s.reason
    assert env.server.served == []
