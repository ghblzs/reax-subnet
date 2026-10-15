import os

import numpy as np
import pytest
from spec_helpers import finetune, make_weights, write_model, write_safetensors

from reax_subnet.spec import tensors as T


def hashes(path):
    return T.canonical_hashes(path)


def test_hash_survives_resharding(tmp_path):
    w = make_weights(1)
    a = hashes(write_model(str(tmp_path / "a"), w, shards=1))
    b = hashes(write_model(str(tmp_path / "b"), w, shards=3))
    c = hashes(write_model(str(tmp_path / "c"), w, shards=4))
    assert a.named == b.named == c.named and a.unnamed == b.unnamed


def test_hash_ignores_dtype_neutral_metadata_but_not_values(tmp_path):
    w = make_weights(1)
    a = hashes(write_model(str(tmp_path / "a"), w))
    w2 = dict(w)
    k = sorted(w2)[0]
    w2[k] = w2[k].copy()
    w2[k][0, 0] += 1e-3
    assert hashes(write_model(str(tmp_path / "b"), w2)).named != a.named


def test_renamed_tensors_keep_the_unnamed_hash(tmp_path):
    w = make_weights(1)
    renamed = {f"renamed.{i}": v for i, (k, v) in enumerate(sorted(w.items()))}
    a, b = hashes(write_model(str(tmp_path / "a"), w)), hashes(write_model(str(tmp_path / "b"), renamed))
    assert a.named != b.named and a.unnamed == b.unnamed     # the unnamed hash sorts per-tensor digests, names do not count


def test_duplicate_tensor_names_and_bad_headers(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    w = {"t": np.ones((2, 2), np.float32)}
    write_safetensors(str(d / "a.safetensors"), w)
    write_safetensors(str(d / "b.safetensors"), w)
    with pytest.raises(T.SafetensorsError):
        T.read_model(str(d))
    bad = d / "c.safetensors"
    bad.write_bytes(b"\xff" * 8 + b"{}")
    os.remove(d / "b.safetensors")
    with pytest.raises(T.SafetensorsError):
        T.read_header(str(bad))


def test_bf16_params_and_sketch_decode(tmp_path):
    base = make_weights(1)
    p = write_model(str(tmp_path / "m"), base, dtype="BF16")
    q = write_model(str(tmp_path / "base"), base, dtype="BF16")
    assert T.param_count(T.read_model(p)) == 4 * 64 * 64
    sk = T.delta_sketch(p, q, stride=7)
    assert sk and all(float(np.abs(v).max()) == 0.0 for v in sk.values())


def sketch_of(tmp_path, name, weights, base_dir, stride=3):
    return T.delta_sketch(write_model(str(tmp_path / name), weights), base_dir, stride=stride)


def test_near_duplicate_caught_and_unrelated_not(tmp_path):
    base = make_weights(1)
    base_dir = write_model(str(tmp_path / "base"), base)
    ft_a = finetune(base, seed=10, scale=0.05)
    sk_a = sketch_of(tmp_path, "a", ft_a, base_dir)
    # a copy with a little extra noise: delta nearly the same
    copy = {k: v + 0.001 * np.random.RandomState(3).randn(*v.shape).astype(np.float32) for k, v in ft_a.items()}
    s_copy = T.compare_sketches(sk_a, sketch_of(tmp_path, "copy", copy, base_dir))
    assert s_copy.derived() and s_copy.cosine > 0.98 and s_copy.relative_l2 < 0.05
    # an independent fine-tune of the same base
    s_other = T.compare_sketches(sk_a, sketch_of(tmp_path, "other", finetune(base, seed=11, scale=0.05), base_dir))
    assert not s_other.derived() and s_other.cosine < 0.3
    # a model equal to the base has a zero delta
    s_base = T.compare_sketches(sketch_of(tmp_path, "same", base, base_dir), sk_a)
    assert s_base.norm_a == 0 and not s_base.derived()


def test_sketch_roundtrip(tmp_path):
    base = make_weights(1)
    base_dir = write_model(str(tmp_path / "base"), base)
    sk = sketch_of(tmp_path, "a", finetune(base, 5), base_dir)
    T.save_sketch(sk, str(tmp_path / "s.npz"))
    back = T.load_sketch(str(tmp_path / "s.npz"))
    assert set(back) == set(sk)
    assert T.compare_sketches(sk, back).cosine == pytest.approx(1.0, abs=1e-5)
