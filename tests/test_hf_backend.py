"""Real-model checks for label_logits_v1 (skipped when the model is not available)."""

import os

import pytest

pytestmark = pytest.mark.model

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def backend():
    from reax_subnet.backends import CATALOG, HFBackend

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return HFBackend(CATALOG["qwen2.5-0.5b-instruct"], device="cpu", dtype="float32", max_batch=4)
    except Exception as exc:  # model not downloaded
        pytest.skip(f"model unavailable: {exc}")


def test_distributions_valid_and_batch_invariant(backend):
    from reax_subnet.items import generate_items
    from reax_subnet.scoring import total_variation

    items = generate_items(8, seed=7)
    batched = backend.predict(items)
    single = [backend.predict([it])[0] for it in items]
    for it, p, q in zip(items, batched, single):
        assert len(p) == len(it.options)
        assert abs(sum(p) - 1.0) < 1e-9
        assert total_variation(p, q) < 1e-3
    assert backend.predict(items) == batched  # deterministic on the same hardware


def test_catalog_production_model_is_pinned():
    from reax_subnet.backends import CATALOG, resolve_model

    prod = [s for s in CATALOG.values() if s.production]
    assert len(prod) == 1 and len(prod[0].revision) == 40 and prod[0].temperature == 4.0
    with pytest.raises(ValueError):
        resolve_model("qwen2.5-0.5b-instruct", allow_dev=False)
    with pytest.raises(ValueError):
        resolve_model("unknown", allow_dev=True)
