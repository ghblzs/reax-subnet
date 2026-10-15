import json
import os

import pytest
from spec_helpers import HOTKEYS, REV, make_weights, write_model

from reax_subnet.spec import fetch as F
from reax_subnet.spec import rules as R
from reax_subnet.spec.commitment import SpecCommitment, parse_commitment

GOOD = f"reax-spec/1:demand_forecasting:acme/df-4b@{REV}"


def test_parse_valid_roundtrip():
    c = parse_commitment(GOOD)
    assert c == SpecCommitment("demand_forecasting", "acme/df-4b", REV)
    assert c.text() == GOOD
    assert parse_commitment("  " + GOOD + "\n") == c


@pytest.mark.parametrize("bad", [
    "", None, 5, "reax-s1/1:s1-fast:0123456789abcdef", f"reax-spec/2:demand_forecasting:acme/x@{REV}",
    f"reax-spec/1:Demand:acme/x@{REV}", f"reax-spec/1:demand_forecasting:acme/x@{REV[:39]}",
    f"reax-spec/1:demand_forecasting:acme/x@{'A' * 40}", f"reax-spec/1:demand_forecasting:acme/x@{'g' * 40}",
    f"reax-spec/1:demand_forecasting:acmex@{REV}", f"reax-spec/1:demand_forecasting:a/b/c@{REV}",
    f"reax-spec/1:demand_forecasting:acme/x:y@{REV}", f"reax-spec/1:demand_forecasting:acme/..x@{REV}",
    f"reax-spec/1:demand_forecasting:acme/x--y@{REV}", f"reax-spec/1:demand_forecasting:acme/x@{REV}@{REV}",
    f"reax-spec/1:demand_forecasting:acme/{'n' * 90}@{REV}", f"reax-spec/1:demand_forecasting:acme/x\u00e9@{REV}",
    f"reax-spec/1:demand_forecasting:acme/main@main",
])
def test_parse_rejects(bad):
    assert parse_commitment(bad) is None


def inspect(path, **kw):
    kw.setdefault("category", "demand_forecasting")
    kw.setdefault("hotkey", HOTKEYS[0])
    return F.inspect_dir(path, **kw)


def test_good_model_passes_and_is_hashed(tmp_path):
    ins = inspect(write_model(str(tmp_path / "m"), make_weights(1)))
    assert ins.violations == [] and ins.hashes and ins.hashes.params == 4 * 64 * 64


def test_refuses_more_than_nine_billion_params(tmp_path):
    ins = inspect(write_model(str(tmp_path / "m"), make_weights(1)))
    rules = R.TrackRules(max_params=1000)
    assert any("parameters" in v for v in inspect(ins.path, rules=rules).violations)


def test_refuses_oversized_weights_by_listing():
    big = {"a.safetensors": 21 * R.GIB, "LICENSE": 1, R.MANIFEST_NAME: 1}
    assert any("GiB" in v for v in R.check_listing(big))


def test_refuses_pickle_and_missing_safetensors():
    v = R.check_listing({"pytorch_model.bin": 10, "LICENSE": 1, R.MANIFEST_NAME: 1})
    assert any("pickle" in x for x in v) and any("no .safetensors" in x for x in v)


def test_refuses_licence_not_on_allowlist(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1), licence="llama3")
    assert any("licence" in v for v in inspect(p).violations)
    ok = R.TrackRules(approved_extra=("llama3",))
    assert inspect(p, rules=ok).violations == []


def test_refuses_unknown_architecture(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1), arch="MyCustomForCausalLM")
    assert any("allowlist" in v and "architecture" in v for v in inspect(p).violations)


def test_refuses_remote_code(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1), config_extra={"auto_map": {"AutoModel": "mod.Model"}})
    assert any("auto_map" in v for v in inspect(p).violations)


def test_refuses_wrong_hotkey_category_and_missing_manifest(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1), hotkey=HOTKEYS[1])
    assert any("hotkey" in v for v in inspect(p).violations)
    assert any("category" in v for v in inspect(p, hotkey=HOTKEYS[1], category="gaming").violations)
    os.remove(os.path.join(p, R.MANIFEST_NAME))
    assert any(R.MANIFEST_NAME in v for v in inspect(p).violations)
    assert not any(R.MANIFEST_NAME in v for v in inspect(p, require_manifest=False).violations)


def test_refuses_symlink_and_broken_safetensors(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1))
    os.symlink("/etc/hostname", os.path.join(p, "extra.json"))
    assert any("symlink" in v for v in inspect(p).violations)
    p2 = write_model(str(tmp_path / "m2"), make_weights(1))
    with open(os.path.join(p2, "model-00000-of-00001.safetensors"), "r+b") as fh:
        fh.truncate(100)
    assert any("safetensors" in v for v in inspect(p2).violations)


def test_refuses_missing_tokenizer_and_license(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1))
    os.remove(os.path.join(p, "tokenizer.json"))
    os.remove(os.path.join(p, "LICENSE"))
    v = inspect(p).violations
    assert any("tokenizer" in x for x in v) and any("LICENSE" in x for x in v)


def test_oversized_json_refused(tmp_path):
    p = write_model(str(tmp_path / "m"), make_weights(1))
    with open(os.path.join(p, "config.json"), "w") as fh:
        fh.write(json.dumps({"architectures": ["Qwen3ForCausalLM"], "pad": "x" * (R.MAX_JSON_BYTES + 10)}))
    assert any("config.json" in v for v in inspect(p).violations)


def test_architecture_allowlist_names_match_vllm_0_30_registry():
    from reax_subnet.spec import rules as R

    # Names checked against vllm.model_executor.models.registry in the pinned vllm/vllm-openai image (0.30.0).
    registry_subset = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
                       "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration", "LlamaForCausalLM"}
    assert set(R.ARCH_ALLOWLIST) == registry_subset
