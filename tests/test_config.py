import pytest

from reax_subnet.config import DISABLED_FEATURES, ConfigError, load_miner, load_validator

# On finney (the default network) the engine competition is fixed by the release: validators need its settings.
ENGINE = {"REAX_ENGINE_REFERENCE": "ghcr.io/x/ref@sha256:" + "ab" * 32, "REAX_ENGINE_WEIGHTS_DIR": "/tmp"}
BASE = {"REAX_NETUID": "7", **ENGINE}
DEV = {**BASE, "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "ws://127.0.0.1:9951"}


def test_defaults():
    v = load_validator(BASE)
    assert v.common.netuid == 7 and v.common.mechanism == "s1" and v.common.model == "s1-fast"
    assert load_validator({**BASE, "REAX_MECHANISM": "mpm1"}).common.model == "qwen3-4b-instruct-2507"
    assert not v.s1_pro_pool
    assert v.set_weights and not v.allow_private_ips and not v.common.allow_dev_models
    assert v.scoring.burn_fraction == 0.0
    m = load_miner(BASE)
    assert m.min_validator_stake == 1000.0 and m.port == 8091


def test_netuid_required():
    assert load_validator(ENGINE).common.netuid == 92
    for network in ("test", "ws://127.0.0.1:9951"):
        with pytest.raises(ConfigError, match="REAX_NETUID is required"):
            load_validator({"REAX_NETWORK": network})


def test_network_specific_netuid_is_preserved():
    assert load_validator({"REAX_NETWORK": "test", "REAX_NETUID": "17"}).common.netuid == 17
    assert load_validator({"REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_NETUID": "4"}).common.netuid == 4


@pytest.mark.parametrize("flag", sorted(DISABLED_FEATURES))
def test_disabled_features_refuse_to_start(flag):
    with pytest.raises(ConfigError, match="not available"):
        load_validator({**BASE, flag: "1"})
    load_validator({**BASE, flag: "0"})


@pytest.mark.parametrize(
    "env",
    [
        {"REAX_TAU": "0.9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "local"},
        {"REAX_LATENCY_FAST_S": "30", "REAX_LATENCY_SLOW_S": "20", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "local"},
        {"REAX_DEVICE": "gpu"},
        {"REAX_DTYPE": "int4"},
        {"REAX_BURN_FRACTION": "2"},
        {"REAX_ITEMS_PER_ROUND": "64", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "local"},
        {"REAX_SET_WEIGHTS": "maybe"},
        {"REAX_LOG_LEVEL": "TRACE"},
        {"REAX_NETUID": "abc"},
    ],
)
def test_bad_values(env):
    with pytest.raises(ConfigError):
        load_validator({**BASE, **env})


@pytest.mark.parametrize("name", ["REAX_TAU", "REAX_EMA_ALPHA", "REAX_ITEMS_PER_ROUND", "REAX_WARMUP_ROUNDS"])
def test_scoring_constants_fixed_outside_dev(name):
    with pytest.raises(ConfigError, match="fixed by the release"):
        load_validator({**BASE, name: "0.2"})
    load_validator({**DEV, name: "4" if name in ("REAX_ITEMS_PER_ROUND", "REAX_WARMUP_ROUNDS") else "0.2"})


def test_burn_fraction_is_configurable_in_production():
    assert load_validator({**BASE, "REAX_BURN_FRACTION": "1"}).scoring.burn_fraction == 1.0


def test_external_ip_must_be_ip_and_is_canonical():
    with pytest.raises(ConfigError):
        load_miner({**BASE, "REAX_EXTERNAL_IP": "miner.example.com"})
    assert load_miner({**BASE, "REAX_EXTERNAL_IP": "2001:DB8::1"}).external_ip == "2001:db8::1"


def test_defaults_are_gpu_bf16():
    c = load_validator(BASE).common
    assert (c.device, c.dtype) == ("cuda", "bfloat16")


def test_safe_endpoint_redacts():
    from reax_subnet.cli import safe_endpoint

    assert safe_endpoint("wss://user:pw@rpc.example.com:443/ws?apikey=x") == "wss://rpc.example.com:443"
    assert safe_endpoint("finney") == "finney"


def test_dev_mode_refused_on_finney():
    with pytest.raises(ConfigError, match="finney"):
        load_validator({**BASE, "REAX_ALLOW_DEV_MODELS": "1"})
