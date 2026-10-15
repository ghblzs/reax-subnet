"""Configuration from environment variables (see docs/CONFIG.md).

Secrets never live in configuration: the wallet is referenced by name and
path, and the key files stay on disk with their usual permissions.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .scoring import ScoringParams

# Reserved features that are not part of this release.
# Setting any of these to a true value aborts startup, so nobody can believe a
# feature is active when it is not. docs/FEATURE-FLAGS.md says why each one is
# off and what has to happen before it can come back.
DISABLED_FEATURES: dict[str, str] = {
    "REAX_FEATURE_GDPR_TIER": "regional admission controls (signed admissions and revocation)",
    "REAX_FEATURE_ORGANIC_TRAFFIC": "routing requests from external clients",
    "REAX_FEATURE_IMAGES": "image inputs",
    "REAX_FEATURE_CAPACITY_PROBES": "burst capacity audits and verified-capacity weighting",
    "REAX_FEATURE_MULTI_MODEL": "more than one approved model per subnet in the rc3 (mpm1) mechanism",
    "REAX_FEATURE_SCORE_AND_NOUL_TYPES": "ordered-score and noul question types",
    "REAX_FEATURE_LABELLED_ITEMS": "scoring against labelled (answer-key) items",
}

MECHANISMS = ("s1", "mpm1")
S1_MODELS = ("s1-fast", "s1-pro")

MAINNET_NAMES = {"finney", "archive", "wss://entrypoint-finney.opentensor.ai:443", "wss://entrypoint-finney.opentensor.ai"}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


class ConfigError(ValueError):
    pass


def _env(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    value = env.get(name)
    return default if value is None else value.strip()


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    if raw.lower() in _TRUE:
        return True
    if raw.lower() in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (1/0), got {raw!r}")


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = _env(env, name)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be within [{low}, {high}]")
    return value


def _float(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = _env(env, name)
    try:
        value = default if raw is None else float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be within [{low}, {high}]")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _env(env, name, default) or default
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}")
    return value


def _device(env: Mapping[str, str]) -> str:
    value = _env(env, "REAX_DEVICE", "cuda") or "cuda"
    if value != "cpu" and value != "cuda" and not (value.startswith("cuda:") and value[5:].isdigit()):
        raise ConfigError("REAX_DEVICE must be cuda, cuda:<n> or cpu")
    return value


def _ip(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = _env(env, name)
    if not raw:
        return None
    try:
        return str(ipaddress.ip_address(raw))  # canonical form, as stored on chain
    except ValueError:
        raise ConfigError(f"{name} must be an IP address, not a host name") from None


def check_disabled_features(env: Mapping[str, str]) -> None:
    for name, what in DISABLED_FEATURES.items():
        if _bool(env, name, False):
            raise ConfigError(
                f"{name} ({what}) is not available in this release. See docs/FEATURE-FLAGS.md."
            )


@dataclass(frozen=True)
class CommonConfig:
    mechanism: str
    network: str
    netuid: int
    wallet_name: str
    wallet_hotkey: str
    wallet_path: str
    model: str
    allow_dev_models: bool
    device: str
    dtype: str
    log_level: str
    log_json: bool
    # reax-s1: loopback URL(s) of the model runtime sidecar(s) and the shared internal token file.
    runtime_urls: tuple[tuple[str, str], ...] = ()
    runtime_token_file: str = ""

    def runtime_url(self, model: str) -> str:
        return dict(self.runtime_urls).get(model, "")


@dataclass(frozen=True)
class MinerConfig:
    common: CommonConfig
    host: str
    port: int
    external_ip: Optional[str]
    external_port: int
    serve_axon: bool
    min_validator_stake: float
    metagraph_refresh_s: int
    max_concurrent: int
    per_caller_concurrency: int
    per_caller_per_minute: int
    body_timeout_s: float
    max_pending_per_ip: int
    max_pending_total: int
    publish_commitment: bool = True


@dataclass(frozen=True)
class ValidatorConfig:
    common: CommonConfig
    round_interval_s: int
    items_per_round: int
    request_timeout_s: float
    max_parallel: int
    state_path: str
    set_weights: bool
    weights_interval_blocks: int
    allow_private_ips: bool
    warmup_rounds: int = 6
    scoring: ScoringParams = field(default_factory=ScoringParams)
    s1_pro_pool: bool = True


def _is_loopback_url(url: str) -> bool:
    """http URL whose host is literally localhost or a loopback IP (no userinfo, no DNS names)."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # raises on a malformed port
    except ValueError:
        return False
    if parts.scheme != "http" or not host or "@" in parts.netloc:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_common(env: Mapping[str, str] = os.environ) -> CommonConfig:
    check_disabled_features(env)
    mechanism = _choice(env, "REAX_MECHANISM", "s1", MECHANISMS)
    network = _env(env, "REAX_NETWORK", "finney") or "finney"
    netuid_raw = _env(env, "REAX_NETUID")
    if not netuid_raw:
        if network in MAINNET_NAMES:
            netuid = 92
        else:
            raise ConfigError("REAX_NETUID is required for testnet and local endpoints")
    else:
        netuid = _int(env, "REAX_NETUID", 0, 1, 65535)
    if _bool(env, "REAX_ALLOW_DEV_MODELS", False) and network in MAINNET_NAMES:
        raise ConfigError("REAX_ALLOW_DEV_MODELS=1 is refused on finney (it unlocks test models and scoring overrides)")
    log_level = (_env(env, "REAX_LOG_LEVEL", "INFO") or "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("REAX_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
    default_model = "s1-fast" if mechanism == "s1" else "qwen3-4b-instruct-2507"
    model = _env(env, "REAX_MODEL", default_model) or default_model
    if mechanism == "s1" and model not in S1_MODELS:
        raise ConfigError(f"REAX_MODEL must be one of {', '.join(S1_MODELS)} with REAX_MECHANISM=s1")
    urls = tuple(
        (m, _env(env, f"REAX_{m.upper().replace('-', '_')}_RUNTIME_URL", "") or "") for m in S1_MODELS
    )
    for _m, url in urls:
        if url and url != "hash" and not _is_loopback_url(url):
            raise ConfigError("runtime URLs must be loopback (the model runtime must not be reachable from outside)")
        if url == "hash" and not _bool(env, "REAX_ALLOW_DEV_MODELS", False):
            raise ConfigError("the hash test runtime needs REAX_ALLOW_DEV_MODELS=1")
    token_file = os.path.expanduser(_env(env, "REAX_RUNTIME_TOKEN_FILE", "") or "")
    if mechanism == "s1" and any(url and url != "hash" for _m, url in urls):
        try:
            with open(token_file, encoding="utf-8") as fh:
                ok = bool(fh.read().strip())
        except OSError:
            ok = False
        if not ok:
            raise ConfigError("REAX_RUNTIME_TOKEN_FILE must name a readable, non-empty token file shared with the "
                              "runtime sidecar (S1M_INTERNAL_TOKEN on the sidecar side)")
    return CommonConfig(
        mechanism=mechanism,
        network=network,
        netuid=netuid,
        wallet_name=_env(env, "REAX_WALLET_NAME", "default") or "default",
        wallet_hotkey=_env(env, "REAX_WALLET_HOTKEY", "default") or "default",
        wallet_path=os.path.expanduser(_env(env, "REAX_WALLET_PATH", "~/.bittensor/wallets") or ""),
        model=model,
        allow_dev_models=_bool(env, "REAX_ALLOW_DEV_MODELS", False),
        device=_device(env),
        dtype=_choice(env, "REAX_DTYPE", "bfloat16", ("bfloat16", "float16", "float32")),
        log_level=log_level,
        log_json=_bool(env, "REAX_LOG_JSON", False),
        runtime_urls=urls,
        runtime_token_file=token_file,
    )


def load_miner(env: Mapping[str, str] = os.environ) -> MinerConfig:
    common = load_common(env)
    port = _int(env, "REAX_MINER_PORT", 8091, 1, 65535)
    return MinerConfig(
        common=common,
        host=_env(env, "REAX_MINER_HOST", "0.0.0.0") or "0.0.0.0",
        port=port,
        external_ip=_ip(env, "REAX_EXTERNAL_IP"),
        external_port=_int(env, "REAX_EXTERNAL_PORT", port, 1, 65535),
        serve_axon=_bool(env, "REAX_SERVE_AXON", True),
        min_validator_stake=_float(env, "REAX_MIN_VALIDATOR_STAKE", 1000.0, 0.0, 1e12),
        metagraph_refresh_s=_int(env, "REAX_METAGRAPH_REFRESH_S", 300, 10, 3600),
        max_concurrent=_int(env, "REAX_MAX_CONCURRENT", 4, 1, 256),
        per_caller_concurrency=_int(env, "REAX_PER_CALLER_CONCURRENCY", 2, 1, 64),
        per_caller_per_minute=_int(env, "REAX_PER_CALLER_PER_MINUTE", 20, 1, 6000),
        body_timeout_s=_float(env, "REAX_BODY_TIMEOUT_S", 5.0, 0.5, 60.0),
        max_pending_per_ip=_int(env, "REAX_MAX_PENDING_PER_IP", 8, 1, 1024),
        max_pending_total=_int(env, "REAX_MAX_PENDING_TOTAL", 4096, 16, 100_000),
        publish_commitment=_bool(env, "REAX_PUBLISH_COMMITMENT", True),
    )


# Scoring parameters are part of the release: validators that disagree on them
# disagree on weights. They can only be overridden on development networks.
FIXED_SCORING_VARS = (
    "REAX_TAU", "REAX_MIN_ROUND_FIDELITY", "REAX_LATENCY_FAST_S", "REAX_LATENCY_SLOW_S", "REAX_LATENCY_WEIGHT",
    "REAX_EMA_ALPHA", "REAX_MIN_EMA", "REAX_ITEMS_PER_ROUND", "REAX_REQUEST_TIMEOUT_S", "REAX_WARMUP_ROUNDS",
    # reax-s1: the s1-pro pool changes every validator's weights. It is ON in 1.2 on every network;
    # switching it off is a development-network switch only (REAX_ALLOW_DEV_MODELS=1).
    "REAX_FEATURE_S1_PRO_POOL",
)


def load_validator(env: Mapping[str, str] = os.environ) -> ValidatorConfig:
    common = load_common(env)
    if not common.allow_dev_models:
        overridden = [name for name in FIXED_SCORING_VARS if _env(env, name) is not None]
        if overridden:
            raise ConfigError(
                f"{', '.join(overridden)} are fixed by the release; overrides are only allowed with "
                "REAX_ALLOW_DEV_MODELS=1 on development networks"
            )
    try:
        scoring = _scoring(env)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if _bool(env, "REAX_ALLOW_PRIVATE_IPS", False) and common.network in MAINNET_NAMES:
        raise ConfigError("REAX_ALLOW_PRIVATE_IPS is for local test chains only; refusing it on mainnet")
    # The timeout follows the slow-latency bound unless set explicitly (dev networks only).
    timeout = _float(env, "REAX_REQUEST_TIMEOUT_S", max(1.0, scoring.latency_slow_s), 1.0, 120.0)
    return ValidatorConfig(
        common=common,
        round_interval_s=_int(env, "REAX_ROUND_INTERVAL_S", 120, 10, 3600),
        items_per_round=_int(env, "REAX_ITEMS_PER_ROUND", 12, 4, 16),
        request_timeout_s=timeout,
        max_parallel=_int(env, "REAX_MAX_PARALLEL", 256, 1, 1024),
        state_path=os.path.expanduser(
            _env(env, "REAX_STATE_PATH", "~/.reax/validator-state.json") or ""
        ),
        set_weights=_bool(env, "REAX_SET_WEIGHTS", True),
        weights_interval_blocks=_int(env, "REAX_WEIGHTS_INTERVAL_BLOCKS", 360, 1, 100_000),
        allow_private_ips=_bool(env, "REAX_ALLOW_PRIVATE_IPS", False),
        warmup_rounds=_int(env, "REAX_WARMUP_ROUNDS", 6, 0, 100),
        scoring=scoring,
        s1_pro_pool=_bool(env, "REAX_FEATURE_S1_PRO_POOL", True),
    )


def _scoring(env: Mapping[str, str]) -> ScoringParams:
    return ScoringParams(
        tau=_float(env, "REAX_TAU", ScoringParams.tau, 0.01, 0.5),
        min_round_fidelity=_float(env, "REAX_MIN_ROUND_FIDELITY", ScoringParams.min_round_fidelity, 0.5, 1.0),
        latency_fast_s=_float(env, "REAX_LATENCY_FAST_S", ScoringParams.latency_fast_s, 0.0, 120.0),
        latency_slow_s=_float(env, "REAX_LATENCY_SLOW_S", ScoringParams.latency_slow_s, 0.1, 120.0),
        latency_weight=_float(env, "REAX_LATENCY_WEIGHT", ScoringParams.latency_weight, 0.0, 0.5),
        ema_alpha=_float(env, "REAX_EMA_ALPHA", ScoringParams.ema_alpha, 0.01, 1.0),
        min_ema=_float(env, "REAX_MIN_EMA", ScoringParams.min_ema, 0.0, 0.9),
        burn_fraction=_float(env, "REAX_BURN_FRACTION", ScoringParams.burn_fraction, 0.0, 1.0),
    )
