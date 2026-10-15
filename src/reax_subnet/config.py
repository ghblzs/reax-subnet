"""Configuration from environment variables (see docs/CONFIG.md).

Secrets never live in configuration: the wallet is referenced by name and
path, and the key files stay on disk with their usual permissions.
"""

from __future__ import annotations

import ipaddress
import logging
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

log = logging.getLogger("reax.config")

# Release constant: the reference engine of the engine competition (the s1-fast runtime image, pinned by digest): the
# published 1.2.0rc1 s1-fast runtime (commit 020429e). On finney it is not configurable; development networks may
# override it with REAX_ENGINE_REFERENCE. Only the competition worker (or a validator in REAX_COMPETITIONS_MODE=local)
# pulls it; a validator in remote mode never does.
ENGINE_REFERENCE_IMAGE = "ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:d857a90eaf168d1db3ca982c5111b10fb61a65cc5f7113ea9e391fe6fca17b71"

# Where the competitions (engine competition, specialist tracks) run: "local" = inside the validator process (needs the
# docker socket and the sealed data on the validator host), "remote" = in a separate `reax-competition-worker`; the
# validator only reads the state files the worker writes (no docker, no sealed data), "off" = not at all (100 % inference).
COMPETITIONS_MODES = ("local", "remote", "off")
ENGINE_STATE_FILE = "engine-competition.json"
SPEC_STATE_FILE = "spec-state.json"
REMOTE_MAX_AGE_S = 26 * 3600  # a state file older than this (by its own saved_at) is treated as missing

MECHANISMS = ("s1", "mpm1")
S1_MODELS = ("s1-fast", "s1-pro", "s1-lite", "s1-cpu")
TIERED_MODELS = ("s1-lite", "s1-cpu")

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


LOCAL_NETWORK_NAMES = ("local", "test")  # named development networks (no genesis check needed)


def _is_mainnet(network: str) -> bool:
    """finney under any spelling: the names above, or any endpoint on opentensor.ai's finney/archive hosts."""
    n = network.strip().lower().rstrip("/")
    return n in {m.lower().rstrip("/") for m in MAINNET_NAMES} or ("opentensor.ai" in n and "test" not in n)


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
    # Tiered pools (s1-lite small GPU / Jetson, s1-cpu) and capacity scoring: ON in 1.2 on every network like s1-pro;
    # switching them off is a development-network switch only (docs/DESIGN-POOLS.md).
    tiered_pools: bool = True
    # Engine competition (docs/ENGINE-COMPETITION.md): fixed on for s1 on finney, a flag on development networks.
    engine: Optional["EngineConfig"] = None
    spec_sealed_dir: str = ""  # REAX_SPEC_SEALED_DIR: sealed pools of the use-case tracks (specialist evaluator)
    # REAX_COMPETITIONS_MODE (local | remote | off) and, in remote mode, REAX_COMPETITIONS_STATE_DIR: the directory the
    # competition worker writes engine-competition.json and spec-state.json to (mounted read-only into the validator).
    competitions_mode: str = "local"
    competitions_state_dir: str = ""


@dataclass(frozen=True)
class EngineConfig:
    runner: str  # docker | dev | remote (no runner here: the worker runs the cycles)
    reference: str  # s1-fast reference engine image (<image>@sha256:<digest>) or a dev-map key
    workdir: str
    state_path: str
    interval_s: int
    max_evals: int
    weights_dir: str = ""
    gpu: str = "0"
    docker_runtime: str = ""
    registries: tuple[str, ...] = ("ghcr.io", "docker.io", "quay.io")
    memory: str = "64g"
    cpus: str = "8"
    dev_map: str = ""
    dev_params: str = ""  # JSON overrides of the track and reward parameters (development networks only)
    release_mode: bool = False  # finney: start time and split profile are the release constants
    nvml_check: bool = False  # REAX_ENGINE_NVML_CHECK=1: per-PID GPU activity check of the paused candidate (F5)


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
    if _bool(env, "REAX_ALLOW_DEV_MODELS", False) and _is_mainnet(network):
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
    # Same for the tiered pools (s1-lite, s1-cpu) and capacity scoring.
    "REAX_FEATURE_TIERED_POOLS",
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
    mode = _choice(env, "REAX_COMPETITIONS_MODE", "local", COMPETITIONS_MODES)
    if common.network in MAINNET_NAMES and mode == "local" and _env(env, "REAX_COMPETITIONS_MODE") is None:
        # In-process competitions put the docker socket next to the hotkey: on finney only by explicit choice, never by
        # default (a 1.2.0rc1 compose with the newer code would otherwise switch them on silently).
        if _env(env, "REAX_ENGINE_WEIGHTS_DIR") and _env(env, "REAX_SPEC_SEALED_DIR"):
            log.error("REAX_COMPETITIONS_MODE is not set but REAX_ENGINE_WEIGHTS_DIR and REAX_SPEC_SEALED_DIR are: NOT running "
                      "the competitions in the validator process (that needs REAX_COMPETITIONS_MODE=local, set explicitly). "
                      "Serving-only: this validator sets 100%% inference weights. Use REAX_COMPETITIONS_MODE=remote with "
                      "reax-competition-worker (deploy/validator-1.2/README.md)")
            mode = "off"
        # otherwise the local prerequisites are missing and _engine() logs what is missing and serves only
    state_dir = os.path.expanduser(_env(env, "REAX_COMPETITIONS_STATE_DIR", "") or "")
    engine, sealed = None, ""
    if mode == "remote":
        engine = _remote_engine(env, common, state_dir)
    elif mode == "off":
        _refuse_flag_on_finney(env, common)
        if common.network in MAINNET_NAMES and common.mechanism == "s1":
            log.warning("REAX_COMPETITIONS_MODE=off: this validator pays 100%% inference weights, not the 50/50 split of "
                        "the validators that run the competitions (docs/COMPETITIONS.md)")
    elif _engine_wanted(env, common):
        engine, sealed = _engine(env, common)
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
        tiered_pools=_bool(env, "REAX_FEATURE_TIERED_POOLS", True),
        engine=engine,
        spec_sealed_dir=sealed if engine is not None else "",
        competitions_mode=mode,
        competitions_state_dir=state_dir if mode == "remote" else "",
    )


def _refuse_flag_on_finney(env: Mapping[str, str], common: CommonConfig) -> None:
    if common.network in MAINNET_NAMES and _env(env, "REAX_FEATURE_ENGINE_COMPETITION") is not None:
        raise ConfigError("REAX_FEATURE_ENGINE_COMPETITION is a development-network switch; on finney the "
                          "engine competition is fixed by the release (docs/COMPETITIONS.md)")


def _remote_engine(env: Mapping[str, str], common: CommonConfig, state_dir: str) -> "EngineConfig":
    """REAX_COMPETITIONS_MODE=remote: the validator needs no docker socket and no sealed data, only the state directory."""
    if common.mechanism != "s1":
        raise ConfigError("the competitions need REAX_MECHANISM=s1")
    _refuse_flag_on_finney(env, common)
    if not state_dir:
        raise ConfigError("REAX_COMPETITIONS_MODE=remote needs REAX_COMPETITIONS_STATE_DIR (the directory with "
                          f"{ENGINE_STATE_FILE} and {SPEC_STATE_FILE}, written by reax-competition-worker)")
    if not os.path.isabs(state_dir):
        raise ConfigError("REAX_COMPETITIONS_STATE_DIR must be an absolute path")
    if not os.path.isdir(state_dir):
        log.warning("REAX_COMPETITIONS_STATE_DIR %s does not exist (yet): the competitions pay nothing until the worker "
                    "has written its state files", state_dir)
    ignored = [n for n in ("REAX_ENGINE_WEIGHTS_DIR", "REAX_SPEC_SEALED_DIR", "REAX_ENGINE_REFERENCE", "REAX_ENGINE_RUNNER",
                           "REAX_ENGINE_DEV_MAP") if _env(env, n)]
    if ignored:
        log.warning("REAX_COMPETITIONS_MODE=remote: %s ignored by the validator (only the worker uses them)", ", ".join(ignored))
    release = common.network in MAINNET_NAMES
    dev_params = _env(env, "REAX_ENGINE_DEV_PARAMS", "") or ""
    if dev_params and not common.allow_dev_models:
        raise ConfigError("the engine parameter overrides need REAX_ALLOW_DEV_MODELS=1")
    return EngineConfig(
        runner="remote",
        reference=ENGINE_REFERENCE_IMAGE,
        workdir="",
        state_path=os.path.join(state_dir, ENGINE_STATE_FILE),
        interval_s=6 * 3600,
        max_evals=4,
        dev_params=dev_params,
        release_mode=release,
    )


def _engine_wanted(env: Mapping[str, str], common: CommonConfig) -> bool:
    """On finney the competition is fixed by the release (s1 mechanism only) and the flag is refused: a validator
    that could switch it off would set different weights from everyone else. On every other network the flag
    ``REAX_FEATURE_ENGINE_COMPETITION=1`` switches it on (development and testnets), active immediately."""
    flag = _env(env, "REAX_FEATURE_ENGINE_COMPETITION")
    if common.network in MAINNET_NAMES:
        if flag is not None:
            raise ConfigError("REAX_FEATURE_ENGINE_COMPETITION is a development-network switch; on finney the "
                              "engine competition is fixed by the release (docs/COMPETITIONS.md)")
        return common.mechanism == "s1"
    return _bool(env, "REAX_FEATURE_ENGINE_COMPETITION", False)


def _engine(env: Mapping[str, str], common: CommonConfig, *, want_engine: bool = True, want_spec: bool = True,
            strict: bool = False) -> tuple[Optional["EngineConfig"], str]:
    """The competition configuration and the sealed-pool directory of the specialist tracks. On finney a validator
    that lacks the host side of the competitions (weights, sealed pools, a published reference image) does NOT fail:
    it warns once and serves only, so it pays 100 % inference (``(None, "")``).

    ``want_engine`` / ``want_spec`` / ``strict`` are for the competition worker, which runs only the parts it is asked
    for and fails with a ConfigError instead of serving only."""
    if common.mechanism != "s1":
        raise ConfigError("the engine competition needs REAX_MECHANISM=s1")
    release = common.network in MAINNET_NAMES
    runner = _choice(env, "REAX_ENGINE_RUNNER", "docker", ("docker", "dev"))
    dev_map = os.path.expanduser(_env(env, "REAX_ENGINE_DEV_MAP", "") or "")
    dev_params = _env(env, "REAX_ENGINE_DEV_PARAMS", "") or ""
    if (runner == "dev" or dev_map or dev_params) and not common.allow_dev_models:
        raise ConfigError("the dev engine runner and parameter overrides need REAX_ALLOW_DEV_MODELS=1")
    if runner == "dev" and not dev_map:
        raise ConfigError("REAX_ENGINE_RUNNER=dev needs REAX_ENGINE_DEV_MAP")
    if release and _env(env, "REAX_ENGINE_REFERENCE") is not None:
        raise ConfigError("REAX_ENGINE_REFERENCE is a development-network override; on finney the reference engine "
                          "is the release constant (config.ENGINE_REFERENCE_IMAGE)")
    reference = (_env(env, "REAX_ENGINE_REFERENCE", "") or "") if not release else ""
    reference = reference or ENGINE_REFERENCE_IMAGE
    weights_dir = os.path.expanduser(_env(env, "REAX_ENGINE_WEIGHTS_DIR", "") or "")
    sealed = os.path.expanduser(_env(env, "REAX_SPEC_SEALED_DIR", "") or "")
    if release or strict:
        missing = []
        if want_engine and "<TBD>" in reference:
            missing.append("the release reference engine image (not published in this build)")
        if want_engine and runner == "docker" and not weights_dir:
            missing.append("REAX_ENGINE_WEIGHTS_DIR")
        if want_spec and not sealed:
            missing.append("REAX_SPEC_SEALED_DIR")
        if missing and strict:
            raise ConfigError("the competition worker is missing: " + ", ".join(missing))
        if missing:
            log.warning("competitions are NOT enabled on this validator: missing %s. Running serving-only: this "
                        "validator sets 100%% inference weights until they are configured (docs/COMPETITIONS.md, "
                        "deploy/validator-1.2/README.md)", ", ".join(missing))
            return None, ""
    elif want_engine and "<TBD>" in reference:
        raise ConfigError("REAX_ENGINE_REFERENCE (the reference engine image, <image>@sha256:<digest>) is required")
    if runner == "docker" and want_engine:
        if "@sha256:" not in reference:
            raise ConfigError("REAX_ENGINE_REFERENCE must be pinned by digest")
        if not weights_dir or not os.path.isdir(weights_dir):
            raise ConfigError("REAX_ENGINE_WEIGHTS_DIR must be the directory with the pinned model weights")
    if sealed and not os.path.isdir(sealed):
        raise ConfigError("REAX_SPEC_SEALED_DIR must be the directory with the sealed pools of the use-case tracks")
    registries = tuple(r.strip() for r in (_env(env, "REAX_ENGINE_REGISTRIES", "") or "ghcr.io,docker.io,quay.io")
                       .split(",") if r.strip())
    risky = [r for r in registries if r.startswith(("localhost", "127.", "10.", "192.168.")) or ":" in r]
    if risky and not common.allow_dev_models:
        raise ConfigError("local or private registries in REAX_ENGINE_REGISTRIES need REAX_ALLOW_DEV_MODELS=1")
    gpu = _env(env, "REAX_ENGINE_GPU", "0") or "0"
    if not gpu.isdigit():
        raise ConfigError("REAX_ENGINE_GPU must be a GPU index")
    cfg = EngineConfig(
        runner=runner,
        reference=reference,
        workdir=os.path.expanduser(_env(env, "REAX_ENGINE_WORKDIR", "~/.reax/engines") or ""),
        state_path=os.path.expanduser(_env(env, "REAX_ENGINE_STATE_PATH", "~/.reax/engine-competition.json") or ""),
        interval_s=_int(env, "REAX_ENGINE_EVAL_INTERVAL_S", 6 * 3600, 30 if common.allow_dev_models else 3600,
                        7 * 86400),
        max_evals=_int(env, "REAX_ENGINE_MAX_EVALS", 4, 1, 32),
        weights_dir=weights_dir,
        gpu=gpu,
        docker_runtime=_env(env, "REAX_ENGINE_DOCKER_RUNTIME", "") or "",
        registries=registries,
        memory=_env(env, "REAX_ENGINE_MEMORY", "64g") or "64g",
        cpus=_env(env, "REAX_ENGINE_CPUS", "8") or "8",
        dev_map=dev_map,
        dev_params=dev_params,
        release_mode=release,
        nvml_check=_bool(env, "REAX_ENGINE_NVML_CHECK", False),
    )
    return cfg, sealed


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
