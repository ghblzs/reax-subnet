"""Entry points: ``reax-miner`` and ``reax-validator``.

All settings come from environment variables (docs/CONFIG.md). Logs never
contain key material: wallets are loaded by the SDK and only public ss58
addresses are printed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from urllib.parse import urlsplit

from .config import LOCAL_NETWORK_NAMES, MAINNET_NAMES, CommonConfig, ConfigError, load_miner, load_validator
from .s1protocol import PROTOCOL_S1
from .version import PROTOCOL, WEIGHTS_VERSION_KEY, __version__

log = logging.getLogger("reax")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(common: CommonConfig) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if common.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(common.log_level)
    # The SDK logs RPC traffic at DEBUG; keep it quiet unless asked for.
    logging.getLogger("bittensor").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def _wallet(common: CommonConfig):
    import bittensor as bt

    wallet = bt.Wallet(name=common.wallet_name, hotkey=common.wallet_hotkey, path=common.wallet_path)
    try:
        hotkey = wallet.hotkey.ss58_address
    except Exception as exc:
        raise ConfigError(
            f"cannot load hotkey {common.wallet_name}/{common.wallet_hotkey} from {common.wallet_path}: {exc}"
        ) from None
    return wallet, hotkey


def safe_endpoint(network: str) -> str:
    """Drop credentials, path and query from an RPC URL before logging it."""
    parts = urlsplit(network)
    if not parts.scheme or not parts.hostname:
        return network
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def _banner(role: str, common: CommonConfig, hotkey: str) -> None:
    protocol = PROTOCOL_S1 if common.mechanism == "s1" else PROTOCOL
    log.info(
        "REAX %s %s (mechanism %s, protocol %s, weights version %d) netuid=%d network=%s model=%s hotkey=%s",
        role, __version__, common.mechanism, protocol, WEIGHTS_VERSION_KEY, common.netuid,
        safe_endpoint(common.network), common.model, hotkey,
    )


def _read_token(path: str) -> str:
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def _runtime_token(common: CommonConfig) -> str:
    return _read_token(common.runtime_token_file)


def _s1_backend(common: CommonConfig, model: str):
    from .s1backends import load_s1_backend

    endpoints = tuple((u, _read_token(tf)) for u, tf in dict(common.runtime_endpoints).get(model, ()))
    return load_s1_backend(model, common.runtime_url(model), _runtime_token(common), common.allow_dev_models,
                           common.runtime_inflight, endpoints)


def miner_main() -> None:
    try:
        cfg = load_miner()
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    setup_logging(cfg.common)
    try:
        wallet, hotkey = _wallet(cfg.common)
    except ConfigError as exc:
        sys.exit(str(exc))
    _banner("miner", cfg.common, hotkey)
    asyncio.run(_run_miner(cfg, wallet))


async def _run_miner(cfg, wallet) -> None:
    from aiohttp import web

    from .backends import load_backend
    from .chain import SdkChain
    from .miner import Miner, S1Miner, refresh_validators_forever

    s1 = cfg.common.mechanism == "s1"
    if s1:
        try:
            backend = _s1_backend(cfg.common, cfg.common.model)
        except ValueError as exc:
            sys.exit(f"configuration error: {exc}")
        urls = ", ".join(u for u, _t in cfg.common.endpoints(cfg.common.model))
        if not await backend.probe():
            sys.exit(f"the {cfg.common.model} runtime at {urls} did not answer an authenticated test decision (is it "
                     "running, and do both sides use the same token? With several backends every one must answer; "
                     "the log names the failing one)")
        if getattr(backend, "size", 1) > 1:
            from .selfcheck import check_pool_identity

            bad = await check_pool_identity(backend, cfg.common.model)
            if bad:
                sys.exit("refusing to start: " + "; ".join(bad))
            from .selfcheck import check_backend_identity

            model = cfg.common.model
            backend.readmit_check = lambda b: check_backend_identity(b, model)
            await backend.start()
            log.info("%d runtime backends behind this hotkey: %s", backend.size, urls)
    else:
        backend = await asyncio.to_thread(
            load_backend, cfg.common.model, cfg.common.allow_dev_models, cfg.common.device, cfg.common.dtype
        )
    chain = SdkChain(cfg.common.network, cfg.common.netuid, wallet, commitments=s1)
    snap = await asyncio.to_thread(chain.snapshot)
    uid = snap.uid_of(wallet.hotkey.ss58_address)
    if uid is None:
        sys.exit(f"hotkey {wallet.hotkey.ss58_address} is not registered on subnet {cfg.common.netuid}")
    if cfg.serve_axon:
        if not cfg.external_ip:
            sys.exit("REAX_EXTERNAL_IP is required to publish the miner endpoint (or set REAX_SERVE_AXON=0)")
        current = next(n for n in snap.neurons if n.uid == uid)
        if (current.ip, current.port) != (cfg.external_ip, cfg.external_port):
            try:
                await asyncio.to_thread(chain.serve_axon, cfg.external_ip, cfg.external_port)
                log.info("published endpoint %s:%d for uid %d", cfg.external_ip, cfg.external_port, uid)
            except Exception as exc:
                # Keep serving: the endpoint may be published on the next start, and
                # crashing here would only cause a restart loop.
                log.error("could not publish endpoint (on chain: %s:%s): %s", current.ip, current.port, exc)
    on_snapshot = None
    if s1 and cfg.publish_commitment:
        from .manifest import commitment_text

        wanted = commitment_text(backend.model, backend.manifest.manifest_id)
        last_try = [0.0]

        async def publish() -> None:
            last_try[0] = time.monotonic()
            try:
                await asyncio.to_thread(chain.set_commitment, wanted)
                log.info("published commitment %s", wanted)
            except Exception as exc:
                log.error("could not publish commitment %s (validators will not query this miner until it is "
                          "on chain; retrying every 10 minutes): %s", wanted, exc)

        current = next(n for n in snap.neurons if n.uid == uid)
        if current.commitment != wanted:
            await publish()

        def on_snapshot(s) -> None:
            mine = next((n for n in s.neurons if n.hotkey == wallet.hotkey.ss58_address), None)
            if mine is not None and mine.commitment != wanted and time.monotonic() - last_try[0] > 600:
                asyncio.get_running_loop().create_task(publish())
    miner = (S1Miner if s1 else Miner)(
        wallet,
        backend,
        max_concurrent=cfg.max_concurrent,
        per_caller_concurrency=cfg.per_caller_concurrency,
        per_caller_per_minute=cfg.per_caller_per_minute,
        body_timeout_s=cfg.body_timeout_s,
        max_pending_per_ip=cfg.max_pending_per_ip,
        max_pending_total=cfg.max_pending_total,
    )
    from .miner import allowed_validators

    miner.set_validators(allowed_validators(snap, cfg.min_validator_stake))
    # Close rejected and idle connections promptly instead of draining or keeping them.
    runner = web.AppRunner(miner.app(), access_log=None, lingering_time=0, keepalive_timeout=5)
    await runner.setup()
    await web.TCPSite(runner, cfg.host, cfg.port).start()
    log.info("miner uid %d listening on %s:%d", uid, cfg.host, cfg.port)
    await refresh_validators_forever(miner, chain, cfg.min_validator_stake, cfg.metagraph_refresh_s, on_snapshot)


def validator_main() -> None:
    try:
        cfg = load_validator()
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    setup_logging(cfg.common)
    try:
        wallet, hotkey = _wallet(cfg.common)
    except ConfigError as exc:
        sys.exit(str(exc))
    _banner("validator", cfg.common, hotkey)
    from .backends import load_backend
    from .chain import SdkChain
    from .validator import Validator

    if cfg.common.mechanism == "s1":
        asyncio.run(_run_s1_validator(cfg, wallet))
        return
    started = time.monotonic()
    backend = load_backend(cfg.common.model, cfg.common.allow_dev_models, cfg.common.device, cfg.common.dtype)
    log.info("reference model %s loaded in %.1fs", backend.model_alias, time.monotonic() - started)
    chain = SdkChain(cfg.common.network, cfg.common.netuid, wallet)
    asyncio.run(Validator(cfg, wallet, chain, backend).run_forever())


async def _run_s1_validator(cfg, wallet) -> None:
    from .chain import SdkChain
    from .s1validator import SHARES_BOTH, SHARES_FAST_ONLY, TIERED_POOLS, S1Validator

    models = sorted(SHARES_BOTH if cfg.s1_pro_pool else SHARES_FAST_ONLY)
    if cfg.tiered_pools:
        models = sorted([*models, *TIERED_POOLS])
    backends = {}
    for m in models:
        tiered = m in TIERED_POOLS
        try:
            backends[m] = _s1_backend(cfg.common, m)
        except ValueError as exc:
            if tiered:  # the tiered references are optional for a validator: that pool stays unscored here
                log.warning("tiered pool %s is NOT scored on this validator: %s", m, exc)
                continue
            sys.exit(f"configuration error: {exc}")
        if not await backends[m].probe():
            msg = (f"the {m} reference runtime at {cfg.common.runtime_url(m)} did not answer an authenticated "
                   "test decision (is it running, and do both sides use the same token?)")
            if tiered:
                log.warning("tiered pool %s is NOT scored on this validator: %s", m, msg)
                del backends[m]
                continue
            sys.exit(msg)
        log.info("reference runtime for %s ready (manifest %s)", m, backends[m].manifest.manifest_id)
    engine, drivers = (_engine_competition(cfg) if cfg.engine is not None else (None, None))
    chain = SdkChain(cfg.common.network, cfg.common.netuid, wallet, commitments=True)
    if cfg.common.network not in MAINNET_NAMES and cfg.common.network not in LOCAL_NETWORK_NAMES:
        from .chain import FINNEY_GENESIS

        try:
            genesis = await asyncio.to_thread(chain.genesis_hash)
        except Exception as exc:  # noqa: BLE001 - fail closed
            sys.exit(f"cannot read the chain's genesis hash ({exc}); refusing to start")
        if genesis.lower() == FINNEY_GENESIS:
            sys.exit("REAX_NETWORK is a custom endpoint of finney (mainnet): set REAX_NETWORK=finney so that the "
                     "release-fixed engine competition and split are used (docs/COMPETITIONS.md)")
    await S1Validator(cfg, wallet, chain, backends, engine=engine, competitions=drivers).run_forever()


def _engine_competition(cfg):
    """Engine competition (docs/ENGINE-COMPETITION.md). On finney the start time and the split profile are the release
    constants; development networks (flag) are active at once and may override parameters. Returns the engine
    competition and the driver that runs it, with the specialist evaluator when ``REAX_SPEC_SEALED_DIR`` is set,
    on the one GPU. In ``REAX_COMPETITIONS_MODE=remote`` nothing runs here: the results are read from the state files
    of ``reax-competition-worker`` (remote.py)."""
    from .competitions import CompetitionDriver
    from .worker import build_local_stack

    e = cfg.engine
    if e.runner == "remote":
        return _remote_competition(cfg, e)
    engine, adapter, _ = build_local_stack(e, cfg.spec_sealed_dir, os.environ, award_gate=_award_gate(e))
    return engine, CompetitionDriver(engine, adapter, background=True)


def _award_gate(e):
    """Finney: every award needs the owner's signed approval (award.py), fail closed. Development networks: only when
    REAX_AWARD_APPROVALS is set."""
    from .award import AwardGate

    if e.release_mode or os.environ.get("REAX_AWARD_APPROVALS"):
        gate = AwardGate.from_env(os.environ)
        log.info("competition awards need the owner's signed approval (approvals file %s)", gate.path or "NOT SET")
        if e.release_mode and gate.key_problem:
            log.error("award approvals: %s; NO competition award can be paid on finney until "
                      "REAX_AWARD_APPROVAL_PUBKEY or the release key is valid", gate.key_problem)
        return gate
    return None


def _remote_competition(cfg, e):
    from .competition_split import COMPETITIONS_START_UTC, release_engine_start
    from .competitions import CompetitionDriver
    from .config import SPEC_STATE_FILE
    from .remote import RemoteEngineCompetition, RemoteSpecAdapter, RemoteSpecEvaluator
    from .spec import config as spec_config
    from .worker import engine_params

    tracks_def = spec_config.DEFAULT_TRACKS
    if not e.release_mode and os.environ.get("REAX_SPEC_TRACKS_FILE"):
        tracks_def = spec_config.load_tracks(os.environ["REAX_SPEC_TRACKS_FILE"])
    start = COMPETITIONS_START_UTC if e.release_mode else None
    engine_start = release_engine_start() if e.release_mode else None
    tracks, reward = engine_params(e)
    adapter = RemoteSpecAdapter(RemoteSpecEvaluator(os.path.join(cfg.competitions_state_dir, SPEC_STATE_FILE), tracks_def))
    engine = RemoteEngineCompetition(e.state_path, tracks=tracks, reward=reward, start_utc=start, specialist=adapter,
                                     engine_start_utc=engine_start, award_gate=_award_gate(e))
    log.info("competitions: REMOTE mode, results read from %s (no docker, no sealed data in this process); starts %s",
             cfg.competitions_state_dir, start.isoformat() if start else "immediately (development network)")
    return engine, CompetitionDriver(engine, adapter, background=True)
