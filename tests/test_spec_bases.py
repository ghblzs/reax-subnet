"""spec/bases.py: the evaluator downloads a missing declared base itself (fake hub, no network)."""

import asyncio
import json
import logging
import os
import shutil

import pytest
from spec_helpers import make_weights, write_model
from test_spec_evaluator import DEFAULT_BASE, TRACK, Env

from reax_subnet.spec import bases as B
from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import fetch as F
from reax_subnet.spec import rules as R
from reax_subnet.spec import state as ST

GIB = 1024 ** 3
OTHER_BASE = "google/gemma-3-4b-it@" + "d" * 40


class FakeBaseHub:
    """``fetch.Hub`` over local directories: ``bases[<hf id>@<rev>]`` is the repo content at that revision."""

    def __init__(self, root):
        self.root = str(root)
        self.bases: dict[str, str] = {}
        self.gated: set[str] = set()
        self.sizes: dict[str, dict[str, int]] = {}    # listing size overrides
        self.listings: list[str] = []
        self.downloads: list[str] = []
        self.during_download = None                    # called with (base, dest) after the files are written
        self.fail_download: set[str] = set()

    def add(self, base, weights=None, **kw):
        path = os.path.join(self.root, "hub-bases", base.replace("/", "__"))
        write_model(path, weights if weights is not None else make_weights(999), **kw)
        os.unlink(os.path.join(path, R.MANIFEST_NAME))  # a base has no manifest (and needs none)
        self.bases[base] = path
        return path

    def listing(self, repo, revision):
        base = f"{repo}@{revision}"
        self.listings.append(base)
        if base not in self.bases:
            raise F.FetchError("unavailable", "repo or revision not found")
        files = F.list_dir(self.bases[base])
        files.update(self.sizes.get(base, {}))
        return F.RepoListing(sha=revision, files=files, private=False, gated=base in self.gated)

    async def download(self, repo, revision, dest):
        base = f"{repo}@{revision}"
        self.downloads.append(base)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(self.bases[base], dest)
        if self.during_download is not None:
            await self.during_download(base, dest)
        if base in self.fail_download:
            raise F.FetchError("unavailable", "connection reset")


def benv(tmp_path, *, budget_gb=40, min_free_gb=50, free_gb=500, retry=6, **kw):
    """An evaluator with an empty base directory and a base fetcher on a fake hub."""
    root = tmp_path / "bases"
    env = Env(tmp_path, {}, base_dir=str(root))
    hub = FakeBaseHub(tmp_path)
    env.free = free_gb * GIB
    env.bf = B.BaseFetcher(str(root), hub, retry_cycles=retry, budget_bytes=int(budget_gb * GIB),
                           min_free_bytes=int(min_free_gb * GIB), free_bytes=lambda p: env.free)
    env.basehub = hub
    env.ev = E.SpecEvaluator(env.cfg, fetcher=env.hub, backend_factory=env.server, clock=env.clock, base_fetcher=env.bf, **kw)
    return env


def base_path(env, base=DEFAULT_BASE):
    return os.path.join(env.cfg.base_dir, C.base_dir_name(base))


async def test_missing_base_is_fetched_and_the_submission_verified_in_the_same_cycle(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and s.delta_checked and s.base_id and env.server.served == ["a/m1"]
    assert os.path.isfile(os.path.join(base_path(env), "config.json"))
    assert os.listdir(env.cfg.base_dir) == [C.base_dir_name(DEFAULT_BASE)]       # no temp dir left behind
    assert env.basehub.downloads == [DEFAULT_BASE]
    env.add(2, "b/m2", 0.5)                                                        # cached: not downloaded again
    await env.cycle({1: "a/m1", 2: "b/m2"}, block=1010)
    assert env.sub(2).delta_checked and env.basehub.downloads == [DEFAULT_BASE]


async def test_a_pending_submission_gets_its_base_in_a_later_cycle(tmp_path):
    """The hub had no such base at first (not found): pending; once it appears and the retry is due, it is checked."""
    env = benv(tmp_path, retry=2)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and "not available on the hub" in env.sub(1).reason
    env.basehub.add(DEFAULT_BASE)
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and env.basehub.downloads == []   # retry not due yet (2 cycles)
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.sub(1).delta_checked and env.server.served == ["a/m1"]


async def test_gated_base_stays_pending_with_a_reason_and_is_retried_once_per_n_cycles(tmp_path):
    env = benv(tmp_path, retry=3)
    env.basehub.add(DEFAULT_BASE)
    env.basehub.gated.add(DEFAULT_BASE)
    env.add(1, "a/m1", 0.9)
    for i in range(7):
        await env.cycle({1: "a/m1"})
        env.clock.advance(3600)
        s = env.sub(1)
        assert s.status == ST.PENDING_VERIFY and "gated or private" in s.reason, (i, s.reason)
    assert env.basehub.listings == [DEFAULT_BASE] * 3      # cycles 1, 4 and 7
    assert env.basehub.downloads == [] and env.server.served == [] and not os.listdir(env.cfg.base_dir)


async def test_oversize_base_is_pending_and_not_downloaded(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    env.basehub.sizes[DEFAULT_BASE] = {"model-00000-of-00001.safetensors": 30 * GIB}
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and "base refused" in s.reason and "GiB" in s.reason
    assert env.basehub.downloads == []


async def test_base_with_custom_code_or_an_unknown_architecture_is_refused(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE, config_extra={"auto_map": {"AutoModel": "x.Y"}})
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and "auto_map" in env.sub(1).reason
    assert not os.listdir(env.cfg.base_dir)                 # downloaded to a temp dir, refused, removed


async def test_licence_and_pickle_files_that_are_not_downloaded_do_not_matter_for_a_base(tmp_path):
    env = benv(tmp_path)
    path = env.basehub.add(DEFAULT_BASE, licence="llama3")
    os.unlink(os.path.join(path, "LICENSE"))
    env.basehub.sizes[DEFAULT_BASE] = {"original/consolidated.00.pth": 10}   # in the repo, never downloaded
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).delta_checked and env.sub(1).status == ST.PENDING_CONFIRM


async def test_half_written_temp_dir_is_never_resolved(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    seen = []

    async def look(base, dest):     # mid-download: the files exist, but only under the temp name
        assert os.path.isdir(dest) and os.path.basename(dest).startswith(B.TMP_PREFIX)
        seen.append(env.ev.base_provider(base))

    env.basehub.during_download = look
    env.basehub.fail_download.add(DEFAULT_BASE)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert seen == [None]
    assert env.sub(1).status == ST.PENDING_VERIFY and "connection reset" in env.sub(1).reason
    assert not os.listdir(env.cfg.base_dir)                 # the failed download left nothing behind
    # a temp dir another (crashed) process left: same content as a real base, still never used
    stale = os.path.join(env.cfg.base_dir, B.TMP_PREFIX + C.base_dir_name(DEFAULT_BASE) + "-dead")
    shutil.copytree(env.basehub.bases[DEFAULT_BASE], stale)
    assert env.ev.base_provider(DEFAULT_BASE) is None
    assert C.dir_base_provider(env.cfg.base_dir)(B.TMP_PREFIX + "x/y@" + "b" * 40) is None


def test_stale_temp_dirs_are_removed_at_start(tmp_path):
    root = tmp_path / "bases"
    old, new = root / ".tmp-old", root / ".tmp-new"
    old.mkdir(parents=True)
    new.mkdir()
    os.utime(old, (1, 1))
    B.BaseFetcher(str(root), FakeBaseHub(tmp_path))
    assert sorted(os.listdir(root)) == [".tmp-new"]


async def test_budget_exceeded_defers_the_base_without_counting_an_attempt(tmp_path):
    env = benv(tmp_path, budget_gb=0.000001, retry=6)       # smaller than the fake base
    env.basehub.add(DEFAULT_BASE)
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_VERIFY and "deferred" in s.reason and env.basehub.downloads == []
    env.bf.budget_bytes = 40 * GIB
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})                            # the very next cycle, not after 6
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.basehub.downloads == [DEFAULT_BASE]


async def test_budget_is_per_cycle(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    env.basehub.add(OTHER_BASE)
    size = sum(F.list_dir(env.basehub.bases[DEFAULT_BASE]).values())
    env.bf.budget_bytes = size + 10                         # room for one base per cycle
    env.bf.new_cycle()
    assert await env.bf.ensure(DEFAULT_BASE) == base_path(env)
    assert await env.bf.ensure(OTHER_BASE) is None and "budget" in env.bf.reason(OTHER_BASE)
    env.bf.new_cycle()
    assert await env.bf.ensure(OTHER_BASE) == base_path(env, OTHER_BASE)


async def test_low_disk_skips_the_base_with_a_log_line(tmp_path, caplog):
    env = benv(tmp_path, free_gb=50)                        # 50 GB free, 50 GB must stay free
    env.basehub.add(DEFAULT_BASE)
    env.add(1, "a/m1", 0.9)
    with caplog.at_level(logging.WARNING, logger="reax.spec"):
        await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and "disk space" in env.sub(1).reason
    assert env.basehub.downloads == [] and any("must stay free" in r.message for r in caplog.records)
    env.free = 500 * GIB
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_CONFIRM


async def test_unpinned_or_malformed_base_names_are_not_fetched(tmp_path):
    env = benv(tmp_path)
    for name in ("Qwen/Qwen3-4B@main", "../../etc@" + "b" * 40, "a/../x@" + "b" * 40, "Qwen/Qwen3-4B@" + "b" * 39):
        assert await env.bf.ensure(name) is None and "40-hex" in env.bf.reason(name)
    assert env.basehub.listings == []
    assert C.base_dir_name("a/../../x@" + "b" * 40) is None and C.base_dir_name("a/b@..") is None
    assert C.base_dir_name(DEFAULT_BASE) == "Qwen__Qwen3-4B@" + "b" * 40


async def test_preseed_runs_in_the_background(tmp_path):
    release = asyncio.Event()

    async def slow(base, dest):
        await release.wait()

    env = benv(tmp_path)
    env.cfg.base_preseed = (OTHER_BASE, "nobody/missing@" + "e" * 40)
    env.basehub.add(OTHER_BASE)
    env.basehub.during_download = slow
    await asyncio.wait_for(env.cycle({}), 5)                # the cycle does not wait for the preseed download
    assert env.ev.base_provider(OTHER_BASE) is None and env.bf.reason(OTHER_BASE) == "" and OTHER_BASE in env.bf._inflight
    assert await env.bf.ensure(OTHER_BASE) is None and "in progress" in env.bf.reason(OTHER_BASE)
    release.set()
    await asyncio.wait_for(env.bf._preseed, 5)
    assert env.ev.base_provider(OTHER_BASE) == base_path(env, OTHER_BASE)
    assert "not available" in env.bf.reason("nobody/missing@" + "e" * 40)
    await env.cycle({})
    assert env.basehub.downloads == [OTHER_BASE]             # started once per process


async def test_the_unverified_champion_of_an_older_release_is_checked_after_the_base_is_fetched(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    os.makedirs(base_path(env))
    write_model(base_path(env), make_weights(999))
    from test_spec_evaluator import crown
    await crown(env, 1, "a/champ", 0.8)
    champ = env.sub(1)
    champ.delta_checked, champ.base_id = False, ""          # as crowned by a release that let it pass
    shutil.rmtree(base_path(env))                           # and the base is not on this host any more
    await env.cycle({1: "a/champ"})                         # fetched at once: not held, keeps the title
    assert env.sub(1).status == ST.CHAMPION and env.basehub.downloads == [DEFAULT_BASE]
    env.add(2, "a/next", 0.5)                               # served again: checked first, on the fetched base
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).status == ST.CHAMPION and env.sub(1).delta_checked and env.sub(1).base_id
    assert env.basehub.downloads == [DEFAULT_BASE]


def test_config_from_env():
    cfg = C.from_env({"REAX_SPEC_SEALED_DIR": "/s", "REAX_SPEC_BASE_DIR": "/b", "REAX_SPEC_BASE_RETRY_CYCLES": "3",
                      "REAX_SPEC_BASE_BUDGET_GB": "10", "REAX_SPEC_BASE_MIN_FREE_GB": "70",
                      "REAX_SPEC_BASE_PRESEED": f" {DEFAULT_BASE} ,{OTHER_BASE},"})
    assert (cfg.base_fetch, cfg.base_retry_cycles, cfg.base_budget_bytes, cfg.base_min_free_bytes) == (True, 3, 10 * GIB, 70 * GIB)
    assert cfg.base_preseed == (DEFAULT_BASE, OTHER_BASE)
    d = C.from_env({"REAX_SPEC_SEALED_DIR": "/s"})
    assert (d.base_retry_cycles, d.base_budget_bytes, d.base_min_free_bytes, d.base_preseed) == (6, 40 * GIB, 50 * GIB, ())
    assert C.from_env({"REAX_SPEC_SEALED_DIR": "/s", "REAX_SPEC_BASE_FETCH": "0"}).base_fetch is False


def test_production_evaluator_gets_a_base_fetcher_tests_and_opt_out_do_not(tmp_path):
    def cfg(**kw):
        return C.SpecConfig(sealed_dir=str(tmp_path), work_dir=str(tmp_path / "w"), state_path=str(tmp_path / "s.json"), **kw)
    ev = E.SpecEvaluator(cfg(base_dir=str(tmp_path / "b")))
    assert isinstance(ev.base_fetcher, B.BaseFetcher) and isinstance(ev.base_fetcher.hub, F.HfHub)
    assert E.SpecEvaluator(cfg(base_dir=str(tmp_path / "b"), base_fetch=False)).base_fetcher is None
    assert E.SpecEvaluator(cfg()).base_fetcher is None     # no base directory: nowhere to put a base
    assert E.SpecEvaluator(cfg(base_dir=str(tmp_path / "b")), fetcher=object()).base_fetcher is None


async def test_hf_fetcher_uses_the_hub_for_submissions(tmp_path):
    """The submission path keeps its rules on top of the shared hub calls (pinned revision, gated, listing checks)."""
    hub = FakeBaseHub(tmp_path)
    src = os.path.join(tmp_path, "sub")
    write_model(src, make_weights(5))
    hub.bases["a/m1@" + "a" * 40] = src
    f = F.HfFetcher(hub=hub)
    from reax_subnet.spec.commitment import SpecCommitment
    from spec_helpers import HOTKEYS
    ins = await f.fetch(SpecCommitment(TRACK, "a/m1", "a" * 40), str(tmp_path / "dest"), category=TRACK, hotkey=HOTKEYS[0])
    assert ins.violations == [] and json.load(open(tmp_path / "dest" / R.MANIFEST_NAME))["category"] == TRACK
    hub.gated.add("a/m1@" + "a" * 40)
    with pytest.raises(F.FetchError) as exc:
        await f.fetch(SpecCommitment(TRACK, "a/m1", "a" * 40), str(tmp_path / "dest"), category=TRACK, hotkey=HOTKEYS[0])
    assert exc.value.kind == "unavailable"
