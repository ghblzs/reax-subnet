"""Mechanism v2: keep-alive for the 8 strongest waiting entries per track; first-pass queue ordered by immunity end."""

import pytest
from spec_helpers import REV, neuron, snapshot
from test_keepalive import DF, NOW, _snap, _sub

from reax_subnet import competition_split as CS
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.competition_split import combine_weights
from reax_subnet.spec import evaluator as E
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_keepalive

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
N = CS.KEEPALIVE_MAX_ENTRIES


def test_release_constant():
    assert N == 8


def test_more_than_eight_qualifying_entries_only_the_top_eight_by_nonproc_score_are_paid():
    st = ST.SpecState()
    # 12 entries on 12 coldkeys; the non-proc score (what qualified them) differs from the total score on purpose
    subs = [_sub(st, u, ST.PENDING_CONFIRM, 50.0 - u, nonproc=60.0 + u, block=u) for u in range(1, 13)]
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in out] == [12, 11, 10, 9, 8, 7, 6, 5]         # highest non-proc score first
    assert dict(out)[12] == 50.0 - 12                                 # the pair still carries the total score


def test_tie_goes_to_the_earlier_commitment_block_then_the_uid():
    st = ST.SpecState()
    blocks = {1: 50, 2: 10, 3: 30, 4: 10, 5: 20, 6: 40, 7: 60, 8: 70, 9: 80, 10: 5}
    subs = [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=70.0, block=b) for u, b in blocks.items()]
    out = compute_keepalive(st, [DF], _snap_cb(subs, blocks), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in out] == [10, 2, 4, 5, 3, 6, 1, 7]           # block 5, 10 (uid 2 before 4), 20, 30, 40, 50, 60


def _snap_cb(subs, cblocks):
    ns = []
    for s in subs:
        n = neuron(s.uid, SpecCommitment(s.track_id, s.repo, REV).text(), coldkey=s.coldkey)
        ns.append(Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=n.commitment,
                         commitment_block=cblocks.get(s.uid, 0)))
    return snapshot(1000, ns)


def test_tie_uses_the_onchain_commitment_block_not_the_local_first_seen_block():
    st = ST.SpecState()
    # first_seen_block (local observation) runs 1..10 by uid; the on-chain commitment blocks run the other way
    subs = [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=70.0, block=u) for u in range(1, 11)]
    cb = {u: 1000 - u for u in range(1, 11)}
    out = compute_keepalive(st, [DF], _snap_cb(subs, cb), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in out] == [10, 9, 8, 7, 6, 5, 4, 3]           # latest uid committed first on chain


def test_tie_with_unknown_commitment_block_falls_back_to_uid_not_first_seen():
    st = ST.SpecState()
    subs = [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=70.0, block=100 - u) for u in range(1, 11)]
    out = compute_keepalive(st, [DF], _snap_cb(subs, {}), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in out] == [1, 2, 3, 4, 5, 6, 7, 8]            # first_seen order would give 10, 9, ...


def test_two_per_coldkey_rule_comes_before_the_top_eight_cut():
    st = ST.SpecState()
    # four strongest entries share one coldkey: only its two best (by priority/commitment order) count, so the cut
    # then fills up with the weaker entries of other coldkeys
    subs = [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=90.0 - u, block=u) for u in (1, 2, 3, 4)]
    subs += [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=60.0 + u, block=u) for u in range(5, 14)]
    same = {u: "cold-A" for u in (1, 2, 3, 4)}
    out = compute_keepalive(st, [DF], _snap(subs, same), {DF: 0.0}, now=NOW)[DF]
    uids = [u for u, _ in out]
    assert uids[:2] == [1, 2] and 3 not in uids and 4 not in uids     # two slots for the coldkey, not four
    assert len(uids) == N and uids[2:] == [13, 12, 11, 10, 9, 8]


def test_eight_or_fewer_are_all_kept_and_an_unapproved_champion_is_never_cut():
    # review fix: a crowned champion waiting for the owner's approval keeps its slot even against stronger pending entries
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0, nonproc=65.0)] + [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=60.0 + u) for u in range(2, 9)]
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    assert len(out) == 8 and {u for u, _ in out} == set(range(1, 9))
    subs.append(_sub(st, 9, ST.PENDING_CONFIRM, 40.0, nonproc=99.0))
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    # rc8 (review-rc8 F2): the champion is listed on top of the 8 pending entries, in rank order (combine_weights cuts to 8 after it
    # dropped an approved champion), so an unapproved champion keeps its slot and the weakest pending entry (uid 2) is cut there
    assert len(out) == 9 and [u for u, _ in out[:2]] == [1, 9] and out[-1][0] == 2
    # nine stronger pending entries: the champion (non-proc 65, weakest of all) still holds a slot
    subs += [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=90.0 + u) for u in range(10, 18)]
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    assert len(out) == 9 and out[0][0] == 1


def test_an_approved_champion_leaves_eight_pending_entries_paid_an_unapproved_one_seven():
    # review-rc8 F2: approved champion 1 is paid by its rank; all 8 strongest pending entries keep their keep-alive
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0, nonproc=65.0)] + [_sub(st, u, ST.PENDING_CONFIRM, 40.0, nonproc=60.0 + u) for u in range(2, 11)]
    entries = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in entries] == [1, 10, 9, 8, 7, 6, 5, 4, 3]
    serving = {u: 1.0 for u in range(50, 90)}
    ranks = {DF: [(1, 70.0)]}
    approved = combine_weights(serving, None, 0.0, ranks, keepalive={DF: entries}, floors={DF: 0.0},
                               award_ok=lambda t, u: u == 1)
    assert {u for u in approved if u < 50} == {1} | set(range(3, 11))          # champion by rank + 8 pending
    pending = combine_weights(serving, None, 0.0, ranks, keepalive={DF: entries}, floors={DF: 0.0},
                              award_ok=lambda t, u: False)
    assert {u for u in pending if u < 50} == {1} | set(range(4, 11))           # champion's keep-alive slot + 7 pending



def test_combine_weights_pays_exactly_eight_each_the_capped_share():
    entries = [(u, 60.0 - u) for u in range(1, 13)]                   # rank order, as compute_keepalive returns it
    serving = {u: 1.0 for u in range(50, 90)}
    w = combine_weights(serving, None, 0.0, {}, keepalive={DF: entries}, floors={DF: 0.0})
    paid = {u for u in w if u < 50}
    assert paid == set(range(1, 9))
    assert all(w[u] > 0 for u in paid) and len({round(w[u], 12) for u in paid}) == 1   # equal pay
    assert sum(w[u] for u in paid) <= CS.KEEPALIVE_MAX_SHARE * CS.profile_for(CS.SPLIT_PROFILE).tracks[0] + 1e-9
    # the 4 outside the top 8 get nothing; the per-entry amount is that of 8 entries, not 12
    w12 = combine_weights(serving, None, 0.0, {}, keepalive={DF: entries[:8]}, floors={DF: 0.0})
    assert w == pytest.approx(w12)


# ------------------------------------------------------------------------------------------------ queue order
def _q(uid, hotkey=None):
    s = ST.Submission(key=f"k{uid}", uid=uid, hotkey=hotkey or f"hk{uid}", coldkey=f"c{uid}", track_id=DF, repo="m/m",
                      revision="a" * 40, first_seen_block=uid, first_seen_ts=1.0, status=ST.QUEUED)
    return s


def _snap_reg(block, regs):
    return Snapshot(block=block, neurons=tuple(Neuron(u, f"hk{u}", f"c{u}", False, 0.0, None, None, registered_block=r)
                                               for u, r in regs.items()), owner_hotkey=None)


def test_immunity_constant():
    assert E.IMMUNITY_BLOCKS_ASSUMED == 5000


def test_queue_goes_by_soonest_immunity_end_and_expired_entries_first():
    now = 100_000
    regs = {1: now - 1000,        # 4000 blocks left
            2: now - 4500,        # 500 left: first among those with time left
            3: now - 6000,        # already past immunity
            4: now - 5000,        # ends exactly now: past
            5: now - 2000,        # 3000 left
            6: 0}                 # unknown registration block: after the others, in commitment order
    snap = _snap_reg(now, regs)
    got = [s.uid for s in E.queue_order([_q(u) for u in (1, 2, 3, 4, 5, 6)], snap)]
    assert got == [3, 4, 2, 5, 1, 6]                                  # past-immunity keep commitment order (3 before 4)


def test_queue_order_ignores_hotkeys_missing_from_the_snapshot():
    snap = _snap_reg(10_000, {1: 9000})
    got = [s.uid for s in E.queue_order([_q(2), _q(1)], snap)]
    assert got == [1, 2]


async def test_evaluator_serves_the_entry_whose_immunity_ends_first(tmp_path):
    from spec_helpers import commit
    from test_spec_evaluator import TRACK, Env
    env = Env(tmp_path, {}, max_evals=1)                              # one evaluation per hour: the order decides
    env.add(1, "a/m1", 0.9)
    env.add(2, "a/m2", 0.9)
    # uid 1 committed first but registered recently; uid 2 registered long ago (immunity almost out)
    neurons = [neuron(1, commit(TRACK, "a/m1")), neuron(2, commit(TRACK, "a/m2"))]
    neurons = [Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=n.commitment,
                      registered_block=reg) for n, reg in zip(neurons, (990, 100))]
    await env.ev.evaluate_cycle(snapshot(1000, neurons))
    assert env.sub(2).status == ST.PENDING_CONFIRM and env.sub(1).status == ST.QUEUED
