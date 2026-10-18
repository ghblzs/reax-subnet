"""Run once with each PYTHONPATH: actual 107cbf5 source versus merged source."""
import json
import pytest
from unittest.mock import patch
from test_spec_evaluator import Env
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

async def scripted_states(tmp_path):
    env = Env(tmp_path, {})
    assert not getattr(env.cfg, 'v3_decisions', False)
    assert getattr(env.cfg, 'x_tier_mode',
        'off') == 'off'
    states = []
    def checkpoint(name):
        with patch.object(ST.time, "time", env.clock):
            raw = ST._to_json(env.ev.state)
        # New optional defaults are the only permissible schema difference. No history, score, event,
        # budget, draw, serve log, timing or existing state field is normalized.
        for ts in raw['tracks'].values():
            for sub in ts['submissions'].values():
                for field, default in [('decision_rule',
        ''), ('close_draws', []), ('confirm_draws', [])]:
                    if field in sub:
                        assert sub[field] == default
                        del sub[field]
        states.append([name, raw])
    env.add(1, 'a/champ', 0.65)
    await env.cycle({1: 'a/champ'})
    checkpoint('first draw vacant')
    env.clock.advance(13 * 3600)
    await env.cycle({1: 'a/champ'})
    checkpoint('crown vacant')
    env.add(2, 'a/weak', 0.3)
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak'})
    checkpoint('reject weak')
    env.add(3, 'a/better', 0.95)
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('pass better')
    env.clock.advance(13 * 3600)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('replace incumbent')
    env.ev = env.make()
    env.clock.advance(3601)
    await env.cycle({1: 'a/champ', 2: 'a/weak', 3: 'a/better'})
    checkpoint('persist restart idle')
    return states

# Golden states produced by compare-flags-off.py with the actual 107cbf5 source (not the merged evaluator).
# Only absent v3 defaults are projected away; the serialization clock is the scenario clock.
async def test_flags_off_matches_107cbf5_state_at_every_checkpoint(tmp_path):
    import hashlib
    states = await scripted_states(tmp_path)
    got = [hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest() for _, raw in states]
    assert got == [
        'b61d6d617b4552d1b23d90e1c1f5045659b2d93c47c9e9e961ad96a4c689d413',
        '766588159c53065b3a78b01db0c3fc9c076e1a077f1c2482859334d2ebcf2b35',
        'a0fccc350bcc7285d5b4428fa25015e2b525a336b4800f07bd298b9ceaa3ad93',
        '0b1a78df06d9cfaa98d2f320803c78e44cddb13bf7e440a88c9b2e003ad7ec27',
        '5e3b9a146111cc9dc97112a90cdac47d33e3e83ba81854c58cc0a59e2caebc79',
        '7d2a2222cb60c85ed21e90b14c7666562b6c49cf79f65ab681ee2bbd523e81c9'
    ]
