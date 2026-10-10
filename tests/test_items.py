from reax_subnet.items import generate_items
from reax_subnet.protocol import DecideRequest, parse_request


def test_items_are_valid_protocol_items():
    items = generate_items(16, seed=1)
    req = DecideRequest("r", "hash-test", tuple(items))
    assert parse_request(req.to_bytes()) == req


def test_many_items_valid_and_unique():
    seen = set()
    for seed in range(200):
        items = generate_items(16, seed=seed)
        parse_request(DecideRequest("r", "hash-test", tuple(items)).to_bytes())
        for it in items:
            assert it.item_id not in seen
            seen.add(it.item_id)


def test_seeded_is_reproducible_and_unseeded_is_fresh():
    assert generate_items(8, seed=42) == generate_items(8, seed=42)
    a, b = generate_items(8), generate_items(8)
    assert {i.item_id for i in a}.isdisjoint({i.item_id for i in b})
    assert [i.question for i in a] != [i.question for i in b] or [i.options for i in a] != [i.options for i in b]


def test_item_space_is_diverse():
    # Fixed-text templates repeat their question; the option sets and orders differ.
    pairs = [(it.question, it.options) for s in range(50) for it in generate_items(16, seed=s)]
    assert len(set(pairs)) == len(pairs)
    assert len({q for q, _ in pairs}) > 400
