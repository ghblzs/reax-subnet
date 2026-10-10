"""Fresh validator items.

Scoring compares a miner's distribution with the validator's own reference
computation, so items need no correct answer. That removes the usual leakage
problem: there is no answer key and no private pool to steal or memorise.
Every round draws new items from a cryptographically seeded generator, so a
cache of past items is worthless.

The generator favours open, slightly ambiguous questions (preferences,
plausibility, style). On such items the reference model's distribution is
spread out and specific to that model, which makes it a good fingerprint:
a different or heavily degraded model disagrees on many of them, whereas
purely factual or computable questions (arithmetic, number sequences) can be
answered by a few lines of code and are therefore not used.
"""

from __future__ import annotations

import random
import secrets
from typing import Callable

from .protocol import MAX_OPTIONS, MIN_OPTIONS, Item

_NAMES = (
    "Ada", "Bruno", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiro", "Ines", "Jonas", "Kofi", "Lena",
    "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tariq", "Uma", "Viktor", "Wen", "Ximena",
    "Yusuf", "Zofia", "Amara", "Bao", "Carmen", "Dario", "Esther", "Femi", "Goran", "Hana", "Ivo", "Jana",
)
_ANIMALS = (
    "cat", "dog", "parrot", "tortoise", "rabbit", "goat", "hedgehog", "ferret", "goldfish", "pony", "owl",
    "lizard", "hamster", "alpaca", "duck", "snail", "crow", "otter", "fox", "beetle",
)
_ADJECTIVES = (
    "sleepy", "curious", "grumpy", "elegant", "tiny", "loud", "clumsy", "brave", "shy", "restless", "ancient",
    "cheerful", "mysterious", "stubborn", "gentle", "chaotic", "proud", "nervous", "lazy", "dramatic",
)
_PLACES = (
    "lighthouse", "bakery", "library", "train station", "vineyard", "submarine", "mountain hut", "harbour",
    "greenhouse", "castle", "laundromat", "observatory", "farm", "rooftop garden", "museum", "ferry",
)
_PROFESSIONS = (
    "chef", "architect", "nurse", "pilot", "teacher", "plumber", "violinist", "judge", "farmer", "astronomer",
    "tailor", "lawyer", "carpenter", "journalist", "dentist", "firefighter", "beekeeper", "translator",
)
_EVENTS = (
    "wedding", "job interview", "first day at work", "picnic", "funeral", "birthday party", "hiking trip",
    "conference talk", "dinner with in-laws", "chess tournament", "camping weekend", "museum opening",
)
_OBJECTS = (
    "umbrella", "notebook", "scarf", "pocket knife", "thermos", "camera", "bicycle", "compass", "guitar",
    "lantern", "map", "sunhat", "backpack", "watch", "kettle", "book of poems", "rope", "flashlight",
    "raincoat", "sandwich", "deck of cards", "pair of binoculars", "first-aid kit", "harmonica",
)
_CITIES = (
    "Lisbon", "Oslo", "Kyoto", "Nairobi", "Lima", "Reykjavik", "Hanoi", "Marrakesh", "Vienna", "Montreal",
    "Tbilisi", "Cape Town", "Valparaiso", "Krakow", "Seville", "Auckland", "Istanbul", "Edinburgh",
)
_ACTIVITIES = (
    "cycling holiday", "food tour", "photography trip", "quiet writing retreat", "family visit",
    "surfing course", "architecture tour", "winter getaway", "music festival", "budget backpacking trip",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)
_COLOURS = (
    "red", "teal", "mustard yellow", "navy blue", "olive green", "lavender", "charcoal grey", "coral",
    "burgundy", "sky blue", "orange", "cream",
)
_TRAITS = (
    "patient", "funny", "reliable", "creative", "honest", "ambitious", "calm", "generous", "organised",
    "adventurous", "modest", "curious",
)
_ROLES = (
    "a team leader", "a travel companion", "a flatmate", "a babysitter", "a business partner",
    "a teacher for a shy child", "a crisis negotiator", "a ship's captain", "a wedding planner",
)
_FOODS = (
    "lentil soup", "sushi", "pancakes", "grilled cheese", "paella", "falafel", "ramen", "apple pie",
    "curry", "tacos", "risotto", "dumplings", "a green salad", "fish and chips", "porridge",
)
_SENTENCE_STARTS = (
    "Could you possibly", "Please", "I need you to", "Would you mind if we", "Hey, just", "Kindly",
    "It would be great if you could", "You must", "Let's", "I was wondering whether you might",
)
_SENTENCE_ENDS = (
    "send me the report by Friday", "move the meeting to next week", "turn the music down",
    "share the slides after the call", "water the plants while I am away", "check the numbers again",
    "pick up some bread on the way home", "review my draft before noon",
)
_GENRES = (
    "mystery novel", "science fiction story", "children's book", "travel memoir", "cookbook",
    "horror story", "romantic comedy", "history book", "poetry collection", "business guide",
)
_TITLE_WORDS = (
    "Silent", "River", "Glass", "Midnight", "Garden", "Iron", "Paper", "Last", "Hidden", "Winter", "Golden",
    "Echo", "Salt", "Lantern", "Orchard", "Thunder", "Velvet", "Harbour", "Copper", "Feather",
)


def _pick(rng: random.Random, pool: tuple[str, ...], k: int) -> list[str]:
    return rng.sample(pool, k)


def _n_options(rng: random.Random, low: int = MIN_OPTIONS, high: int = 6) -> int:
    return rng.randint(max(low, MIN_OPTIONS), min(high, MAX_OPTIONS))


def _t_pet_name(rng: random.Random) -> tuple[str, list[str]]:
    q = (
        f"What is the best name for a {rng.choice(_ADJECTIVES)} {rng.choice(_ANIMALS)} "
        f"that lives in a {rng.choice(_PLACES)}?"
    )
    return q, _pick(rng, _NAMES, _n_options(rng))


def _t_pack_object(rng: random.Random) -> tuple[str, list[str]]:
    q = (
        f"A {rng.choice(_PROFESSIONS)} may bring only one item to a {rng.choice(_EVENTS)}. "
        "Which item is the most sensible choice?"
    )
    return q, _pick(rng, _OBJECTS, _n_options(rng))


def _t_city_trip(rng: random.Random) -> tuple[str, list[str]]:
    q = f"Which city is the better destination for a {rng.choice(_ACTIVITIES)} in {rng.choice(_MONTHS)}?"
    return q, _pick(rng, _CITIES, _n_options(rng))


def _t_colour(rng: random.Random) -> tuple[str, list[str]]:
    q = f"Which colour suits the front door of a {rng.choice(_ADJECTIVES)} {rng.choice(_PLACES)} best?"
    return q, _pick(rng, _COLOURS, _n_options(rng))


def _t_trait(rng: random.Random) -> tuple[str, list[str]]:
    q = f"Which quality matters most in {rng.choice(_ROLES)}?"
    return q, _pick(rng, _TRAITS, _n_options(rng))


def _t_food(rng: random.Random) -> tuple[str, list[str]]:
    name = rng.choice(_NAMES)
    q = (
        f"{name} is a {rng.choice(_PROFESSIONS)} who just finished a long shift on a cold "
        f"{rng.choice(_MONTHS)} evening. What should {name} eat?"
    )
    return q, _pick(rng, _FOODS, _n_options(rng))


def _t_politeness(rng: random.Random) -> tuple[str, list[str]]:
    end = rng.choice(_SENTENCE_ENDS)
    starts = _pick(rng, _SENTENCE_STARTS, _n_options(rng, 3, 5))
    return "Which request sounds the most polite?", [f"{s} {end}." for s in starts]


def _t_book_title(rng: random.Random) -> tuple[str, list[str]]:
    genre = rng.choice(_GENRES)
    titles: set[str] = set()
    n = _n_options(rng)
    while len(titles) < n:
        a, b = rng.sample(_TITLE_WORDS, 2)
        titles.add(f"The {a} {b}")
    return f"Which title fits a {genre} best?", sorted(titles)


def _t_yes_no(rng: random.Random) -> tuple[str, list[str]]:
    q = (
        f"Would a {rng.choice(_ADJECTIVES)} {rng.choice(_ANIMALS)} enjoy living in a "
        f"{rng.choice(_PLACES)} in {rng.choice(_CITIES)}?"
    )
    opts = ["Yes", "No"] if rng.random() < 0.5 else ["No", "Yes"]
    if rng.random() < 0.4:
        opts.insert(rng.randint(0, 2), "Unclear")
    return q, opts


def _t_compare(rng: random.Random) -> tuple[str, list[str]]:
    p1, p2 = rng.sample(_PROFESSIONS, 2)
    q = f"Who is more likely to own a {rng.choice(_OBJECTS)}: a {p1} or a {p2}?"
    opts = [f"The {p1}", f"The {p2}", "Equally likely"]
    rng.shuffle(opts)
    return q, opts


_TEMPLATES: tuple[Callable[[random.Random], tuple[str, list[str]]], ...] = (
    _t_pet_name, _t_pack_object, _t_city_trip, _t_colour, _t_trait, _t_food, _t_politeness,
    _t_book_title, _t_yes_no, _t_compare,
)


def generate_items(n: int, seed: int | None = None) -> list[Item]:
    """Return ``n`` fresh items. ``seed`` is for tests only; production passes None."""
    rng = random.Random(secrets.randbits(128) if seed is None else seed)
    items: list[Item] = []
    for _ in range(n):
        question, options = rng.choice(_TEMPLATES)(rng)
        rng.shuffle(options)
        item_id = "%032x" % rng.getrandbits(128)
        items.append(Item(item_id, question, tuple(options)))
    return items
