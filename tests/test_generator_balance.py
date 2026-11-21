from __future__ import annotations

import random
from collections import Counter

import pytest

from word_search_generator import WordSearch
from word_search_generator.core.word import Direction, Word
from word_search_generator.utils import get_random_words
from word_search_generator.word_search._generator import WordSearchGenerator, _axis


def _line_counts(ws: WordSearch, secret: bool = False) -> Counter[tuple[int, int]]:
    return Counter(
        _axis(w.direction)  # type: ignore[arg-type]
        for w in ws.placed_words
        if w.secret is secret
    )


def _prepared_generator(level: int = 3) -> WordSearchGenerator:
    """A generator with an empty 15x15 puzzle, ready for placement calls."""
    ws = WordSearch("placeholder", level=level, size=15)
    gen = WordSearchGenerator()
    gen.game = ws  # type: ignore[assignment]  # GameType is an unbound TypeVar
    gen.puzzle = ws._build_puzzle(ws.size, "")
    gen._axis_counts = {False: Counter(), True: Counter()}
    gen._line = None
    return gen


def test_axis_pairs_opposite_directions():
    for d in Direction:
        opposite = Direction((-d.r_move, -d.c_move))
        assert _axis(d) == _axis(opposite)
    assert len({_axis(d) for d in Direction}) == 4


def test_lines_by_use_puts_least_used_first():
    gen = _prepared_generator()
    gen._axis_counts[False][_axis(Direction.S)] = 3
    gen._axis_counts[False][_axis(Direction.E)] = 2
    gen._axis_counts[False][_axis(Direction.SE)] = 1
    random.seed(42)
    for _ in range(25):
        assert gen._lines_by_use(Word("cat")) == [
            _axis(Direction.NE),
            _axis(Direction.SE),
            _axis(Direction.E),
            _axis(Direction.S),
        ]


def test_lines_by_use_breaks_ties_randomly():
    gen = _prepared_generator()
    random.seed(42)
    firsts = {gen._lines_by_use(Word("cat"))[0] for _ in range(100)}
    assert len(firsts) == 4


def test_lines_by_use_only_offers_allowed_lines():
    gen = _prepared_generator(level=1)
    assert sorted(gen._lines_by_use(Word("cat"))) == sorted(
        [_axis(Direction.E), _axis(Direction.S)]
    )


def test_find_a_fit_stays_on_current_line():
    gen = _prepared_generator()
    gen._line = _axis(Direction.NE)
    random.seed(42)
    picked = {gen.find_a_fit(Word("cat"), (7, 7))[0] for _ in range(50)}
    assert picked == {"NE", "SW"}  # both ways along the line get used


def test_word_falls_back_to_next_line_when_blocked(monkeypatch):
    random.seed(3)
    original = WordSearchGenerator.test_a_fit

    def no_vertical(self, word, position, direction):
        if _axis(direction) == _axis(Direction.S):
            return []
        return original(self, word, position, direction)

    monkeypatch.setattr(WordSearchGenerator, "test_a_fit", no_vertical)
    ws = WordSearch("apple banana cherry grape lemon mango", level=3, size=15)
    assert len(ws.placed_words) == 6
    assert _axis(Direction.S) not in _line_counts(ws)


def test_secret_words_counted_separately():
    random.seed(7)
    ws = WordSearch(
        "apple banana cherry grape lemon mango",
        level=3,
        size=15,
        secret_words="melon",
        secret_level=1,
    )
    gen = ws.generator
    assert isinstance(gen, WordSearchGenerator)
    assert sum(gen._axis_counts[False].values()) == len(ws.placed_hidden_words)
    assert gen._axis_counts[False] == _line_counts(ws)
    assert gen._axis_counts[True] == _line_counts(ws, secret=True)


@pytest.mark.parametrize(
    ("level", "lines"),
    [
        (3, 4),
        (2, 4),
        (1, 2),
        (7, 2),
    ],
)
def test_words_spread_evenly_across_lines(level: int, lines: int):
    random.seed(1234)
    total: Counter[tuple[int, int]] = Counter()
    spreads = []
    placed = 0
    for _ in range(30):
        ws = WordSearch(get_random_words(12), level=level, size=15)
        counts = _line_counts(ws)
        assert all(w.direction in ws.directions for w in ws.placed_words)
        assert len(counts) == lines
        total.update(counts)
        spreads.append(max(counts.values()) - min(counts.values()))
        placed += len(ws.placed_words)

    assert placed >= 30 * 11  # no drop in how many words get placed
    share = 1 / lines
    for count in total.values():
        assert count / placed == pytest.approx(share, abs=0.03)
    assert sum(spreads) / len(spreads) <= 0.5
