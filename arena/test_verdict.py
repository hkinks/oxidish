"""arena.verdict decision logic on synthetic results (no server).

    uv run pytest arena/test_verdict.py
"""

from .verdict import sprt, status


def rows(wins: int, losses: int) -> list[dict]:
    out = []
    for s in range(wins + losses):
        cand = 7.0 if s < wins else 6.0
        out += [{"policy": "cand", "seed": s, "log_mass": cand}, {"policy": "opp", "seed": s, "log_mass": 6.5}]
    return out


def test_sprt_signs():
    assert sprt(60, 100, 0.5, 0.6) > 0 > sprt(40, 100, 0.5, 0.6)


def test_clear_winner_passes():
    assert status(rows(130, 70), "cand", 0.5, 0.6, 0.05, 0.05) == {"opp": "PASS"}


def test_coin_flip_fails_h1():
    assert status(rows(100, 100), "cand", 0.5, 0.6, 0.05, 0.05) == {"opp": "FAIL"}


def test_few_matches_stay_open():
    assert status(rows(7, 5), "cand", 0.5, 0.6, 0.05, 0.05) == {"opp": "open"}
