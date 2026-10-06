"""Food router tests (pure arithmetic, no server).

    uv run pytest bot/test_farm.py
"""

import time

from .farm import FoodRouter
from .game import Cell, Snapshot

BORDER = {"left": -3500, "right": 3500, "top": -3500, "bottom": 3500}


def _walk(router: FoodRouter) -> tuple[float, float]:
    """Visit square (0, 1) once, then wait 10 decisions in the corner square (0, 0) with no food in
    view: (0, 1) was visited 0.8 game-s ago, so exploring must head for the other neighbour (1, 0)."""
    path = [(-3400, -3400 + 875)] + [(-3400, -3400)] * 10
    for x, y in path:
        s = Snapshot([Cell(1, x, y, 40, "mine", "")], [], {}, 0, 0, BORDER, False)
        target = router.next_waypoint(s, [], lambda c: False)
    return target


def test_explore_ignores_wall_clock(monkeypatch):
    """Exploration memory is in game time: a slow simulation or a fast lockstep server must
    explore the same way (it used time.monotonic, so it depended on the run's speed)."""
    base = _walk(FoodRouter())
    assert base == (-3500 + 1.5 * 875, -3500 + 0.5 * 875), base   # square (1, 0)
    t = [0.0]

    def fast_clock():
        t[0] += 50.0      # every call 50 wall-seconds apart
        return t[0]

    monkeypatch.setattr(time, "monotonic", fast_clock)
    assert _walk(FoodRouter()) == base
