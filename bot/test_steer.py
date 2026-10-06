"""Steering regression tests (pure arithmetic, no server).

    uv run python -m bot.test_steer            # or: pytest bot/test_steer.py
"""

import math
import sys
import traceback

from .game import Cell, Snapshot
from .params import Params
from .steer import Plan, steer

BORDER = {"left": -3500, "right": 3500, "top": -3500, "bottom": 3500}
CHAMP = Params.from_dict({"border_margin": 30.0, "target_bonus": 6.21, "lead": 0.0})


def cell(i, x, y, mass, kind="player"):
    return Cell(i, x, y, math.sqrt(mass * 100), kind, "")


def aim_dir(s: Snapshot, plan: Plan, p: Params = CHAMP) -> tuple[float, float]:
    x, y, _ = steer(s, plan, None, p)
    me = s.main
    d = math.hypot(x - me.x, y - me.y)
    return (x - me.x) / d, (y - me.y) / d


def test_hunt_into_corner():
    """Prey tucked into the corner beyond our wall margin (showcase 2026-10-06: champ-0 shook
    there for a minute). The wall push turned the aim away from the prey; we must head for it."""
    me = cell(1, 3388, -3406, 76, "mine")
    prey = cell(44, 3473, -3471, 53)
    s = Snapshot([me], [prey], {}, 0, 0, BORDER, False)
    ux, uy = aim_dir(s, Plan("hunt", 44))
    px, py = prey.x - me.x, prey.y - me.y
    cos = (ux * px + uy * py) / math.hypot(px, py)
    assert cos > 0.9, f"aim ({ux:.2f}, {uy:.2f}) is {math.degrees(math.acos(cos)):.0f} deg off the prey"


def test_farm_keeps_off_walls():
    """Without a hunt target the wall push still applies: nothing visible in a corner -> move out."""
    me = cell(1, 3420, -3420, 76, "mine")
    s = Snapshot([me], [], {}, 0, 0, BORDER, False)
    ux, uy = aim_dir(s, Plan("farm"))
    assert ux < 0 and uy > 0, f"aim ({ux:.2f}, {uy:.2f}) does not leave the corner"


def test_hunt_target_away_from_wall():
    """Target on the open side: the wall push still keeps us off the wall we are near."""
    me = cell(1, 3420, 0, 76, "mine")
    prey = cell(44, 3420, 300, 53)
    s = Snapshot([me], [prey], {}, 0, 0, BORDER, False)
    ux, _ = aim_dir(s, Plan("hunt", 44))
    assert ux < 0, f"aim x {ux:.2f}: no push off the right wall"


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
