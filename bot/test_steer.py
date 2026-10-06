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


FLEE = Params.from_dict({"border_margin": 30.0, "flee_plan": 1.0})


def test_flee_slides_along_wall():
    """Pressed on the right wall with a bigger chaser coming from the open side, up-left: pushing
    straight away from it pins us on the wall (48% of the rule bots' mass lost, 2026-10-06 mining).
    The flee planner slides along the wall, away from the chaser."""
    me = cell(1, 3500 - 260, 0, 2500, "mine")
    chaser = cell(2, 3500 - 260 - 700, -350, 3800)
    s = Snapshot([me], [chaser], {}, 0, 0, BORDER, False)
    ux, uy = aim_dir(s, Plan("flee"), FLEE)
    assert uy > 0.8, f"aim ({ux:.2f}, {uy:.2f}) does not slide down the wall, away from the chaser"


def test_flee_open_field_runs_straight():
    me = cell(1, 0, 0, 500, "mine")
    chaser = cell(2, -600, 0, 900)
    s = Snapshot([me], [chaser], {}, 0, 0, BORDER, False)
    ux, uy = aim_dir(s, Plan("flee"), FLEE)
    assert ux > 0.9, f"aim ({ux:.2f}, {uy:.2f}) is not straight away from the chaser"


def test_flee_out_of_corner():
    """Top-right corner, chaser coming along the top wall: escape down the right wall. (With the
    chaser on the diagonal every heading is caught and staying put lasts longest: no test.)"""
    me = cell(1, 3500 - 260, -3500 + 260, 2500, "mine")
    chaser = cell(2, 3500 - 1000, -3500 + 300, 3800)
    s = Snapshot([me], [chaser], {}, 0, 0, BORDER, False)
    ux, uy = aim_dir(s, Plan("flee"), FLEE)
    assert uy > 0.8, f"aim ({ux:.2f}, {uy:.2f}) does not escape down the right wall"


PIECES = Params.from_dict({"border_margin": 30.0, "flee_plan": 1.0, "flee_pieces": 1.0, "flee_margin": 1.0})


def test_threat_to_a_piece_triggers_flee():
    """Split into a 2000 main cell and an 800 piece; a 1700 hunter (neither prey nor threat to the main cell) closes on
    the piece. 56% of safe_fp's lost mass was lost while split (2026-10-06 A/B): flee for the piece too."""
    from .modes import heuristic_plan
    main_cell, piece = cell(1, 0, 0, 2000, "mine"), cell(2, 0, 700, 800, "mine")
    hunter = cell(3, 0, 700 + 650, 1700)
    s = Snapshot([main_cell, piece], [hunter], {}, 0, 0, BORDER, False)
    assert heuristic_plan(s, PIECES).mode == "flee"
    ux, uy = aim_dir(s, heuristic_plan(s, PIECES), PIECES)
    assert uy < -0.8, f"aim ({ux:.2f}, {uy:.2f}) does not take the piece away from the hunter"


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
