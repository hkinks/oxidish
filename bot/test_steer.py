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


CONTEND = Params.from_dict({"border_margin": 30.0, "contender": 0.75, "contender_gap": 200.0})


def test_contender_is_a_threat():
    """Hunting a 1369 prey while overlapping a 2043 near-equal (we are 2275): the neighbour can't eat us,
    but one meal (that prey) lets it. 30% of r300_fp's lost mass went this way (2026-10-06 mining)."""
    from .modes import heuristic_plan
    me = cell(1, 0, 0, 2275, "mine")
    rival = cell(2, 300, 0, 2043)
    prey = cell(3, 0, 700, 1369)
    s = Snapshot([me], [rival, prey], {}, 0, 0, BORDER, False)
    plan = heuristic_plan(s, CONTEND)
    assert plan.mode == "flee", plan
    ux, _ = aim_dir(s, plan, CONTEND)
    assert ux < -0.5, f"aim x {ux:.2f} does not move away from the rival"


WALLD = Params.from_dict({"border_margin": 30.0, "wall_danger": 400.0})


def test_wall_danger_flees_earlier_at_a_wall():
    """A 1400 chaser 1700 from a 600 cell: far enough to farm in the open, but pinned on a wall (no room to run)
    the bot flees. Off (0) it does not; mid-map the gene changes nothing."""
    from .modes import heuristic_plan
    base = Params.from_dict({"border_margin": 30.0})
    at_wall = Snapshot([cell(1, 3500 - 120, 0, 600, "mine")], [cell(2, 3500 - 120 - 1700, 0, 1400)], {}, 0, 0, BORDER, False)
    mid = Snapshot([cell(1, 0, 0, 600, "mine")], [cell(2, -1700, 0, 1400)], {}, 0, 0, BORDER, False)
    assert heuristic_plan(at_wall, base).mode == "farm"
    assert heuristic_plan(at_wall, WALLD).mode == "flee"
    assert heuristic_plan(mid, WALLD).mode == heuristic_plan(mid, base).mode

def test_aim_leads_every_own_piece():
    """Split, with a 1197 piece 775 ahead (left) of our 1832 main cell, heading left: the mouse 600 ahead of
    the main cell was behind the piece, so the piece turned back into the main cell and both crawled
    (ramp300 parked 10+ s like this, 2026-10-06). Every own cell must head the same way."""
    me = cell(1, 2728, 3053, 1832, "mine")
    piece = cell(2, 2728 - 775, 3053, 1197, "mine")
    s = Snapshot([me, piece], [cell(9, 2728 - 300, 3053, 30), cell(10, 2728 - 1500, 3053, 30)], {}, 0, 0, BORDER, False)
    x, y, _ = steer(s, Plan("hunt", 10), None, CHAMP)
    d = math.hypot(x - me.x, y - me.y)
    ux, uy = (x - me.x) / d, (y - me.y) / d
    assert ux < -0.5, f"heading ({ux:.2f}, {uy:.2f}) is not toward the prey on the left"
    ahead = (x - piece.x) * ux + (y - piece.y) * uy      # mouse beyond the piece, along the heading
    assert ahead > 300, f"mouse only {ahead:.0f} ahead of the piece: it turns back into the main cell"


def test_wall_wary_keeps_off_the_wall_near_a_big_player():
    """Farming 500 off the right wall: with nobody near the wall margin (30) leaves us alone; with a bigger player
    within 1500 the widened margin pushes us off the wall."""
    wary = Params.from_dict({"border_margin": 30.0, "wall_wary": 400.0})
    me = cell(1, 3500 - 500, 0, 600, "mine")
    big = cell(2, 3500 - 500 - 1200, 0, 700)
    alone = Snapshot([me], [], {}, 0, 0, BORDER, False)
    near = Snapshot([me], [big], {}, 0, 0, BORDER, False)
    assert aim_dir(alone, Plan("farm"), wary)[0] == 1.0
    assert aim_dir(near, Plan("farm"), wary)[0] < aim_dir(near, Plan("farm"), CHAMP)[0] - 0.01


def test_piece_steers_around_virus():
    """Split: main cell ahead, a 500 piece behind it with a virus right on the piece's way. 42 of 53 own
    virus pops happened while split, mostly by pieces (2026-10-07, Node): the detour must cover pieces."""
    p = Params.from_dict({"border_margin": 30.0, "piece_detour": 1.0})
    me = cell(1, 0, 0, 1200, "mine")
    piece = cell(2, -900, 300, 500, "mine")
    virus = cell(3, -900 + 330, 300 + 5, 100, "virus")
    food = [cell(10 + i, 900 + 40 * i, 0, 1, "food") for i in range(5)]
    s = Snapshot([me, piece], [virus] + food, {}, 0, 0, BORDER, False)
    x, y, _ = steer(s, Plan("farm"), None, p)
    # the piece heads for the mouse: its line to the mouse must clear the virus
    dx, dy = x - piece.x, y - piece.y
    d = math.hypot(dx, dy)
    t = ((virus.x - piece.x) * dx + (virus.y - piece.y) * dy) / d
    off = abs((virus.x - piece.x) * dy - (virus.y - piece.y) * dx) / d
    assert t <= 0 or off > virus.size + piece.size, f"piece passes {off:.0f} from the virus centre (needs > {virus.size + piece.size:.0f})"


def test_split_onto_split_enemy_pieces():
    """An enemy split into pieces near us, none of which can eat us or our half: split onto them (the fixed
    split_reach of ~300 refused; the user saw bots shy away from 'bigger, split enemies they could gobble up')."""
    from .modes import heuristic_plan
    p = Params.from_dict({"border_margin": 30.0, "split_ev": 1.0, "split_reach": 0.36, "split_range": 842.0})
    me = cell(1, 0, 0, 3000, "mine")
    pieces = [Cell(10 + i, 980 + 60 * i, 40 * i, math.sqrt(500 * 100), "player", "enemy") for i in range(3)]
    s = Snapshot([me], pieces, {}, 0, 0, BORDER, False)
    plan = heuristic_plan(s, p)
    assert plan.mode == "hunt" and plan.split_ok, plan
    _, _, split = steer(s, plan, None, p)
    assert split, "no split onto three edible pieces within split range"


def test_no_ev_split_into_a_threat():
    """Same pieces, but a big enemy cell next to the landing spot could eat our half: no split."""
    from .modes import heuristic_plan
    p = Params.from_dict({"border_margin": 30.0, "split_ev": 1.0, "split_range": 842.0})
    me = cell(1, 0, 0, 3000, "mine")
    pieces = [Cell(10 + i, 980 + 60 * i, 40 * i, math.sqrt(500 * 100), "player", "enemy") for i in range(3)]
    big = Cell(20, 1350, 0, math.sqrt(2500 * 100), "player", "enemy")
    s = Snapshot([me], pieces + [big], {}, 0, 0, BORDER, False)
    _, _, split = steer(s, heuristic_plan(s, p), None, p)
    assert not split, "split next to a cell that eats our half"


def test_escape_split_when_cornered():
    """Pinned in a corner, a 1.6x chaser closing from the open side: every heading is caught within the look-ahead,
    so split away along the best escape heading (the user: 'wall deaths can be avoided by splitting')."""
    from .modes import heuristic_plan
    p = Params.from_dict({"border_margin": 30.0, "escape_split": 1.0, "flee_margin": 1.0})
    me = cell(1, 3500 - 260, -3500 + 260, 2500, "mine")
    chaser = cell(2, me.x - 500, me.y + 300, 4000)
    s = Snapshot([me], [chaser], {}, 0, 0, BORDER, False)
    plan = heuristic_plan(s, p)
    assert plan.mode == "flee", plan
    x, y, split = steer(s, plan, None, p)
    assert split, "no escape split although every heading is caught"
    # the lunging half lands out of the chaser's reach (it eats a cell whose centre is within its size - ours/3)
    half = math.sqrt(2500 / 2 * 100)
    d = math.hypot(x - me.x, y - me.y)
    lx = min(max(me.x + (x - me.x) / d * 0.8 * 650, -3500 + half / 2), 3500 - half / 2)
    ly = min(max(me.y + (y - me.y) / d * 0.8 * 650, -3500 + half / 2), 3500 - half / 2)
    assert math.hypot(lx - chaser.x, ly - chaser.y) > chaser.size - half / 3, (x - me.x, y - me.y)


def test_no_escape_split_when_escape_is_possible():
    """Open field, chaser far behind: running works, no split."""
    from .modes import heuristic_plan
    p = Params.from_dict({"border_margin": 30.0, "escape_split": 1.0, "flee_margin": 1.0})
    me = cell(1, 0, 0, 2500, "mine")
    chaser = cell(2, -1300, 0, 4000)
    s = Snapshot([me], [chaser], {}, 0, 0, BORDER, False)
    _, _, split = steer(s, heuristic_plan(s, p), None, p)
    assert not split


def test_shoot_a_bigger_rival_through_a_virus():
    """The human's virus shots were at rivals his own size or bigger, the virus between them (2026-10-07). With
    shoot_threats the bot feeds the virus at such a threat; without it, the threat vetoes shooting."""
    from .steer import Plan as _Plan
    from .tactics import Tactics
    me = cell(1, 0, 0, 3000, "mine")
    virus = cell(2, 700, 0, 100, "virus")
    rival = cell(3, 1600, 0, 4500)
    s = Snapshot([me], [virus, rival], {}, 0, 0, BORDER, False)
    base = {"border_margin": 30.0, "shoot_enable": 1.0, "shoot_min_mass": 300.0}
    off = Tactics().act(s, _Plan("farm"), Params.from_dict(base))
    on = Tactics().act(s, _Plan("farm"), Params.from_dict({**base, "shoot_threats": 1.0}))
    assert off is None, off
    assert on is not None and on[3], f"no shot at the rival behind the virus: {on}"


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
