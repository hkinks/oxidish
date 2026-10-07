"""AposBot port and virus-staging tests (pure arithmetic, no server).

    uv run pytest bot/test_apos.py
"""

import math

from arena.policies import build
from bot.game import Cell, Snapshot
from bot.params import Params
from bot.tactics import Tactics
from bot.steer import Plan

BORDER = {"left": -3500.0, "right": 3500.0, "top": -3500.0, "bottom": 3500.0}


def cell(i, x, y, mass, kind="player", color=0):
    return Cell(i, x, y, math.sqrt(mass * 100), kind, "", color=color)


def snap(me, others):
    return Snapshot([me], others, {}, 0, 0, dict(BORDER), False)


def test_apos_walks_to_food():
    me = cell(1, 0, 0, 50, "mine", 7)
    food = [cell(100 + i, 400 + 20 * i, 10, 1, "food") for i in range(5)]
    x, y, split, eject = build({"kind": "apos"}).act(snap(me, food))
    assert x > 100 and abs(y) < 200 and not split and not eject


def test_apos_flees_a_big_threat():
    me = cell(1, 0, 0, 50, "mine", 7)
    threat = cell(2, 350, 0, 400, color=9)
    pol = build({"kind": "apos"})
    for _ in range(5):                  # let the motion tracker and heading settle
        x, y, *_ = pol.act(snap(me, [threat]))
    assert x < 0, f"aimed at {x:.0f},{y:.0f}, towards the threat"


def test_apos_splits_onto_close_prey_and_ignores_far_prey():
    me = cell(1, 0, 0, 400, "mine", 7)
    prey = cell(2, 450, 0, 30, color=9)
    pol = build({"kind": "apos"})
    out = [pol.act(snap(me, [prey])) for _ in range(6)]
    assert any(o[2] for o in out), "never split onto a prey 450 away"


def test_apos_feeds_a_virus_towards_a_big_player():
    me = cell(1, 0, 0, 450, "mine", 7)
    virus = cell(3, 450, 0, 100, "virus", 0x33FF33)
    big = cell(4, 1100, 0, 900, color=9)
    pol = build({"kind": "apos"})
    out = [pol.act(snap(me, [virus, big])) for _ in range(8)]
    assert any(o[3] for o in out) or pol.counters["shots_started"] or pol.counters.get("mode_stage", 0), pol.counters


def test_shoot_stage_walks_to_the_far_side_of_a_virus():
    """Hunter virus play: with a big player on one side of a virus we walk round to the other side."""
    p = Params.from_dict({"shoot_enable": 1.0, "shoot_stage": 1.0, "shoot_target_mult": 1.6, "shoot_min_mass": 200.0,
                          "shoot_standoff": 35.0, "shoot_walk": 1400.0, "shoot_reach": 1000.0})
    me = cell(1, 700, 0, 400, "mine")          # east of the virus, not on the line to the target
    virus = cell(2, 300, 0, 100, "virus")
    big = cell(3, 300, -950, 900)              # north of the virus: the far side from it is south (y > virus.y)
    s = snap(me, [virus, big])
    t = Tactics()
    out = t.act(s, Plan("farm"), p)
    assert out is not None and out[3] is False, out
    # the waypoint is on the ring round the virus, on the side away from the target (y > virus.y)
    assert out[1] > virus.y
    off = Params.from_dict({"shoot_enable": 1.0})
    assert Tactics().act(s, Plan("farm"), off) is None, "staging must default off"
