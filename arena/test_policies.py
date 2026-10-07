"""HeuristicPolicy state across steps (no server).

    uv run pytest arena/test_policies.py
"""

import math

from bot.game import Cell, Snapshot

from .policies import HeuristicPolicy

BORDER = {"left": -3500, "right": 3500, "top": -3500, "bottom": 3500}


def cell(i, x, y, mass, kind="player"):
    return Cell(i, x, y, math.sqrt(mass * 100), kind, "")


def heading(out, me):
    d = math.hypot(out[0] - me.x, out[1] - me.y)
    return (out[0] - me.x) / d, (out[1] - me.y) / d


def _two_steps(params):
    """Fleeing from a chaser that drifts sideways: the raw flee heading swings ~80 deg between two steps."""
    pol = HeuristicPolicy("route", params)
    me = cell(1, 0, 0, 500, "mine")
    s1 = Snapshot([me], [cell(2, -500, -300, 1500)], {}, 0, 0, BORDER, False)
    s2 = Snapshot([me], [cell(2, -450, 380, 1500)], {}, 0, 0, BORDER, False)
    return heading(pol.act(s1), me), heading(pol.act(s2), me)


def test_flee_smooth_commits_to_an_escape():
    (a1, b1), (a2, b2) = _two_steps({"flee_margin": 1.0, "flee_smooth": 0.7})
    raw1, raw2 = _two_steps({"flee_margin": 1.0})
    swing = math.degrees(math.acos(max(-1, min(1, raw1[0] * raw2[0] + raw1[1] * raw2[1]))))
    smooth = math.degrees(math.acos(max(-1, min(1, a1 * a2 + b1 * b2))))
    assert 60 < swing < 100, f"scenario should swing 60-100 deg raw, got {swing:.0f}"
    assert smooth < swing - 20, f"smoothed swing {smooth:.0f} vs raw {swing:.0f}"


def test_flee_smooth_lets_a_reversal_through():
    """The chaser jumps to the other side: no blending (never keep fleeing toward it)."""
    pol = HeuristicPolicy("route", {"flee_margin": 1.0, "flee_smooth": 0.7})
    me = cell(1, 0, 0, 500, "mine")
    pol.act(Snapshot([me], [cell(2, -500, 0, 1500)], {}, 0, 0, BORDER, False))
    ux, _ = heading(pol.act(Snapshot([me], [cell(3, 500, 0, 1500)], {}, 0, 0, BORDER, False)), me)
    assert ux < -0.8, f"heading x {ux:.2f}: should flee left from the chaser now on the right"
