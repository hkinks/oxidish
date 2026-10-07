"""Rule-based mode selection: the free, instant stand-in for Jev in arena runs."""

import math

from .game import Snapshot
from .params import DEFAULT, Params
from .steer import EAT_RATIO, Plan, classify, danger_radius, eaters_of, guarded_cells, half_of, split_threats


def heuristic_plan(s: Snapshot, p: Params = DEFAULT) -> Plan:
    me = s.main
    _, prey, threats, _ = classify(s, p)
    gap = lambda c: me.dist(c) - c.size - me.size
    if any(gap(t) < danger_radius(t, me, p, s.border) * p.flee_margin for t in threats):
        return Plan("flee")
    if p.flee_pieces > 0.5:     # a threat to one of our bigger pieces (not the main cell) also makes us flee
        for q in guarded_cells(s, p):
            if q is not me and any(q.dist(t) - t.size - q.size < danger_radius(t, q, p) * p.flee_margin
                                   for t in eaters_of(s, q)):
                return Plan("flee")
    rng, floor = p.hunt_range, me.mass * p.min_prey_frac
    if p.big_hunt > 0.5 and me.mass > p.big_mass:
        # Big: food barely offsets decay, so chase players across the whole (wider)
        # view, and keep the prey floor where it was at big_mass instead of letting
        # it rise until nothing on the map qualifies.
        rng *= math.sqrt(me.mass / p.big_mass)
        floor = p.big_mass * p.min_prey_frac
    worth = [c for c in prey if gap(c) < rng and c.mass >= floor]
    if worth:
        target = max(worth, key=lambda c: c.mass / (max(gap(c), 0) + 100))
        side = half_of(me) if p.half_split_safety > 0.5 else me
        safe = not any(gap(t) < danger_radius(t, side, p) * p.split_safety for t in split_threats(s, me, p))
        if p.split_ev > 0:    # split by expected value: steer() weighs gain against risk
            return Plan("hunt", target.id, me.mass / 2 > target.mass * EAT_RATIO)
        return Plan("hunt", target.id, safe and me.mass / 2 > target.mass * EAT_RATIO)
    return Plan("farm")
