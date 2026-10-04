"""Virus skills that override normal steering: shooting viruses and hiding behind them.

Shooting: a virus (mass 100) fires a new virus once fed to mass 200; each eject
adds 13, so ~8 ejects (~136 of our mass). The new virus flies along the last
eject's direction, so we aim through the virus at a big player behind it; a
virus pops any cell more than 1.33x its mass.
Hiding: a cell lighter than a virus can sit on it; a bigger chaser can't follow
without popping, so under threat we move to the virus's far side.
Both are gated by evolvable genes (`*_enable` > 0.5) so evolution decides if they pay.
"""

import math

from .game import Cell, Snapshot
from .params import Params
from .steer import EAT_RATIO, Plan, classify, danger_radius

VIRUS_POP_RATIO = 1.33


def _unit(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


class Tactics:
    def __init__(self):
        self._feeding: int | None = None  # id of the virus we're shooting
        self._second_split: tuple[float, float] | None = None  # double split in progress
        self.counters = {"ejects": 0, "shots_started": 0, "shots_fired": 0, "hide_ticks": 0,
                         "selffeeds": 0, "double_splits": 0}

    def reset(self) -> None:
        self._feeding = None
        self._second_split = None

    def act(self, s: Snapshot, plan: Plan, p: Params):
        """Return (x, y, split, eject) to override steering, or None."""
        me = s.main
        _, _, threats, viruses = classify(s)
        gap = lambda c: me.dist(c) - c.size - me.size
        by_id = {v.id: v for v in viruses}
        if self._feeding is not None and self._feeding not in by_id:
            self.counters["shots_fired"] += 1  # the virus we fed fired (or left view)
            self._feeding = None

        if self._second_split is not None:  # finish a double split: second press, same aim
            x, y = self._second_split
            self._second_split = None
            return x, y, True, False

        safe = not any(gap(t) < danger_radius(t, me, p) for t in threats)

        # Self-feeding: while split and safe, aim at our biggest piece and eject, so the
        # small pieces' W blobs feed it and the mass is back in one cell before the
        # merge timer would allow (ready to split-attack again).
        if (p.selffeed_enable > 0.5 and len(s.mine) >= 2 and safe
                and s.total_mass >= p.selffeed_min_mass and me.mass / s.total_mass < p.selffeed_frac):
            self.counters["selffeeds"] += 1
            self.counters["ejects"] += 1
            return me.x, me.y, False, True

        # Double split: reach a target one split can't, if a quarter of us still eats it.
        if (p.double_split_enable > 0.5 and plan.mode == "hunt" and len(s.mine) == 1 and safe
                and plan.target_id is not None):
            target = next((c for c in s.others if c.id == plan.target_id), None)
            if target is not None and me.mass / 4 > target.mass * EAT_RATIO:
                k = p.lead * min(1.0, me.dist(target) / 600)
                tx, ty = target.x + target.vx * k, target.y + target.vy * k
                reach = p.split_range * p.split_reach
                d = math.hypot(tx - me.x, ty - me.y) - me.size
                if reach < d < reach * p.double_split_reach:
                    self.counters["double_splits"] += 1
                    self._second_split = (tx, ty)
                    return tx, ty, True, False

        if p.hide_enable > 0.5 and threats:
            near = [t for t in threats if gap(t) < danger_radius(t, me, p) * p.flee_margin]
            covers = [v for v in viruses if v.mass > me.mass * EAT_RATIO and gap(v) < p.hide_range]
            if near and covers:
                chaser = min(near, key=gap)
                v = min(covers, key=me.dist)
                ux, uy = _unit(chaser.x, chaser.y, v.x, v.y)  # far side, away from the chaser
                self.counters["hide_ticks"] += 1
                return v.x + ux * v.size * 0.6, v.y + uy * v.size * 0.6, False, False

        if p.shoot_enable > 0.5 and len(s.mine) == 1 and me.mass >= p.shoot_min_mass:
            if any(gap(t) < danger_radius(t, me, p) * 0.7 for t in threats):
                self._feeding = None
                return None  # survival first
            targets = [c for c in s.others if c.kind == "player"]
            best, best_score = None, 0.0
            for v in viruses:
                if gap(v) > p.shoot_range or me.mass > v.mass * VIRUS_POP_RATIO and gap(v) < 30:
                    continue
                ax, ay = _unit(me.x, me.y, v.x, v.y)
                for t in targets:
                    if t.mass < v.mass * VIRUS_POP_RATIO * p.shoot_target_mult:  # only worth it vs big cells
                        continue
                    d = v.dist(t)
                    if d > p.shoot_reach:
                        continue
                    bx, by = _unit(v.x, v.y, t.x, t.y)
                    if ax * bx + ay * by < p.shoot_align:
                        continue
                    score = t.mass / (d + 100) * (2 if v.id == self._feeding else 1)
                    if score > best_score:
                        best, best_score = v, score
            if best is not None:
                if best.id != self._feeding:
                    self.counters["shots_started"] += 1
                self._feeding = best.id
                self.counters["ejects"] += 1
                return best.x, best.y, False, True
            self._feeding = None
        return None
