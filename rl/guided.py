"""Guided (residual) neural bot: the network sees the champion's suggestion and decides
whether to follow it or override it.

Observation = rl.obs grid at G=48 over the server's real view, plus:
- velocity (vx, vy) of every moving cell (players incl. ours, viruses, ejected mass),
  measured over VEL_STEPS steps. One step is too short: cells move 17-39 units per
  step, less than one grid square;
- merge readiness of our own pieces and of enemy pieces: MultiOgar lets a cell remerge
  once its age >= max(30 s, 0.2 * size) * 25 ticks; age is counted from when we
  first saw the cell (exact for our own pieces, a lower bound for enemies);
- owner mass: every enemy piece is painted with log(its player's visible total mass /
  ours), grouped by name, so a big player split into small pieces still reads as big;
- virus fullness replaces the plain virus mask (0.3 at mass 100 -> 1.0 at 200, when
  the next feed fires it);
- hint scalars: the champion's aim direction and distance, its split/eject, its mode.
Action: move in {16 headings, stay, FOLLOW}; FOLLOW executes the champion's exact action
(including its split/eject, so the split/eject heads are ignored on FOLLOW).
"""

import math
from collections import defaultdict, deque

import numpy as np

from arena.policies import HeuristicPolicy
from bot.game import Snapshot
from bot.modes import heuristic_plan

from . import obs as O

G = 48                      # finer than the plain net's 32: a spawn cell is ~1.5 squares
EXTRA = ("vx", "vy", "own_merge", "enemy_merge", "owner_mass")
C = O.C + len(EXTRA)
S = O.S + 8                 # + hint: dir x, dir y, dist, split, eject, farm, hunt, flee
FOLLOW = 17                 # move index meaning "do what the champion does"
N_MOVE = 18
HEADS = (N_MOVE, 2, 2)
VEL_STEPS = 3               # velocity measured over this many steps (~240 ms)
VEL_SCALE = 40.0            # units/step that maps to 1.0 (a spawn-size cell's top speed)
SYNC_TICKS = 2              # server ticks per step
MERGE_SECONDS = 30          # MultiOgar playerMergeTime
VIRUS_MIN, VIRUS_MAX = 100.0, 200.0  # virus mass range (fires at max)
CH_VX, CH_VY, CH_OWN_MERGE, CH_ENEMY_MERGE, CH_OWNER = range(O.C, C)


class GuidedEncoder:
    """Per-agent: runs the champion (guide) and builds guided observations."""

    def __init__(self, guide_spec: dict):
        self.guide = HeuristicPolicy(guide_spec.get("farm", "route"), guide_spec.get("params"))
        self._p = self.guide._p
        self.reset()

    def reset(self) -> None:
        if self.guide:
            self.guide.reset()
        self.hist: deque[dict[int, tuple[float, float]]] = deque(maxlen=VEL_STEPS + 1)
        self.first_seen: dict[int, int] = {}
        self.step = 0
        self.suggestion = None  # champion action for the current observation

    def encode(self, s: Snapshot):
        grid0, scal0, cx, cy, ext = O.encode(s, G=G, server_view=True)
        grid = np.zeros((C, G, G), np.float32)
        scal = np.zeros(S, np.float32)
        grid[:O.C], scal[:O.S] = grid0, scal0
        if not s.mine:
            return grid, scal, cx, cy, ext
        self.step += 1
        k = G / (2 * ext)
        to_g = lambda c: ((c.x - cx) * k + G / 2, (c.y - cy) * k + G / 2, max(c.size * k, 0.7))

        movers = [c for c in s.mine + s.others if c.kind != "food"]
        for c in movers:
            self.first_seen.setdefault(c.id, self.step)
        self.hist.append({c.id: (c.x, c.y) for c in movers})
        old, steps = self.hist[0], len(self.hist) - 1
        my_total = s.total_mass
        groups = defaultdict(float)
        for c in s.others:
            if c.kind == "player":
                groups[c.name] += c.mass

        for c in movers:
            gx, gy, gr = to_g(c)
            if steps and c.id in old:
                vx = (c.x - old[c.id][0]) / steps / VEL_SCALE
                vy = (c.y - old[c.id][1]) / steps / VEL_SCALE
                if abs(vx) + abs(vy) > 0.02:
                    O._disc(grid, CH_VX, gx, gy, gr, max(-1.0, min(1.0, vx)), overwrite=True)
                    O._disc(grid, CH_VY, gx, gy, gr, max(-1.0, min(1.0, vy)), overwrite=True)
            if c.kind in ("mine", "player"):
                age_ticks = (self.step - self.first_seen[c.id]) * SYNC_TICKS
                ready = min(age_ticks / (max(MERGE_SECONDS, 0.2 * c.size) * 25), 1.0)
                ch = CH_OWN_MERGE if c.kind == "mine" else CH_ENEMY_MERGE
                O._disc(grid, ch, gx, gy, gr, max(ready, 0.05))
            if c.kind == "player":
                ratio = math.log(max(groups[c.name], 1.0) / max(my_total, 1.0))
                O._disc(grid, CH_OWNER, gx, gy, gr, max(-1.0, min(1.0, ratio / 3)), overwrite=True)
            elif c.kind == "virus":
                fill = (c.mass - VIRUS_MIN) / (VIRUS_MAX - VIRUS_MIN)
                O._disc(grid, 1, gx, gy, gr, 0.3 + 0.7 * max(0.0, min(1.0, fill)), overwrite=True)
        seen_now = {c.id for c in movers}
        if len(self.first_seen) > 4 * len(seen_now) + 200:  # forget long-gone cells
            self.first_seen = {i: t for i, t in self.first_seen.items() if i in seen_now}

        if self.guide is None:  # rl.nohint: same observation, hint scalars stay zero
            return grid, scal, cx, cy, ext
        x, y, split, *rest = self.guide.act(s)
        eject = bool(rest and rest[0])
        self.suggestion = (x, y, split, eject)
        plan = heuristic_plan(s, self._p)
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy) or 1.0
        scal[O.S:] = (dx / d, dy / d, min(d / ext, 2.0) / 2, float(split), float(eject),
                      float(plan.mode == "farm"), float(plan.mode == "hunt"),
                      float(plan.mode == "flee"))
        return grid, scal, cx, cy, ext

    def decode(self, move: int, split: int, eject: int, cx: float, cy: float, ext: float):
        if move == FOLLOW and self.suggestion is not None:
            return self.suggestion
        from .actions import decode
        return decode(min(move, 16), split, eject, cx, cy, ext)
