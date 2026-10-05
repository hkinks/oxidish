"""Neural policy as an arena policy (spec {"kind": "nn", "ckpt": path, "greedy": bool}).

Guided checkpoints (with a FOLLOW action) run their champion alongside the net.
"""

import math

import torch

from bot.game import Snapshot
from sim.engine import VIRUS_EAT_MULT

from . import actions as A
from .common import load
from .obs import encode

torch.set_num_threads(1)  # many agents per process; keep each one light


class NNPolicy:
    def __init__(self, ckpt: str, greedy: bool | None = None, nosplit: bool = False, guard: bool = False):
        self.nosplit = nosplit          # diagnostics: never split or eject (the net's move only)
        self.guard = int(guard)         # 1: steer around viruses we'd pop on; 2: also keep out of threats' reach;
                                        # 3: also the champion's split-kill
        self.model = load(ckpt, "cpu").eval()
        if self.model.config.get("kind") == "entity":   # sim-trained transformer over object tokens
            from .ppo_sim import from_checkpoint          # pads older nets' inputs to the current layout
            self.model = from_checkpoint(ckpt, "cpu").eval()
            self.model.binary_eps = 0.0
        # Guided nets play greedily outside training: sampled exploration (random
        # overrides) costs far more than it's worth in real play; greedy = follow the
        # champion except where the net has learned an override is better.
        self.greedy = greedy if greedy is not None else self.model.follow is not None
        self.counters = {"ejects": 0, "splits": 0, "follow": 0, "steps": 0}
        self.encoder = None
        if self.model.config.get("kind") == "entity":
            from .entity_obs import RealEntityEncoder
            self.encoder = RealEntityEncoder()
        elif self.model.follow is not None:
            from .guided import GuidedEncoder
            self.encoder = GuidedEncoder(self.model.guide)
        else:
            from .guided import C as GUIDED_C
            if self.model.config.get("c", 0) >= GUIDED_C:  # sim-trained: guided layout (+ sim channels), no champion
                from .nohint import NoHintEncoder
                self.encoder = NoHintEncoder(self.model.config["c"])

    def reset(self) -> None:
        if self.encoder:
            self.encoder.reset()
        self.held = None

    held = None   # frame-skip nets (sim-trained with skip k) decide every k steps and hold the target

    def act(self, s: Snapshot):
        x, y, sp, ej = self._act(s)
        if self.guard >= 2:
            x, y = threat_guard(s, x, y)
        if self.guard:
            x, y = virus_guard(s, x, y)
        if self.guard >= 3 and not sp:
            target = split_kill(s)
            if target is not None:
                x, y, sp = target.x, target.y, True
                self.counters["splits"] += 1
        return x, y, sp, ej

    def _act(self, s: Snapshot):
        g, sc, cx, cy, ext = self.encoder.encode(s) if self.encoder else encode(s)
        skip = getattr(self.model, "skip", 1)
        if self.held is not None and self.counters["steps"] % skip:
            self.counters["steps"] += 1
            return self.held
        a, _, _ = self.model.act(torch.from_numpy(g)[None], torch.from_numpy(sc)[None], self.greedy)
        move, split, eject = (int(v) for v in a[0])
        if self.nosplit:
            split = eject = 0
        self.counters["steps"] += 1
        if self.encoder:
            self.counters["follow"] += self.model.follow is not None and move == self.model.follow
            x, y, sp, ej = self.encoder.decode(move, split, eject, cx, cy, ext)
        else:
            x, y, sp, ej = A.decode(move, split, eject, cx, cy, ext)
        self.counters["ejects"] += ej
        self.counters["splits"] += sp
        self.held = (x, y, False, False)        # split / eject fire once per decision
        return x, y, sp, ej


VIRUS_AVOID = 227.5       # the champion's virus_avoid gene (runs/final_champion.json)


def virus_guard(s: Snapshot, x: float, y: float) -> tuple[float, float]:
    """bot/steer.py's virus detour around a net's mouse target: if the nearest virus our largest cell
    would pop on lies across its path to (x, y), aim beside it on the side we lean to instead."""
    if not s.mine:
        return x, y
    me = max(s.mine, key=lambda c: c.size)
    viruses = [c for c in s.others if c.kind == "virus" and me.size >= VIRUS_EAT_MULT * c.size]
    dx, dy = x - me.x, y - me.y
    n = math.hypot(dx, dy)
    if not viruses or n < 1e-6:
        return x, y
    ux, uy = dx / n, dy / n
    blocking = []
    for v in viruses:
        vx, vy = v.x - me.x, v.y - me.y
        t = vx * ux + vy * uy                       # distance ahead along our path
        clear = v.size + me.size + VIRUS_AVOID * 0.2
        h = vx * uy - vy * ux                       # signed sideways offset of the virus
        if 0 < t < min(clear + VIRUS_AVOID + 200, n + clear) and abs(h) < clear:
            blocking.append((t, v, h, clear))
    if not blocking:
        return x, y
    _, v, h, clear = min(blocking, key=lambda b: b[0])
    side = 1.0 if h > 0 else -1.0                   # pass on the side that needs the smaller turn
    tx, ty = v.x - uy * side * clear * 1.15, v.y + ux * side * clear * 1.15
    ddx, ddy = tx - me.x, ty - me.y
    dn = math.hypot(ddx, ddy) or 1.0
    return me.x + ddx / dn * n, me.y + ddy / dn * n


DANGER_MARGIN, SPLIT_RANGE, FLEE_WEIGHT = 177.0, 842.0, 3.0   # the champion's genes (danger_margin, split_range)


def threat_guard(s: Snapshot, x: float, y: float) -> tuple[float, float]:
    """The champion's threat push around a net's mouse target: every cell that could eat our largest
    cell and is inside its danger radius (its reach, plus the split range when it is big enough to
    split onto us, plus a margin) pushes us away by (r - d) / r; the push is added to the net's
    heading with weight FLEE_WEIGHT."""
    from bot.steer import EAT_RATIO, SPLIT_KILL_RATIO
    if not s.mine:
        return x, y
    me = max(s.mine, key=lambda c: c.size)
    mine = {c.id for c in s.mine}
    fx = fy = 0.0
    for c in s.others:
        if c.kind != "player" or c.id in mine or c.mass <= me.mass * EAT_RATIO:
            continue
        dx, dy = me.x - c.x, me.y - c.y
        dist = math.hypot(dx, dy) or 1.0
        d = dist - c.size - me.size
        r = c.size + me.size + (SPLIT_RANGE if c.mass > me.mass * SPLIT_KILL_RATIO else 0.0) + DANGER_MARGIN
        if d < r:
            w = (r - d) / r
            fx += dx / dist * w
            fy += dy / dist * w
    if fx == 0.0 and fy == 0.0:
        return x, y
    dx, dy = x - me.x, y - me.y
    n = math.hypot(dx, dy)
    ux, uy = (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)
    ux, uy = ux + FLEE_WEIGHT * fx, uy + FLEE_WEIGHT * fy
    un = math.hypot(ux, uy) or 1.0
    reach = max(n, 600.0)
    return me.x + ux / un * reach, me.y + uy / un * reach


SPLIT_REACH = 0.474   # the champion's split_reach gene: split when prey is within split_range * this


def split_kill(s: Snapshot):
    """The champion's split-kill (bot/steer.py): with one cell, the biggest prey that half our mass can
    eat and that is within reach, unless a threat could reach us after the split. Returns it or None."""
    from bot.steer import EAT_RATIO, SPLIT_KILL_RATIO
    if len(s.mine) != 1:
        return None
    me = s.mine[0]
    best = None
    for c in s.others:
        if c.kind != "player":
            continue
        d = math.hypot(c.x - me.x, c.y - me.y)
        if c.mass > me.mass * EAT_RATIO:            # a threat near the landing zone: no split at all
            r = c.size + me.size + (SPLIT_RANGE if c.mass > me.mass * SPLIT_KILL_RATIO else 0.0) + DANGER_MARGIN
            if d < r + SPLIT_RANGE:
                return None
        elif (me.mass / 2 > c.mass * EAT_RATIO and d - me.size < SPLIT_RANGE * SPLIT_REACH
              and (best is None or c.mass > best.mass)):
            best = c
    return best
