"""Entity observation from a real-server Snapshot: the same tokens as sim/entities.py, so entity nets
trained in the simulator (rl.entity_model) can play on MultiOgar (rl.policy.NNPolicy picks this
encoder for checkpoints with config kind "entity").

The server only sends what is in view, so visibility is already applied. Token order within a kind
is nearest first, as in the sim; own cells in id order (the transformer has no positions, only the
pointer indices depend on the order, and they are decoded against this same list). Velocities and
merge timers come from the encoder's memory of earlier snapshots (it is called every step, the
policy decides every `skip` steps and holds its move in between); the last decision (pointer
target object, direction, stay) is remembered for the commitment inputs.
"""

import math

import numpy as np

from bot.game import Snapshot
from sim.engine import MERGE_SIZE_COEF
from sim.entities import FD, K_CELL, K_EJECT, K_FOOD, K_VIRUS, REGIONS, T_CELL, T_EJECT, T_FOOD, T_OWN, T_VIRUS
from sim.obs import EAT_RATIO, MIN_SCALE, S, SPLIT_KILL_RATIO, VEL_SCALE, VIEW_X, VIEW_Y, VIRUS_MAX, VIRUS_MIN

SYNC_TICKS = 2
DIRS = 16
REGION_KEY = -1_000_000          # pointer-target keys of the cluster regions (cell ids are >= 0)


class RealEntityEncoder:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.prev: dict[int, tuple[float, float]] = {}
        self.first_seen: dict[int, int] = {}
        self.target_key: int | None = None
        self.last_dir = (0.0, 0.0)
        self.last_stay = 0.0
        self.tok_key: list[int] = []
        self.tok_pos: list[tuple[float, float]] = []

    def encode(self, s: Snapshot):
        """-> (ent [NT, FD] float32, scal [S] float32, cx, cy, ext)."""
        self.step += 1
        nt = 16 + K_CELL + K_FOOD + K_VIRUS + K_EJECT + REGIONS * REGIONS
        ent = np.zeros((nt, FD), np.float32)
        scal = np.zeros(S, np.float32)
        keys = [-2] * nt
        pos = [(0.0, 0.0)] * nt
        mine = sorted(s.mine, key=lambda c: c.id)
        if not mine:
            self.tok_key, self.tok_pos = keys, pos
            return ent, scal, 0.0, 0.0, 1000.0
        fs = [math.floor(c.size) for c in mine]
        masses = [f * f / 100 for f in fs]
        total = max(sum(masses), 1.0)
        cx = sum(int(c.x) * m for c, m in zip(mine, masses)) / total
        cy = sum(int(c.y) * m for c, m in zip(mine, masses)) / total
        ext = 960.0 / min(64.0 / max(sum(fs), 1.0), 1.0) ** 0.4
        me = max(max(masses), 1.0)
        sscale = max(min(64.0 / max(sum(c.size for c in mine), 1.0), 1.0) ** 0.4, MIN_SCALE)
        hx, hy = VIEW_X / sscale, VIEW_Y / sscale
        ccx = sum(c.x for c in mine) / len(mine)
        ccy = sum(c.y for c in mine) / len(mine)
        for c in mine:
            self.first_seen.setdefault(c.id, self.step)

        def row(i, kind, c, mass, extra=()):
            dx, dy = int(c.x) - cx, int(c.y) - cy
            f = ent[i]
            f[0] = 1.0
            f[1 + kind] = 1.0
            f[6], f[7], f[8] = dx / 1000, dy / 1000, math.hypot(dx, dy) / 1000
            if kind != T_FOOD and c.id in self.prev:
                px, py = self.prev[c.id]
                f[9] = max(-2.0, min(2.0, (c.x - px) / VEL_SCALE))
                f[10] = max(-2.0, min(2.0, (c.y - py) / VEL_SCALE))
            ratio = mass / me
            f[11] = math.log1p(mass) / 8
            f[12] = max(-2.0, min(2.0, math.log(max(ratio, 1e-6)) / 3))
            f[13] = float(ratio * EAT_RATIO < 1)
            f[14] = float(ratio > EAT_RATIO)
            for j, v in extra:
                f[j] = v
            f[20] = float(self.target_key is not None and c.id == self.target_key)
            keys[i] = c.id
            pos[i] = (c.x, c.y)

        def nearest(cells, k, rank=None):
            rank = rank or (lambda c: (int(c.x) - cx) ** 2 + (int(c.y) - cy) ** 2)
            return sorted(cells, key=rank)[:k]

        i = 0
        for c, f, m in zip(mine[:16], fs, masses):
            age = (self.step - self.first_seen[c.id]) * SYNC_TICKS
            row(i, T_OWN, c, m, [(17, min(max(age / (max(MERGE_SIZE_COEF * c.size, 30.0) * 25.0), 0.0), 1.0))])
            i += 1
        i = 16
        players = [c for c in s.others if c.kind == "player"]
        owner_mass: dict[str, float] = {}
        for c in players:
            owner_mass[c.name] = owner_mass.get(c.name, 0.0) + math.floor(c.size) ** 2 / 100
        for c in nearest(players, K_CELL):
            m = math.floor(c.size) ** 2 / 100
            r = m / me
            row(i, T_CELL, c, m, [(15, float(r * SPLIT_KILL_RATIO < 1)), (16, float(r > SPLIT_KILL_RATIO)),
                                  (18, max(-2.0, min(2.0, math.log(max(owner_mass[c.name], 1.0) / total) / 3)))])
            i += 1
        i = 16 + K_CELL
        food = [c for c in s.others if c.kind == "food"]
        mine_xy = [(int(c.x), int(c.y)) for c in mine]
        cell_d2 = lambda c: min((int(c.x) - x) ** 2 + (int(c.y) - y) ** 2 for x, y in mine_xy)
        for c in nearest(food, K_FOOD, cell_d2):
            row(i, T_FOOD, c, 1.0, [(21, min(math.sqrt(cell_d2(c)) / 1000, 5.0))])
            i += 1
        i = 16 + K_CELL + K_FOOD
        for c in nearest([c for c in s.others if c.kind == "virus"], K_VIRUS):
            m = math.floor(c.size) ** 2 / 100
            row(i, T_VIRUS, c, m, [(19, min(max((m - VIRUS_MIN) / (VIRUS_MAX - VIRUS_MIN), 0.0), 1.0))])
            i += 1
        i = 16 + K_CELL + K_FOOD + K_VIRUS
        for c in nearest([c for c in s.others if c.kind == "ejected"], K_EJECT):
            row(i, T_EJECT, c, math.floor(c.size) ** 2 / 100)
            i += 1
        # food clusters: visible pellets in REGIONS x REGIONS regions of the view box
        i = 16 + K_CELL + K_FOOD + K_VIRUS + K_EJECT
        R = REGIONS
        cnt = np.zeros(R * R)
        sx = np.zeros(R * R)
        sy = np.zeros(R * R)
        for c in food:
            dx, dy = int(c.x) - cx, int(c.y) - cy
            bx = min(max(math.floor((int(c.x) - ccx + hx) / (2 * hx) * R), 0), R - 1)
            by = min(max(math.floor((int(c.y) - ccy + hy) / (2 * hy) * R), 0), R - 1)
            r = by * R + bx
            cnt[r] += 1
            sx[r] += dx
            sy[r] += dy
        for r in range(R * R):
            j = i + r
            keys[j] = REGION_KEY - r
            if cnt[r] == 0:
                continue
            dx, dy = sx[r] / cnt[r], sy[r] / cnt[r]
            f = ent[j]
            f[0], f[1 + T_FOOD] = 1.0, 1.0
            f[6], f[7], f[8] = dx / 1000, dy / 1000, math.hypot(dx, dy) / 1000
            ratio = cnt[r] / me
            f[11] = math.log1p(cnt[r]) / 8
            f[12] = max(-2.0, min(2.0, math.log(max(ratio, 1e-6)) / 3))
            f[13], f[14] = float(ratio * EAT_RATIO < 1), float(ratio > EAT_RATIO)
            f[20] = float(self.target_key == keys[j])
            pos[j] = (cx + dx, cy + dy)
        # memory for the next call: positions (velocities); forget long-gone ids
        self.prev = {c.id: (c.x, c.y) for c in [*mine, *s.others] if c.kind != "food"}
        if len(self.first_seen) > 200:
            ids = {c.id for c in mine}
            self.first_seen = {k: v for k, v in self.first_seen.items() if k in ids}
        half_w = (s.border["right"] - s.border["left"]) / 2 if s.border else 3500.0
        mid_x = (s.border["right"] + s.border["left"]) / 2 if s.border else 0.0
        mid_y = (s.border["bottom"] + s.border["top"]) / 2 if s.border else 0.0
        scal[0] = math.log(total) / 8
        scal[1] = len(mine) / 16
        scal[2] = me / total
        scal[3] = (cx - mid_x) / half_w
        scal[4] = (cy - mid_y) / half_w
        scal[5] = ext / 3000
        scal[6] = float(me >= 36)
        scal[7] = math.log(me) / 8
        scal[8], scal[9] = self.last_dir
        scal[10] = self.last_stay
        self.tok_key, self.tok_pos = keys, pos
        return ent, scal, cx, cy, ext

    def decode(self, move: int, split: int, eject: int, cx: float, cy: float, ext: float):
        """Action -> (x, y, split, eject) in world coordinates; remembers the decision."""
        if move == DIRS:
            ux = uy = 0.0
        elif move < DIRS:
            a = 2 * math.pi * move / DIRS
            ux, uy = math.cos(a), math.sin(a)
        else:
            tx, ty = self.tok_pos[move - DIRS - 1]
            n = max(math.hypot(tx - cx, ty - cy), 1e-3)
            ux, uy = (tx - cx) / n, (ty - cy) / n
        self.target_key = self.tok_key[move - DIRS - 1] if move > DIRS else None
        self.last_dir = (ux, uy)
        self.last_stay = float(move == DIRS)
        return cx + ux * ext, cy + uy * ext, bool(split), bool(eject)
