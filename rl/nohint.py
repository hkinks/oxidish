"""Observation for nets trained in the GPU simulator (rl/ppo_sim.py) when they play on the
real server: the guided net's grid and scalars, but no champion runs, so the 8 hint scalars
stay zero (as in the sim) and there is no FOLLOW action (17 moves: 16 headings + stay).

Sim nets with 15 input channels also get the two sim-only channels (sim/obs.py):
`killable` (enemy cells a split half of our largest cell could eat) and `density`
(log(1 + enemy mass) summed per pixel / 9)."""

import math

import numpy as np

from bot.game import Snapshot

from . import obs as O
from .guided import G, GuidedEncoder

SPLIT_KILL_RATIO = 2.6


class NoHintEncoder(GuidedEncoder):
    def __init__(self, channels: int = 13):
        self.guide = None
        self.channels = channels
        self.reset()

    def encode(self, s: Snapshot):
        grid, scal, cx, cy, ext = super().encode(s)
        if self.channels <= grid.shape[0]:
            return grid, scal, cx, cy, ext
        extra = np.zeros((self.channels - grid.shape[0], G, G), np.float32)
        if s.mine:
            k = G / (2 * ext)
            me = max(c.mass for c in s.mine)
            den = np.zeros((G, G), np.float32)
            for c in s.others:
                if c.kind != "player":
                    continue
                gx, gy = (c.x - cx) * k + G / 2, (c.y - cy) * k + G / 2
                if c.mass * SPLIT_KILL_RATIO < me:
                    O._disc(extra, 0, gx, gy, max(c.size * k, 0.5), 1.0)
                ix, iy = math.floor(gx), math.floor(gy)
                if 0 <= ix < G and 0 <= iy < G:
                    den[iy, ix] += c.mass
            extra[1] = np.minimum(np.log1p(den) / 9, 1.0)
        return np.concatenate([grid, extra]), scal, cx, cy, ext

    def decode(self, move: int, split: int, eject: int, cx: float, cy: float, ext: float):
        from .actions import decode
        return decode(move, split, eject, cx, cy, ext)
