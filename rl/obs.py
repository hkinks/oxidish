"""Observation encoder: Snapshot -> (grid [C,G,G], scalars [S]) as float32 numpy.

The grid is centred on our mass-weighted centroid and scaled with our size, like
the game camera. Other players are encoded relative to our largest cell, with
separate channels for prey, threats and split-kill threats so the network
doesn't have to learn the 1.3x / 2.6x rules from raw sizes.
"""

import math

import numpy as np

from bot.game import Snapshot

G = 32                 # grid cells per side
EAT_RATIO = 1.3
SPLIT_KILL_RATIO = 2.6
CHANNELS = ("food", "virus", "mine", "prey", "threat", "split_threat", "ejected", "outside")
C = len(CHANNELS)
S = 8                  # scalar features


def view_half_extent(total_mass: float) -> float:
    """Original plain-net view (narrower than the server's; kept for old checkpoints)."""
    size = math.sqrt(100 * max(total_mass, 1.0))
    return min(max(2.5 * size + 500, 600.0), 3000.0)


def server_half_extent(s: Snapshot) -> float:
    """Half-width the server actually shows us (MultiOgar: 1920 / scale / 2,
    scale = min(64 / sum of cell sizes, 1) ** 0.4), so nothing visible is cut off."""
    total_size = sum(c.size for c in s.mine) or 1.0
    return 960 / min(64 / total_size, 1.0) ** 0.4


def centre(s: Snapshot) -> tuple[float, float]:
    m = sum(c.mass for c in s.mine) or 1.0
    return (sum(c.x * c.mass for c in s.mine) / m, sum(c.y * c.mass for c in s.mine) / m)


def _disc(grid, ch, gx, gy, gr, value, overwrite: bool = False):
    """Paint a disc (grid units) into grid[ch]: max-blend, or overwrite (signed values)."""
    n = grid.shape[-1]
    x0, x1 = max(int(gx - gr), 0), min(int(gx + gr) + 1, n)
    y0, y1 = max(int(gy - gr), 0), min(int(gy + gr) + 1, n)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs + 0.5 - gx) ** 2 + (ys + 0.5 - gy) ** 2 <= max(gr, 0.5) ** 2
    region = grid[ch, y0:y1, x0:x1]
    if overwrite:
        region[inside] = value
    else:
        np.maximum(region, np.where(inside, value, 0.0), out=region)


def encode(s: Snapshot, G: int = G, server_view: bool = False
           ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Returns grid, scalars, centre x, centre y, half extent (for decoding actions).

    Defaults reproduce the original plain-net encoding; the guided net uses
    G=48 and server_view=True."""
    grid = np.zeros((C, G, G), dtype=np.float32)
    scal = np.zeros(S, dtype=np.float32)
    if not s.mine:
        return grid, scal, 0.0, 0.0, 600.0
    total = s.total_mass
    me = s.main
    cx, cy = centre(s)
    ext = server_half_extent(s) if server_view else view_half_extent(total)
    k = G / (2 * ext)  # world -> grid scale
    to_g = lambda x, y: ((x - cx) * k + G / 2, (y - cy) * k + G / 2)

    food = [c for c in s.others if c.kind == "food"]
    if food:
        fx = np.array([c.x for c in food]); fy = np.array([c.y for c in food])
        ix = ((fx - cx) * k + G / 2).astype(int); iy = ((fy - cy) * k + G / 2).astype(int)
        ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
        np.add.at(grid[0], (iy[ok], ix[ok]), 0.25)
        np.minimum(grid[0], 1.0, out=grid[0])

    for c in s.mine:
        gx, gy = to_g(c.x, c.y)
        _disc(grid, 2, gx, gy, c.size * k, 1.0)
    for c in s.others:
        if c.kind == "food":
            continue
        gx, gy = to_g(c.x, c.y)
        gr = c.size * k
        if c.kind == "virus":
            _disc(grid, 1, gx, gy, gr, 1.0)
        elif c.kind == "ejected":
            _disc(grid, 6, gx, gy, gr, 1.0)
        else:
            ratio = c.mass / max(me.mass, 1.0)
            if ratio * EAT_RATIO < 1:
                _disc(grid, 3, gx, gy, gr, min(1.0, 0.3 + ratio))
            elif ratio > EAT_RATIO:
                _disc(grid, 4, gx, gy, gr, min(1.0, math.log(ratio) / 2))
                if ratio > SPLIT_KILL_RATIO:  # reach of their split lunge
                    _disc(grid, 5, gx, gy, gr + 650 * k, 1.0)

    b = s.border
    xs = cx + (np.arange(G) + 0.5 - G / 2) / k
    ys = cy + (np.arange(G) + 0.5 - G / 2) / k
    out_x = (xs < b["left"]) | (xs > b["right"])
    out_y = (ys < b["top"]) | (ys > b["bottom"])
    grid[7] = (out_y[:, None] | out_x[None, :]).astype(np.float32)

    w = (b["right"] - b["left"]) or 1.0
    h = (b["bottom"] - b["top"]) or 1.0
    scal[:] = (math.log(total) / 8, len(s.mine) / 16, me.mass / total,
               (cx - b["left"]) / w * 2 - 1, (cy - b["top"]) / h * 2 - 1,
               ext / 3000, float(me.mass >= 36), math.log(me.mass) / 8)
    return grid, scal, cx, cy, ext
