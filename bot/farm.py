"""Food routing: plan a short pellet route instead of averaging pulls from all pellets.

Summing attraction over every visible pellet makes the cell drift toward the
centroid between clusters and dither when two clusters compete. Instead:

1. Drop pellets inside a threat's danger zone.
2. Beam-search routes of up to ROUTE_DEPTH stops over the nearest CANDIDATES,
   scoring pellets eaten per unit of path length. Pellets that pass within our
   radius of a leg are eaten on the way and count too (once).
3. Commit to the chosen first pellet until it's gone, unless a new route is
   clearly better (hysteresis), so the cell doesn't flip-flop between targets.
4. With no safe food in view, explore toward the least recently visited part
   of the map.
"""

import math
import time

import numpy as np

from .game import Cell, Snapshot

CANDIDATES = 14       # nearest pellets considered as route stops
ROUTE_DEPTH = 4       # pellets per planned route
BEAM = 24             # partial routes kept per depth
SWITCH_MARGIN = 1.25  # default: new route must score this much better to drop the committed target
GRID = 8              # exploration grid cells per map side


def _sweep_masks(sx, sy, px, py, reach):
    """masks[i][j]: bitmask of pellets k within `reach` of segment start_i -> pellet_j.

    sx/sy: segment starts (our cell, then each pellet); px/py: pellets. Bit j is
    always set: the endpoint itself is eaten.
    """
    ax, ay = sx[:, None, None], sy[:, None, None]          # (S,1,1)
    bx, by = px[None, :, None], py[None, :, None]          # (1,N,1)
    qx, qy = px[None, None, :], py[None, None, :]          # (1,1,N)
    dx, dy = bx - ax, by - ay
    seg2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg2, 0.0, 1.0)
    hit = np.hypot(ax + t * dx - qx, ay + t * dy - qy) < reach   # (S,N,N)
    n = len(px)
    hit[:, np.arange(n), np.arange(n)] = True
    weights = 1 << np.arange(n, dtype=np.int64)
    return (hit * weights).sum(axis=2).tolist()


class FoodRouter:
    def __init__(self):
        self._target_id: int | None = None
        self._visits: dict[tuple[int, int], float] = {}
        self._visit_grid = None  # numpy mirror of _visits for _explore
        self.route: list[Cell] = []

    def reset(self) -> None:
        self._target_id = None
        self.route = []

    def _plan(self, me: Cell, food: list[Cell], first: Cell | None = None
              ) -> tuple[list[Cell], float]:
        """Best route by pellets eaten per unit of path; optionally forced to start at `first`."""
        near = sorted(food, key=me.dist)[:CANDIDATES]
        if first is not None and first not in near:
            near.append(first)
        n = len(near)
        if not n:
            return [], 0.0
        px = np.array([c.x for c in near], dtype=float)
        py = np.array([c.y for c in near], dtype=float)
        sx, sy = np.concatenate(([me.x], px)), np.concatenate(([me.y], py))
        masks = _sweep_masks(sx, sy, px, py, me.size * 0.9)       # start index 0 = us
        dist = np.hypot(sx[:, None] - px[None, :], sy[:, None] - py[None, :]).tolist()
        first_j = near.index(first) if first is not None else None
        beam = [(0.0, 0.0, 0, 0, [])]  # (score, path length, eaten mask, start idx, stops)
        best = (0.0, [])
        for depth in range(ROUTE_DEPTH):
            grown = []
            for _, length, eaten, at, route in beam:
                options = [first_j] if depth == 0 and first_j is not None else range(n)
                row_m, row_d = masks[at], dist[at]
                for j in options:
                    if eaten >> j & 1:
                        continue
                    now = eaten | row_m[j]
                    new_len = length + row_d[j]
                    grown.append((now.bit_count() / (new_len + me.size), new_len, now,
                                  j + 1, route + [j]))
            if not grown:
                break
            grown.sort(key=lambda r: -r[0])
            beam = grown[:BEAM]
            if beam[0][0] > best[0]:
                best = (beam[0][0], beam[0][4])
        return [near[j] for j in best[1]], best[0]

    def _explore(self, s: Snapshot, me: Cell) -> tuple[float, float]:
        b = s.border
        w, h = (b["right"] - b["left"]) / GRID, (b["bottom"] - b["top"]) / GRID
        now = time.monotonic()
        here = (int((me.x - b["left"]) // w), int((me.y - b["top"]) // h))
        self._visits[here] = now
        # Vectorised over the GRID x GRID cells; same scores (and first-max tie-break)
        # as scoring each cell in row-major order.
        if self._visit_grid is None:
            self._visit_grid = np.full((GRID, GRID), -1e9)
        if 0 <= here[0] < GRID and 0 <= here[1] < GRID:
            self._visit_grid[here] = now
        centres = (np.arange(GRID) + 0.5)
        dist = np.hypot((b["left"] + centres * w - me.x)[:, None],
                        (b["top"] + centres * h - me.y)[None, :])
        score = np.minimum(now - self._visit_grid, 300) - dist / 50
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        return b["left"] + (i + 0.5) * w, b["top"] + (j + 0.5) * h

    def next_waypoint(self, s: Snapshot, food: list[Cell], unsafe,
                      switch_margin: float = SWITCH_MARGIN) -> tuple[float, float]:
        """World point to head for. `unsafe(cell)` is True for pellets in danger zones."""
        me = s.main
        safe = [c for c in food if not unsafe(c)]
        explore_point = self._explore(s, me)  # also records the visit
        if not safe:
            self.route = []
            self._target_id = None
            return explore_point

        route, score = self._plan(me, safe)
        committed = next((c for c in safe if c.id == self._target_id), None)
        if committed is not None and route and route[0].id != committed.id:
            kept_route, kept = self._plan(me, safe, first=committed)
            if score < kept * switch_margin:
                route = kept_route
        self.route = route
        self._target_id = route[0].id if route else None
        if not route:
            return explore_point
        return route[0].x, route[0].y
