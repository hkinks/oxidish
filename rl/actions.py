"""Discrete action space: heading (16 directions + stay), split, eject (W)."""

import math

DIRS = 16
N_MOVE = DIRS + 1        # last index = stay (mouse on our centre)
HEADS = (N_MOVE, 2, 2)   # move, split, eject


def decode(move: int, split: int, eject: int, cx: float, cy: float, ext: float):
    """Action indices -> (x, y, split, eject) in world coordinates."""
    if move == DIRS:
        return cx, cy, bool(split), bool(eject)
    a = 2 * math.pi * move / DIRS
    return cx + math.cos(a) * ext, cy + math.sin(a) * ext, bool(split), bool(eject)


def encode(x: float, y: float, split: bool, eject: bool, cx: float, cy: float):
    """World aim (e.g. a teacher policy's) -> action indices."""
    dx, dy = x - cx, y - cy
    if math.hypot(dx, dy) < 30:
        move = DIRS
    else:
        move = round(math.atan2(dy, dx) / (2 * math.pi) * DIRS) % DIRS
    return move, int(bool(split)), int(bool(eject))
