"""Observation for nets trained in the GPU simulator (rl/ppo_sim.py) when they play on the
real server: the guided net's grid and scalars, but no champion runs, so the 8 hint scalars
stay zero (as in the sim) and there is no FOLLOW action (17 moves: 16 headings + stay)."""

from .guided import GuidedEncoder


class NoHintEncoder(GuidedEncoder):
    def __init__(self):
        self.guide = None
        self.reset()

    def decode(self, move: int, split: int, eject: int, cx: float, cy: float, ext: float):
        from .actions import decode
        return decode(move, split, eject, cx, cy, ext)
