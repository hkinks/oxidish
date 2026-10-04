"""Neural policy as an arena policy (spec {"kind": "nn", "ckpt": path, "greedy": bool}).

Guided checkpoints (with a FOLLOW action) run their champion alongside the net.
"""

import torch

from bot.game import Snapshot

from . import actions as A
from .common import load
from .obs import encode

torch.set_num_threads(1)  # many agents per process; keep each one light


class NNPolicy:
    def __init__(self, ckpt: str, greedy: bool | None = None):
        self.model = load(ckpt, "cpu").eval()
        # Guided nets play greedily outside training: sampled exploration (random
        # overrides) costs far more than it's worth in real play; greedy = follow the
        # champion except where the net has learned an override is better.
        self.greedy = greedy if greedy is not None else self.model.follow is not None
        self.counters = {"ejects": 0, "splits": 0, "follow": 0, "steps": 0}
        self.encoder = None
        if self.model.follow is not None:
            from .guided import GuidedEncoder
            self.encoder = GuidedEncoder(self.model.guide)
        else:
            from .guided import C as GUIDED_C
            if self.model.config.get("c") == GUIDED_C:  # sim-trained: guided layout, no champion
                from .nohint import NoHintEncoder
                self.encoder = NoHintEncoder()

    def reset(self) -> None:
        if self.encoder:
            self.encoder.reset()

    def act(self, s: Snapshot):
        g, sc, cx, cy, ext = self.encoder.encode(s) if self.encoder else encode(s)
        a, _, _ = self.model.act(torch.from_numpy(g)[None], torch.from_numpy(sc)[None], self.greedy)
        move, split, eject = (int(v) for v in a[0])
        self.counters["steps"] += 1
        if self.encoder:
            self.counters["follow"] += self.model.follow is not None and move == self.model.follow
            x, y, sp, ej = self.encoder.decode(move, split, eject, cx, cy, ext)
        else:
            x, y, sp, ej = A.decode(move, split, eject, cx, cy, ext)
        self.counters["ejects"] += ej
        self.counters["splits"] += sp
        return x, y, sp, ej
