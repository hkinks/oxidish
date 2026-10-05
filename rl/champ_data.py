"""Imitation data from the rule-based champions playing on the real server, in the entity nets' format.

    uv run python -m rl.champ_data [--matches 12] [--out runs/rl/champ_data.pt]

Each match (MultiOgar, 2x food, 8 server bots) seats champ x2 and r2g9 x2; every `--every`-th step of
each of them becomes one sample, like rl/human_data.py: the entity tokens and scalars an entity net
would see (rl/entity_obs.RealEntityEncoder, fed every frame so velocities and timers build up as in
play) and the champion's action as the label (heading of 16 from the view centre to its mouse target,
stay within 30 units; split; eject). The champions farm along planned pellet routes (bot/farm.py),
which the sim nets do not match without virus popping (real server, alone 2 min: 240 vs 187).
"""

import argparse
import asyncio
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from arena.client import AgentClient
from arena.match import free_port, parse_lineup, start_server
from arena.policies import build
from rl.common import ROOT

from .entity_obs import RealEntityEncoder

DIRS = 16


class Recorder:
    """Wraps a policy: plays it unchanged and records (tokens, scalars, action) samples."""

    def __init__(self, policy, every: int):
        self.policy, self.every = policy, every
        self.counters = getattr(policy, "counters", {})
        self.enc = RealEntityEncoder()
        self.n = 0
        self.ents, self.scals, self.acts = [], [], []

    def reset(self) -> None:
        self.enc.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, s):
        ent, scal, cx, cy, ext = self.enc.encode(s)
        x, y, split, *rest = self.policy.act(s)
        eject = bool(rest and rest[0])
        dx, dy = x - cx, y - cy
        move = DIRS if math.hypot(dx, dy) < 30 else round(math.atan2(dy, dx) / (2 * math.pi) * DIRS) % DIRS
        self.enc.decode(move, int(split), int(eject), cx, cy, ext)
        self.n += 1
        if self.n % self.every == 0:
            self.ents.append(ent.astype(np.float16))
            self.scals.append(scal)
            self.acts.append((move, int(split), int(eject)))
        return x, y, split, eject


def _match(i: int, ticks: int, seed_base: int, every: int):
    port, seed = free_port(), seed_base + i
    lineup = parse_lineup(["champ=runs/final_champion.json:2", "r2g9=runs/r2/champion_009.json:2"])
    server = start_server(port, 8, seed, len(lineup), 2, 7000, food_mult=2.0)
    recs = [Recorder(build(spec), every) for _, spec in lineup]
    try:
        url = f"ws://127.0.0.1:{port}"

        async def play():
            return await asyncio.gather(*[AgentClient(url, name, r, ticks, 2).run()
                                          for (name, _), r in zip(lineup, recs)])
        asyncio.run(play())
    finally:
        server.terminate()
    return ([e for r in recs for e in r.ents], [s for r in recs for s in r.scals],
            [a for r in recs for a in r.acts])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=12)
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--every", type=int, default=2, help="record every n-th step (the nets decide every 2)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=80000)
    p.add_argument("--out", default=str(ROOT / "runs" / "rl" / "champ_data.pt"))
    args = p.parse_args()
    ents, scals, acts = [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_match, i, args.ticks, args.seed_base, args.every) for i in range(args.matches)]
        for f in as_completed(futs):
            e, s, a = f.result()
            ents += e
            scals += s
            acts += a
    act = torch.tensor(acts, dtype=torch.long)
    torch.save({"ent": torch.from_numpy(np.stack(ents)), "scal": torch.from_numpy(np.stack(scals)), "act": act},
               args.out)
    print(f"{len(act)} samples -> {args.out}: splits {int(act[:, 1].sum())}, W {int(act[:, 2].sum())}, "
          f"stay {(act[:, 0] == DIRS).float().mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
