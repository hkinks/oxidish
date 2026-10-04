"""Measure how often bots get stuck, and whether a virus is involved.

    uv run python -m arena.stuck --agents champ=runs/final_champion.json:4 --matches 4

Stuck = alive and the main cell moved less than STUCK_DIST over the last WINDOW steps
(2 s of game time; a free cell covers 400+ units in that time). A stuck step is
"at a virus" when a virus we would pop on (we are 1.3x heavier) is within
VIRUS_NEAR of our main cell's edge.
"""

import argparse
import asyncio
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from .client import AgentClient
from .match import free_port, parse_lineup, start_server
from .policies import build

WINDOW = 25
STUCK_DIST = 150.0
VIRUS_NEAR = 250.0


class StuckMeter:
    """Wraps a policy; counts alive / stuck / stuck-at-virus steps."""

    def __init__(self, policy):
        self.policy = policy
        self.pos = deque(maxlen=WINDOW + 1)
        self.n = {"alive": 0, "stuck": 0, "stuck_virus": 0, "near_virus": 0}

    def reset(self):
        self.pos.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, s):
        me = s.main
        if me is not None and not s.dead:
            self.n["alive"] += 1
            self.pos.append((me.x, me.y))
            near = any(c.kind == "virus" and me.mass > c.mass * 1.3
                       and me.dist(c) - c.size - me.size < VIRUS_NEAR for c in s.others)
            self.n["near_virus"] += near
            if len(self.pos) > WINDOW:
                (x0, y0), (x1, y1) = self.pos[0], self.pos[-1]
                if math.hypot(x1 - x0, y1 - y0) < STUCK_DIST:
                    self.n["stuck"] += 1
                    self.n["stuck_virus"] += near
        else:
            self.pos.clear()
        return self.policy.act(s)


def one(lineup, seed, ticks, food_mult):
    port = free_port()
    srv = start_server(port, 8, seed, len(lineup), 2, 7000, food_mult=food_mult)
    try:
        clients = [AgentClient(f"ws://127.0.0.1:{port}", n, StuckMeter(build(sp)), ticks, 2)
                   for n, sp in lineup]

        async def go():
            await asyncio.gather(*(c.run() for c in clients))
        asyncio.run(go())
        return [(c.name.rsplit("-", 1)[0], c.policy.n, c.stats.avg_mass) for c in clients]
    finally:
        srv.terminate()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--ticks", type=int, default=9000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--seed-base", type=int, default=555000)
    args = p.parse_args()
    lineup = parse_lineup(args.agents)
    tot: dict[str, dict] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, lineup, args.seed_base + i, args.ticks, args.food_mult)
                for i in range(args.matches)]
        for f in futs:
            for key, n, mass in f.result():
                t = tot.setdefault(key, {"alive": 0, "stuck": 0, "stuck_virus": 0,
                                         "near_virus": 0, "mass": []})
                for k in n:
                    t[k] += n[k]
                t["mass"].append(mass)
    print(f"{'policy':<10} {'stuck %':>8} {'stuck@virus %':>14} {'near virus %':>13} {'avg mass':>9}")
    for key, t in tot.items():
        a = max(t["alive"], 1)
        print(f"{key:<10} {100 * t['stuck'] / a:8.1f} {100 * t['stuck_virus'] / a:14.1f} "
              f"{100 * t['near_virus'] / a:13.1f} {sum(t['mass']) / len(t['mass']):9.0f}")


if __name__ == "__main__":
    main()
