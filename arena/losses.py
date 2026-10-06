"""Mine big losses: what was around a rule bot in the seconds before it lost mass or died.

    uv run python -m arena.losses --agents safe=bot/champ_safe.json:2 champ=runs/final_champion.json:2 \
        --matches 4 --out runs/losses_safe.json

A loss event = the bot's total mass fell by >= DROP of itself in one step (eaten, popped) or it died.
Each event keeps the last HIST steps: our cells, plan mode, and the nearest players and viruses
(mass, gap, cell count of the owner). `--report` sorts events into rough causes:
  popped     - our cell count jumped by 3+ just before (virus pop), then the loss
  split_kill - a split player (2+ cells, together heavier than our main cell) touches us at the loss
  unseen     - no cell able to eat our smallest piece was in view 1 s earlier
  fled       - we were fleeing in the last 1 s and still got eaten
  no_flee    - an eater was in view but we were not fleeing
"""

import argparse
import asyncio
import json
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from bot.modes import heuristic_plan
from bot.steer import EAT_RATIO

from .client import AgentClient
from .match import free_port, limit_cpus, parse_lineup, start_server
from .policies import build

DROP = 0.3
HIST = 30      # steps kept (sync 2: 30 steps = 2.4 s)
LOOKBACK = 12  # "1 s earlier" in steps


def _scene(s, p):
    me = s.main
    plan = heuristic_plan(s, p).mode if p is not None else "-"
    owners: dict[str, list] = {}
    for c in s.others:
        if c.kind == "player":
            owners.setdefault(c.name, []).append(c)
    near = []
    for c in s.others:
        if c.kind not in ("player", "virus"):
            continue
        gap = me.dist(c) - c.size - me.size
        if gap > 900:
            continue
        own = owners.get(c.name, [c]) if c.kind == "player" else [c]
        near.append({"k": c.kind, "name": c.name[:14], "m": round(c.mass), "gap": round(gap),
                     "dx": round(c.x - me.x), "dy": round(c.y - me.y),
                     "own_n": len(own), "own_m": round(sum(o.mass for o in own))})
    near.sort(key=lambda d: d["gap"])
    return {"plan": plan, "me": [round(me.x), round(me.y), round(me.mass)], "n": len(s.mine),
            "cells": sorted(round(c.mass) for c in s.mine), "total": round(s.total_mass), "near": near[:6]}


class LossMeter:
    def __init__(self, name, policy):
        self.name, self.policy = name, policy
        self.hist = deque(maxlen=HIST)
        self.events = []
        self.prev_total = None
        self.t = 0

    def reset(self):
        if self.prev_total and self.hist:
            self._event("death", 0.0)
        self.hist.clear()
        self.prev_total = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _event(self, kind, after):
        self.events.append({"bot": self.name, "t": self.t, "kind": kind, "before": round(self.prev_total),
                            "after": round(after), "hist": list(self.hist)})

    def act(self, s):
        self.t += 1
        out = self.policy.act(s)
        if s.main is None or s.dead:
            return out
        total = s.total_mass
        if self.prev_total and total < self.prev_total * (1 - DROP):
            self._event("drop", total)
        self.hist.append(_scene(s, getattr(self.policy, "_p", None)))
        self.prev_total = total
        return out


def one(lineup, seed, ticks, food_mult):
    port = free_port()
    srv = start_server(port, 8, seed, len(lineup), 2, 7000, food_mult=food_mult)
    try:
        clients = [AgentClient(f"ws://127.0.0.1:{port}", n, LossMeter(n, build(sp)), ticks, 2) for n, sp in lineup]

        async def go():
            await asyncio.gather(*(c.run() for c in clients))
        asyncio.run(go())
        return [e for c in clients for e in c.policy.events]
    finally:
        srv.terminate()


def cause(e) -> str:
    h = e["hist"]
    if not h:
        return "unknown"
    last, early = h[-1], h[max(0, len(h) - 1 - LOOKBACK)]
    if any(b["n"] >= a["n"] + 3 for a, b in zip(h[-6:], h[-5:])):
        return "popped"
    my_main = early["me"][2]
    smallest = min(early["cells"])
    eaters = [c for c in early["near"] if c["k"] == "player" and c["m"] > smallest * EAT_RATIO]
    if any(c["k"] == "player" and c["own_n"] >= 2 and c["gap"] < 0 and c["own_m"] > my_main * EAT_RATIO
           for c in last["near"]):
        return "split_kill"
    if not eaters:
        return "unseen"
    if any(x["plan"] == "flee" for x in h[-LOOKBACK:]):
        return "fled"
    return "no_flee"


def report(events) -> None:
    by: dict[str, dict[str, list]] = {}
    for e in events:
        by.setdefault(e["bot"].rsplit("-", 1)[0], {}).setdefault(cause(e), []).append(e["before"] - e["after"])
    for bot, causes in sorted(by.items()):
        tot = sum(sum(v) for v in causes.values()) or 1
        n = sum(len(v) for v in causes.values())
        print(f"{bot}: {n} loss events, {tot:.0f} mass lost")
        for c, v in sorted(causes.items(), key=lambda kv: -sum(kv[1])):
            print(f"   {c:<11} {len(v):4} events  {sum(v):8.0f} mass ({100 * sum(v) / tot:4.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+")
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ticks", type=int, default=22500)
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--seed-base", type=int, default=600000)
    p.add_argument("--cpus", type=int, default=11, help="pin to this many CPUs (0 = all): stay under 90% CPU")
    p.add_argument("--out", default="runs/losses.json")
    p.add_argument("--report", help="only print the cause table of a saved --out file")
    a = p.parse_args()
    if a.report:
        report(json.load(open(a.report)))
        return
    limit_cpus(a.cpus)
    lineup = parse_lineup(a.agents)
    events = []
    with ProcessPoolExecutor(a.workers) as ex:
        for evs in ex.map(one, [lineup] * a.matches, range(a.seed_base, a.seed_base + a.matches),
                          [a.ticks] * a.matches, [a.food_mult] * a.matches):
            events += evs
            print(f"  match done: {len(evs)} events", flush=True)
    with open(a.out, "w") as f:
        json.dump(events, f)
    report(events)


if __name__ == "__main__":
    main()
