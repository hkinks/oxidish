"""Parity + speed tools for the Rust arena client (rust/arena_client).

    # record snapshots and Python decisions from a real lockstep match
    uv run python -m arena.rust_tools dump out.jsonl --agents route:2 champ=runs/final_champion.json:2 --ticks 2000
    # replay them through the Rust policy and compare
    rust/arena_client/target/release/arena_client parity out.jsonl
    # decisions/s, Python then Rust
    uv run python -m arena.rust_tools bench out.jsonl
    rust/arena_client/target/release/arena_client bench out.jsonl

Dump format (JSON lines): {"t":"agent","a":i,"spec":{...}} once per agent, then per step
{"t":"s","a":i,"mine":[[id,x,y,size]],"others":[[id,x,y,size,kind]],"b":[left,top,right,bottom],"out":[x,y,split,eject]}
and {"t":"r","a":i} where the policy was reset (death).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from bot.game import Cell, Snapshot

from .client import AgentClient
from .match import free_port, parse_lineup, start_server
from .policies import build


class DumpPolicy:
    """Wraps a policy; appends every snapshot it sees and its decision to `sink`."""

    def __init__(self, index: int, policy, sink: list):
        self.index, self.policy, self.sink = index, policy, sink

    @property
    def counters(self):
        return self.policy.counters

    def reset(self):
        self.sink.append(json.dumps({"t": "r", "a": self.index}))
        self.policy.reset()

    def act(self, s):
        rec = {"t": "s", "a": self.index,
               "mine": [[c.id, c.x, c.y, c.size] for c in s.mine],
               "others": [[c.id, c.x, c.y, c.size, c.kind] for c in s.others],
               "b": [s.border["left"], s.border["top"], s.border["right"], s.border["bottom"]]}
        out = self.policy.act(s)
        rec["out"] = [float(out[0]), float(out[1]), bool(out[2]), bool(out[3]) if len(out) > 3 else False]
        self.sink.append(json.dumps(rec, separators=(",", ":")))
        return out


def dump(path: str, agents: list[str], ticks: int, seed: int, food_mult: float, bots: int) -> None:
    lineup = parse_lineup(agents)
    port = free_port()
    server = start_server(port, bots, seed, len(lineup), 2, 7000, food_mult=food_mult)
    sink: list[str] = []
    try:
        for i, (name, spec) in enumerate(lineup):
            sink.append(json.dumps({"t": "agent", "a": i, "spec": spec}))
        clients = [AgentClient(f"ws://127.0.0.1:{port}", name, DumpPolicy(i, build(spec), sink), ticks, 2)
                   for i, (name, spec) in enumerate(lineup)]

        async def go():
            await asyncio.gather(*(c.run() for c in clients))
        asyncio.run(go())
    finally:
        server.terminate()
    Path(path).write_text("\n".join(sink) + "\n")
    print(f"{len(sink)} records -> {path}")


def fuzz(path: str, specs: list[str], steps: int, seed: int) -> None:
    """Synthetic but varied snapshots (many pieces, viruses, big players near us) so every tactic and gene
    fires; Python decisions recorded in the same dump format. Cells persist and drift between steps."""
    import random
    rng = random.Random(seed)
    lineup = parse_lineup(specs)
    sink: list[str] = []
    for i, (name, spec) in enumerate(lineup):
        sink.append(json.dumps({"t": "agent", "a": i, "spec": spec}))
    pols = [DumpPolicy(i, build(spec), sink) for i, (_, spec) in enumerate(lineup)]
    border = {"left": -3500.0, "top": -3500.0, "right": 3500.0, "bottom": 3500.0}
    nid = 1
    for i, pol in enumerate(pols):
        cx, cy = rng.uniform(-3000, 3000), rng.uniform(-3000, 3000)
        others: list[Cell] = []
        mine: list[Cell] = []
        for step in range(steps):
            if step % 400 == 0 or not mine:   # new scenario / respawn
                if step:
                    pol.reset()
                cx, cy = rng.uniform(-3300, 3300), rng.uniform(-3300, 3300)
                size = rng.choice([30, 60, 120, 200, 300, 450, 700])
                mine = [Cell(nid := nid + 1, cx, cy, size, "mine", "")]
                for _ in range(rng.choice([0, 0, 1, 2, 3])):
                    mine.append(Cell(nid := nid + 1, cx + rng.uniform(-150, 150), cy + rng.uniform(-150, 150),
                                     size * rng.uniform(0.3, 1.0), "mine", ""))
                others = []
                for _ in range(rng.randint(10, 50)):
                    others.append(Cell(nid := nid + 1, cx + rng.uniform(-1500, 1500), cy + rng.uniform(-1500, 1500), 10, "food", ""))
                for _ in range(rng.randint(0, 4)):
                    others.append(Cell(nid := nid + 1, cx + rng.uniform(-1200, 1200), cy + rng.uniform(-1200, 1200), rng.choice([100, 110, 130]), "virus", ""))
                for _ in range(rng.randint(0, 6)):
                    others.append(Cell(nid := nid + 1, cx + rng.uniform(-1800, 1800), cy + rng.uniform(-1800, 1800),
                                       size * rng.choice([0.2, 0.4, 0.7, 1.0, 1.2, 1.5, 2.5, 4.0]), "player", "p"))
                if rng.random() < 0.6:   # a virus between us and a big player: the virus-shot setup
                    import math
                    a = rng.uniform(0, 6.283)
                    others.append(Cell(nid := nid + 1, cx + 350 * math.cos(a), cy + 350 * math.sin(a), 100, "virus", ""))
                    others.append(Cell(nid := nid + 1, cx + 800 * math.cos(a + rng.uniform(-0.1, 0.1)),
                                       cy + 800 * math.sin(a + rng.uniform(-0.1, 0.1)), rng.choice([300, 450]), "player", "p"))
                for _ in range(rng.randint(0, 3)):
                    others.append(Cell(nid := nid + 1, cx + rng.uniform(-600, 600), cy + rng.uniform(-600, 600), 38, "ejected", ""))
            for c in others:     # players and viruses drift; some food gets eaten and respawns
                if c.kind in ("player", "ejected"):
                    c.x += rng.uniform(-20, 20)
                    c.y += rng.uniform(-20, 20)
            for k in range(len(others)):
                if others[k].kind == "food" and rng.random() < 0.03:
                    others[k] = Cell(nid := nid + 1, mine[0].x + rng.uniform(-1500, 1500), mine[0].y + rng.uniform(-1500, 1500), 10, "food", "")
            snap = Snapshot([Cell(c.id, c.x, c.y, c.size, "mine", "") for c in mine],
                            [Cell(c.id, c.x, c.y, c.size, c.kind, "") for c in others], {}, 0, 0, dict(border), False)
            out = pol.act(snap)
            for c in mine:      # move toward the aim point like a cell would
                dx, dy = out[0] - c.x, out[1] - c.y
                d = max((dx * dx + dy * dy) ** 0.5, 1.0)
                v = min(60.0, d)
                c.x = min(max(c.x + dx / d * v, -3490), 3490)
                c.y = min(max(c.y + dy / d * v, -3490), 3490)
            if out[2] and len(mine) < 4:   # a split halves the cell
                m = mine[0]
                m.size /= 2 ** 0.5
                mine.append(Cell(nid := nid + 1, m.x, m.y, m.size, "mine", ""))
    Path(path).write_text("\n".join(sink) + "\n")
    print(f"{len(sink)} records -> {path}")


def bench(path: str, repeat: int) -> None:
    agents: dict[int, dict] = {}
    events = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["t"] == "agent":
            agents[r["a"]] = r["spec"]
        events.append(r)
    total, n = 0.0, 0
    for _ in range(repeat):
        pols = {a: build(spec) for a, spec in agents.items()}
        for r in events:
            if r["t"] == "r":
                pols[r["a"]].reset()
            elif r["t"] == "s":
                mine = [Cell(i, x, y, sz, "mine", "") for i, x, y, sz in r["mine"]]
                others = [Cell(i, x, y, sz, k, "") for i, x, y, sz, k in r["others"]]
                b = r["b"]
                s = Snapshot(mine, others, {}, 0, 0, {"left": b[0], "top": b[1], "right": b[2], "bottom": b[3]}, False)
                t0 = time.perf_counter()
                pols[r["a"]].act(s)
                total += time.perf_counter() - t0
                n += 1
    print(f"python: {n} decisions in {total:.3f}s = {n / total:.0f} decisions/s ({total / n * 1e6:.1f} us each)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("path")
    d.add_argument("--agents", nargs="+", required=True)
    d.add_argument("--ticks", type=int, default=2000)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--food-mult", type=float, default=2.0)
    d.add_argument("--bots", type=int, default=8)
    f = sub.add_parser("fuzz")
    f.add_argument("path")
    f.add_argument("--agents", nargs="+", required=True)
    f.add_argument("--steps", type=int, default=2000)
    f.add_argument("--seed", type=int, default=1)
    b = sub.add_parser("bench")
    b.add_argument("path")
    b.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "dump":
        dump(args.path, args.agents, args.ticks, args.seed, args.food_mult, args.bots)
    elif args.cmd == "fuzz":
        fuzz(args.path, args.agents, args.steps, args.seed)
    else:
        bench(args.path, args.repeat)


if __name__ == "__main__":
    main()
