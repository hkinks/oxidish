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
    b = sub.add_parser("bench")
    b.add_argument("path")
    b.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "dump":
        dump(args.path, args.agents, args.ticks, args.seed, args.food_mult, args.bots)
    else:
        bench(args.path, args.repeat)


if __name__ == "__main__":
    main()
