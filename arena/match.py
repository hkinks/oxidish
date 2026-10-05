"""Run one lockstep match: start an arena server, connect agents, report stats.

    uv run python -m arena.match --agents route:4 field:4 --ticks 3000 --seed 1 [--watch]

Game time is ticks * 40 ms (3000 ticks = 2 game-minutes), independent of wall time.
"""

import argparse
import asyncio
import collections
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .client import AgentClient, AgentStats
from .policies import POLICIES, build

ROOT = Path(__file__).resolve().parent.parent
SERVER_JS = ROOT / "arena" / "server" / "arena.js"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, bots: int, seed: int, agents: int, sync: int, map_size: int,
                 max_tps: int = 0, follow: str = "", food_mult: float = 1.0) -> subprocess.Popen:
    cmd = ["node", str(SERVER_JS), "--port", str(port), "--bots", str(bots), "--seed", str(seed),
           "--agents", str(agents), "--sync", str(sync), "--map", str(map_size),
           "--max-tps", str(max_tps), "--food-mult", str(food_mult)] + (["--follow", follow] if follow else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:  # wait until it is listening
        if line.startswith("[arena] port="):
            break
    else:
        raise RuntimeError(f"arena server failed to start (exit {proc.poll()})")
    # Keep draining output so the pipe never fills; the tail is kept for error reports.
    proc.tail = collections.deque(maxlen=20)
    threading.Thread(target=lambda: proc.tail.extend(proc.stdout), daemon=True).start()
    return proc


async def run_match(lineup: list[tuple[str, dict]], ticks: int, port: int, sync: int,
                    on_ready=None) -> list[AgentStats]:
    """lineup: [(agent name, policy spec)]"""
    url = f"ws://127.0.0.1:{port}"
    clients = [AgentClient(url, name, build(spec), ticks, sync) for name, spec in lineup]
    tasks = [asyncio.create_task(c.run()) for c in clients]
    if on_ready:
        on_ready()
    return await asyncio.gather(*tasks)


def parse_lineup(specs: list[str]) -> list[tuple[str, dict]]:
    """"route:4 field:4" -> named policy specs; "champ=path.json:4" loads a saved genome."""
    lineup = []
    for spec in specs:
        head, _, tail = spec.rpartition(":")  # count is after the last colon (paths may have C:)
        key, count = (head, tail) if tail.isdigit() else (spec, "1")
        if "=" in key:
            key, path = key.split("=", 1)
            if path.startswith("jev:"):   # "jevbot=jev:runs/final_champion.json": Jev + a champion's routines
                policy = {"kind": "jev", **json.loads(Path(path[4:]).read_text())}
            elif path.endswith(".pt"):  # neural network checkpoint
                policy = {"kind": "nn", "ckpt": path}
            else:
                policy = json.loads(Path(path).read_text())
        elif key in POLICIES:
            policy = POLICIES[key]
        else:
            raise SystemExit(f"unknown policy {key!r}; known: {', '.join(POLICIES)} or name=file.json")
        lineup += [(f"{key}-{i}", policy) for i in range(int(count or 1))]
    return lineup


def print_table(stats: list[AgentStats], wall: float, ticks: int) -> None:
    print(f"\n{'agent':<10} {'avg mass':>9} {'peak':>7} {'deaths':>7} {'kills':>6}")
    for s in sorted(stats, key=lambda s: -s.avg_mass):
        print(f"{s.name:<10} {s.avg_mass:9.1f} {s.peak_mass:7.0f} {s.deaths:7d} {s.kills:6d}")
    by_policy: dict[str, list[AgentStats]] = {}
    for s in stats:
        by_policy.setdefault(s.name.rsplit("-", 1)[0], []).append(s)
    print()
    for key, group in by_policy.items():
        n = len(group)
        print(f"{key:<10} mean avg-mass {sum(s.avg_mass for s in group) / n:7.1f}  "
              f"deaths/agent {sum(s.deaths for s in group) / n:4.2f}  "
              f"kills/agent {sum(s.kills for s in group) / n:4.2f}")
    print(f"\n{ticks} ticks ({ticks * 0.04 / 60:.1f} game-min) in {wall:.1f}s wall "
          f"= {ticks * 0.04 / wall:.1f}x realtime")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["route:4", "field:4"])
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=0, help="0 = any free port")
    p.add_argument("--bots", type=int, default=8, help="built-in MultiOgar bots")
    p.add_argument("--map", type=int, default=7000)
    p.add_argument("--sync", type=int, default=2, help="server ticks per agent decision")
    p.add_argument("--watch", action="store_true", help="open a spectator browser window")
    p.add_argument("--max-tps", type=int, default=0, help="cap ticks/s (25 = realtime)")
    p.add_argument("--follow", default="", help="spectator follows agents with this name prefix")
    args = p.parse_args()

    lineup = parse_lineup(args.agents)
    args.port = args.port or free_port()
    server = start_server(args.port, args.bots, args.seed, len(lineup), args.sync, args.map,
                          args.max_tps, args.follow)
    watcher = None
    try:
        def ready():
            nonlocal watcher
            if args.watch:
                watcher = subprocess.Popen([sys.executable, "-m", "arena.watch",
                                            "--port", str(args.port)], cwd=ROOT)
        started = time.monotonic()
        stats = asyncio.run(run_match(lineup, args.ticks, args.port, args.sync, ready))
        print_table(stats, time.monotonic() - started, args.ticks)
    finally:
        server.terminate()
        if watcher:
            watcher.terminate()


if __name__ == "__main__":
    main()
