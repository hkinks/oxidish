"""Add agents to an already running arena server; they play until this process is stopped.

    uv run python -m arena.join --port 9500 --agents r3=runs/r3/champion_final.json:2 route:1

Used with the persistent showcase server (arena/showcase.py): new champions
join the live game without restarting the server or the spectator window.
Stopping this process removes its agents (their cells disconnect).
"""

import argparse
import asyncio

from .match import parse_lineup, run_match

FOREVER = 10 ** 12


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--sync", type=int, default=2, help="must match the server's --sync")
    args = p.parse_args()
    lineup = parse_lineup(args.agents)
    print(f"joining {len(lineup)} agents to port {args.port}: "
          f"{', '.join(name for name, _ in lineup)}", flush=True)
    asyncio.run(run_match(lineup, FOREVER, args.port, args.sync))


if __name__ == "__main__":
    main()
