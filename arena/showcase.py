"""Persistent real-time arena for watching and playing; agents are added with arena.join.

    uv run python -m arena.showcase [--port 9500] [--follow champ]

Starts the arena server (no match end, 25 ticks/s = real time) and a spectator
window with the follow/play panel. Leave it running; add or remove bots with
`python -m arena.join --port 9500 --agents name=file.json:N`.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

from .match import ROOT, SERVER_JS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9500)
    p.add_argument("--bots", type=int, default=8)
    p.add_argument("--map", type=int, default=7000)
    p.add_argument("--follow", default="champ")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--food-mult", type=float, default=1.0, help="food density multiplier")
    p.add_argument("--log-players", type=float, default=5.0, help="log player masses every N s")
    p.add_argument("--record", default="", help="record human players' inputs + views to this dir (imitation data)")
    args = p.parse_args()
    server = subprocess.Popen(["node", str(SERVER_JS), "--port", str(args.port), "--bots",
                               str(args.bots), "--map", str(args.map), "--seed", str(args.seed),
                               "--sync", "2", "--max-tps", "25", "--follow", args.follow,
                               "--food-mult", str(args.food_mult),
                               "--log-players", str(args.log_players),
                               *(["--record", str(Path(args.record).resolve())] if args.record else [])],
                              stdin=subprocess.DEVNULL)
    time.sleep(1.5)
    watcher = subprocess.Popen([sys.executable, "-m", "arena.watch", "--port", str(args.port)],
                               cwd=ROOT)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.terminate()
        server.terminate()


if __name__ == "__main__":
    main()
