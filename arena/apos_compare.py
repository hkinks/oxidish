"""Native AposBot port vs the real AposBot (the JS, in headless browsers) in one real-time world.

    uv run python -m arena.apos_compare --minutes 12 --native 2 --real 2 --extra champ=runs/final_champion.json:1

Starts a Rust-engine server (`ogar serve`, real time, 8 server bots) with `--native` in-process native `apos`
agents (+ `--extra` rule-bot agents for scale), a static web client for the browsers, and `arena.apos --count N`
(the original `createAposBot` running in headless Chromium through the BOT-API shim). The real bots plan at
the page's speed, so this only works in real time (wall minutes = game minutes). After `--minutes` it reads the
server's `[players]` log and prints, per family: mean mass over the samples (dead = 0), share of samples alive,
peak mass, deaths (alive -> dead transitions) and leader share. Needs `vendor/` (web client, apos bundle) and
Playwright's Chromium.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .endless import ogar_bin
from .match import free_port

ROOT = Path(__file__).resolve().parent.parent


def family(name: str) -> str:
    return name.rsplit("-", 1)[0]


def summarize(samples: list[dict], families: set[str]) -> None:
    names = sorted({p["n"] for rec in samples for p in rec["p"] if family(p["n"]) in families})
    mass = {n: [] for n in names}
    leaders: dict[str, int] = defaultdict(int)
    for rec in samples:
        here = {p["n"]: p["m"] for p in rec["p"]}
        for n in names:
            mass[n].append(here.get(n, 0))
        if rec["p"]:
            leaders[family(max(rec["p"], key=lambda p: p["m"])["n"])] += 1
    print(f"{len(samples)} samples; per bot (dead counts as 0 mass)")
    print(f"{'bot':<14}{'mean mass':>11}{'alive':>8}{'peak':>8}{'deaths':>8}")
    fam: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    for n in names:
        s = mass[n]
        deaths = sum(1 for a, b in zip(s, s[1:]) if a > 0 and b == 0)
        row = (sum(s) / len(s), sum(1 for v in s if v) / len(s), max(s), deaths)
        fam[family(n)].append(row)
        print(f"{n:<14}{row[0]:11.0f}{row[1]:8.0%}{row[2]:8d}{row[3]:8d}")
    print(f"\n{'family':<14}{'mean mass':>11}{'alive':>8}{'peak':>8}{'deaths/bot':>12}{'leader':>8}")
    for f, rows in fam.items():
        k = len(rows)
        print(f"{f:<14}{sum(r[0] for r in rows) / k:11.0f}{sum(r[1] for r in rows) / k:8.0%}"
              f"{sum(r[2] for r in rows) / k:8.0f}{sum(r[3] for r in rows) / k:12.1f}{leaders[f] / max(len(samples), 1):8.0%}")
    other = sum(v for f, v in leaders.items() if f not in families)
    print(f"(leaderboard #1 was a server bot in {other / max(len(samples), 1):.0%} of samples)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=12.0)
    p.add_argument("--native", type=int, default=2, help="native apos agents (in-process)")
    p.add_argument("--real", type=int, default=2, help="real AposBot browsers (family aposjs)")
    p.add_argument("--extra", nargs="*", default=[], help="more in-process agents, e.g. champ=runs/final_champion.json:1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--log-s", type=float, default=5.0)
    args = p.parse_args()

    port, web = free_port(), free_port()
    server = subprocess.Popen(
        [str(ogar_bin()), "serve", "--port", str(port), "--bind", "127.0.0.1", "--bots", "8", "--map", "7000",
         "--food-mult", str(args.food_mult), "--seed", str(args.seed), "--max-tps", "25", "--log-players", str(args.log_s),
         "--agents", f"apos=bot/sparring/apos.json:{args.native}", *args.extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
        encoding="utf-8", errors="replace")
    http = subprocess.Popen([sys.executable, "-m", "http.server", str(web), "--bind", "127.0.0.1"],
                            cwd=ROOT / "vendor" / "agarian-client", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    real = None
    samples: list[dict] = []
    import threading

    def read():
        for line in server.stdout:
            if line.startswith("[players] "):
                samples.append(json.loads(line[len("[players] "):]))
    threading.Thread(target=read, daemon=True).start()
    try:
        time.sleep(2.0)
        if args.real:
            real = subprocess.Popen([sys.executable, "-m", "arena.apos", "--port", str(port), "--count", str(args.real),
                                     "--name", "aposjs", "--client", f"http://127.0.0.1:{web}/"],
                                    cwd=ROOT, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        print(f"server :{port}, client :{web}; running {args.minutes:.0f} min ...", flush=True)
        time.sleep(args.minutes * 60)
    finally:
        for proc in (real, http, server):
            if proc is not None:
                proc.kill()
    fams = {"apos", "aposjs"} | {a.split("=")[0].split(":")[0] for a in args.extra}
    summarize(samples, fams)


if __name__ == "__main__":
    main()
