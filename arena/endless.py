"""Dominance in an endless world (like the showcase server): how often is a bot family on top?

    uv run python -m arena.endless --candidate c3_4 --agents c3_4=bot/c3_4.json:2 champ=runs/final_champion.json:2 ... \
        --worlds 4 --game-min 240 --window-min 10

Runs `--worlds` independent Rust-engine worlds (rust/ogar `serve`, unthrottled, rule bots in-process, 8 MultiOgar
server bots) for `--game-min` game-minutes each, never resetting (dead players respawn, giants keep growing). Every
world is cut into `--window-min` game-minute windows; in each window every bot family's mass is averaged over its
copies and the window's samples. Dominance = share of windows in which the candidate family is the heaviest bot
family (server bots excluded); also per opponent, the share of windows the candidate is ahead of it. The user's bar
(2026-10-07): ~90% dominance.
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .match import free_port, limit_cpus, parse_lineup
from .pooled import wilson

ROOT = Path(__file__).resolve().parent.parent
TICKS_PER_MIN = 25 * 60


def ogar_bin() -> Path:
    exe = "ogar.exe" if os.name == "nt" else "ogar"
    return Path(os.environ.get("OGAR_BIN", ROOT / "rust" / "ogar" / "target" / "release" / exe))


def run_world(specs: list[str], seed: int, game_min: float, log_s: float, food_mult: float) -> list[dict]:
    """One endless world; returns its [players] samples ({t, p: [{n, m, c}]})."""
    port = free_port()
    end_tick = int(game_min * TICKS_PER_MIN)
    cmd = [str(ogar_bin()), "serve", "--port", str(port), "--bind", "127.0.0.1", "--bots", "8", "--map", "7000",
           "--food-mult", str(food_mult), "--seed", str(seed), "--max-tps", "0", "--log-players", str(log_s),
           "--agents", *specs]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace")
    samples = []
    try:
        for line in proc.stdout:
            if line.startswith("[players] "):
                rec = json.loads(line[len("[players] "):])
                samples.append(rec)
                if rec["t"] >= end_tick:
                    break
    finally:
        proc.kill()
    return samples


def family_masses(samples: list[dict], families: dict[str, int], window_ticks: int) -> list[dict[str, float]]:
    """Per window: family -> mean mass (over copies and samples); dead copies count 0."""
    out = []
    by_win: dict[int, list[dict]] = defaultdict(list)
    for rec in samples:
        by_win[rec["t"] // window_ticks].append(rec)
    for w in sorted(by_win):
        acc: dict[str, list[float]] = defaultdict(list)
        for rec in by_win[w]:
            per: dict[str, list[float]] = defaultdict(list)
            for pl in rec["p"]:
                fam = pl["n"].rsplit("-", 1)[0]
                if fam in families:
                    per[fam].append(pl["m"])
            for fam, copies in families.items():
                acc[fam].append(sum(per[fam]) / copies)
        out.append({fam: sum(v) / len(v) for fam, v in acc.items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--game-min", type=float, default=240.0)
    p.add_argument("--window-min", type=float, default=10.0)
    p.add_argument("--log-s", type=float, default=15.0, help="mass sample every N game-seconds")
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--seed-base", type=int, default=700)
    p.add_argument("--skip-min", type=float, default=0.0, help="ignore the first N game-minutes of each world")
    p.add_argument("--cpus", type=int, default=11)
    p.add_argument("--out", help="save every world's per-window family masses (JSON)")
    a = p.parse_args()
    limit_cpus(a.cpus)
    lineup = parse_lineup(a.agents)
    families: dict[str, int] = defaultdict(int)     # family -> copies (in-process agents are named <spec>-<i>)
    for n, _ in lineup:
        families[n.rsplit("-", 1)[0]] += 1
    t0 = time.time()
    with ThreadPoolExecutor(a.worlds) as ex:
        worlds = list(ex.map(lambda i: run_world(a.agents, a.seed_base + i, a.game_min, a.log_s, a.food_mult),
                             range(a.worlds)))
    window_ticks = int(a.window_min * TICKS_PER_MIN)
    skip = int(a.skip_min / a.window_min)
    wins = [fm for s in worlds for fm in family_masses(s, families, window_ticks)[skip:]]
    if a.out:
        with open(a.out, "w") as f:
            json.dump(wins, f)
    c = a.candidate
    # leader share: samples where the #1 player of the whole leaderboard (server bots included) is ours
    lead_n = lead_k = 0
    leaders: dict[str, int] = defaultdict(int)
    for s_ in worlds:
        for rec in s_:
            if rec["t"] < a.skip_min * TICKS_PER_MIN or not rec["p"]:
                continue
            fam = max(rec["p"], key=lambda pl: pl["m"])["n"].rsplit("-", 1)[0]
            leaders[fam if fam in families else "(server bot)"] += 1
            lead_n += 1
            lead_k += fam == c
    top = sum(1 for w in wins if max(w, key=w.get) == c)
    lo, hi = wilson(top, len(wins))
    print(f"{len(worlds)} worlds x {a.game_min:.0f} game-min, {len(wins)} windows of {a.window_min:.0f} min "
          f"({time.time() - t0:.0f}s wall)")
    print(f"\n{c} DOMINANCE (heaviest bot family): {top}/{len(wins)} = {top / max(len(wins), 1):.0%}  "
          f"[95% {lo:.0%}-{hi:.0%}]   bar: ~90%")
    llo, lhi = wilson(lead_k, lead_n)
    print(f"{c} LEADER SHARE (#1 on the leaderboard): {lead_k / max(lead_n, 1):.0%}  [95% {llo:.0%}-{lhi:.0%}] of "
          f"{lead_n} samples (correlated within a world)   goal: 80%")
    print("   leader shares: " + ", ".join(f"{k} {v / max(lead_n, 1):.0%}" for k, v in sorted(leaders.items(), key=lambda kv: -kv[1])))
    print(f"\n{'family':<12} {'mean mass':>10} {'top share':>10} {c + ' ahead':>14}")
    for fam in sorted(families, key=lambda f: -sum(w[f] for w in wins)):
        mean = sum(w[fam] for w in wins) / max(len(wins), 1)
        tops = sum(1 for w in wins if max(w, key=w.get) == fam) / max(len(wins), 1)
        ahead = "" if fam == c else f"{sum(1 for w in wins if w[c] > w[fam]) / max(len(wins), 1):.0%}"
        print(f"{fam:<12} {mean:10.0f} {tops:10.0%} {ahead:>14}")


if __name__ == "__main__":
    main()
