"""Rival-position prediction error of the lookahead planner (AGAR_PLAN_VAL), per rival-model scenario.

    uv run python -m arena.plan_val --spec bot/plan2_c34.json --scen 0 1 2 3 --game-min 15 --seed 3000

For every scenario runs one Rust-engine world with planner agents among rule bots; the planner logs, at 0.4 / 1 / 2 s
after each gated decision, the mean distance between the simulated and the real rival cells (scenario = the rival
model used for the logged prediction). Prints the mean px error per horizon.
"""

import argparse
import os
import re
import subprocess
from collections import defaultdict

from .endless import ogar_bin, ROOT, TICKS_PER_MIN
from .match import free_port

FIELD = ["champ=runs/final_champion.json:2", "r2g9=runs/r2/champion_009.json:2", "safe=bot/champ_safe.json:2",
         "ramp_r2=bot/r2g9_ramp800.json:2", "ramp300=runs/r2g9_ramp300.json:2"]
VAL = re.compile(r"VAL dec=\S+ h=(\d+) matched=(\d+) of (\d+) perr=([\d.]+) serr=(-?[\d.]+) near=(\d+) nerr=([\d.]+)")


def run(spec: str, scen: int, game_min: float, seed: int) -> dict[int, tuple[float, int, float]]:
    env = dict(os.environ, AGAR_PLAN_VAL="1", AGAR_PLAN_VAL_SCEN=str(scen))
    cmd = [str(ogar_bin()), "serve", "--port", str(free_port()), "--bind", "127.0.0.1", "--bots", os.environ.get("VAL_BOTS", "8"), "--map", "7000",
           "--food-mult", "2", "--seed", str(seed), "--max-tps", "0", "--log-players", "15",
           "--agents", f"plan={spec}:2", *FIELD]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", env=env)
    acc: dict[int, list[float]] = defaultdict(list)
    near: dict[int, list[float]] = defaultdict(list)    # rivals within 600 px gap of us at decision time
    end = int(game_min * TICKS_PER_MIN)
    try:
        for line in proc.stdout:
            if line.startswith("[players] "):
                if int(re.search(r'"t": *(\d+)', line).group(1)) >= end:
                    break
                continue
            m = VAL.match(line)
            if m and int(m.group(2)) > 0:
                h = int(m.group(1)) % 1000
                acc[h].extend([float(m.group(4))] * int(m.group(2)))   # weighted by matched rivals
                near[h].extend([float(m.group(7))] * int(m.group(6)))
    finally:
        proc.kill()
    return {h: (sum(v) / len(v), len(v), sum(near[h]) / max(len(near[h]), 1)) for h, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="bot/plan2_c34.json")
    p.add_argument("--scen", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--game-min", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=3000)
    a = p.parse_args()
    print(f"{'scen':>5} " + " ".join(f"{'h=' + str(h / 25) + 's':>14}" for h in (10, 24, 50)))
    for sc in a.scen:
        r = run(a.spec, sc, a.game_min, a.seed)
        print(f"{sc:>5} " + " ".join(f"{r[h][0]:>6.0f} / near {r[h][2]:>4.0f}" if h in r else f"{'-':>14}" for h in (10, 24, 50)), flush=True)


if __name__ == "__main__":
    main()
