"""Summarise per-player mass from the showcase's [players] log lines.

    uv run python -m training.eval.showcase_stats [--log logs/showcase.txt] [--since-min 0]

Bots with numbered copies (champ-0, champ-1) are grouped by prefix. A "death" is
counted when a player disappears from the list or drops below 30% of its previous
mass in one 5 s sample (eaten while split also counts).
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUMAN = ""   # the user plays with an empty name: the benchmark the bots should beat
SAMPLE_S = 5  # showcase --log-players interval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(ROOT / "logs" / "showcase.txt"))
    p.add_argument("--since-min", type=float, default=0.0, help="ignore the first N game-minutes")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # player names are full of symbols

    masses, deaths = defaultdict(list), defaultdict(int)
    prev: dict[str, int] = {}
    for line in Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[players] "):
            continue
        rec = json.loads(line[len("[players] "):])
        if rec["t"] / 25 / 60 < args.since_min:
            continue
        now = {pl["n"]: pl["m"] for pl in rec["p"]}
        for name, m in prev.items():
            if name not in now or now[name] < 0.3 * m:
                deaths[name] += 1
        boosted = {pl["n"] for pl in rec["p"] if pl.get("b")}   # lives that used "+": no benchmark mass
        for name, m in now.items():
            if name not in boosted:
                masses[name].append(m)
        prev = now

    groups = defaultdict(lambda: {"m": [], "d": 0})
    for name, ms in masses.items():
        key = re.sub(r"-\d+$", "", name)
        groups[key]["m"] += ms
        groups[key]["d"] += deaths[name]
    rows = sorted(groups.items(), key=lambda kv: -statistics.fmean(kv[1]["m"]))
    human = groups.get(HUMAN)
    ref = statistics.fmean(human["m"]) if human else None
    print(f"{'player':<22}{'avg mass':>10}{'peak':>8}{'minutes':>9}{'deaths':>8}{'deaths/h':>9}"
          f"{'avg vs you':>11}")
    for name, g in rows:
        avg, minutes = statistics.fmean(g["m"]), len(g["m"]) * SAMPLE_S / 60
        label = "YOU (human)" if name == HUMAN else name[:21]
        print(f"{label:<22}{avg:10.0f}{max(g['m']):8d}{minutes:9.0f}{g['d']:8d}"
              f"{g['d'] / max(minutes / 60, 1e-9):9.1f}"
              f"{(f'{avg / ref:10.0%}' if ref else '         -'):>11}")


if __name__ == "__main__":
    main()
