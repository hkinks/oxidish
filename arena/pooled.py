"""Pool several tournament result files (arena.tournament --out) and compare the policies over all matches.

    uv run python -m arena.pooled runs/bench/safe_long_7*.json runs/bench/safe_long_8*.json

Per policy: mean log-mass over matches (the same per-match averaging as arena.tournament) with a 95% CI;
then the head-to-head win rate over every match two policies shared, with a 95% Wilson interval: a
policy "consistently beats" another when the interval's lower end stays above 50%.
"""

import argparse
import glob
import json
import math
import statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    team: dict[str, dict[tuple, float]] = {}
    for pattern in args.files:
        for path in sorted(glob.glob(pattern)):
            rows = json.load(open(path))
            by: dict[tuple, list[float]] = {}
            for r in rows:
                by.setdefault((r["policy"], path, r["seed"]), []).append(r["log_mass"])
            for (pol, f, seed), v in by.items():
                team.setdefault(pol, {})[(f, seed)] = statistics.fmean(v)
    names = sorted(team, key=lambda n: -statistics.fmean(team[n].values()))
    print(f"{'policy':<10} {'matches':>7} {'log-mass':>9} {'+-95%':>7}")
    for n in names:
        v = list(team[n].values())
        ci = 1.96 * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"{n:<10} {len(v):7d} {statistics.fmean(v):9.3f} {ci:7.3f}")
    print("\nhead-to-head (row beats column): win rate [95% Wilson interval], matches")
    for a in names:
        cells = []
        for b in names:
            if a == b:
                continue
            common = team[a].keys() & team[b].keys()
            k = sum(team[a][m] > team[b][m] for m in common)
            lo, hi = wilson(k, len(common))
            cells.append(f"{b} {k / max(len(common), 1):.0%} [{lo:.0%}-{hi:.0%}] n={len(common)}")
        print(f"  {a:<9} " + " | ".join(cells))


if __name__ == "__main__":
    main()
