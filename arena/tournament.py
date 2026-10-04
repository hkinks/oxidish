"""Run many seeded matches in parallel and compare policies with confidence intervals.

    uv run python -m arena.tournament --agents route:4 field:4 --matches 24 --workers 6

Each match gets its own arena server (free port) and seed i+1. The same seeds
give the same results, so variants can be compared on identical matches.
"""

import argparse
import asyncio
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .match import free_port, parse_lineup, run_match, start_server

METRICS = ("log_mass", "avg_mass", "peak_mass", "deaths", "kills")


def _play(index: int, specs: list[str], ticks: int, bots: int, map_size: int,
          sync: int, seed_base: int, food_mult: float = 1.0) -> list[dict]:
    port, seed = free_port(), seed_base + index
    lineup = parse_lineup(specs)
    server = start_server(port, bots, seed, len(lineup), sync, map_size, food_mult=food_mult)
    try:
        stats = asyncio.run(run_match(lineup, ticks, port, sync))
    finally:
        server.terminate()
    return [{"policy": s.name.rsplit("-", 1)[0], "seed": seed, "avg_mass": s.avg_mass,
             "log_mass": math.log(max(s.avg_mass, 1.0)),
             "peak_mass": s.peak_mass, "deaths": s.deaths, "kills": s.kills} for s in stats]


def _ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, float("nan")
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["route:4", "field:4"])
    p.add_argument("--matches", type=int, default=12)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--bots", type=int, default=8)
    p.add_argument("--map", type=int, default=7000)
    p.add_argument("--sync", type=int, default=2)
    p.add_argument("--food-mult", type=float, default=1.0, help="2 = the showcase's food")
    p.add_argument("--seed-base", type=int, default=1, help="use fresh seeds for unbiased checks")
    args = p.parse_args()

    started = time.monotonic()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_play, i, args.agents, args.ticks, args.bots,
                               args.map, args.sync, args.seed_base, args.food_mult) for i in range(args.matches)]
        for n, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            print(f"  match {n}/{args.matches} done", flush=True)
    wall = time.monotonic() - started

    # Average agents of the same policy within a match first: matches are the samples.
    per_match: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        per_match.setdefault(r["policy"], {}).setdefault(r["seed"], []).append(r)
    print(f"\n{'policy':<10}" + "".join(f"{m:>20}" for m in METRICS))
    for policy, by_seed in per_match.items():
        cells = []
        for m in METRICS:
            samples = [statistics.fmean(r[m] for r in group) for group in by_seed.values()]
            mean, half = _ci95(samples)
            cells.append(f"{mean:10.2f} +- {half:6.2f}")
        print(f"{policy:<10}" + "".join(f"{c:>20}" for c in cells))
    # Head-to-head: share of matches where the row policy's mean log mass beats the column's.
    names = list(per_match)
    if len(names) > 1:
        team = {n: {seed: statistics.fmean(r["log_mass"] for r in g) for seed, g in by.items()}
                for n, by in per_match.items()}
        print("\nhead-to-head win rate (row beats column, same matches)")
        print(f"{'':<10}" + "".join(f"{n:>10}" for n in names))
        for a in names:
            row = []
            for b in names:
                if a == b:
                    row.append(f"{'-':>10}")
                    continue
                seeds = team[a].keys() & team[b].keys()
                row.append(f"{sum(team[a][s] > team[b][s] for s in seeds) / len(seeds):>10.0%}")
            print(f"{a:<10}" + "".join(row))
    game_min = args.matches * args.ticks * 0.04 / 60
    print(f"\n{args.matches} matches, {game_min:.0f} game-min in {wall:.0f}s wall "
          f"= {game_min * 60 / wall:.0f}x realtime aggregate (+-: 95% CI over matches)")


if __name__ == "__main__":
    main()
