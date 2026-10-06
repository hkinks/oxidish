"""Run many seeded matches in parallel and compare policies with confidence intervals.

    uv run python -m arena.tournament --agents route:4 field:4 --matches 24 --workers 6

Each match gets its own arena server (free port) and seed i+1. The same seeds
give the same results, so variants can be compared on identical matches.
"""

import argparse
import asyncio
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .match import free_port, limit_cpus, parse_lineup, run_match, start_server

METRICS = ("log_mass", "avg_mass", "peak_mass", "deaths", "kills")


def _play(index: int, specs: list[str], ticks: int, bots: int, map_size: int,
          sync: int, seed_base: int, food_mult: float = 1.0, rust: bool = False,
          engine: str = "node") -> list[dict]:
    seed = seed_base + index
    if engine == "ogar":   # the Rust re-implementation of the server (rust/ogar): whole match in one process
        from . import rust_runner
        if not rust_runner.supported(specs):
            raise SystemExit("--engine ogar: only rule bots (route, field, name=genome.json) are supported")
        return rust_runner.play_ogar(specs, ticks, sync, seed, bots, map_size, food_mult)
    port = free_port()
    if rust:   # rule-bot lineups only: the Rust client plays the whole lineup in one process
        from . import rust_runner
        if not rust_runner.supported(specs):
            raise SystemExit("--rust: only rule bots (route, field, name=genome.json) are supported")
        server = start_server(port, bots, seed, len(parse_lineup(specs)), sync, map_size, food_mult=food_mult)
        try:
            return rust_runner.play_rust(specs, ticks, port, sync, seed)
        finally:
            server.terminate()
    lineup = parse_lineup(specs)
    server = start_server(port, bots, seed, len(lineup), sync, map_size, food_mult=food_mult)
    try:
        stats = asyncio.run(run_match(lineup, ticks, port, sync))
    finally:
        server.terminate()
    return [{"policy": s.name.rsplit("-", 1)[0], "seed": seed, "avg_mass": s.avg_mass,
             "log_mass": math.log(max(s.avg_mass, 1.0)),
             "peak_mass": s.peak_mass, "deaths": s.deaths, "kills": s.kills, "losses": s.losses,
             "counters": s.counters, "popped": s.popped}
            for s in stats]


def play_many(specs: list[str], matches: int, workers: int = 8, ticks: int = 6000, bots: int = 8,
              map_size: int = 7000, sync: int = 2, seed_base: int = 1, food_mult: float = 2.0,
              engine: str = "node") -> list[dict]:
    """`matches` seeded matches of the lineup `specs` in parallel; one row per agent per match."""
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_play, i, specs, ticks, bots, map_size, sync, seed_base, food_mult, False, engine)
                   for i in range(matches)]
        for fut in as_completed(futures):
            try:
                rows += fut.result()
            except Exception as e:     # one broken match must not sink the whole evaluation
                print(f"  match failed: {e!r}", flush=True)
    return rows


def print_losses(rows: list[dict]) -> None:
    """Where each policy's mass went: eaten by whom, and how often a split was involved."""
    names = {r["policy"] for r in rows}
    group = lambda by: by.rsplit("-", 1)[0] if by.rsplit("-", 1)[0] in names else (
        by if by in ("virus", "unseen", "ejected") else "server bot")
    print("\nmass lost to (share of each policy's eaten mass); split-kill = killer had 2+ cells,"
          " while split = we had 2+ cells")
    for policy in sorted(names):
        ls = [l for r in rows if r["policy"] == policy for l in r["losses"]]
        total = sum(l["mass"] for l in ls)
        if not total:
            continue
        by: dict[str, float] = {}
        for l in ls:
            by[group(l["by"])] = by.get(group(l["by"]), 0.0) + l["mass"]
        split_kill = sum(l["mass"] for l in ls if l["killer_cells"] > 1) / total
        while_split = sum(l["mass"] for l in ls if l["my_cells"] > 1) / total
        top = ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:6])
        print(f"  {policy:<10} {len(ls):5d} cells, {total:9.0f} mass | split-kill {split_kill:4.0%}"
              f" | while split {while_split:4.0%} | {top}")


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
    p.add_argument("--out", help="also write every agent's per-match stats (incl. losses) to this JSON")
    p.add_argument("--rust", action="store_true", help="play rule-bot lineups with the Rust client (rust/arena_client)")
    p.add_argument("--engine", choices=("node", "ogar"), default="node",
                   help="ogar = the Rust engine (rust/ogar), rule bots only; ~10x faster than node + clients")
    p.add_argument("--cpus", type=int, default=11, help="pin to this many CPUs (0 = all): stay under 90% CPU")
    args = p.parse_args()
    limit_cpus(args.cpus)

    started = time.monotonic()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_play, i, args.agents, args.ticks, args.bots,
                               args.map, args.sync, args.seed_base, args.food_mult, args.rust, args.engine) for i in range(args.matches)]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception as e:     # one broken match (e.g. a dropped socket) must not lose the run
                print(f"  match {n}/{args.matches} failed: {e!r}", flush=True)
                continue
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
    print_losses(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows))
    game_min = args.matches * args.ticks * 0.04 / 60
    print(f"\n{args.matches} matches, {game_min:.0f} game-min in {wall:.0f}s wall "
          f"= {game_min * 60 / wall:.0f}x realtime aggregate (+-: 95% CI over matches)")


if __name__ == "__main__":
    main()
