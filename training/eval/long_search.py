"""CMA-ES over the champion's genes, scored in long Rust-engine games (`oxidish-server match`) against a fixed field.

    uv run python -m training.eval.long_search --run long1 --init runs/champ_safe.json [--popsize 6] [--matches 6]

Long games (15 game-min, 2x food, 8 MultiOgar bots) reward survival (training/eval/params.py caution genes) far
more than the 3-4 min matches the earlier evolutions used. Each generation every candidate plays
`--matches` matches as 2 copies among a fixed field (champ x3, r2g9 x3: the reckless champion family);
all candidates share the generation's seeds (common random numbers). Fitness = mean over its matches
of (candidate log-mass - mean log-mass of the field in that match). State and every candidate's
result go to runs/<run>/gen_###.json; the best candidate so far to runs/<run>/best.json. Validate a
winner in the full field (training.eval.tournament + training.eval.pooled) before believing it.
"""

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="Could not import matplotlib")
import cma  # noqa: E402

from .params import BOUNDS, Params, from_unit, to_unit

from .util import limit_cpus
from .tournament import _play

GENES = ["danger_margin", "threat_force", "flee_margin", "farm_threat", "hunt_threat", "flee_threat",
         "split_safety", "hunt_range", "min_prey_frac", "food_pull", "target_bonus", "virus_avoid",
         "farm_prey", "switch_margin", "split_reach"]
FIELD = ["champ=bots/specs/final_champion.json:3", "r2g9=bots/specs/r2/champion_009.json:3"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="long1")
    p.add_argument("--init", default="runs/champ_safe.json")
    p.add_argument("--popsize", type=int, default=6)
    p.add_argument("--matches", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--ticks", type=int, default=22500)
    p.add_argument("--field", nargs="+", default=FIELD, help="the fixed opponents (arena lineup specs)")
    p.add_argument("--genes", nargs="+", default=GENES, help="the searched genes (others stay as in --init)")
    p.add_argument("--cpus", type=int, default=11, help="pin to this many CPUs (0 = all): stay under 90% CPU")
    args = p.parse_args()
    limit_cpus(args.cpus)
    root = Path("runs") / args.run
    root.mkdir(parents=True, exist_ok=True)
    init = json.loads(Path(args.init).read_text())
    base = Params.from_dict(init["params"])
    names = list(BOUNDS)
    unit = to_unit(base)
    genes = args.genes
    idx = [names.index(g) for g in genes]
    es = cma.CMAEvolutionStrategy([unit[i] for i in idx], args.sigma,
                                  {"popsize": args.popsize, "bounds": [0, 1], "seed": 1, "verbose": -9})
    best = {"fitness": -1e9}
    gen = 0
    while gen < args.gens and not es.stop():
        t0 = time.time()
        xs = es.ask()
        specs = []
        for x in xs:
            u = list(unit)
            for k, i in enumerate(idx):
                u[i] = float(x[k])
            params = from_unit(u).to_dict()
            for k in ("half_split_safety", "piece_threats", "caution_mass"):   # keep the non-searched genes
                if k not in genes:
                    params[k] = init["params"].get(k, params.get(k))
            specs.append({**init, "params": params})
        seeds = [900_000 + 1000 * gen + j for j in range(args.matches)]
        jobs = []
        for c, spec in enumerate(specs):
            f = root / f"cand_{gen:03d}_{c}.json"
            f.write_text(json.dumps(spec, indent=1))
            lineup = [f"cand{c}={f}:2"] + args.field
            jobs += [(c, s, lineup) for s in seeds]
        scores: dict[int, list[float]] = {c: [] for c in range(len(specs))}
        with ProcessPoolExecutor(args.workers) as pool:
            futs = {pool.submit(_play, s - 1, lineup, args.ticks, 8, 7000, 2, 1, 2.0): c for c, s, lineup in jobs}
            for fut, c in futs.items():
                try:
                    rows = fut.result()
                except Exception as e:      # a broken match must not sink the generation
                    print(f"  match failed: {e!r}", flush=True)
                    continue
                mine = [r["log_mass"] for r in rows if r["policy"].startswith("cand")]
                field = [r["log_mass"] for r in rows if not r["policy"].startswith("cand")]
                scores[c].append(statistics.fmean(mine) - statistics.fmean(field))
        fit = [statistics.fmean(v) if v else -9.0 for c, v in sorted(scores.items())]
        es.tell(xs, [-f for f in fit])
        for c, f in enumerate(fit):
            if f > best["fitness"]:
                best = {"fitness": f, "gen": gen, "cand": c, "spec": specs[c]}
                (root / "best.json").write_text(json.dumps(specs[c], indent=1))
        (root / f"gen_{gen:03d}.json").write_text(json.dumps(
            {"gen": gen, "fitness": fit, "scores": scores, "best": {k: best[k] for k in ("fitness", "gen", "cand")},
             "mean_genes": dict(zip(genes, map(float, es.mean)))}, indent=1))
        print(f"gen {gen}: fitness " + " ".join(f"{f:+.3f}" for f in fit)
              + f"  (best so far {best['fitness']:+.3f}, gen {best['gen']} cand {best['cand']})"
              + f"  {time.time() - t0:.0f}s", flush=True)
        gen += 1


if __name__ == "__main__":
    main()
