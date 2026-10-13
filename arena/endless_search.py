"""CMA-ES over a rule bot's genes, scored by leader share in endless Rust-engine worlds (arena.endless).

    uv run python -m arena.endless_search --run es1 --init bot/c3_4.json --field champ=runs/final_champion.json:2 ...

Each candidate plays (2 copies) in `--worlds` endless worlds of `--game-min` game-minutes against the fixed field; the
generation's candidates share the world seeds (common random numbers). Fitness = share of mass samples in which the
#1 player of the leaderboard is one of the candidate's copies (the user's goal: ~80% on the Rust server). State per
generation in runs/<run>/gen_###.json, the best candidate so far in runs/<run>/best.json.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="Could not import matplotlib")
import cma  # noqa: E402

from .params import BOUNDS, Params, from_unit, to_unit

from .endless import run_world
from .util import limit_cpus

GENES = ["danger_margin", "threat_force", "flee_margin", "farm_threat", "hunt_threat", "flee_threat", "split_safety",
         "hunt_range", "min_prey_frac", "food_pull", "target_bonus", "virus_avoid", "farm_prey", "split_reach",
         "caution_mass", "flee_margin_big", "split_safety_big", "wall_danger", "wall_wary", "wall_wary_ratio",
         "split_max_cells", "big_hunt", "big_mass", "shoot_enable", "hide_enable", "double_split_enable"]


def leader_share(samples: list[dict], name: str) -> float:
    n = k = 0
    for rec in samples:
        if rec["p"]:
            n += 1
            k += max(rec["p"], key=lambda pl: pl["m"])["n"].rsplit("-", 1)[0] == name
    return k / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--field", nargs="+", required=True)
    p.add_argument("--genes", nargs="+", default=GENES)
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--worlds", type=int, default=12)
    p.add_argument("--game-min", type=float, default=120.0)
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--cpus", type=int, default=11)
    a = p.parse_args()
    limit_cpus(a.cpus)
    root = Path("runs") / a.run
    root.mkdir(parents=True, exist_ok=True)
    init = json.loads(Path(a.init).read_text())
    names = list(BOUNDS)
    unit = to_unit(Params.from_dict(init["params"]))
    idx = [names.index(g) for g in a.genes]
    es = cma.CMAEvolutionStrategy([unit[i] for i in idx], a.sigma,
                                  {"popsize": a.popsize, "bounds": [0, 1], "seed": 1, "verbose": -9})
    best = {"fitness": -1.0}
    for gen in range(a.gens):
        if es.stop():
            break
        t0 = time.time()
        xs = es.ask()
        files = []
        for c, x in enumerate(xs):
            u = list(unit)
            for k, i in enumerate(idx):
                u[i] = float(x[k])
            spec = {**init, "params": {**init["params"], **{g: from_unit(u).to_dict()[g] for g in a.genes}}}
            f = root / f"cand_{gen:03d}_{c}.json"
            f.write_text(json.dumps(spec, indent=1))
            files.append(f)
        seeds = [50_000 + 1000 * gen + w for w in range(a.worlds)]
        jobs = [(c, s) for c in range(len(xs)) for s in seeds]
        with ThreadPoolExecutor(a.threads) as ex:
            res = list(ex.map(lambda j: leader_share(run_world([f"cand{j[0]}={files[j[0]]}:2"] + a.field, j[1],
                                                               a.game_min, 15.0, 2.0), f"cand{j[0]}"), jobs))
        fit = [sum(res[c * len(seeds):(c + 1) * len(seeds)]) / len(seeds) for c in range(len(xs))]
        es.tell(xs, [-f for f in fit])
        for c, f in enumerate(fit):
            if f > best["fitness"]:
                best = {"fitness": f, "gen": gen, "cand": c}
                (root / "best.json").write_text(files[c].read_text())
        (root / f"gen_{gen:03d}.json").write_text(json.dumps({"gen": gen, "fitness": fit, "best": best,
                                                              "mean": dict(zip(a.genes, map(float, es.mean)))}, indent=1))
        print(f"gen {gen}: leader share " + " ".join(f"{f:.0%}" for f in fit)
              + f"  (best {best['fitness']:.0%}, gen {best['gen']} cand {best['cand']})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
