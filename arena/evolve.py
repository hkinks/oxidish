"""Evolve bot Params with CMA-ES until improvement saturates.

    uv run python -m arena.evolve --run r1 [--popsize 16] [--patience 6]

Each generation:
1. Training: CMA-ES samples `popsize` genomes. Each plays `--train-matches`
   matches as 4 candidates vs 4 opponents (route + field baselines and
   hall-of-fame champions) plus the server's built-in bots. Fitness is the mean
   of log(avg mass), where avg mass is the mass-time integral / match time. The
   log makes it a geometric mean: steady play beats rare jackpots.
2. Validation: CMA's current mean genome ("challenger") plays the incumbent
   champion 4v4 on fixed validation seeds. Score per match = challenger team mean
   log mass - incumbent team mean log mass, a paired comparison on identical
   matches. If the 95% CI lower bound is > 0, the challenger becomes champion
   and joins the hall of fame.
3. Saturation: stop after `--patience` generations without a new champion.

Outputs go to runs/<run>/: gen_###.json logs and champion_###.json policy specs
(usable as `--agents champ=runs/<run>/champion_###.json:4`).
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="Could not import matplotlib")
import cma  # noqa: E402

from bot.params import DEFAULT, Params, from_unit, to_unit

from .match import free_port, run_match, start_server
from .policies import hunter_spec

ROOT = Path(__file__).resolve().parent.parent
BASE_ROUTE = {"farm": "route"}
BASE_FIELD = {"farm": "field"}


def play(lineup: list[tuple[str, dict]], seed: int, ticks: int, bots: int, map_size: int,
         sync: int, food_mult: float = 1.0) -> dict[str, float]:
    """Run one match in a fresh arena; returns {agent name: avg mass}, or {} if it failed."""
    port = free_port()
    server = start_server(port, bots, seed, len(lineup), sync, map_size, food_mult=food_mult)
    try:
        stats = asyncio.run(run_match(lineup, ticks, port, sync))
    except Exception:  # one broken match must not kill a long run; it is logged and skipped
        print(f"match seed={seed} failed:\n{traceback.format_exc()}", flush=True)
        return {}
    finally:
        server.terminate()
    return {s.name: max(s.avg_mass, 1.0) for s in stats}


def spec(unit) -> dict:
    return {"farm": "route", "params": from_unit(unit).to_dict()}


def ci95(xs: list[float]) -> tuple[float, float]:
    m = statistics.fmean(xs)
    return m, (1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf"))


class Evolution:
    def __init__(self, args):
        self.a = args
        self.dir = ROOT / "runs" / args.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = ProcessPoolExecutor(args.workers)
        # gen 0 champion: the hand-tuned defaults, or a saved genome to continue from
        self.champion = (json.loads(Path(args.init).read_text()) if args.init
                         else {"farm": "route", "params": DEFAULT.to_dict()})
        self.hall: list[dict] = [{"farm": "route", "params": DEFAULT.to_dict()}]
        for path in args.hall or []:  # earlier champions to seed the opponent league
            self.hall.append(json.loads(Path(path).read_text()))
        if args.init:
            self.hall.append(self.champion)
        start = to_unit(Params.from_dict(self.champion["params"]))
        self.es = cma.CMAEvolutionStrategy(start, args.sigma, {
            "popsize": args.popsize, "bounds": [0, 1], "seed": args.seed, "verbose": -9})
        self.rng = random.Random(args.seed)

    def _run(self, jobs):
        a = self.a
        futures = [self.pool.submit(play, lineup, seed, a.ticks, a.bots, a.map, a.sync, a.food_mult)
                   for lineup, seed in jobs]
        return [f.result() for f in futures]

    def train(self, gen: int) -> tuple[list, list[float]]:
        xs = self.es.ask()
        slots = [i for i in range(len(xs)) for _ in range(self.a.train_matches)]
        self.rng.shuffle(slots)
        jobs = []
        for m in range(0, len(slots), 4):
            group = slots[m:m + 4]
            if self.a.league:  # strong opponents: recent champions, one baseline as anchor
                recent = self.hall[-5:]
                opponents = [BASE_ROUTE] + [self.rng.choice(recent) for _ in range(3)]
                if self.a.hunters:  # aggressive, human-like sparring partners
                    opponents[1] = hunter_spec(self.champion)
            else:
                opponents = [BASE_ROUTE, BASE_FIELD] + self.rng.sample(self.hall, min(2, len(self.hall)))
            while len(opponents) < 4:
                opponents.append(BASE_ROUTE)
            lineup = [(f"c{i}-{k}", spec(xs[i])) for k, i in enumerate(group)]
            lineup += [(f"opp-{k}", o) for k, o in enumerate(opponents)]
            jobs.append((lineup, 100_000 * gen + m))
        scores: dict[int, list[float]] = {i: [] for i in range(len(xs))}
        for result in self._run(jobs):
            for name, mass in result.items():
                if name.startswith("c"):
                    scores[int(name[1:].split("-")[0])].append(math.log(mass))
        floor = min((v for vs in scores.values() for v in vs), default=0.0)
        fitness = [statistics.fmean(scores[i]) if scores[i] else floor for i in range(len(xs))]
        self.es.tell(xs, [-f for f in fitness])  # CMA minimises
        return xs, fitness

    def duel(self, challenger: dict, incumbent: dict, seeds) -> tuple[float, float, float]:
        """Paired 4v4 duel; returns (mean log-mass advantage, CI half width, win rate)."""
        lineup = ([(f"ch-{k}", challenger) for k in range(4)] +
                  [(f"in-{k}", incumbent) for k in range(4)])
        diffs = []
        for result in self._run([(lineup, s) for s in seeds]):
            if not result:
                continue
            ch = statistics.fmean(math.log(v) for k, v in result.items() if k.startswith("ch"))
            inc = statistics.fmean(math.log(v) for k, v in result.items() if k.startswith("in"))
            diffs.append(ch - inc)
        mean, half = ci95(diffs)
        return mean, half, sum(d > 0 for d in diffs) / len(diffs)

    def run(self) -> None:
        a = self.a
        stale = 0
        for gen in range(1, a.max_gens + 1):
            t0 = time.monotonic()
            _, fitness = self.train(gen)
            challenger = spec(self.es.mean)
            # Fresh validation seeds every generation: re-using one fixed set let lucky
            # challengers through (winner's curse) in earlier runs.
            val_seeds = [900_000 + 1000 * gen + i for i in range(a.val_matches)]
            adv, half, win = self.duel(challenger, self.champion, val_seeds)
            promoted = adv - half > 0
            if promoted:
                self.champion = challenger
                self.hall.append(challenger)
                (self.dir / f"champion_{gen:03d}.json").write_text(json.dumps(challenger, indent=1))
                stale = 0
            else:
                stale += 1
            log = {"gen": gen, "fitness": {"best": max(fitness), "mean": statistics.fmean(fitness)},
                   "sigma": self.es.sigma, "challenger_vs_champion": {
                       "log_mass_adv": adv, "ci95": half, "win_rate": win, "promoted": promoted},
                   "challenger": challenger, "wall_s": time.monotonic() - t0}
            (self.dir / f"gen_{gen:03d}.json").write_text(json.dumps(log, indent=1))
            print(f"gen {gen:3d}  fit best {max(fitness):5.2f} mean {statistics.fmean(fitness):5.2f}"
                  f"  sigma {self.es.sigma:.3f}  duel {adv:+.3f}+-{half:.3f} win {win:4.0%}"
                  f"  {'PROMOTED' if promoted else f'stale {stale}'}"
                  f"  ({log['wall_s']:.0f}s)", flush=True)
            if stale >= a.patience:
                print(f"saturated: no significant improvement for {a.patience} generations")
                break
        (self.dir / "champion_final.json").write_text(json.dumps(self.champion, indent=1))
        self.pool.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="r1")
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--train-matches", type=int, default=3, help="matches per candidate")
    p.add_argument("--val-matches", type=int, default=18)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-gens", type=int, default=80)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--ticks", type=int, default=4500, help="3 game-minutes")
    p.add_argument("--bots", type=int, default=8)
    p.add_argument("--map", type=int, default=7000)
    p.add_argument("--sync", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", help="champion json to start from (continue a previous run)")
    p.add_argument("--hall", nargs="*", help="earlier champion jsons added to the opponent pool")
    p.add_argument("--food-mult", type=float, default=1.0, help="food density (showcase uses 2)")
    p.add_argument("--hunters", action="store_true", help="put an aggressive hunter in every league match")
    p.add_argument("--league", action="store_true",
                   help="train against 3 recent champions + 1 baseline instead of 2 baselines + 2 champions")
    Evolution(p.parse_args()).run()


if __name__ == "__main__":
    main()
