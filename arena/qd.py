"""Quality-diversity evolution (MAP-Elites) over bot Params, to encourage creative tactics.

    uv run python -m arena.qd --run q1 --init runs/final_champion.json --food-mult 2

Plain CMA-ES keeps only the single best genome, so a new tactic that costs mass while
its thresholds are still bad gets bred out at once. Here the archive keeps the best
bot of every *playstyle* (niche), so new ways of playing survive and get refined:

  niche = (eject rate, split rate, hunting share), 3 bins each = 27 playstyles,
          measured from what the bot actually did in its matches (policy counters).

Each round:
- pick parents from filled niches (rarely tried niches first: curiosity),
- mutate: Gaussian on the normalised genome, plus macro-mutations that flip a whole
  tactic on/off (shoot, hide, self-feed, double split) or re-draw one gene at random,
- evaluate children in 15-minute matches (fresh seeds) together with the champion,
  a hunter and two archive elites (whose results also refine their running averages),
- fitness = mean log(average mass); insert a child if its niche is empty or it beats
  the niche's elite.
Every --confirm-every rounds the best elite duels the champion on fresh seeds; a
significant win makes it the new champion (runs/<run>/champion_###.json).
Saturation: no confirmed champion, no fitter elite and no new niche for --patience rounds.
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

import numpy as np

from bot.params import BOUNDS, Params, from_unit, to_unit

from .match import free_port, run_match, start_server
from .policies import hunter_spec

ROOT = Path(__file__).resolve().parent.parent
SWITCHES = ["shoot_enable", "hide_enable", "selffeed_enable", "double_split_enable"]
KEYS = list(BOUNDS)
SYNC = 2


def play(lineup, seed, ticks, food_mult):
    """One match; {name: (avg_mass, counters, alive_steps)}, or {} on failure."""
    port = free_port()
    server = start_server(port, 8, seed, len(lineup), SYNC, 7000, food_mult=food_mult)
    try:
        stats = asyncio.run(run_match(lineup, ticks, port, SYNC))
    except Exception:
        print(f"match seed={seed} failed:\n{traceback.format_exc()}", flush=True)
        return {}
    finally:
        server.terminate()
    return {s.name: (max(s.avg_mass, 1.0), s.counters, s.steps_alive) for s in stats}


def descriptor(counters: dict, alive_steps: int) -> tuple[int, int, int]:
    minutes = max(alive_steps * SYNC * 0.04 / 60, 0.1)
    ejects = counters.get("ejects", 0) / minutes
    splits = counters.get("splits", 0) / minutes
    hunt = counters.get("mode_hunt", 0) / max(counters.get("steps", 0), 1)
    b = lambda v, lo, hi: 0 if v < lo else 1 if v < hi else 2
    return b(ejects, 0.2, 3), b(splits, 0.5, 3), b(hunt, 0.15, 0.4)


def spec(unit) -> dict:
    return {"farm": "route", "params": from_unit(unit).to_dict()}


class Archive:
    def __init__(self, path: Path):
        self.path = path
        self.cells: dict[str, dict] = {}  # niche key -> {unit, scores, tries}

    def key(self, d) -> str:
        return "".join(map(str, d))

    def fitness(self, e) -> float:
        # Mean minus a small uncertainty penalty: a single lucky match can't hold a niche.
        s = e["scores"]
        return statistics.fmean(s) - (0.3 / math.sqrt(len(s)))

    def offer(self, unit, score: float, d) -> str:
        k = self.key(d)
        cur = self.cells.get(k)
        cand = {"unit": list(map(float, unit)), "scores": [score], "tries": 0}
        if cur is None:
            self.cells[k] = cand
            return "new niche"
        if self.fitness(cand) > self.fitness(cur):
            cand["tries"] = cur["tries"]
            self.cells[k] = cand
            return "improved"
        return ""

    def best(self, min_evals: int = 1):
        ok = [kv for kv in self.cells.items() if len(kv[1]["scores"]) >= min_evals]
        return max(ok, key=lambda kv: self.fitness(kv[1])) if ok else None

    def load(self) -> None:
        if self.path.exists():
            self.cells = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.cells, indent=1))


class QD:
    def __init__(self, args):
        self.a = args
        self.dir = ROOT / "runs" / args.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(args.seed)
        self.np_rng = np.random.default_rng(args.seed)
        self.pool = ProcessPoolExecutor(args.workers)
        self.champion = json.loads(Path(args.init).read_text())
        self.archive = Archive(self.dir / "archive.json")
        self.seed_no = 0
        self.first_round = 1
        if args.resume:  # continue from the saved archive and latest champion
            self.archive.load()
            champs = sorted(self.dir.glob("champion_[0-9]*.json"))
            if champs:
                self.champion = json.loads(champs[-1].read_text())
            self.first_round = args.resume
            self.seed_no = 1000 * args.resume  # fresh seeds, never reuse earlier ones

    def _seed(self) -> int:
        self.seed_no += 1
        return 1_000_000 * self.a.seed + self.seed_no

    def _run(self, jobs):
        futs = [self.pool.submit(play, lineup, seed, self.a.ticks, self.a.food_mult)
                for lineup, seed in jobs]
        return [f.result() for f in futs]

    def mutate(self, unit) -> list[float]:
        x = np.array(unit) + self.np_rng.normal(0, self.a.sigma, len(unit))
        if self.rng.random() < self.a.macro:  # macro-mutation: change *what* it does
            if self.rng.random() < 0.6:
                i = KEYS.index(self.rng.choice(SWITCHES))
                x[i] = 0.75 if x[i] <= 0.5 else 0.25  # flip a tactic on/off
            else:
                x[self.rng.randrange(len(x))] = self.rng.random()  # re-draw one gene
        return list(np.clip(x, 0, 1))

    def parents(self, n: int) -> list[tuple[str, dict]]:
        cells = list(self.archive.cells.items())
        cells.sort(key=lambda kv: (kv[1]["tries"], self.rng.random()))  # curiosity first
        picks = [cells[i % len(cells)] for i in range(n)]
        for _, e in picks:
            e["tries"] += 1
        return picks

    def evaluate_round(self, children: list[list[float]]):
        """4 children per match + champion + hunter + 2 archive elites; returns child results."""
        results = {i: [] for i in range(len(children))}
        elite_keys = list(self.archive.cells)
        jobs, metas = [], []
        for rep in range(self.a.evals):
            order = list(range(len(children)))
            self.rng.shuffle(order)
            for m in range(0, len(order), 4):
                group = order[m:m + 4]
                lineup = [(f"kid{i}-{k}", spec(children[i])) for k, i in enumerate(group)]
                lineup += [("champ-0", self.champion), ("hunter-0", hunter_spec(self.champion))]
                elites = self.rng.sample(elite_keys, min(2, len(elite_keys)))
                lineup += [(f"e{k}-{j}", spec(self.archive.cells[k]["unit"])) for j, k in enumerate(elites)]
                jobs.append((lineup, self._seed()))
                metas.append(elites)
        for res, elites in zip(self._run(jobs), metas):
            for name, (mass, counters, alive) in res.items():
                if name.startswith("kid"):
                    results[int(name[3:].split("-")[0])].append((math.log(mass), counters, alive))
                elif name.startswith("e"):
                    k = name[1:].rsplit("-", 1)[0]
                    if k in self.archive.cells:  # refine the elite's running average
                        self.archive.cells[k]["scores"].append(math.log(mass))
        return results

    def confirm(self, unit) -> tuple[float, float, float]:
        cand = spec(unit)
        lineup = [(f"ch-{k}", cand) for k in range(4)] + [(f"in-{k}", self.champion) for k in range(4)]
        diffs = []
        for res in self._run([(lineup, self._seed()) for _ in range(self.a.confirm_matches)]):
            if not res:
                continue
            ch = statistics.fmean(math.log(v[0]) for k, v in res.items() if k.startswith("ch"))
            inc = statistics.fmean(math.log(v[0]) for k, v in res.items() if k.startswith("in"))
            diffs.append(ch - inc)
        m = statistics.fmean(diffs)
        half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
        return m, half, sum(d > 0 for d in diffs) / len(diffs)

    def run(self) -> None:
        a = self.a
        start = to_unit(Params.from_dict(self.champion["params"]))
        # Seed the archive: the champion plus one variant per tactic switched on.
        seeds = [start] + [self.mutate(start) for _ in range(a.batch - 1)]
        for idx, sw in enumerate(SWITCHES, start=1):
            if idx < len(seeds):
                seeds[idx][KEYS.index(sw)] = 0.75
        stale, best_fit = 0, -1e9
        for rnd in range(self.first_round, a.max_rounds + 1):
            t0 = time.monotonic()
            if rnd == 1 and not self.archive.cells:
                children = seeds
            else:
                children = [self.mutate(e["unit"]) for _, e in self.parents(a.batch)]
            results = self.evaluate_round(children)
            events = []
            for i, rs in results.items():
                if not rs:
                    continue
                score = statistics.fmean(r[0] for r in rs)
                counters = {}
                for r in rs:
                    for k, v in r[1].items():
                        counters[k] = counters.get(k, 0) + v
                d = descriptor(counters, sum(r[2] for r in rs))
                ev = self.archive.offer(children[i], score, d)
                if ev:
                    events.append(f"{ev} {''.join(map(str, d))} ({score:.2f})")
            key, elite = self.archive.best()
            fit = self.archive.fitness(elite)
            improved = fit > best_fit + 0.02 or any(e.startswith("new niche") for e in events)
            best_fit = max(best_fit, fit)
            line = (f"round {rnd:3d} niches {len(self.archive.cells):2d}/27 best {key} fit {fit:.2f} "
                    f"({len(elite['scores'])} evals) | {'; '.join(events) or 'no change'}")
            promoted = False
            # Only elites with enough matches may challenge: a single lucky match is not evidence.
            proven = self.archive.best(a.min_evals)
            if rnd % a.confirm_every == 0 and proven:
                pkey, pelite = proven
                m, half, win = self.confirm(pelite["unit"])
                promoted = m - half > 0
                line += (f" | confirm {pkey} ({len(pelite['scores'])} evals) vs champion "
                         f"{m:+.3f}+-{half:.3f} win {win:.0%}")
                if promoted:
                    self.champion = spec(pelite["unit"])
                    (self.dir / f"champion_{rnd:03d}.json").write_text(json.dumps(self.champion, indent=1))
                    line += " PROMOTED"
            stale = 0 if (improved or promoted) else stale + 1
            self.archive.save()
            print(f"{line}  ({time.monotonic() - t0:.0f}s)", flush=True)
            if stale >= a.patience:
                print(f"saturated: no new niche, fitter elite or champion for {a.patience} rounds")
                break
        (self.dir / "champion_final.json").write_text(json.dumps(self.champion, indent=1))
        self.pool.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="q1")
    p.add_argument("--init", default=str(ROOT / "runs" / "final_champion.json"))
    p.add_argument("--batch", type=int, default=8, help="children per round")
    p.add_argument("--evals", type=int, default=2, help="matches per child per round")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--macro", type=float, default=0.35, help="chance of a macro-mutation")
    p.add_argument("--confirm-every", type=int, default=3)
    p.add_argument("--confirm-matches", type=int, default=12)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-evals", type=int, default=4, help="matches an elite needs to challenge")
    p.add_argument("--resume", type=int, default=0, help="continue from the saved archive at this round")
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--ticks", type=int, default=22500, help="15 game-minutes")
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    QD(p.parse_args()).run()


if __name__ == "__main__":
    main()
