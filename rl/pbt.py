"""Continuous evolution of a population in ONE process and ONE simulator (population-based training).

    uv run python -m rl.pbt --name p1 --from-evo e1 [--b 12 --skip 2 --eval-minutes 20]
    uv run python -m rl.pbt --name p1 --seed runs/rl/s1/latest.pt --pop 4

Every member (a network + its genome, see rl.evo) owns `--b` arenas of one shared SimEnv and
learns there with rl.ppo_sim.Learner: the members' arenas step together, so a simulator step
costs the same for 4 members as for one (it is ~100 small GPU jobs whose cost barely depends
on the batch). Members decide every `--skip` env steps.

Evaluation never stops training: every `--eval-minutes` the members are snapshotted and a side
process (`--eval-job`) plays them against each other and the newest hall-of-fame net in shared
50-player arenas (sim.tourney: fitness = mean log of average mass), plus the farming-alone and
10-minute checks. When the results come back:
  - the best member is copied to the hall of fame (runs/rl/<name>/hof/), which every member's
    league plays against;
  - the bottom half is replaced by mutated copies (rl.evo.mutate) of the top half of the
    ranking, which includes the hall-of-fame net: a member that regressed below it is replaced
    by a copy of the better old net instead of carrying on from its worse weights;
  - a child with its parent's architecture starts from the parent's evaluated weights, one with
    a new architecture first imitates the parent, and is protected from the next selection.
Replaced members' arenas restart with the child's drills and rewards.

State: runs/rl/<name>/evo.json (the rl.evo layout, so sim.watch shows the population; "gen"
counts evaluation rounds). Logs: logs/pbt-<name>.txt (events, results),
logs/pbt-<name>-<member>.txt (training lines), TensorBoard <name>/<member>.
"""

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from rl.common import ROOT
import math

from rl.evo import DEFAULT_ARCH, DEFAULT_HP, FARM_HP_KINDS, HP_KINDS, mutate
from rl.ppo_sim import Learner, collect, from_checkpoint, frozen, new_model, setup_cuda
from sim.env import SimEnv

FIXED_HP = dict(league_frac=0.3, bot_frac=0.1, league_models=2, league_every=25, league_size=8,
                save_every=20, vf_warmup=0, lam=0.95, clip=0.2, target_kl=0.02, vf=0.5)
# --objective farm: farming arenas only (every slot a learner), mass is the only reward; fitness is
# the free farming check (alone for 2 min, split / eject allowed: whatever farms fastest)
FARM_HP = dict(farm_frac=1.0, hunt_frac=0.0, flee_frac=0.0, giant_frac=0.0, league_frac=0.0, bot_frac=0.0,
               kill_reward=[0.0, 0.0], virus_reward=[0.0, 0.0, 0.0], growth=None, action_bonus=[0.0, 0.0])


def log(msg: str):
    print(msg, flush=True)


def eval_job(path: str) -> None:
    """Side process: tournament + checks on frozen snapshots; writes the JSON result."""
    from sim.tourney import farm, run
    job = json.loads(Path(path).read_text())
    setup_cuda(job.get("vram_gb", 5.0))
    nets = job["nets"]
    mem = {k: v for k, v in nets.items() if k != "hof_last"}
    if job.get("objective") == "farm":
        # fitness: both farming tests (split / eject allowed), each relative to its routine:
        # from 10 mass (routines ~300) and the gain from 1000 mass (routine +374; split-farming +467)
        small = farm(nets, B=64, free=True)
        big = farm(nets, B=64, free=True, start=1000.0, gain=True)
        res = {k: {"log_mass": math.log(max((small[k] / 300 + max(big[k], 1.0) / 374) / 2, 1e-3)), "ci": 0.0,
                   "avg_mass": small[k], "gain_1000": big[k], "deaths": 0.0, "kills": 0.0} for k in nets}
        Path(job["out"]).write_text(json.dumps({"results": res, "farm": small, "long": {}}, indent=1))
        return
    res = run(nets, B=job["eval_b"], P=job["p"], steps=job["steps"], seed=job["seed"])
    out = {"results": res, "farm": farm(mem)}
    out["long"] = {k: v["avg_mass"] for k, v in run(mem, B=4, P=job["p"], steps=job["long_steps"],
                                                     seed=job["seed"] + 2000).items()}
    Path(job["out"]).write_text(json.dumps(out, indent=1))


class Population:
    def __init__(self, args):
        self.args = args
        self.root = ROOT / "runs" / "rl" / args.name
        (self.root / "hof").mkdir(parents=True, exist_ok=True)
        (self.root / "eval").mkdir(exist_ok=True)
        self.state_f = self.root / "evo.json"
        self.rng = random.Random(args.rng)
        self.state = self.load_state()
        self.dev = "cuda"
        pop = len(self.state["members"])
        kinds = {m["genome"]["arch"].get("kind", "grid") for m in self.state["members"]}
        self.env = SimEnv(B=pop * args.b, P=args.p, seed=args.rng, episode_steps=20000, curriculum_p=args.curriculum_p,
                          curriculum_mass=tuple(float(x) for x in args.curriculum_mass.split(",")),
                          obs="both" if len(kinds) > 1 or args.hof_extra else kinds.pop(),
                          obs_dtype=torch.float16, skip=args.skip)
        self.hof_files = [f for d in [self.root / "hof", *map(Path, args.hof_extra)] for f in sorted(d.glob("*.pt"))]
        self.hof = [frozen(from_checkpoint(str(f), self.dev)) for f in self.hof_files]
        self.learners = [self.make_learner(m, k) for k, m in enumerate(self.state["members"])]
        self.eval_proc, self.eval_t = None, time.time()

    # ---- state
    def load_state(self) -> dict:
        a = self.args
        if self.state_f.exists():
            st = json.loads(self.state_f.read_text())
            log(f"resuming {a.name} at round {st['gen']}")
            return st
        if a.from_evo:      # continue an rl.evo population: its members, genomes and weights
            src = json.loads((ROOT / "runs" / "rl" / a.from_evo / "evo.json").read_text())
            members = []
            for m in src["members"]:
                ck = ROOT / "runs" / "rl" / a.from_evo / m["id"] / "latest.pt"
                members.append({**m, "ckpt": str(ck if ck.exists() else m["ckpt"]), "history": [],
                                "origin": f"{a.from_evo}/{m['id']}: {m['origin']}"})
                members[-1].pop("new_arch_gen", None)
            st = {"gen": 0, "members": members, "next_id": src["next_id"], "log": [], "hof": []}
        else:
            if not a.seed:
                raise SystemExit("--seed or --from-evo is required for a new population")
            base = {"arch": json.loads(a.arch) if a.arch else DEFAULT_ARCH, "hp": DEFAULT_HP}
            members = []
            kinds = FARM_HP_KINDS if a.objective == "farm" else HP_KINDS
            for k in range(a.pop):
                g, note = (copy.deepcopy(base), "seed") if k == 0 else mutate(base, self.rng, hp_kinds=kinds)
                members.append({"id": f"m{k}", "genome": g, "ckpt": a.seed, "it": 0, "seed": 100 + k,
                                "origin": note, "history": []})
            st = {"gen": 0, "members": members, "next_id": a.pop, "log": [], "hof": []}
        self.state_f.write_text(json.dumps(st, indent=1))
        return st

    def save_state(self):
        for m, L in zip(self.state["members"], self.learners):
            m["it"] = L.it
            if (L.out / "latest.pt").exists():
                m["ckpt"] = str(L.out / "latest.pt")
        self.state_f.write_text(json.dumps(self.state, indent=1))

    # ---- members
    def make_learner(self, m: dict, k: int) -> Learner:
        a = self.args
        hp = {**DEFAULT_HP, **m["genome"]["hp"], **FIXED_HP, "mb": a.mb}
        if a.objective == "farm":
            hp.update(FARM_HP, farm_players=a.p)
        model = new_model(**m["genome"]["arch"]).to(self.dev)
        parent = from_checkpoint(m["ckpt"], self.dev)
        teacher = None
        if parent.config == model.config:
            model = parent
        else:
            teacher = frozen(parent)
        logf = open(ROOT / "logs" / f"pbt-{a.name}-{m['id']}.txt", "a", encoding="utf-8")
        L = Learner(f"{a.name}/{m['id']}", model, self.env, k * a.b, a.b, hp, self.root / m["id"], hof=self.hof,
                    teacher=teacher, distill_iters=30 if teacher is not None else 0, start_it=m["it"], log=logf,
                    skip=a.skip)
        L.print(f"--- {m['id']} in arenas {k * a.b}-{(k + 1) * a.b - 1}: {m['origin']}  genome {m['genome']}"
                f"{'  (distilling a new architecture)' if teacher is not None else ''}")
        return L

    # ---- evaluation
    def launch_eval(self):
        a, rnd = self.args, self.state["gen"]
        nets = {}
        for m, L in zip(self.state["members"], self.learners):
            f = self.root / "eval" / f"g{rnd:03d}_{m['id']}.pt"
            L.save(f)
            nets[m["id"]] = str(f)
        hof_files = sorted((self.root / "hof").glob("*.pt")) or self.hof_files
        if hof_files:
            nets["hof_last"] = str(hof_files[-1])
        self.eval_nets = nets
        job = {"nets": nets, "eval_b": a.eval_b, "p": a.p, "steps": a.eval_steps, "seed": 7000 + rnd,
               "long_steps": a.long_steps, "out": str(self.root / "eval" / f"g{rnd:03d}.json"), "vram_gb": 5.0,
               "objective": a.objective}
        jf = self.root / "eval" / f"g{rnd:03d}_job.json"
        jf.write_text(json.dumps(job))
        self.eval_out = Path(job["out"])
        self.eval_log = open(ROOT / "logs" / f"pbt-{a.name}-eval.txt", "a", encoding="utf-8")
        self.eval_proc = subprocess.Popen([sys.executable, "-m", "rl.pbt", "--eval-job", str(jf)], cwd=ROOT,
                                          stdout=self.eval_log, stderr=subprocess.STDOUT)
        self.save_state()
        log(f"\n=== round {rnd}: evaluating {', '.join(nets)} (training continues)")

    def poll_eval(self):
        if self.eval_proc is None:
            if time.time() - self.eval_t > self.args.eval_minutes * 60:
                self.launch_eval()
            return
        if self.eval_proc.poll() is None:
            return
        self.eval_proc, self.eval_t = None, time.time()
        self.eval_log.close()
        if not self.eval_out.exists():
            log(f"  evaluation failed: see logs/pbt-{self.args.name}-eval.txt")
            return
        self.select(json.loads(self.eval_out.read_text()))

    def select(self, ev: dict):
        rnd, res = self.state["gen"], ev["results"]
        members = self.state["members"]
        log(f"--- round {rnd} results" + (" (fitness: free farming, 2 min alone, split / eject allowed)"
                                          if self.args.objective == "farm" else ""))
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]):
            if "gain_1000" in r:
                log(f"  {name:<9} fitness {r['log_mass']:+.3f}  from 10 -> {r['avg_mass']:.0f}  "
                    f"from 1000: {r['gain_1000']:+.0f}  (routines ~300 / +374, split-farming +467)")
                continue
            log(f"  {name:<9} log_mass {r['log_mass']:.3f} +-{r['ci']:.3f}  mass {r['avg_mass']:.0f}  "
                f"deaths {r['deaths']:.2f}  kills {r['kills']:.2f}  splits/min {r.get('splits_per_min', 0):.2f}")
        if self.args.objective != "farm":
            log("  farming alone, mass after 2 min (routines ~300): " +
                "  ".join(f"{k} {v:.0f}" for k, v in sorted(ev["farm"].items(), key=lambda kv: -kv[1])))
        if ev["long"]:
            log(f"  long game ({self.args.long_steps * 0.08 / 60:.0f} min from mass 10), avg mass: " +
                "  ".join(f"{k} {v:.0f}" for k, v in sorted(ev["long"].items(), key=lambda kv: -kv[1])))
        ids = {m["id"] for m in members}
        ranking = [k for k, _ in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]) if k in ids or k == "hof_last"]
        ranked = [next(m for m in members if m["id"] == k) for k in ranking if k in ids]
        for m in ranked:
            m["history"].append({"gen": rnd, **res[m["id"]], "farm": ev["farm"].get(m["id"]),
                                 "long_mass": ev["long"].get(m["id"])})
        best = ranked[0]
        hof_f = self.root / "hof" / f"g{rnd:03d}_{best['id']}.pt"
        shutil.copy(self.eval_nets[best["id"]], hof_f)
        self.state["hof"].append({"file": str(hof_f), "genome": best["genome"], "from": best["id"]})
        self.hof.append(frozen(from_checkpoint(str(hof_f), self.dev)))
        self.state["log"].append({"gen": rnd, "results": res, "farm": ev["farm"], "long": ev["long"],
                                  "best": best["id"], "time": time.time()})
        # parents: the top half of the ranking, the last hall-of-fame net included (if its genome is known)
        n_keep = max(1, len(members) // 2)
        grace = [m for m in ranked if m.get("new_arch_gen") == rnd]
        losers = [m for m in ranked[n_keep:] if m not in grace]
        hof_prev = self.state["hof"][-2] if len(self.state["hof"]) > 1 else None
        parents = []
        for k in ranking:
            if len(parents) == n_keep:
                break
            if k == "hof_last":
                if hof_prev and Path(hof_prev["file"]) == Path(self.eval_nets["hof_last"]):
                    parents.append({"id": f"hof {Path(hof_prev['file']).stem}", "genome": hof_prev["genome"],
                                    "ckpt": hof_prev["file"]})
            elif all(k != m["id"] for m in losers):
                m = next(m for m in members if m["id"] == k)
                parents.append({"id": k, "genome": m["genome"], "ckpt": self.eval_nets[k], "it": m["it"]})
        for m in grace:
            log(f"  {m['id']} (new architecture) protected this round")
        for j, dead in enumerate(losers):
            par = parents[j % len(parents)]
            g, note = mutate(par["genome"], self.rng,
                             hp_kinds=FARM_HP_KINDS if self.args.objective == "farm" else HP_KINDS)
            cid = f"m{self.state['next_id']}"
            self.state["next_id"] += 1
            same = g["arch"] == par["genome"]["arch"]
            child = {"id": cid, "genome": g, "ckpt": par["ckpt"], "it": par.get("it", 0) if same else 0,
                     "seed": self.rng.randrange(10 ** 6), "origin": f"r{rnd} child of {par['id']}: {note}",
                     "history": [], **({} if same else {"new_arch_gen": rnd + 1})}
            k = members.index(dead)
            old = self.learners[k]
            old.save(old.out / "latest.pt")
            old.tb.close()
            old.log_f.close()
            members[k] = child
            self.learners[k] = self.make_learner(child, k)
            mask = torch.zeros(self.env.B, dtype=torch.bool, device=self.dev)
            mask[k * self.args.b:(k + 1) * self.args.b] = True
            self.env.reset_arenas(mask)
            log(f"  {dead['id']} out -> {cid} = {par['id']} + {note}")
        self.state["gen"] = rnd + 1
        self.save_state()

    # ---- main loop
    def run(self):
        a = self.args
        g, s, _, _, v = self.env.reset()
        mem = self.state["members"]
        log(f"PBT {a.name}: {len(mem)} members x {a.b} arenas x {a.p} slots in one simulator, skip {a.skip}, "
            f"{a.steps} decisions/iter, evaluation every {a.eval_minutes} min")
        for m, L in zip(mem, self.learners):
            log(f"  {m['id']}: {L.N} learners  {m['origin']}")
        t_start, n_it = time.time(), 0
        while not a.max_minutes or time.time() - t_start < a.max_minutes * 60:
            g, s, v = collect(self.env, self.learners, (g, s, v), a.steps, a.skip)
            for L in self.learners:
                L.update()
            n_it += 1
            if n_it % 20 == 0:
                self.save_state()
            self.poll_eval()
        for L in self.learners:
            L.save(L.out / "latest.pt")
        self.save_state()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-job", help=argparse.SUPPRESS)
    p.add_argument("--name", default="p1")
    p.add_argument("--from-evo", help="start from this rl.evo population's members (e.g. e1)")
    p.add_argument("--seed", help="checkpoint for a new population")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--b", type=int, default=12, help="arenas per member")
    p.add_argument("--p", type=int, default=50)
    p.add_argument("--skip", type=int, default=2, help="env steps (80 ms) per decision")
    p.add_argument("--steps", type=int, default=32, help="decisions per iteration")
    p.add_argument("--mb", type=int, default=2048)
    p.add_argument("--eval-minutes", type=float, default=20)
    p.add_argument("--eval-b", type=int, default=12)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--long-steps", type=int, default=7500, help="the long check game (7500 = 10 min)")
    p.add_argument("--hof-extra", nargs="*", default=[], help="more hall-of-fame dirs (league opponents only)")
    p.add_argument("--max-minutes", type=float, default=0)
    p.add_argument("--curriculum-p", type=float, default=0.15, help="share of learner respawns that start big (50-5000)")
    p.add_argument("--curriculum-mass", default="50,5000", help="their start mass range (log-uniform)")
    p.add_argument("--arch", help="genome architecture of a new population (JSON), e.g. '{\"kind\": \"entity\"}'")
    p.add_argument("--objective", choices=("arena", "farm"), default="arena",
                   help="farm: farming arenas only, mass reward, fitness = free farming check")
    p.add_argument("--rng", type=int, default=1)
    p.add_argument("--vram-gb", type=float, default=20.0)
    args = p.parse_args()
    if args.eval_job:
        return eval_job(args.eval_job)
    setup_cuda(args.vram_gb, args.rng)
    Population(args).run()


if __name__ == "__main__":
    main()
