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
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from rl.common import ROOT

from rl.evo import DEFAULT_ARCH, DEFAULT_HP, FARM_HP_KINDS, HP_KINDS, HUNT_HP_KINDS, mutate
from rl.ppo_sim import ARENA_NAMES, Learner, collect, from_checkpoint, frozen, new_model, setup_cuda
from sim.env import SimEnv

FIXED_HP = dict(league_frac=0.3, bot_frac=0.1, league_models=2, league_every=25, league_size=8,
                save_every=20, vf_warmup=0, clip=0.2, target_kl=0.02, vf=0.5)
# --objective hunt: hunting arenas only - a few hunters (learners, 300-3000 mass) among respawning prey
# (20-300): half scripted bots (they flee: splitting is how to catch them), half frozen hall-of-fame
# farmer nets at 300-5000 (they never flee and pop on viruses into edible pieces: big bait); mass is the
# only reward and death costs nothing (fearless); fitness is the hunting test (sim.tourney.hunt_batch)
HUNT_HP = dict(hunt_frac=1.0, farm_frac=0.0, flee_frac=0.0, giant_frac=0.0, league_frac=0.0, bot_frac=0.0,
               hunt_players=3, hunt_spawn=[300.0, 3000.0], prey_spawn=[20.0, 300.0], prey_nets=0.5,
               prey_net_spawn=[300.0, 5000.0],
               league_pool="hof", league_models=4, death_scale=0.0, kill_reward=[0.0, 0.0],
               virus_reward=[0.0, 0.0, 0.0], growth=None, action_bonus=[0.0, 0.0], eject_explore=False)
# --objective evade: escape arenas - evaders (learners, 10-3000 mass) among big scripted hunters
# (1500-6000) that chase anything they can eat; pellets as usual; mass reward with the full death penalty
# (dying costs everything farmed); fitness = farming under threat (sim.tourney.evade_batch) x farming alone
EVADE_HP = dict(flee_frac=1.0, farm_frac=0.0, hunt_frac=0.0, giant_frac=0.0, league_frac=0.0, bot_frac=0.0,
                flee_players=12, flee_hunters=6, flee_spawn=[10.0, 3000.0], flee_hunter_spawn=[1500.0, 6000.0],
                flee_hunter_nets=0.5, league_pool="hof", league_models=4,
                death_scale=2.0, kill_reward=[0.0, 0.0], virus_reward=[0.0, 0.0, 0.0], growth=None,
                action_bonus=[0.02, 0.0], eject_explore=False,   # a little "fun" per split the game carries out
                split_explore_min_mass=70.0)   # split escapes: evaders are mostly 100-600 mass when hunted
# --objective general: one net for every situation, all of them in every batch (interleaved): solo
# farming (2 per arena, 10-3000 mass), hunting (prey bots + prey nets), escaping (hunter bots + hunter nets);
# full death penalty; per-situation advantage normalisation; kickstarting from the specialists (--teachers
# farm=..,hunt=..,flee=..: a KL pull toward each situation's specialist, fading 1 -> 0.1 over 300 iterations);
# fitness = all four tests relative to the specialists (farm from 10, farm from 1000, hunt, evade)
GENERAL_HP = dict(farm_frac=0.3, hunt_frac=0.35, flee_frac=0.35, giant_frac=0.0, league_frac=0.0, bot_frac=0.0,
                  farm_players=2, farm_spawn=[10.0, 3000.0], hunt_players=3, hunt_spawn=[300.0, 3000.0],
                  prey_spawn=[20.0, 300.0], prey_nets=0.5, prey_net_spawn=[300.0, 5000.0], flee_players=10,
                  flee_hunters=6, flee_spawn=[10.0, 3000.0], flee_hunter_spawn=[1500.0, 6000.0], flee_hunter_nets=0.5,
                  league_pool="hof", league_models=4, death_scale=1.0, kill_reward=[0.0, 0.0],
                  virus_reward=[0.0, 0.0, 0.0], growth=None, action_bonus=[0.02, 0.0], adv_per_type=True,
                  kick_lambda=1.0, kick_iters=300, kick_min=0.1,
                  # imitation of recorded human play (rl/human_data.py), fading 0.5 -> 0.05 over 300 iterations
                  human_data=str(ROOT / "runs" / "human" / "dataset.pt"), human_lambda=0.5, human_iters=300,
                  human_min=0.05, human_batch=512, human_rare_weight=20.0)
# --objective ffa: the real server's game - free-for-all arenas with MultiOgar's own bots (sim.bots.
# MultiOgarBots: they split-kill small pieces, which is how the sim nets die on the real server) and
# league nets (own snapshots + the --hof-extra populations), plus some solo farming / hunting / escaping
# drills against forgetting; no split bonus, no imitation; kickstarting (--teachers) fades as in general.
# fitness = a real MultiOgar tournament every round (fresh seeds): members (with the play-time guards,
# rl.policy.virus_guard / threat_guard) vs champ / r2g9 and the server's bots, scored as log-mass
# margin over the rule-based champions in the same matches
# solo farming drills under the guards (no virus eating possible: fp2's trick, deadly in a real game),
# with imitation of the nearest-pellet farmer there (222 mass in 2 min under the guards vs the nets'
# ~175; champ ~240): the nets' remaining gap to the champions is growth, i.e. farming
FFA_HP = dict(bot_frac=0.4, league_frac=0.15, farm_frac=0.2, hunt_frac=0.1, flee_frac=0.15, giant_frac=0.0,
              farm_players=2, farm_spawn=[10.0, 3000.0], hunt_players=3, hunt_spawn=[300.0, 3000.0],
              prey_spawn=[20.0, 300.0], prey_nets=0.5, prey_net_spawn=[300.0, 5000.0], flee_players=10,
              flee_hunters=6, flee_spawn=[10.0, 3000.0], flee_hunter_spawn=[1500.0, 6000.0], flee_hunter_nets=0.5,
              league_models=4, death_scale=1.0, kill_reward=[0.0, 0.0], virus_reward=[0.0, 0.0, 0.0],
              growth=None, action_bonus=[0.0, 0.0], adv_per_type=True, kick_lambda=1.0, kick_iters=300,
              kick_min=0.1, split_explore_min_mass=150.0,
              pop_penalty=0.5,    # popping on a virus costs half our mass (the pieces the bots would eat)
              guard=2,            # train under the play-time guards (sim.bots.guard_mouse), as deployed
              farm_imitate=1.0, farm_imitate_min=0.1, farm_imitate_iters=300)
FFA_HP_KINDS = ["lr", "ent", "gamma", "explore", "explore", "epochs", "lam", "lam", "split_min"]
FFA_REFS = ["champ=runs/final_champion.json:2", "r2g9=runs/r2/champion_009.json:2"]   # the rule-based yardsticks
GENERAL_REFS = {"farm10": 288.0, "farm1000": 889.0, "hunt": 1000.0, "evade": 494.0}   # the specialists' scores
LAYOUT_KEYS = ("farm_frac", "hunt_frac", "flee_frac", "giant_frac", "league_frac", "bot_frac", "kill_reward",
               "virus_reward", "growth", "action_bonus", "hunt_players", "hunt_spawn", "prey_spawn", "prey_nets",
               "prey_net_spawn", "flee_players", "flee_hunters", "flee_spawn", "flee_hunter_spawn", "flee_hunter_nets",
               "farm_players", "adv_per_type", "kick_lambda", "kick_iters", "kick_min", "human_data",
               "human_lambda", "human_iters", "human_min", "human_batch", "human_rare_weight",
               "league_pool", "death_scale")
# --objective farm: farming arenas only (every slot a learner), mass is the only reward; fitness is
# the free farming check (alone for 2 min, split / eject allowed: whatever farms fastest)
FARM_HP = dict(farm_frac=1.0, hunt_frac=0.0, flee_frac=0.0, giant_frac=0.0, league_frac=0.0, bot_frac=0.0,
               kill_reward=[0.0, 0.0], virus_reward=[0.0, 0.0, 0.0], growth=None, action_bonus=[0.0, 0.0],
               farm_spawn=[10.0, 3000.0], split_explore_min_mass=300.0, eject_explore=False, epochs=4)


def log(msg: str):
    print(msg, flush=True)


def eval_job(path: str) -> None:
    """Side process: tournament + checks on frozen snapshots; writes the JSON result."""
    from sim.tourney import evade_batch, farm, farm_batch, hunt_batch, run
    job = json.loads(Path(path).read_text())
    setup_cuda(job.get("vram_gb", 5.0))
    nets = job["nets"]
    mem = {k: v for k, v in nets.items() if k != "hof_last"}
    if job.get("objective") == "general":
        small = farm_batch(nets, B=64, free=True)
        big = farm_batch(nets, B=64, free=True, start=1000.0, gain=True)
        hb = hunt_batch(nets, B=32, prey_nets=job.get("prey_nets"))
        ev = evade_batch(nets, B=32)
        R = GENERAL_REFS
        res = {}
        for k in nets:
            sc = {"farm10": small[k], "farm1000": big[k], "hunt": hb[k]["gain"], "evade": ev[k]["mass"]}
            fit = sum(math.log(max(sc[t], 1.0) / R[t]) for t in R) / len(R)
            res[k] = {"log_mass": fit, "ci": 0.0, "avg_mass": small[k], "deaths": ev[k]["deaths_per_min"],
                      "kills": hb[k]["kills_per_min"], "splits_per_min": hb[k]["splits_per_min"], **sc}
        Path(job["out"]).write_text(json.dumps({"results": res, "farm": {}, "long": {}}, indent=1))
        return
    if job.get("objective") == "ffa":
        Path(job["out"]).write_text(json.dumps(ffa_eval(nets, job), indent=1))
        return
    if job.get("objective") == "evade":
        ev = evade_batch(nets, B=32)
        small = farm_batch(nets, B=64, free=True)
        # survival first (deaths per minute under threat), mass under threat and farming alone secondary
        res = {k: {"log_mass": -math.log(v["deaths_per_min"] + 0.05) + 0.25 * math.log(max(v["mass"], 1.0) / 200)
                   + 0.25 * math.log(max(small[k], 1.0) / 300),
                   "ci": 0.0, "avg_mass": v["mass"], "deaths": v["deaths_per_min"], "farm10": small[k], "kills": 0.0,
                   "splits_per_min": v["splits_per_min"]}
               for k, v in ev.items()}
        Path(job["out"]).write_text(json.dumps({"results": res, "farm": small, "long": {}}, indent=1))
        return
    if job.get("objective") == "hunt":
        hb = hunt_batch(nets, B=32, prey_nets=job.get("prey_nets"))
        res = {k: {"log_mass": v["gain"] / 1000, "ci": 0.0, "avg_mass": v["gain"], "kills": v["kills_per_min"],
                   "splits_per_min": v["splits_per_min"], "deaths": 0.0} for k, v in hb.items()}
        Path(job["out"]).write_text(json.dumps({"results": res, "farm": {}, "long": {}}, indent=1))
        return
    if job.get("objective") == "farm":
        # fitness: both farming tests (split / eject allowed), each relative to its routine:
        # from 10 mass (routines ~300) and the gain from 1000 mass (routine +374; split-farming +467)
        small = farm_batch(nets, B=64, free=True)
        big = farm_batch(nets, B=64, free=True, start=1000.0, gain=True)
        # product of the two (sum of logs): averaging let a member trade a collapsed small-size
        # score for a big-size one (m4: 50 from 10, +1100 from 1000)
        res = {k: {"log_mass": (math.log(max(small[k], 1.0) / 300) + math.log(max(big[k], 1.0) / 374)) / 2, "ci": 0.0,
                   "avg_mass": small[k], "gain_1000": big[k], "deaths": 0.0, "kills": 0.0} for k in nets}
        Path(job["out"]).write_text(json.dumps({"results": res, "farm": small, "long": {}}, indent=1))
        return
    res = run(nets, B=job["eval_b"], P=job["p"], steps=job["steps"], seed=job["seed"])
    out = {"results": res, "farm": farm(mem)}
    out["long"] = {k: v["avg_mass"] for k, v in run(mem, B=4, P=job["p"], steps=job["long_steps"],
                                                     seed=job["seed"] + 2000).items()}
    Path(job["out"]).write_text(json.dumps(out, indent=1))


def ffa_eval(nets: dict, job: dict) -> dict:
    """Real-server fitness: `job["matches"]` MultiOgar matches (2x food, 8 server bots, fresh seeds) of
    every net (2 copies; hof_last 1) with champ and r2g9. Fitness = the net's mean log mass minus the mean
    of champ's and r2g9's in the same matches (paired by seed), so > 0 means it beats the rule bots."""
    import statistics
    from arena.tournament import play_many
    # the nets play as deployed: with the virus detour and the champion's threat push (rl.policy guards)
    specs = [f"{k}=guard2:{v}:{1 if k == 'hof_last' else 2}" for k, v in nets.items()] + FFA_REFS
    rows = play_many(specs, job.get("matches", 24), workers=job.get("workers", 8), seed_base=job["seed"])
    by: dict[str, dict[int, list[float]]] = {}
    dead: dict[str, list[float]] = {}
    pops: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["policy"], {}).setdefault(r["seed"], []).append(r["log_mass"])
        dead.setdefault(r["policy"], []).append(r["deaths"])
        pops.setdefault(r["policy"], []).append(r.get("popped", 0))
    mean = {k: {s: statistics.fmean(v) for s, v in d.items()} for k, d in by.items()}
    ref = {s: statistics.fmean(mean[n][s] for n in ("champ", "r2g9") if s in mean.get(n, {}))
           for s in set().union(*(mean.get(n, {}).keys() for n in ("champ", "r2g9")))}
    res = {}
    for k in list(nets) + ["champ", "r2g9"]:
        seeds = [s for s in mean.get(k, {}) if s in ref]
        if not seeds:
            res[k] = {"log_mass": -9.0, "ci": 0.0, "avg_mass": 0.0, "deaths": 0.0, "kills": 0.0, "wins": 0.0}
            continue
        margin = [mean[k][s] - ref[s] for s in seeds]
        res[k] = {"log_mass": statistics.fmean(margin),
                  "ci": 1.96 * statistics.stdev(margin) / math.sqrt(len(margin)) if len(margin) > 1 else 0.0,
                  "avg_mass": statistics.fmean(math.exp(mean[k][s]) for s in seeds),
                  "deaths": statistics.fmean(dead[k]), "kills": 0.0, "popped": statistics.fmean(pops[k]),
                  "wins": sum(m > 0 for m in margin) / len(margin), "matches": len(seeds)}
    return {"results": res, "farm": {}, "long": {}, "rows": rows}


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
                          obs="both" if len(kinds) > 1 else kinds.pop(),
                          arena_steps=args.arena_steps,
                          obs_dtype=torch.float16, skip=args.skip,
                          bot_kind=args.bot_kind or ("multiogar" if args.objective == "ffa" else "scripted"))
        self.hof_files = [f for d in [self.root / "hof", *map(Path, args.hof_extra)] for f in sorted(d.glob("*.pt"))]
        self.hof = [frozen(from_checkpoint(str(f), self.dev)) for f in self.hof_files]
        # hunt / evade: the league slots (prey nets, hunter nets) are played by the --hof-extra populations'
        # halls of fame only, rescanned every round: an arms race with the other population
        self.teachers = {}                     # --teachers: arena kind -> frozen specialist (kickstarting)
        for spec in args.teachers:
            kind, path = spec.split("=", 1)
            self.teachers[ARENA_NAMES[kind]] = frozen(from_checkpoint(path, self.dev))
        self.opp_files: list[Path] = []
        self.opp: list = []
        self.refresh_opponents()
        self.learners = [self.make_learner(m, k) for k, m in enumerate(self.state["members"])]
        self.eval_proc, self.eval_t = None, time.time()

    # ---- state
    def load_state(self) -> dict:
        """Resume from evo.json, or build generation 0 (from an rl.evo population or one --seed checkpoint)."""
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
            kinds = {"farm": FARM_HP_KINDS, "hunt": HUNT_HP_KINDS, "evade": HUNT_HP_KINDS, "general": HUNT_HP_KINDS, "ffa": FFA_HP_KINDS}.get(a.objective, HP_KINDS)
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
        """Learner for member `m` owning arenas [k*b, (k+1)*b) of the shared env; its genome supplies hyper-parameters."""
        a = self.args
        farm = {"farm": FARM_HP, "hunt": HUNT_HP, "evade": EVADE_HP, "general": GENERAL_HP, "ffa": FFA_HP}.get(a.objective, {})
        # farming defaults sit below the genome (its genes may override them: lambda, split exploration,
        # start sizes, epochs), the arena layout of the farm objective above it
        hp = {**DEFAULT_HP, "lam": 0.95, **farm, **m["genome"]["hp"], **FIXED_HP, "mb": a.mb}
        if farm:
            hp.update({"farm_players": a.p, **{k: farm[k] for k in LAYOUT_KEYS if k in farm}})
            if "fun" in m["genome"]["hp"]:     # the genome's taste for splitting overrides the objective's bonus
                hp["action_bonus"] = [m["genome"]["hp"]["fun"], hp["action_bonus"][1]]
        model = new_model(**m["genome"]["arch"]).to(self.dev)
        parent = from_checkpoint(m["ckpt"], self.dev)
        teacher = None
        if parent.config == model.config:
            model = parent
        else:
            teacher = frozen(parent)
        logf = open(ROOT / "logs" / f"pbt-{a.name}-{m['id']}.txt", "a", encoding="utf-8")
        opp = self.opp if a.objective in ("hunt", "evade", "general", "ffa") and self.opp else self.hof
        L = Learner(f"{a.name}/{m['id']}", model, self.env, k * a.b, a.b, hp, self.root / m["id"], hof=opp,
                    teacher=teacher, distill_iters=30 if teacher is not None else 0, start_it=m["it"], log=logf,
                    skip=a.skip)
        L.kick = self.teachers
        L.print(f"--- {m['id']} in arenas {k * a.b}-{(k + 1) * a.b - 1}: {m['origin']}  genome {m['genome']}"
                f"{'  (distilling a new architecture)' if teacher is not None else ''}")
        return L

    # ---- evaluation
    def launch_eval(self):
        """Snapshot every member and start the side evaluation process (training continues meanwhile)."""
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
               "objective": a.objective, "matches": a.eval_matches,
               # hunting test: half the prey = the first --hof-extra population's newest nets (the evaders)
               "prey_nets": [str(f) for f in sorted(Path(a.hof_extra[0]).glob("*.pt"))[-3:]]
               if a.objective in ("hunt", "general") and a.hof_extra else None}
        jf = self.root / "eval" / f"g{rnd:03d}_job.json"
        jf.write_text(json.dumps(job))
        self.eval_out = Path(job["out"])
        self.eval_log = open(ROOT / "logs" / f"pbt-{a.name}-eval.txt", "a", encoding="utf-8")
        self.eval_proc = subprocess.Popen([sys.executable, "-m", "rl.pbt", "--eval-job", str(jf)], cwd=ROOT,
                                          stdout=self.eval_log, stderr=subprocess.STDOUT)
        self.save_state()
        log(f"\n=== round {rnd}: evaluating {', '.join(nets)} (training continues)")

    def poll_eval(self):
        """Start an evaluation when due; when a running one has finished, run the selection on its results."""
        if self.eval_proc is None:
            if time.time() - self.eval_t > self.args.eval_minutes * 60:
                self.launch_eval()
            return
        if self.eval_proc.poll() is None:
            return
        self.eval_proc, self.eval_t = None, time.time()
        self.eval_log.close()
        self.refresh_opponents()
        if not self.eval_out.exists():
            log(f"  evaluation failed: see logs/pbt-{self.args.name}-eval.txt")
            return
        self.select(json.loads(self.eval_out.read_text()))

    def select(self, ev: dict):
        """One selection round: rank the evaluated members, promote the best to the hall of fame,
        replace the bottom half by mutated copies of the top half (see the module docstring)."""
        rnd, res = self.state["gen"], ev["results"]
        members = self.state["members"]
        self.log_results(rnd, res, ev)
        ids = {m["id"] for m in members}
        ranking = [k for k, _ in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]) if k in ids or k == "hof_last"]
        ranked = [next(m for m in members if m["id"] == k) for k in ranking if k in ids]
        for m in ranked:
            m["history"].append({"gen": rnd, **res[m["id"]], "farm": ev["farm"].get(m["id"]),
                                 "long_mass": ev["long"].get(m["id"])})
        best = ranked[0]
        self.promote_to_hof(rnd, best)
        self.state["log"].append({"gen": rnd, "results": res, "farm": ev["farm"], "long": ev["long"],
                                  "best": best["id"], "time": time.time()})
        n_keep = max(1, len(members) // 2)
        grace = [m for m in ranked if m.get("new_arch_gen") == rnd]
        losers = [m for m in ranked[n_keep:] if m not in grace]
        parents = self.pick_parents(ranking, losers, n_keep)
        for m in grace:
            log(f"  {m['id']} (new architecture) protected this round")
        for j, dead in enumerate(losers):
            self.replace_member(dead, parents[j % len(parents)], rnd)
        self.state["gen"] = rnd + 1
        self.save_state()

    def log_results(self, rnd: int, res: dict, ev: dict):
        """The "--- round N results" block of the log (other tools grep its lines)."""
        log(f"--- round {rnd} results" + {"farm": " (fitness: free farming, 2 min alone, split / eject allowed)",
                                           "hunt": " (fitness: hunting test, 1000 mass among 15 prey - half scripted, half the evaders' newest nets - 2 min)",
                                           "evade": " (fitness: survival first - deaths/min among 7 big hunters; mass under threat and farming alone secondary)",
                                           "general": " (fitness: farm 10 / farm 1000 / hunt / evade vs the specialists: 288 / +889 / +1000 / 494)",
                                           "ffa": " (fitness: real MultiOgar matches, log-mass margin over champ / r2g9 in the same matches)"
                                           }.get(self.args.objective, ""))
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]):
            if self.args.objective == "general":
                log(f"  {name:<9} fitness {r['log_mass']:+.3f}  farm 10->{r['farm10']:.0f} 1000:{r['farm1000']:+.0f}  "
                    f"hunt {r['hunt']:+.0f} ({r['kills']:.1f} kills/min, {r['splits_per_min']:.2f} splits/min)  "
                    f"evade {r['evade']:.0f} ({r['deaths']:.2f} deaths/min)")
                continue
            if self.args.objective == "ffa":
                log(f"  {name:<9} margin {r['log_mass']:+.3f} +-{r['ci']:.3f}  beats the rule bots in {r.get('wins', 0):.0%} "
                    f"of {r.get('matches', 0)} matches  avg mass {r['avg_mass']:.0f}  deaths/match {r['deaths']:.2f}  "
                    f"virus pops/match {r.get('popped', 0):.1f}")
                continue
            if self.args.objective == "evade":
                log(f"  {name:<9} fitness {r['log_mass']:+.3f}  under threat: mass {r['avg_mass']:.0f} "
                    f"deaths/min {r['deaths']:.2f} splits/min {r.get('splits_per_min', 0):.2f}  "
                    f"farming alone from 10 -> {r['farm10']:.0f}")
                continue
            if self.args.objective == "hunt":
                log(f"  {name:<9} gain {r['avg_mass']:+6.0f}  kills/min {r['kills']:.2f}  splits/min {r['splits_per_min']:.2f}")
                continue
            if "gain_1000" in r:
                log(f"  {name:<9} fitness {r['log_mass']:+.3f}  from 10 -> {r['avg_mass']:.0f}  "
                    f"from 1000: {r['gain_1000']:+.0f}  (routines ~300 / +374, split-farming +467)")
                continue
            log(f"  {name:<9} log_mass {r['log_mass']:.3f} +-{r['ci']:.3f}  mass {r['avg_mass']:.0f}  "
                f"deaths {r['deaths']:.2f}  kills {r['kills']:.2f}  splits/min {r.get('splits_per_min', 0):.2f}")
        if ev["farm"] and self.args.objective not in ("farm", "evade"):
            log("  farming alone, mass after 2 min (routines ~300): " +
                "  ".join(f"{k} {v:.0f}" for k, v in sorted(ev["farm"].items(), key=lambda kv: -kv[1])))
        if ev["long"]:
            log(f"  long game ({self.args.long_steps * 0.08 / 60:.0f} min from mass 10), avg mass: " +
                "  ".join(f"{k} {v:.0f}" for k, v in sorted(ev["long"].items(), key=lambda kv: -kv[1])))

    def refresh_opponents(self, per_dir: int = 3):
        """Load the newest `per_dir` checkpoints of every --hof-extra dir into self.opp, in place (the
        learners hold this list; their league picks from it at the next reassignment)."""
        want = [f for d in map(Path, self.args.hof_extra) for f in sorted(d.glob("*.pt"))[-per_dir:]]
        if want == self.opp_files:
            return
        have = dict(zip(self.opp_files, self.opp))
        self.opp[:] = [have[f] if f in have else frozen(from_checkpoint(str(f), self.dev)) for f in want]
        new = [f.stem for f in want if f not in have]
        self.opp_files = want
        if new and self.learners_ready():
            log(f"  opponents refreshed: {', '.join(f'{f.parent.parent.name}/{f.stem}' for f in want)}")

    def learners_ready(self) -> bool:
        return bool(getattr(self, "learners", None))

    def promote_to_hof(self, rnd: int, best: dict):
        """Copy the round's best member's evaluated checkpoint to the hall of fame (disk, state, league)."""
        hof_f = self.root / "hof" / f"g{rnd:03d}_{best['id']}.pt"
        shutil.copy(self.eval_nets[best["id"]], hof_f)
        self.state["hof"].append({"file": str(hof_f), "genome": best["genome"], "from": best["id"]})
        self.hof.append(frozen(from_checkpoint(str(hof_f), self.dev)))

    def pick_parents(self, ranking: list, losers: list, n_keep: int) -> list[dict]:
        """The top half of the ranking (minus the replaced), the last hall-of-fame net included if its
        genome is known. Each parent: {id, genome, ckpt[, it]}."""
        members = self.state["members"]
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
        return parents

    def replace_member(self, dead: dict, par: dict, rnd: int):
        """Swap `dead` for a mutated child of `par`: new learner, its arenas restarted."""
        members = self.state["members"]
        g, note = mutate(par["genome"], self.rng,
                         hp_kinds={"farm": FARM_HP_KINDS, "hunt": HUNT_HP_KINDS, "evade": HUNT_HP_KINDS, "general": HUNT_HP_KINDS, "ffa": FFA_HP_KINDS}.get(self.args.objective, HP_KINDS))
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

    # ---- main loop
    def run(self):
        """Train all members together until --max-minutes; evaluation and selection happen inside the loop."""
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
    p.add_argument("--arena-steps", type=int, default=0, help="per-arena restart clock in steps (staggered; 0: off)")
    p.add_argument("--arch", help="genome architecture of a new population (JSON), e.g. '{\"kind\": \"entity\"}'")
    p.add_argument("--teachers", nargs="*", default=[], help="kickstarting: kind=checkpoint (farm, hunt, flee)")
    p.add_argument("--objective", choices=("arena", "farm", "hunt", "evade", "general", "ffa"), default="arena",
                   help="farm: farming arenas only, mass reward, fitness = free farming check")
    p.add_argument("--rng", type=int, default=1)
    p.add_argument("--vram-gb", type=float, default=20.0)
    p.add_argument("--bot-kind", choices=("scripted", "multiogar"),
                   help="the sim's scripted bots (default: multiogar for --objective ffa, else scripted)")
    p.add_argument("--eval-matches", type=int, default=24, help="ffa: real-server matches per evaluation")
    args = p.parse_args()
    if args.eval_job:
        return eval_job(args.eval_job)
    setup_cuda(args.vram_gb, args.rng)
    Population(args).run()


if __name__ == "__main__":
    main()
