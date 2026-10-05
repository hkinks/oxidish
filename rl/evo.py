"""Evolution over network designs: each member's weights are learned by PPO in the simulator
(rl.ppo_sim); the genome is the architecture, the inputs it sees and the training settings.

    uv run python -m rl.evo --name e1 --seed runs/rl/s1/latest.pt [--pop 4 --minutes 30]

A generation:
1. every member trains for `--minutes` in its own rl.ppo_sim process (all at once, sharing the
   GPU); opponents are its own past snapshots plus the hall of fame;
2. all members play each other in shared 50-player sim arenas (sim.tourney); fitness = mean log
   of average mass (being big and staying alive);
3. the best member joins the hall of fame (runs/rl/<name>/hof/), the bottom half is replaced by
   mutated copies of the top half. A child with the parent's architecture inherits its weights;
   one with a new architecture starts fresh and first imitates the parent (rl.ppo_sim --arch).

State: runs/rl/<name>/evo.json (rewritten every generation; the loop resumes from it).
Log: printed, and logs/evo-<name>.txt when started as documented in docs.
"""

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
import time

from rl.common import ROOT
from sim.obs import C as N_CH
from sim.tourney import farm as farm_test, run as tourney

# ---- genome
DEFAULT_ARCH = {"width": 256, "convs": [[32, 1], [64, 2], [64, 2]], "pool": 1, "in_mask": None,
                "s_mask": None, "shared": False}
DEFAULT_HP = {"lr": 1e-4, "ent": 0.001, "gamma": 0.998, "explore": 0.02, "kill_reward": [0.5, 1.0],
              "virus_reward": [0.5, 0.5, 1.0], "explore_reward": 0.02, "epochs": 3, "farm_frac": 0.15, "hunt_frac": 0.15, "flee_frac": 0.1, "giant_frac": 0.1,
              "growth": None, "action_bonus": [0.0, 0.0]}
CH_MINE = 2   # our own cells: always visible to the net


HP_KINDS = ["lr", "ent", "gamma", "explore", "kill", "virus", "explore_reward", "epochs", "farm", "hunt", "hunt",
            "flee", "giant", "growth", "growth", "bonus", "bonus"]
HUNT_HP_KINDS = ["lr", "ent", "gamma", "explore", "explore", "epochs", "lam", "lam", "split_min", "fun", "fun"]
FARM_HP_KINDS = ["lr", "ent", "gamma", "explore", "explore", "explore_reward", "epochs", "lam", "lam",
                 "split_min", "split_min", "farm_spawn"]


def mutate(genome: dict, rng: random.Random, arch_p: float = 0.35, hp_kinds=HP_KINDS) -> tuple[dict, str]:
    """One or two random changes; returns the child genome and a description."""
    g = copy.deepcopy(genome)
    arch, hp = g["arch"], g["hp"]
    notes = []
    for _ in range(rng.choice([1, 1, 2])):
        entity = arch.get("kind") == "entity"
        arch_change = rng.random() < arch_p
        if arch_change and entity:          # transformer genes
            kind = rng.choice(["width", "d", "layers", "heads"])
            cur = {"width": 256, "d": 128, "layers": 2, "heads": 4, **arch}
            opts = {"width": (128, 192, 256, 384, 512), "d": (64, 128, 192, 256), "layers": (1, 2, 3, 4),
                    "heads": (2, 4, 8)}[kind]
            arch[kind] = rng.choice([x for x in opts if x != cur[kind]])
            d, h = arch.get("d", 128), arch.get("heads", 4)
            if d % h:                      # attention heads must divide the token width
                arch["heads"] = max(x for x in (2, 4, 8) if d % x == 0 and x <= h)
            notes.append(f"{kind} {arch[kind]}")
        elif arch_change:
            kind = rng.choice(["width", "channels", "layer", "pool", "input", "shared"])
            if kind == "width":
                arch["width"] = rng.choice([w for w in (128, 192, 256, 384, 512) if w != arch["width"]])
                notes.append(f"width {arch['width']}")
            elif kind == "channels":
                i = rng.randrange(len(arch["convs"]))
                c = arch["convs"][i][0]
                arch["convs"][i][0] = max(16, min(128, int(round(c * rng.choice([0.5, 1.5, 2]) / 16) * 16)))
                notes.append(f"conv{i} {c}->{arch['convs'][i][0]}")
            elif kind == "layer":
                if len(arch["convs"]) > 2 and rng.random() < 0.5:
                    i = rng.randrange(1, len(arch["convs"]))
                    arch["convs"].pop(i)
                    notes.append(f"drop conv{i}")
                elif len(arch["convs"]) < 6:
                    i = rng.randrange(1, len(arch["convs"]) + 1)
                    prev = arch["convs"][i - 1][0]
                    n_stride2 = sum(1 for _, st in arch["convs"] if st == 2)
                    stride = 2 if n_stride2 < 3 and rng.random() < 0.5 else 1
                    arch["convs"].insert(i, [prev, stride])
                    notes.append(f"add conv{i} ({prev}, stride {stride})")
            elif kind == "pool":
                arch["pool"] = 2 if arch["pool"] == 1 else 1
                notes.append(f"pool {arch['pool']}")
            elif kind == "input":
                mask = list(range(N_CH)) if arch["in_mask"] is None else list(arch["in_mask"])
                ch = rng.choice([c for c in range(N_CH) if c != CH_MINE])
                if ch in mask:
                    mask.remove(ch)
                    notes.append(f"drop input {ch}")
                else:
                    mask.append(ch)
                    notes.append(f"add input {ch}")
                arch["in_mask"] = None if sorted(mask) == list(range(N_CH)) else sorted(mask)
            else:
                arch["shared"] = not arch["shared"]
                notes.append(f"shared {arch['shared']}")
        else:
            kind = rng.choice(hp_kinds)
            if kind == "lr":
                hp["lr"] = min(5e-4, max(2e-5, hp["lr"] * rng.choice([0.5, 0.7, 1.4, 2])))
                notes.append(f"lr {hp['lr']:.1e}")
            elif kind == "ent":
                hp["ent"] = min(0.02, max(1e-4, hp["ent"] * rng.choice([0.5, 2, 3])))
                notes.append(f"ent {hp['ent']:.4f}")
            elif kind == "gamma":
                hp["gamma"] = rng.choice([x for x in (0.99, 0.995, 0.998, 0.999) if x != hp["gamma"]])
                notes.append(f"gamma {hp['gamma']}")
            elif kind == "explore":
                hp["explore"] = min(0.15, max(0.005, hp["explore"] * rng.choice([0.5, 2, 3])))
                notes.append(f"split/eject explore {hp['explore']:.3f}")
            elif kind == "kill":
                f = rng.choice([0, 0.5, 2])
                hp["kill_reward"] = [round(x * f, 3) if hp["kill_reward"][0] else round(0.5 * f, 3)
                                     for x in hp["kill_reward"]]
                notes.append(f"kill reward {hp['kill_reward']}")
            elif kind == "virus":
                f = rng.choice([0, 0.5, 2])
                hp["virus_reward"] = [round(x * f, 3) for x in DEFAULT_HP["virus_reward"]] if not any(hp["virus_reward"]) \
                    else [round(x * f, 3) for x in hp["virus_reward"]]
                notes.append(f"virus reward {hp['virus_reward']}")
            elif kind == "explore_reward":
                hp["explore_reward"] = rng.choice([0.0, 0.01, 0.02, 0.05])
                notes.append(f"explore reward {hp['explore_reward']}")
            elif kind == "growth":
                if hp.get("growth") is None or rng.random() < 0.2:
                    hp["growth"] = None if hp.get("growth") else [4.0, 500.0, 1.0]
                else:
                    gr = list(hp["growth"])
                    gr[0] = rng.choice([x for x in (2.0, 3.0, 4.0, 6.0, 10.0) if x != gr[0]])
                    gr[1] = rng.choice([200.0, 500.0, 1000.0])
                    hp["growth"] = gr
                notes.append(f"growth reward {hp['growth']}")
            elif kind == "bonus":
                b = list(hp.get("action_bonus", [0.0, 0.0]))
                i = rng.randrange(2)
                b[i] = rng.choice([x for x in (0.0, 0.005, 0.02, 0.05) if x != b[i]])
                hp["action_bonus"] = b
                notes.append(f"{['split', 'W'][i]} bonus {b[i]}")
            elif kind == "flee":
                hp["flee_frac"] = rng.choice([x for x in (0.0, 0.1, 0.2) if x != hp.get("flee_frac", 0.1)])
                notes.append(f"escape drills {hp['flee_frac']:.0%} of arenas")
            elif kind == "giant":
                hp["giant_frac"] = rng.choice([x for x in (0.0, 0.1, 0.2) if x != hp.get("giant_frac", 0.1)])
                notes.append(f"giant arenas {hp['giant_frac']:.0%}")
            elif kind == "hunt":
                hp["hunt_frac"] = rng.choice([x for x in (0.0, 0.1, 0.2, 0.3) if x != hp.get("hunt_frac", 0.15)])
                notes.append(f"hunt drills {hp['hunt_frac']:.0%} of arenas")
            elif kind == "fun":           # reward per executed split: how much the genome likes splitting
                hp["fun"] = rng.choice([x for x in (0.0, 0.02, 0.05, 0.1) if x != hp.get("fun", 0.02)])
                notes.append(f"split fun {hp['fun']}")
            elif kind == "lam":           # credit horizon: a split pays 15-30 s (100-190 decisions) later
                hp["lam"] = rng.choice([x for x in (0.95, 0.97, 0.99) if x != hp.get("lam", 0.95)])
                notes.append(f"lambda {hp['lam']}")
            elif kind == "split_min":     # split exploration only from this mass
                hp["split_explore_min_mass"] = rng.choice([x for x in (150.0, 300.0, 500.0, 1000.0)
                                                           if x != hp.get("split_explore_min_mass", 300.0)])
                notes.append(f"split exploration from {hp['split_explore_min_mass']:.0f} mass")
            elif kind == "farm_spawn":    # farm start sizes
                hp["farm_spawn"] = rng.choice([x for x in ([10.0, 3000.0], [100.0, 6000.0], [300.0, 6000.0])
                                               if x != hp.get("farm_spawn", [10.0, 3000.0])])
                notes.append(f"farm starts {hp['farm_spawn'][0]:.0f}-{hp['farm_spawn'][1]:.0f} mass")
            elif kind == "farm":
                hp["farm_frac"] = rng.choice([x for x in (0.0, 0.1, 0.2, 0.3) if x != hp.get("farm_frac", 0.15)])
                notes.append(f"farm drills {hp['farm_frac']:.0%} of arenas")
            else:
                hp["epochs"] = rng.choice([e for e in (2, 3, 4) if e != hp["epochs"]])
                notes.append(f"epochs {hp['epochs']}")
    return g, ", ".join(notes)


def train_cmd(m: dict, name: str, args) -> list[str]:
    """The rl.ppo_sim command line that trains member `m` for one generation."""
    hp = m["genome"]["hp"]
    cmd = [sys.executable, "-m", "rl.ppo_sim", "--run", f"{name}/{m['id']}",
           "--init", m["ckpt"], "--arch", json.dumps(m["genome"]["arch"]),
           "--start-it", str(m["it"]), "--max-minutes", str(args.minutes),
           "--b", str(args.b), "--p", str(args.p), "--steps", "32", "--mb", str(args.mb),
           "--vram-gb", str(args.vram_gb), "--episode-steps", "20000", "--save-every", "20",
           "--vf-warmup", "0", "--league-frac", "0.3", "--bot-frac", "0.1",
           "--hof", str(ROOT / "runs" / "rl" / name / "hof"),
           "--lr", str(hp["lr"]), "--ent", str(hp["ent"]), "--gamma", str(hp["gamma"]),
           "--explore", str(hp["explore"]), "--epochs", str(hp["epochs"]),
           "--kill-reward", ",".join(map(str, hp["kill_reward"])),
           "--virus-reward", ",".join(map(str, hp["virus_reward"])),
           "--explore-reward", str(hp["explore_reward"]), "--seed", str(m["seed"]),
           "--farm-frac", str(hp.get("farm_frac", DEFAULT_HP["farm_frac"])),
           "--hunt-frac", str(hp.get("hunt_frac", DEFAULT_HP["hunt_frac"])),
           "--flee-frac", str(hp.get("flee_frac", DEFAULT_HP["flee_frac"])),
           "--giant-frac", str(hp.get("giant_frac", DEFAULT_HP["giant_frac"])),
           "--action-bonus", ",".join(map(str, hp.get("action_bonus", [0.0, 0.0])))]
    if hp.get("growth"):
        cmd += ["--growth", ",".join(map(str, hp["growth"]))]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="e1")
    p.add_argument("--seed", help="checkpoint every first-generation member starts from")
    p.add_argument("--pop", type=int, default=4)
    p.add_argument("--minutes", type=float, default=30, help="training per member per generation")
    p.add_argument("--gens", type=int, default=1000)
    p.add_argument("--b", type=int, default=24, help="arenas per member")
    p.add_argument("--p", type=int, default=50)
    p.add_argument("--mb", type=int, default=1024)
    p.add_argument("--vram-gb", type=float, default=5.5, help="per member")
    p.add_argument("--eval-b", type=int, default=12)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--rng", type=int, default=1)
    p.add_argument("--long-steps", type=int, default=7500, help="the long check game (7500 = 10 min)")
    args = p.parse_args()

    root = ROOT / "runs" / "rl" / args.name
    (root / "hof").mkdir(parents=True, exist_ok=True)
    state_f = root / "evo.json"
    rng = random.Random(args.rng)
    if state_f.exists():
        state = json.loads(state_f.read_text())
        print(f"resuming {args.name} at generation {state['gen']}", flush=True)
    else:
        if not args.seed:
            raise SystemExit("--seed is required for a new population")
        base = {"arch": DEFAULT_ARCH, "hp": DEFAULT_HP}
        members = []
        for k in range(args.pop):     # member 0 is the seed unchanged, the others start mutated
            g, note = (copy.deepcopy(base), "seed") if k == 0 else mutate(base, rng)
            members.append({"id": f"m{k}", "genome": g, "ckpt": args.seed, "it": 0, "seed": 100 + k,
                            "origin": note, "history": []})
        state = {"gen": 0, "members": members, "next_id": args.pop, "log": []}
        state_f.write_text(json.dumps(state, indent=1))

    while state["gen"] < args.gens:
        gen = state["gen"]
        members = state["members"]
        print(f"\n=== generation {gen}: training {len(members)} members for {args.minutes} min", flush=True)
        for m in members:
            print(f"  {m['id']}: {m['origin']}  arch {m['genome']['arch']}  hp {m['genome']['hp']}", flush=True)
        procs = []
        for m in members:
            log = open(ROOT / "logs" / f"evo-{args.name}-{m['id']}.txt", "a", encoding="utf-8")
            procs.append(subprocess.Popen(train_cmd(m, args.name, args), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        for pr in procs:
            pr.wait()
        alive = []
        for m, pr in zip(members, procs):
            ck = root / m["id"] / "latest.pt"
            if pr.returncode != 0 and ck.exists():
                print(f"  {m['id']} crashed (exit {pr.returncode}); using its last checkpoint - "
                      f"see logs/evo-{args.name}-{m['id']}.txt", flush=True)
            if ck.exists():
                import torch
                m["ckpt"] = str(ck)
                m["it"] = torch.load(ck, map_location="cpu", weights_only=False).get("it", m["it"])
                alive.append(m)
            else:
                print(f"  {m['id']} failed (exit {pr.returncode}); see logs/evo-{args.name}-{m['id']}.txt", flush=True)
        if not alive:
            raise SystemExit("every member failed")

        # ---- fitness: everyone (plus the last hall-of-fame entry, as a fixed yardstick) in shared arenas
        snaps = {}
        for m in alive:                       # freeze the checkpoints being compared
            dst = root / "eval" / f"g{gen:03d}_{m['id']}.pt"
            dst.parent.mkdir(exist_ok=True)
            shutil.copy(m["ckpt"], dst)
            snaps[m["id"]] = str(dst)
        hof = sorted((root / "hof").glob("*.pt"))
        if hof:
            snaps["hof_last"] = str(hof[-1])
        res = tourney(snaps, B=args.eval_b, P=args.p, steps=args.eval_steps, seed=7000 + gen)
        ranked = sorted(alive, key=lambda m: -res[m["id"]]["log_mass"])
        print(f"--- generation {gen} results", flush=True)
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]):
            print(f"  {name:<9} log_mass {r['log_mass']:.3f} +-{r['ci']:.3f}  mass {r['avg_mass']:.0f}  "
                  f"deaths {r['deaths']:.2f}  kills {r['kills']:.2f}  splits/min {r.get('splits_per_min', 0):.2f}", flush=True)
        farm = farm_test({k: v for k, v in snaps.items() if k != "hof_last"})
        long = tourney({k: v for k, v in snaps.items() if k != "hof_last"}, B=4, P=args.p,
                       steps=args.long_steps, seed=9000 + gen)
        print("  farming alone, mass after 2 min (nearest-pellet routine ~320): " +
              "  ".join(f"{k} {v:.0f}" for k, v in sorted(farm.items(), key=lambda kv: -kv[1])), flush=True)
        print(f"  long game ({args.long_steps * 0.08 / 60:.0f} min from mass 10), avg mass: " +
              "  ".join(f"{k} {v['avg_mass']:.0f}" for k, v in sorted(long.items(), key=lambda kv: -kv[1]['avg_mass'])), flush=True)
        for m in alive:
            m["history"].append({"gen": gen, **res[m["id"]], "farm": farm[m["id"]], "long_mass": long[m["id"]]["avg_mass"]})
        best = ranked[0]
        shutil.copy(snaps[best["id"]], root / "hof" / f"g{gen:03d}_{best['id']}.pt")
        state["log"].append({"gen": gen, "results": res, "best": best["id"], "time": time.time()})

        # ---- selection: bottom half -> mutated copies of the top half. A new architecture spent its
        # first generation imitating its parent and catching up, so it gets one generation of grace.
        n_keep = max(1, len(ranked) // 2)
        grace = [m for m in ranked if m.get("new_arch_gen") == gen]
        losers = [m for m in ranked[n_keep:] if m not in grace]      # only the bottom half can go
        survivors, children = [m for m in ranked if m not in losers], []
        for m in grace:
            print(f"  {m['id']} (new architecture) protected this generation", flush=True)
        for k, dead in enumerate(losers):
            parent = ranked[k % n_keep]
            g, note = mutate(parent["genome"], rng)
            cid = f"m{state['next_id']}"
            state["next_id"] += 1
            same_arch = g["arch"] == parent["genome"]["arch"]
            children.append({"id": cid, "genome": g, "ckpt": parent["ckpt"], "it": parent["it"] if same_arch else 0,
                             "seed": rng.randrange(10 ** 6), "origin": f"g{gen} child of {parent['id']}: {note}",
                             "history": [], **({} if same_arch else {"new_arch_gen": gen + 1})})
            print(f"  {dead['id']} out -> {cid} = {parent['id']} + {note}", flush=True)
        state["members"] = survivors + children
        state["gen"] = gen + 1
        state_f.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
