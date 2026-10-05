"""Head-to-head tournament of sim-trained nets in shared simulator arenas.

    uv run python -m sim.tourney --nets a=runs/rl/x/latest.pt b=runs/rl/y/latest.pt [--b 16 --p 50]

Every arena holds every net: the slots are dealt round-robin (shifted per arena so no net
always gets the same spawn order), the last `--bots` slots are scripted bots. Players start at
the default size and respawn on death, as on a server. Per net: mean log of its average mass
per arena (the arena tournament's log_mass, which rewards being big *and* staying alive), its
average mass, deaths and kills per slot, with a 95% interval over arenas.
"""

import argparse
import json
import math

import torch

from rl.ppo_sim import from_checkpoint, obs_for, obs_kind
from sim.bots import farmer_actions, scripted_actions
from sim.env import SimEnv


@torch.no_grad()
def run(nets: dict[str, str], B: int = 16, P: int = 50, steps: int = 3000, bots: int = 5,
        seed: int = 123, dev: str = "cuda", bot_kind: str = "scripted") -> dict:
    """Shared tournament: the nets (name -> checkpoint) and `bots` scripted bots per arena fill B arenas of P
    slots; returns per-net mass statistics (fitness = mean log of average mass)."""
    names = list(nets)
    models = [from_checkpoint(path, dev).eval() for path in nets.values()]
    for m in models:
        m.binary_eps = 0.0
    kinds = {obs_kind(m) for m in models}
    env = SimEnv(B=B, P=P, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0,
                 obs="both" if len(kinds) > 1 else kinds.pop(), bot_kind=bot_kind)
    M, n_play = len(models), P - bots
    owner = torch.full((B, P), -1, dtype=torch.long, device=dev)        # net index per slot, -1 = bot
    for b in range(B):
        for j in range(n_play):
            owner[b, j] = (j + b) % M
    g, s, _, _, _ = env.reset()
    mass_sum = torch.zeros(B, P, device=dev)
    deaths = torch.zeros(B, P, device=dev)
    splits = torch.zeros(B, P, device=dev)
    flat_owner = owner.reshape(-1)
    idx = [(flat_owner == k).nonzero()[:, 0] for k in range(M)]
    a = torch.zeros(B * P, 3, dtype=torch.long, device=dev)
    for step in range(steps):
        for m, ix in zip(models, idx):
            if step % m.skip == 0:          # a frame-skip net decides every m.skip steps
                a[ix] = m.act(obs_for(m, env, g)[ix], s[ix], amp=True)[0]
            else:                           # ... and holds its move; split / eject fire once
                a[ix, 1:] = 0
        if bots:
            a.view(B, P, 3)[:, n_play:] = env.bot_actions()[:, n_play:]
        splits += ((a[:, 1] > 0) & env.prev_alive.reshape(-1)).view(B, P).float()
        g, s, _, d, _ = env.step(a)
        mass_sum += env.prev_mass
        deaths += d.view(B, P).float()
    kills = env.w.kstat[0, :, :P]
    pops = env.w.popped[:, :P]
    out = {}
    for k, name in enumerate(names):
        sel = owner == k
        n_slots = sel.sum(1).clamp(min=1)
        avg = (mass_sum * sel).sum(1) / n_slots / steps                  # per arena
        lm = torch.log(avg.clamp(min=1.0))
        ci = 1.96 * lm.std().item() / math.sqrt(B) if B > 1 else float("nan")
        out[name] = {"log_mass": lm.mean().item(), "ci": ci, "avg_mass": avg.mean().item(),
                     "deaths": ((deaths * sel).sum(1) / n_slots).mean().item(),
                     "kills": ((kills * sel).sum(1) / n_slots).mean().item(),
                     "splits_per_min": ((splits * sel).sum(1) / n_slots).mean().item() / (steps * 0.08 / 60),
                     "pops_per_min": ((pops * sel).sum(1) / n_slots).mean().item() / (steps * 0.08 / 60)}
    if bots:
        sel = owner < 0
        out["bots"] = {"avg_mass": ((mass_sum * sel).sum(1) / sel.sum(1).clamp(min=1) / steps).mean().item(),
                       "deaths": ((deaths * sel).sum(1) / sel.sum(1).clamp(min=1)).mean().item(),
                       "pops_per_min": ((pops * sel).sum(1) / sel.sum(1).clamp(min=1)).mean().item() / (steps * 0.08 / 60)}
    return out


@torch.no_grad()
def farm(nets: dict[str, str], B: int = 16, steps: int = 1500, seed: int = 5, dev: str = "cuda",
         free: bool = False, start: float = 10.0, gain: bool = False) -> dict:
    """Farming alone: one player per arena, no enemies, no splits/ejects; mean mass after `steps`.
    free=True: the net may split and eject too (smaller cells move faster: farming speed by any means).
    start: the starting mass (split-farming only pays for big cells: from 1000 mass splitting adds
    ~25%, from 10 it costs ~33%). gain=True: report the mass gained instead of the final mass.
    (For scale: path "script" is the nearest-pellet farmer, sim.bots.farmer_actions, deciding
    every 2 steps: ~300 mass.)"""
    out = {}
    for name, path in nets.items():
        m = None
        if path != "script":                # "script": the scripted bot (sim.bots) as the reference
            m = from_checkpoint(path, dev).eval()
            m.binary_eps = 0.0
        env = SimEnv(B=B, P=1, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0,
                     obs=obs_kind(m) if m is not None else "grid")
        if start > 10:
            env.spawn_mass[:] = torch.tensor([start, start], device=dev)
        g, s, _, _, _ = env.reset()
        m0 = env.prev_mass.mean().item()
        for step in range(steps):
            if m is None:
                if step % 2 == 0:
                    a = farmer_actions(env)
            elif step % m.skip == 0:
                a = m.act(g, s, amp=True)[0]
            if not free or m is None:
                a[:, 1:] = 0
            elif step % m.skip:
                a[:, 1:] = 0        # held between decisions: split / eject fire once
            g, s, _, _, _ = env.step(a)
        out[name] = env.prev_mass.mean().item() - (m0 if gain else 0.0)
    return out


@torch.no_grad()
def farm_batch(nets: dict[str, str], B: int = 64, steps: int = 1500, seed: int = 5, dev: str = "cuda",
               free: bool = False, start: float = 10.0, gain: bool = False, guard: int = 0) -> dict:
    """farm() for many nets at once: one simulator, `B` solo arenas per net (one pass instead of one
    per net; same numbers up to the arenas' random draws). guard: play under the play-time guards
    (sim.bots.guard_mouse; 1+ = no virus pops), as the nets are deployed."""
    names = list(nets)
    models = [from_checkpoint(path, dev).eval() for path in nets.values()]
    for m in models:
        m.binary_eps = 0.0
    kinds = {obs_kind(m) for m in models}
    env = SimEnv(B=B * len(models), P=1, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0,
                 obs="both" if len(kinds) > 1 else kinds.pop())
    if guard:
        env.guard = torch.ones(env.B, env.NL, dtype=torch.bool, device=dev)
        env.guard_level = guard
    if start > 10:
        env.spawn_mass[:] = torch.tensor([start, start], device=dev)
    g, s, _, _, _ = env.reset()
    m0 = env.prev_mass.view(len(models), B).mean(1)
    a = torch.zeros(B * len(models), 3, dtype=torch.long, device=dev)
    for step in range(steps):
        for k, m in enumerate(models):
            ix = slice(k * B, (k + 1) * B)
            if step % m.skip == 0:
                a[ix] = m.act(obs_for(m, env, g)[ix], s[ix], amp=True)[0]
                if not free:
                    a[ix, 1:] = 0
            else:
                a[ix, 1:] = 0                # held between decisions: split / eject fire once
        g, s, _, _, _ = env.step(a)
    end = env.prev_mass.view(len(models), B).mean(1)
    return {n: (end[k] - (m0[k] if gain else 0.0)).item() for k, n in enumerate(names)}


@torch.no_grad()
def hunt_batch(nets: dict[str, str], B: int = 32, P: int = 16, steps: int = 1500, start: float = 1000.0,
               prey: tuple = (20.0, 300.0), seed: int = 6, dev: str = "cuda", prey_nets: list | None = None) -> dict:
    """Hunting test: per net `B` arenas, each with that net as one hunter starting at `start` mass among
    P - 1 prey that start and respawn at `prey` mass: scripted bots (sim.bots: they flee threats) and,
    with `prey_nets` (checkpoints, e.g. the evaders' newest), the second half of the prey played by those
    nets in turn - prey that has learned to run. Per net: mass gained in `steps` (1500 = 2 min), kills
    and splits per minute. One simulator for all nets."""
    names = list(nets)
    models = [from_checkpoint(path, dev).eval() for path in nets.values()]
    preys = [from_checkpoint(path, dev).eval() for path in (prey_nets or [])]
    for m in models + preys:
        m.binary_eps = 0.0
    M = len(models)
    kinds = {obs_kind(m) for m in models + preys}
    env = SimEnv(B=B * M, P=P, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0, max_player_mass=0,
                 obs="both" if len(kinds) > 1 else kinds.pop())
    env.spawn_mass[:, 0] = torch.tensor([start, start], device=dev)
    env.spawn_mass[:, 1:] = torch.tensor(list(prey), device=dev)
    g, s, _, _, _ = env.reset()
    hunters = torch.arange(B * M, device=dev) * P                 # slot 0 of every arena
    half = P // 2                                                  # prey slots half..P-1: the prey nets
    prey_groups = []
    if preys:
        slots = (torch.arange(B * M, device=dev)[:, None] * P + torch.arange(half, P, device=dev)[None]).reshape(-1)
        prey_groups = [(pm, slots[j::len(preys)]) for j, pm in enumerate(preys)]
    m0 = env.prev_mass[:, 0].view(M, B).mean(1)
    a = torch.zeros(B * M * P, 3, dtype=torch.long, device=dev)
    splits = torch.zeros(M, device=dev)
    for step in range(steps):
        a.view(B * M, P, 3)[:, 1:] = env.bot_actions()[:, 1:]
        for pm, ix in prey_groups:
            if step % pm.skip == 0:
                pa = pm.act(obs_for(pm, env, g)[ix], s[ix], amp=True)[0]
                pa[:, 2] = 0
                pm._held = pa
            pa = pm._held.clone()
            if step % pm.skip:
                pa[:, 1:] = 0
            a[ix] = pa
        for k, m in enumerate(models):
            ix = hunters[k * B:(k + 1) * B]
            if step % m.skip == 0:
                a[ix] = m.act(obs_for(m, env, g)[ix], s[ix], amp=True)[0]
                a[ix, 2] = 0
                splits[k] += ((a[ix, 1] > 0) & env.prev_alive.reshape(-1)[ix]).sum()
            else:
                a[ix, 1:] = 0
        g, s, _, _, _ = env.step(a)
    end = env.prev_mass[:, 0].view(M, B).mean(1)
    kills = env.w.kstat[0][:, 0].view(M, B).mean(1)
    mins = steps * 0.08 / 60
    return {n: {"gain": (end[k] - m0[k]).item(), "kills_per_min": kills[k].item() / mins,
                "splits_per_min": splits[k].item() / B / mins} for k, n in enumerate(names)}


@torch.no_grad()
def evade_batch(nets: dict[str, str], B: int = 32, P: int = 8, steps: int = 1500, start: float = 200.0,
                hunters: tuple = (1500.0, 4000.0), seed: int = 8, dev: str = "cuda") -> dict:
    """Farming under threat: per net `B` arenas, each with that net as one player (starting and
    respawning at `start` mass) among P - 1 big scripted hunters (sim.bots: they chase anything they
    can eat) that start and respawn at `hunters` mass. Per net: mean mass at the end of `steps`
    (1500 = 2 min), and deaths per minute. One simulator for all nets."""
    names = list(nets)
    models = [from_checkpoint(path, dev).eval() for path in nets.values()]
    for m in models:
        m.binary_eps = 0.0
    M = len(models)
    kinds = {obs_kind(m) for m in models}
    env = SimEnv(B=B * M, P=P, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0, max_player_mass=0,
                 obs="both" if len(kinds) > 1 else kinds.pop())
    env.spawn_mass[:, 0] = torch.tensor([start, start], device=dev)
    env.spawn_mass[:, 1:] = torch.tensor(list(hunters), device=dev)
    g, s, _, _, _ = env.reset()
    me = torch.arange(B * M, device=dev) * P
    a = torch.zeros(B * M * P, 3, dtype=torch.long, device=dev)
    deaths = torch.zeros(M, device=dev)
    splits = torch.zeros(M, device=dev)
    for step in range(steps):
        a.view(B * M, P, 3)[:, 1:] = env.bot_actions()[:, 1:]
        for k, m in enumerate(models):
            ix = me[k * B:(k + 1) * B]
            if step % m.skip == 0:
                a[ix] = m.act(obs_for(m, env, g)[ix], s[ix], amp=True)[0]
                a[ix, 2] = 0
                splits[k] += ((a[ix, 1] > 0) & env.prev_alive.reshape(-1)[ix]).sum()
            else:
                a[ix, 1:] = 0
        g, s, _, d, _ = env.step(a)
        deaths += d[me].view(M, B).float().sum(1)
    end = env.prev_mass[:, 0].view(M, B).mean(1)
    mins = steps * 0.08 / 60
    return {n: {"mass": end[k].item(), "deaths_per_min": deaths[k].item() / B / mins,
                "splits_per_min": splits[k].item() / B / mins} for k, n in enumerate(names)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nets", nargs="+", required=True, help="name=checkpoint")
    p.add_argument("--b", type=int, default=16)
    p.add_argument("--p", type=int, default=50)
    p.add_argument("--steps", type=int, default=3000, help="80 ms each: 3000 = 4 min of game time")
    p.add_argument("--bots", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--vram-gb", type=float, default=6.0)
    p.add_argument("--json", help="also write the results here")
    args = p.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2 ** 30 / total))
    nets = dict(x.split("=", 1) for x in args.nets)
    res = run(nets, args.b, args.p, args.steps, args.bots, args.seed)
    print(f"{'net':<16} {'log_mass':>14} {'avg_mass':>9} {'deaths':>7} {'kills':>6} {'splits/min':>10}")
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]):
        print(f"{name:<16} {r['log_mass']:7.3f} +-{r['ci']:5.3f} {r['avg_mass']:9.0f} {r['deaths']:7.2f} {r['kills']:6.2f} {r['splits_per_min']:10.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
