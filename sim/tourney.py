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

from rl.ppo_sim import from_checkpoint
from sim.bots import scripted_actions
from sim.env import SimEnv


@torch.no_grad()
def run(nets: dict[str, str], B: int = 16, P: int = 50, steps: int = 3000, bots: int = 5,
        seed: int = 123, dev: str = "cuda") -> dict:
    names = list(nets)
    models = [from_checkpoint(path, dev).eval() for path in nets.values()]
    for m in models:
        m.binary_eps = 0.0
    env = SimEnv(B=B, P=P, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0)
    M, n_play = len(models), P - bots
    owner = torch.full((B, P), -1, dtype=torch.long, device=dev)        # net index per slot, -1 = bot
    for b in range(B):
        for j in range(n_play):
            owner[b, j] = (j + b) % M
    g, s, _, _, _ = env.reset()
    mass_sum = torch.zeros(B, P, device=dev)
    deaths = torch.zeros(B, P, device=dev)
    flat_owner = owner.reshape(-1)
    idx = [(flat_owner == k).nonzero()[:, 0] for k in range(M)]
    for _ in range(steps):
        a = torch.zeros(B * P, 3, dtype=torch.long, device=dev)
        for m, ix in zip(models, idx):
            a[ix] = m.act(g[ix], s[ix], amp=True)[0]
        if bots:
            a.view(B, P, 3)[:, n_play:] = scripted_actions(env.w, env.bot_split_prob)[:, n_play:]
        g, s, _, d, _ = env.step(a)
        mass_sum += env.prev_mass
        deaths += d.view(B, P).float()
    kills = env.w.kstat[0, :, :P]
    out = {}
    for k, name in enumerate(names):
        sel = owner == k
        n_slots = sel.sum(1).clamp(min=1)
        avg = (mass_sum * sel).sum(1) / n_slots / steps                  # per arena
        lm = torch.log(avg.clamp(min=1.0))
        ci = 1.96 * lm.std().item() / math.sqrt(B) if B > 1 else float("nan")
        out[name] = {"log_mass": lm.mean().item(), "ci": ci, "avg_mass": avg.mean().item(),
                     "deaths": ((deaths * sel).sum(1) / n_slots).mean().item(),
                     "kills": ((kills * sel).sum(1) / n_slots).mean().item()}
    return out


@torch.no_grad()
def farm(nets: dict[str, str], B: int = 16, steps: int = 1500, seed: int = 5, dev: str = "cuda") -> dict:
    """Farming alone: one player per arena, no enemies, no splits/ejects; mean mass after `steps`.
    (For scale: going straight to the nearest pellet reaches ~320 in 1500 steps.)"""
    out = {}
    for name, path in nets.items():
        m = from_checkpoint(path, dev).eval()
        m.binary_eps = 0.0
        env = SimEnv(B=B, P=1, seed=seed, episode_steps=10 ** 9, curriculum_p=0.0)
        g, s, _, _, _ = env.reset()
        for _ in range(steps):
            a = m.act(g, s, amp=True)[0]
            a[:, 1:] = 0
            g, s, _, _, _ = env.step(a)
        out[name] = env.prev_mass.mean().item()
    return out


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
    print(f"{'net':<16} {'log_mass':>14} {'avg_mass':>9} {'deaths':>7} {'kills':>6}")
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["log_mass"]):
        print(f"{name:<16} {r['log_mass']:7.3f} +-{r['ci']:5.3f} {r['avg_mass']:9.0f} {r['deaths']:7.2f} {r['kills']:6.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
