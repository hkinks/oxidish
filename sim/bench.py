"""Throughput benchmark: agent-steps/s of SimEnv (physics + observation encoding).

    uv run python -m sim.bench [--B 64 128 256] [--P 16] [--steps 60]

Actions are random headings (10% split / eject) so cells split, merge and eat as in training.
Also prints the physics-only rate and peak GPU memory.
"""

import argparse
import time

import torch

from .env import SimEnv


def rand_actions(n, dev):
    move = torch.randint(0, 17, (n,), device=dev)
    split = (torch.rand(n, device=dev) < 0.05).long()
    eject = (torch.rand(n, device=dev) < 0.05).long()
    return torch.stack([move, split, eject], 1)


def bench(B: int, P: int, steps: int, warm: int = 10, obs: bool = True) -> tuple[float, float]:
    env = SimEnv(B=B, P=P, seed=0)
    env.reset()
    dev = env.dev
    torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        a = rand_actions(env.n_agents, dev)
        if obs:
            env.step(a)
        else:  # physics only: skip the encoder
            enc, env.enc = env.enc, None
            _physics_only(env, a)
            env.enc = enc
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return B * P * steps / dt, torch.cuda.max_memory_allocated() / 2 ** 30


def _physics_only(env, a):
    import math
    w = env.w
    B, P = env.B, env.P
    mouse = torch.zeros(B, P, 2, device=env.dev)
    ang = 2 * math.pi * a[:, 0].clamp(max=16).float().view(B, P) / 16
    mouse[..., 0] = w.cx.view(B, P, -1).mean(-1) + torch.cos(ang) * 1000
    mouse[..., 1] = w.cy.view(B, P, -1).mean(-1) + torch.sin(ang) * 1000
    alive = w.alive_players()
    w.step(mouse, a[:, 1].view(B, P) > 0, a[:, 2].view(B, P) > 0, alive, respawn=~alive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    print(f"{'B':>5} {'P':>3} {'agents':>7} {'with obs':>14} {'physics only':>14} {'GPU GB':>7}")
    for B in args.B:
        full, mem = bench(B, args.P, args.steps)
        phys, _ = bench(B, args.P, args.steps, obs=False)
        print(f"{B:5d} {args.P:3d} {B * args.P:7d} {full:11,.0f}/s {phys:11,.0f}/s {mem:7.2f}")


if __name__ == "__main__":
    main()
