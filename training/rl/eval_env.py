"""In-env performance: learners driven purely by the student (or by the teacher, for the bar) in the endless field.

    python training/rl/eval_env.py --mode teacher --steps 12000            # the bar
    python training/rl/eval_env.py --mode student --ckpt runs/vision/bc1/best.pt --steps 12000

Opponent field = training.eval.endless field (champ, r2g9, safe, ramp_r2, ramp300) + 8 server bots; learners per world share
the world (like the 2 copies of a candidate). Reports mean mass, fraction of decisions as #1 in the world (leader),
deaths per learner-hour (1 decision = 2 ticks = 80 ms of game time). Run from the repo root.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import FIELD, ROOT, TEACHER, pin_cores
from evalval import load_model
from model import decode_target

import oxidish_gym

DT = 0.08  # seconds of game time per decision


def run(mode, ckpt, worlds, learners, steps, threads, seed, split_thr=None, no_split=False, warmup=0, log_every=0):
    label = mode == "teacher"
    env = oxidish_gym.VecEnv(worlds, learners, opponents=FIELD, teacher=TEACHER, px=256, threads=threads, seed=seed,
                         warmup=warmup, label=label)
    n = env.n
    env.set_teacher(label)
    if mode == "student":
        model, ck = load_model(ckpt)
        thr = split_thr if split_thr is not None else ck.get("split_thr", 0.5)
    obs = env.reset()
    mass_sum = np.zeros(n)
    leader_sum = np.zeros(n)
    top3_sum = np.zeros(n)
    deaths = np.zeros(n)
    splits = 0
    t0 = time.time()
    act = np.zeros((n, 4), np.float32)
    for t in range(steps):
        if mode == "student":
            view = torch.from_numpy(obs["view"]).cuda()
            wide = torch.from_numpy(obs["wide"]).cuda()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                heat, s, e, _ = model(view, wide)
            act[:, :2] = decode_target(heat.float()).cpu().numpy()
            act[:, 2] = 0 if no_split else (torch.sigmoid(s.float()) > thr).cpu().numpy()
            act[:, 3] = (torch.sigmoid(e.float()) > 0.5).cpu().numpy()
            splits += int(act[:, 2].sum())
        obs, _, done, info = env.step(act)
        mass_sum += info["mass"]
        leader_sum += info["leader"]
        top3_sum += info["rank"] < 3
        deaths += done
        if log_every and (t + 1) % log_every == 0:
            print(f"  {t + 1}/{steps} leader {leader_sum.sum() / n / (t + 1):.3f} mass {mass_sum.sum() / n / (t + 1):.0f} "
                  f"deaths/h {deaths.sum() / n / ((t + 1) * DT / 3600):.1f}  {(t + 1) * n / (time.time() - t0):.0f} dec/s",
                  flush=True)
    hours = steps * DT / 3600
    per_world = leader_sum.reshape(worlds, learners).sum(1) / (steps * learners)
    res = dict(mode=mode, ckpt=ckpt, worlds=worlds, learners=learners, steps=steps, game_min=steps * DT / 60,
               mean_mass=float(mass_sum.sum() / n / steps), leader_frac=float(leader_sum.sum() / n / steps),
               top3_frac=float(top3_sum.sum() / n / steps), deaths_per_hour=float(deaths.sum() / n / hours),
               leader_frac_world_sd=float(per_world.std()), seed=seed, wall_s=time.time() - t0)
    if mode == "student":
        res["splits_per_hour"] = splits / n / hours
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teacher", "student"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--learners", type=int, default=2)
    p.add_argument("--steps", type=int, default=12000, help="decisions per learner (12000 = 16 game minutes)")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--split-thr", type=float, default=None)
    p.add_argument("--no-split", action="store_true")
    p.add_argument("--out", default=None, help="append the result json line here")
    p.add_argument("--log-every", type=int, default=3000)
    a = p.parse_args()
    os.chdir(ROOT)
    pin_cores()
    r = run(a.mode, a.ckpt, a.worlds, a.learners, a.steps, a.threads, a.seed, a.split_thr, a.no_split,
            log_every=a.log_every)
    print(json.dumps(r))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
