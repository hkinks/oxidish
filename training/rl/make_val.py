"""Collect the fixed held-out validation set once (teacher-driven, seeds different from training).

    python training/rl/make_val.py --out runs/vision/val --windows 4 --worlds 64 --learners 4 --threads 10

Writes win_XX.npz: time-major windows of T = --len + 1 consecutive decisions for N learners (the first row is only
the previous-frame context). Evaluated frames = windows * N * len (default 4 * 256 * 50 = 51k).
"""
import argparse
import os
import time
import zipfile

import numpy as np

from common import FIELD, TEACHER, pin_cores

import oxidish_gym


def save(path, buf):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for k, v in buf.items():
            with z.open(k + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, v)


def mine_events(env, a, obs, info, first):
    """Frames where the teacher splits, as time-major windows of T = 2 (previous frame, event frame)."""
    n = env.n
    prev_v, prev_w = obs["view"][:, 0].copy(), obs["wide"][:, 0].copy()
    out = dict(view=[], wide=[], teacher=[], first=[], mass=[], rank=[])
    steps = 0
    while sum(len(x) for x in out["mass"]) < a.events:
        split = info["teacher"][:, 2] > 0.5
        idx = np.nonzero(split & (first == 0))[0]
        if len(idx):
            out["view"].append(np.stack([prev_v[idx], obs["view"][idx, 0]]))
            out["wide"].append(np.stack([prev_w[idx], obs["wide"][idx, 0]]))
            tt = info["teacher"][idx]
            out["teacher"].append(np.stack([tt, tt]))
            out["first"].append(np.zeros((2, len(idx)), np.uint8))
            out["mass"].append(info["mass"][idx])
            out["rank"].append(info["rank"][idx])
        prev_v, prev_w = obs["view"][:, 0].copy(), obs["wide"][:, 0].copy()
        obs, _, done, info = env.step(info["teacher"].copy())
        first = done.copy()
        steps += 1
        if steps % 2000 == 0:
            print(f"{steps} steps, {sum(len(x) for x in out['mass'])} events", flush=True)
    buf = {k: np.concatenate(v, 1) if k not in ("mass", "rank") else np.concatenate(v) for k, v in out.items()}
    buf["mass"] = np.stack([buf["mass"]] * 2)
    buf["rank"] = np.stack([buf["rank"]] * 2)
    buf["first"][0] = 1
    save(os.path.join(a.out, "events.npz"), buf)
    print(f"{buf['view'].shape[1]} split events saved", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/vision/val")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--len", type=int, default=50)
    p.add_argument("--gap", type=int, default=700, help="decisions between windows")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--learners", type=int, default=4)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--seed", type=int, default=777000)
    p.add_argument("--events", type=int, default=0, help="instead: mine this many teacher-split frames (+ previous frame)")
    a = p.parse_args()
    pin_cores()
    os.makedirs(a.out, exist_ok=True)
    env = oxidish_gym.VecEnv(a.worlds, a.learners, opponents=FIELD, teacher=TEACHER, px=256, threads=a.threads,
                         seed=a.seed, warmup=0)
    env.set_teacher(True)
    obs = env.reset()
    info = env.info()
    n, T = env.n, a.len + 1
    first = np.ones(n, np.uint8)
    t0 = time.time()
    if a.events:
        return mine_events(env, a, obs, info, first)
    steps = 0
    for w in range(a.windows):
        for _ in range(a.gap):
            obs, _, done, info = env.step(info["teacher"].copy())
            first = done.copy()
            steps += 1
        buf = dict(view=np.empty((T, n, 256, 256), np.uint8), wide=np.empty((T, n, 128, 128), np.uint8),
                   teacher=np.empty((T, n, 4), np.float32), first=np.empty((T, n), np.uint8),
                   mass=np.empty((T, n), np.float32), rank=np.empty((T, n), np.int32))
        for t in range(T):
            buf["view"][t] = obs["view"][:, 0]
            buf["wide"][t] = obs["wide"][:, 0]
            buf["teacher"][t] = info["teacher"]
            buf["first"][t] = first
            buf["mass"][t], buf["rank"][t] = info["mass"], info["rank"]
            obs, _, done, info = env.step(info["teacher"].copy())
            first = done.copy()
            steps += 1
        buf["first"][0] = 1  # row 0 has no usable predecessor inside the file; it is never evaluated
        save(os.path.join(a.out, f"win_{w:02d}.npz"), buf)
        tt = buf["teacher"][1:]
        print(f"window {w}: {a.len * n} frames; teacher split {tt[..., 2].mean():.4f} eject {tt[..., 3].mean():.4f} "
              f"mean mass {buf['mass'].mean():.0f}  ({steps * n / (time.time() - t0):.0f} dec/s)", flush=True)


if __name__ == "__main__":
    main()
