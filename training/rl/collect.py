"""Collect behaviour-cloning shards from the oxidish_env VecEnv (M2 input).

    python training/rl/collect.py --out data/vision/bc0 --shards 4 --steps 500 --worlds 32 --learners 4 --threads 10

Each shard (shard_00000.npz) is time-major, T = --steps decisions for N = worlds * learners learners:
    view    uint8  [T, N, px, px]      camera frame at decision t (current frame only; the stack's previous frame is
                                       view[t-1] of the same learner, unless done[t-1] (new life) or t == 0)
    wide    uint8  [T, N, px/2, px/2]  2x wider view
    teacher float32 [T, N, 4]          BC label for the frame: target pixel (u, v), split, eject
    action  float32 [T, N, 4]          what was executed (= teacher, or teacher + gaussian pixel noise with --noise)
    reward  float32 [T, N]             change of own mass caused by that action (death: -mass)
    done    uint8   [T, N]             the action ended the life (frame t+1 is the new life's first frame)
    mass, rank, leader  [T, N]         at frame t
    first   uint8   [T, N]             1 where there is no previous frame (first step of a shard or new life)
Shards continue the same worlds, so consecutive shards of one run are consecutive in time (`first` marks the start).
`--noise S` executes teacher + N(0, S px) so the data also covers states the teacher alone would not reach
(DART-style); the label stays the clean teacher action. Prints bytes per decision and a per-hour size estimate.
"""
import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import oxidish_env


def save_shard(save, path, buf):
    if save is None:  # fast deflate (level 1), np.load-compatible .npz
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for k, v in buf.items():
                with z.open(k + ".npy", "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, v)
    else:
        save(path, **buf)
    return os.path.getsize(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--steps", type=int, default=500, help="decisions per shard (per learner)")
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--learners", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--px", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--teacher", default="bots/specs/plan_c34.json")
    p.add_argument("--opponent", default="bots/specs/c3_4.json")
    p.add_argument("--opponents", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.0, help="std (px) of noise added to executed targets")
    p.add_argument("--writers", type=int, default=2, help="shard writer threads (each pending shard holds its buffers in RAM)")
    p.add_argument("--compress", action="store_true", help="deflate level 1: ~50x smaller than raw")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    env = oxidish_env.VecEnv(a.worlds, a.learners, opponents=[("opp", a.opponent, a.opponents)], teacher=a.teacher,
                         px=a.px, threads=a.threads, seed=a.seed, warmup=a.warmup)
    env.set_teacher(a.noise == 0.0)
    rng = np.random.default_rng(a.seed)
    n, T, px = env.n, a.steps, a.px
    obs = env.reset()
    info = env.info()
    first = np.ones(n, np.uint8)
    save = None if a.compress else np.savez  # None = fast deflate
    writer, writes = ThreadPoolExecutor(a.writers), []
    t_start, total = time.perf_counter(), 0
    for shard in range(a.shards):
        buf = dict(
            view=np.empty((T, n, px, px), np.uint8), wide=np.empty((T, n, px // 2, px // 2), np.uint8),
            teacher=np.empty((T, n, 4), np.float32), action=np.empty((T, n, 4), np.float32),
            reward=np.empty((T, n), np.float32), done=np.empty((T, n), np.uint8),
            mass=np.empty((T, n), np.float32), rank=np.empty((T, n), np.int32),
            leader=np.empty((T, n), np.uint8), first=np.empty((T, n), np.uint8))
        for t in range(T):
            buf["view"][t] = obs["view"][:, 0]
            buf["wide"][t] = obs["wide"][:, 0]
            buf["teacher"][t] = info["teacher"]
            buf["mass"][t], buf["rank"][t], buf["leader"][t] = info["mass"], info["rank"], info["leader"]
            buf["first"][t] = first
            act = info["teacher"].copy()
            if a.noise > 0:
                act[:, :2] = np.clip(act[:, :2] + rng.normal(0, a.noise, (n, 2)), 0, px)
            buf["action"][t] = act
            obs, reward, done, info = env.step(act)
            buf["reward"][t], buf["done"][t] = reward, done
            first = done.copy()
        total += T * n
        path = os.path.join(a.out, f"shard_{shard:05d}.npz")
        while sum(not w.done() for w in writes) >= a.writers:
            time.sleep(0.05)  # back-pressure: bounded RAM
        writes.append(writer.submit(save_shard, save, path, buf))  # zlib / file IO release the GIL
        rate = total / (time.perf_counter() - t_start)
        print(f"shard {shard}: {T * n} decisions collected, {rate:.0f} decisions/s so far")
        first[:] = 1  # a new shard has no history for its first frame
    sizes = [w.result() for w in writes]
    per = sum(sizes) / total
    rate = total / (time.perf_counter() - t_start)
    print(f"{total} decisions, {sum(sizes) / 1e6:.0f} MB on disk = {per / 1e3:.1f} KB/decision "
          f"({'compressed' if a.compress else 'raw'}); at {rate:.0f} decisions/s that is {rate * 3600 / 1e6:.0f}M decisions/h "
          f"= {rate * 3600 * per / 1e9:.0f} GB/h")


if __name__ == "__main__":
    main()
