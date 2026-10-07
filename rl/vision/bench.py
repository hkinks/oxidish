"""Throughput of the ogar_py VecEnv: learner decisions per second (render + teacher + world ticks + opponents).

    python rl/vision/bench.py --worlds 32 --learners 4 --threads 4 8 10 --steps 300
"""
import argparse
import time

import numpy as np
import ogar_py


def run(args, threads):
    env = ogar_py.VecEnv(
        args.worlds, args.learners, opponents=[("c34", args.opponent, args.opponents)], teacher=args.teacher,
        px=args.px, threads=threads, seed=1, warmup=args.warmup, label=not args.no_label)
    env.set_teacher(True)
    env.reset()
    act = np.zeros((env.n, 4), np.float32)
    for _ in range(20):
        env.step(act)
    t0 = env.timing()
    t = time.perf_counter()
    for _ in range(args.steps):
        env.step(act)
    wall = time.perf_counter() - t
    t1 = env.timing()
    dec = args.steps * env.n
    row = {k: t1[k] - t0[k] for k in ("tick", "opponents", "snapshot", "render", "teacher")}
    tot = sum(row.values())
    print(f"threads {threads:2d}: {dec / wall:8.0f} decisions/s  ({wall:.1f}s)  "
          + "  ".join(f"{k} {1e6 * v / dec:.0f}us ({100 * v / tot:.0f}%)" for k, v in row.items())
          + "  [cpu-us per decision]")
    c = env.teacher_counters()
    if c.get("plan_steps"):
        print(f"           planner: deep eval on {100 * c['plan_gated'] / c['plan_steps']:.1f}% of steps, "
              f"{c['plan_us'] / max(c['plan_gated'], 1) / 1000:.2f} ms each, {c['plan_us'] / c['plan_steps']:.0f} us/decision avg")
    return dec / wall


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--learners", type=int, default=4)
    p.add_argument("--threads", type=int, nargs="+", default=[4, 8, 10])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--px", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1500, help="ticks before measuring (60 s of game time)")
    p.add_argument("--teacher", default="bot/plan_c34.json")
    p.add_argument("--opponent", default="bot/c3_4.json")
    p.add_argument("--opponents", type=int, default=4)
    p.add_argument("--no-label", action="store_true")
    a = p.parse_args()
    for th in a.threads:
        run(a, th)
