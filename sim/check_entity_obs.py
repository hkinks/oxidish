"""Check that the real-server entity encoder (rl/entity_obs.py, from a server Snapshot) produces the
same tokens as the simulator's (sim/entities.py) for the same scene.

    uv run python -m sim.check_entity_obs

Plays a few hundred steps of a simulator arena, then for several agents builds the Snapshot a
MultiOgar client would receive (objects in the server's view box, sizes floored to integers,
positions truncated) and feeds the same two consecutive frames to both encoders. Tokens are matched
per kind by position; reported: unmatched tokens and the largest difference per feature.
"""

import math

import numpy as np
import torch

from bot.game import Cell, Snapshot
from rl.entity_obs import RealEntityEncoder
from sim.bots import scripted_actions
from sim.engine import ES
from sim.entities import FD, K_CELL, K_EJECT, K_FOOD, K_VIRUS, REGIONS
from sim.env import SimEnv
from sim.obs import MIN_SCALE, VIEW_X, VIEW_Y

FEATURES = ["valid", "t_food", "t_cell", "t_own", "t_virus", "t_eject", "dx", "dy", "dist", "vx", "vy",
            "logmass", "logratio", "can_eat", "can_be_eaten", "split_kill", "split_threat", "merge",
            "owner_mass", "virus_fill", "last_target", "dist_to_own_cell"]
KINDS = [("own", 0, 16), ("cell", 16, K_CELL), ("food", 16 + K_CELL, K_FOOD),
         ("virus", 16 + K_CELL + K_FOOD, K_VIRUS), ("eject", 16 + K_CELL + K_FOOD + K_VIRUS, K_EJECT),
         ("region", 16 + K_CELL + K_FOOD + K_VIRUS + K_EJECT, REGIONS * REGIONS)]


def snapshot(env: SimEnv, b: int, p: int) -> Snapshot:
    """What a MultiOgar client of player p in arena b would see now."""
    w = env.w
    K = w.K
    al = w.calive[b].cpu().numpy()
    cx, cy, cr2, cid = (t[b].cpu().numpy() for t in (w.cx, w.cy, w.cr2, w.cid))
    size = np.sqrt(cr2)
    own = [j for j in range(p * K, (p + 1) * K) if al[j]]
    mine = [Cell(int(cid[j]), math.trunc(float(cx[j])), math.trunc(float(cy[j])), math.floor(float(size[j])), "mine", f"p{p}") for j in own]
    if not mine:
        return Snapshot([], [], {}, 1920, 1080, {}, True)
    sscale = max(min(64.0 / max(sum(size[j] for j in own), 1.0), 1.0) ** 0.4, MIN_SCALE)
    hx, hy = VIEW_X / sscale, VIEW_Y / sscale
    ccx = sum(cx[j] for j in own) / len(own)
    ccy = sum(cy[j] for j in own) / len(own)
    vis = lambda x, y, s: abs(x - ccx) <= hx + s and abs(y - ccy) <= hy + s
    others = []
    for j in np.nonzero(al)[0]:
        if j in own or not vis(cx[j], cy[j], size[j]):
            continue
        others.append(Cell(int(cid[j]), math.trunc(float(cx[j])), math.trunc(float(cy[j])), math.floor(float(size[j])), "player", f"p{j // K}"))
    for alive, xs, ys, kind, rr, base in (
            (w.falive, w.fx, w.fy, "food", None, 30_000_000),
            (w.valive, w.vx, w.vy, "virus", w.vr2, 20_000_000),
            (w.ealive, w.ex, w.ey, "ejected", None, 10_000_000)):
        a_, x_, y_ = alive[b].cpu().numpy(), xs[b].cpu().numpy(), ys[b].cpu().numpy()
        r_ = np.sqrt(rr[b].cpu().numpy()) if rr is not None else None
        ids = {"virus": w.vid, "ejected": w.eid}.get(kind)
        ids = ids[b].cpu().numpy() if ids is not None else None
        for j in np.nonzero(a_)[0]:
            s = r_[j] if r_ is not None else (10.0 if kind == "food" else ES)
            if not vis(x_[j], y_[j], s):
                continue
            oid = base + (int(ids[j]) if ids is not None else int(j))
            others.append(Cell(oid, math.trunc(float(x_[j])), math.trunc(float(y_[j])), math.floor(float(s)), kind, ""))
    half = w.map / 2
    return Snapshot(mine, others, {}, 1920, 1080, {"left": -half, "right": half, "top": -half, "bottom": half}, False)


def compare(sim_tok: np.ndarray, real_tok: np.ndarray):
    """Match tokens per kind by position; returns per-feature max |diff| and unmatched counts."""
    diffs = np.zeros(FD)
    unmatched = {}
    for name, lo, n in KINDS:
        S = [t for t in sim_tok[lo:lo + n] if t[0] > 0.5]
        R = [t for t in real_tok[lo:lo + n] if t[0] > 0.5]
        miss = 0
        for t in S:
            d = [np.hypot(r[6] - t[6], r[7] - t[7]) for r in R]
            if not d or min(d) > 0.003:                    # 3 world units
                miss += 1
                continue
            r = R[int(np.argmin(d))]
            diffs = np.maximum(diffs, np.abs(r - t))
        unmatched[name] = (miss, len(S), len(R))
    return diffs, unmatched


def main() -> None:
    torch.manual_seed(0)
    env = SimEnv(B=4, P=20, seed=3, episode_steps=10 ** 9, obs="entity", curriculum_p=0.5)
    env.reset()
    for _ in range(300):
        env.step(scripted_actions(env.w, 0.5).reshape(-1, 3))
    agents = [(b, p) for b in range(4) for p in (0, 5, 11)]
    encs = {a: RealEntityEncoder() for a in agents}
    for a in agents:                                   # frame 1: history for velocities
        encs[a].encode(snapshot(env, *a))
    env.step(scripted_actions(env.w, 0.5).reshape(-1, 3))
    ent = env.ent.float().view(env.B, env.P, -1, FD).cpu().numpy()
    total = np.zeros(FD)
    for a in agents:
        s = snapshot(env, *a)
        if s.dead:
            continue
        real, _, _, _, _ = encs[a].encode(s)
        d, un = compare(ent[a], real)
        total = np.maximum(total, d)
        print(f"arena {a[0]} player {a[1]:2d}: unmatched sim tokens (missing / sim / real) " +
              "  ".join(f"{k} {m}/{n}/{r}" for k, (m, n, r) in un.items()))
    print("largest feature differences over matched tokens:")
    for i in np.argsort(-total)[:10]:
        print(f"  {FEATURES[i]:<17} {total[i]:.4f}")


if __name__ == "__main__":
    main()
