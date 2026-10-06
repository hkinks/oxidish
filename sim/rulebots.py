"""Arena policies (rule bots, and any policy that acts on a bot.game.Snapshot) playing in the GPU sim.

    uv run python -m sim.rulebots --agents champ=runs/final_champion.json:1 r2g9=runs/r2/champion_009.json:1 \
        --b 32 --steps 11250 --food-mult 2

The same lineup as arena.tournament, but every match is a sim arena: the policy code is unchanged
(each step builds the Snapshot the real client would see: the cells inside the server's view box)
and the last `--bots` slots are the sim's MultiOgar bots. Scores as arena.tournament: per match
log(average mass, dead time counting 0), and head-to-head = share of arenas where the row's
log-mass beats the column's. One step = 2 ticks (sync 2): 11250 steps = 15 game-min.
"""

import argparse
import json
import math
import statistics
import time

import numpy as np
import torch

from arena.match import limit_cpus, parse_lineup
from arena.policies import build
from bot.game import Cell, Snapshot

from .bots import MO_MIN_SCALE, MO_VIEW_X, MO_VIEW_Y, MultiOgarBots
from .engine import EJECT_R2, FOOD_R2, World

DIRS = 16
FOOD_ID, EJECT_ID = 1 << 40, 1 << 41   # keep pellet / blob ids apart from cell and virus ids


def view(w_np: dict, b: int, p: int, K: int):
    """Our cells and the view box (centre x, y, half widths) of player p in arena b, or None if dead."""
    sl = slice(p * K, (p + 1) * K)
    al = w_np["calive"][b, sl]
    if not al.any():
        return None
    x, y, r2, ids = w_np["cx"][b, sl][al], w_np["cy"][b, sl][al], w_np["cr2"][b, sl][al], w_np["cid"][b, sl][al]
    size = np.sqrt(r2)
    scale = max(min(64.0 / max(size.sum(), 1.0), 1.0) ** 0.4, MO_MIN_SCALE)
    return (x, y, size, ids), (x.mean(), y.mean(), MO_VIEW_X / scale, MO_VIEW_Y / scale)


def snapshot(w_np: dict, b: int, p: int, K: int, names: list[str], border: dict) -> Snapshot:
    v = view(w_np, b, p, K)
    if v is None:
        return Snapshot([], [], {}, 0, 0, border, True)
    (x, y, size, ids), (mx, my, hx, hy) = v
    name = names[p]
    mine = [Cell(int(i), float(a), float(c), float(s), "mine", name) for a, c, s, i in zip(x, y, size, ids)]
    others = []

    def add(px, py, ps, alive, idv, kind, owner=None):
        ok = alive & (np.abs(px - mx) <= hx + ps) & (np.abs(py - my) <= hy + ps)
        for j in np.nonzero(ok)[0]:
            nm = names[owner[j]] if owner is not None else ""
            others.append(Cell(int(idv[j]), float(px[j]), float(py[j]), float(ps[j]), kind, nm))

    C = w_np["cx"].shape[1]
    owner = np.arange(C) // K
    other_alive = w_np["calive"][b] & (owner != p)
    add(w_np["cx"][b], w_np["cy"][b], np.sqrt(w_np["cr2"][b]), other_alive, w_np["cid"][b], "player", owner)
    F = w_np["fx"].shape[1]
    add(w_np["fx"][b], w_np["fy"][b], np.full(F, math.sqrt(FOOD_R2)), w_np["falive"][b], FOOD_ID + np.arange(F), "food")
    add(w_np["vx"][b], w_np["vy"][b], np.sqrt(w_np["vr2"][b]), w_np["valive"][b], w_np["vid"][b], "virus")
    add(w_np["ex"][b], w_np["ey"][b], np.full(w_np["ex"].shape[1], math.sqrt(EJECT_R2)), w_np["ealive"][b],
        EJECT_ID + w_np["eid"][b], "ejected")
    cam = {"x": float(mx), "y": float(my), "z": 1.0}
    return Snapshot(mine, others, cam, int(2 * hx), int(2 * hy), border, False)


FIELDS = ("cx", "cy", "cr2", "calive", "cid", "fx", "fy", "falive", "vx", "vy", "vr2", "valive", "vid",
          "ex", "ey", "ealive", "eid")


@torch.no_grad()
def run(lineup: list[tuple[str, dict]], B: int = 16, steps: int = 11250, bots: int = 8, food_mult: float = 2.0,
        seed: int = 1, dev: str = "cuda", log_every: int = 1000) -> dict:
    R = len(lineup)
    P = R + bots
    w = World(B, P, dev, 7000.0, food_mult, seed=seed)
    mo = MultiOgarBots(w) if bots else None
    names = [n for n, _ in lineup] + [f"bot{j}" for j in range(bots)]
    pols = [[build(spec) for _, spec in lineup] for _ in range(B)]
    border = {"left": w.lo, "right": w.hi, "top": w.lo, "bottom": w.hi}
    K = w.K
    mass_sum = np.zeros((B, R))
    peak = np.zeros((B, R))
    deaths = np.zeros((B, R))
    was = np.zeros((B, R), dtype=bool)
    mouse = torch.zeros(B, P, 2, device=w.dev)
    split = torch.zeros(B, P, dtype=torch.bool, device=w.dev)
    eject = torch.zeros(B, P, dtype=torch.bool, device=w.dev)
    m_np = np.zeros((B, R, 2), dtype=np.float32)
    s_np = np.zeros((B, R), dtype=bool)
    e_np = np.zeros((B, R), dtype=bool)
    t0 = time.time()
    for step in range(steps):
        w_np = {f: getattr(w, f).cpu().numpy() for f in FIELDS}
        for b in range(B):
            for p in range(R):
                s = snapshot(w_np, b, p, K, names, border)
                if s.dead:
                    if was[b, p]:
                        deaths[b, p] += 1
                        pols[b][p].reset() if hasattr(pols[b][p], "reset") else None
                    was[b, p] = False
                    s_np[b, p] = e_np[b, p] = False
                    continue
                was[b, p] = True
                tm = s.total_mass
                mass_sum[b, p] += tm
                peak[b, p] = max(peak[b, p], tm)
                out = pols[b][p].act(s)
                m_np[b, p] = out[0], out[1]
                s_np[b, p] = bool(out[2])
                e_np[b, p] = bool(out[3]) if len(out) > 3 else False
        mouse[:, :R] = torch.from_numpy(m_np).to(w.dev)
        split[:, :R] = torch.from_numpy(s_np).to(w.dev)
        eject[:, :R] = torch.from_numpy(e_np).to(w.dev)
        if bots:  # as sim.env: MultiOgar bots aim at a far point along their heading
            ba = mo()[:, R:]
            r2 = (w.cr2 * w.calive).view(B, P, K)[:, R:]
            n = r2.sum(-1).clamp(min=1)
            cx = (w.cx.view(B, P, K)[:, R:] * r2).sum(-1) / n
            cy = (w.cy.view(B, P, K)[:, R:] * r2).sum(-1) / n
            ang = 2 * math.pi * ba[..., 0].clamp(max=DIRS).float() / DIRS
            far = torch.where(ba[..., 0] >= DIRS, torch.zeros_like(cx), torch.full_like(cx, 1000.0))
            mouse[:, R:, 0] = cx + torch.cos(ang) * far
            mouse[:, R:, 1] = cy + torch.sin(ang) * far
            split[:, R:] = ba[..., 1] > 0
        alive = w.alive_players()
        w.step(mouse, split, eject, alive, respawn=~alive)
        if log_every and (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  step {step + 1}/{steps}  {el:.0f}s  ({(step + 1) / el:.1f} steps/s)", flush=True)
    avg = mass_sum / steps
    return {"names": [n for n, _ in lineup], "log_mass": np.log(np.maximum(avg, 1.0)), "avg_mass": avg,
            "peak": peak, "deaths": deaths}


def report(res: dict) -> None:
    names, lm = res["names"], res["log_mass"]
    B = lm.shape[0]
    base = [n.rsplit("-", 1)[0] for n in names]
    keys = list(dict.fromkeys(base))
    team = {k: lm[:, [i for i, n in enumerate(base) if n == k]].mean(1) for k in keys}
    print(f"\n{B} sim arenas\n{'policy':<10} {'log_mass':>14} {'avg_mass':>9} {'peak':>8} {'deaths':>7}")
    for k in sorted(keys, key=lambda k: -team[k].mean()):
        ix = [i for i, n in enumerate(base) if n == k]
        ci = 1.96 * statistics.stdev(team[k]) / math.sqrt(B) if B > 1 else 0.0
        print(f"{k:<10} {team[k].mean():6.2f} +- {ci:4.2f} {res['avg_mass'][:, ix].mean():9.0f} "
              f"{res['peak'][:, ix].mean():8.0f} {res['deaths'][:, ix].mean():7.2f}")
    print("\nhead-to-head win rate (row beats column, same arenas)")
    print(f"{'':<10}" + "".join(f"{k:>10}" for k in keys))
    for a in keys:
        print(f"{a:<10}" + "".join(f"{'-':>10}" if a == c else f"{100 * float(np.mean(team[a] > team[c])):9.0f}%"
                                   for c in keys))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--b", type=int, default=16, help="arenas (= matches)")
    p.add_argument("--steps", type=int, default=11250, help="2 ticks each: 11250 = 15 game-min")
    p.add_argument("--bots", type=int, default=8)
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpus", type=int, default=8, help="pin to this many CPUs (0 = all): stay under 90% CPU")
    p.add_argument("--out", help="save the per-arena results (JSON); --merge reports several of them together")
    p.add_argument("--merge", nargs="+", help="only report these saved --out files as one tournament")
    a = p.parse_args()
    if a.merge:
        parts = [json.load(open(f)) for f in a.merge]
        res = {"names": parts[0]["names"]}
        for k in ("log_mass", "avg_mass", "peak", "deaths"):
            res[k] = np.concatenate([np.array(r[k]) for r in parts])
        report(res)
        return
    limit_cpus(a.cpus)
    torch.set_num_threads(1)   # policies run one at a time; several runner processes share the CPUs
    res = run(parse_lineup(a.agents), a.b, a.steps, a.bots, a.food_mult, a.seed)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in res.items()}, f)
    report(res)


if __name__ == "__main__":
    main()
