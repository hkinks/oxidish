"""Compare the simulator's observation encoder with the real Python encoder (rl/guided.py).

Builds a bot.game.Snapshot from the sim state exactly as the client would see it (view box,
floored sizes, truncated positions) and feeds it to GuidedEncoder with its history.

    uv run python -m sim.compare_obs [steps] [seed]
"""

import math

import numpy as np
import torch

from . import obs as SO
from .bots import scripted_actions
from .env import SimEnv
from bot.game import Cell, Snapshot
from rl.guided import GuidedEncoder

def snapshot(w, b, p):
    K = w.K
    sl = slice(p*K, (p+1)*K)
    al = w.calive[b].cpu().numpy(); cx = w.cx[b].cpu().numpy(); cy = w.cy[b].cpu().numpy()
    r2 = w.cr2[b].cpu().numpy(); cid = w.cid[b].cpu().numpy()
    size = np.sqrt(r2)
    mine = [Cell(int(cid[i]), float(int(cx[i])), float(int(cy[i])), float(int(size[i])), "mine", f"p{p}") for i in range(p*K,(p+1)*K) if al[i]]
    # view box
    n = len(mine)
    ids = [i for i in range(p*K,(p+1)*K) if al[i]]
    ccx = np.mean(cx[ids]); ccy = np.mean(cy[ids])
    ssum = size[ids].sum(); sc = max(min(64/ssum,1)**.4, .15)
    hx, hy = 960/sc, 540/sc
    vis = lambda x,y,s: abs(x-ccx) <= hx+s and abs(y-ccy) <= hy+s
    others = []
    for i in range(w.C):
        if al[i] and i//K != p and vis(cx[i],cy[i],size[i]):
            others.append(Cell(int(cid[i]), float(int(cx[i])), float(int(cy[i])), float(int(size[i])), "player", f"p{i//K}"))
    ea = w.ealive[b].cpu().numpy(); ex=w.ex[b].cpu().numpy(); ey=w.ey[b].cpu().numpy(); eid=w.eid[b].cpu().numpy()
    for j in range(w.E):
        if ea[j] and vis(ex[j],ey[j],36.06):
            others.append(Cell(int(eid[j]), float(int(ex[j])), float(int(ey[j])), 36.0, "ejected", ""))
    va = w.valive[b].cpu().numpy(); vx=w.vx[b].cpu().numpy(); vy=w.vy[b].cpu().numpy(); vr=w.vr2[b].cpu().numpy(); vid=w.vid[b].cpu().numpy()
    for j in range(w.V):
        if va[j] and vis(vx[j],vy[j],math.sqrt(vr[j])):
            others.append(Cell(int(vid[j]), float(int(vx[j])), float(int(vy[j])), float(int(math.sqrt(vr[j]))), "virus", ""))
    fa = w.falive[b].cpu().numpy(); fx=w.fx[b].cpu().numpy(); fy=w.fy[b].cpu().numpy()
    for j in range(w.F):
        if fa[j] and vis(fx[j],fy[j],10.0):
            others.append(Cell(-1-j, float(int(fx[j])), float(int(fy[j])), 10.0, "food", ""))
    others.sort(key=lambda c: c.id if c.id > 0 else -1)  # insertion ~ id order
    return Snapshot(mine, others, {"x":0,"y":0,"z":1}, 0, 0, {"left":-3500,"right":3500,"top":-3500,"bottom":3500}, not mine)



def compare(steps=60, seed=0, B=2, P=8, verbose=False):
    """Returns {channel: fraction of agent-observations with any pixel differing by > 1e-3}."""
    env = SimEnv(B=B, P=P, seed=seed, curriculum_p=0.7, curriculum_mass=(50, 3000))
    g, s, r, d, v = env.reset()
    pyenc = {(b, p): GuidedEncoder({"farm": "route"}) for b in range(B) for p in range(P)}
    worst = {}
    for step in range(steps):
        g_cpu, s_cpu = g.cpu().numpy(), s.cpu().numpy()
        for b in range(B):
            for p in range(P):
                a = b * P + p
                if not v[a]:
                    pyenc[(b, p)].reset()
                    continue
                gg, ss, cx, cy, ext = pyenc[(b, p)].encode(snapshot(env.w, b, p))
                for ch in range(13):
                    worst.setdefault(ch, []).append(int((np.abs(gg[ch] - g_cpu[a, ch]) > 1e-3).sum()))
                worst.setdefault("scal", []).append(float(np.abs(ss[:8] - s_cpu[a, :8]).max()))
        acts = scripted_actions(env.w)[:, :env.NL].reshape(-1, 3)
        acts[:, 1] |= (torch.rand(acts.shape[0], device=env.dev) < 0.03).long()
        acts[:, 2] = (torch.rand(acts.shape[0], device=env.dev) < 0.03).long()
        g, s, r, d, v = env.step(acts)
    names = list(SO.BASE_CH) + list(SO.EXTRA_CH)
    out = {}
    for k, vv in worst.items():
        vv = np.array(vv)
        nm = names[k] if k != "scal" else "scalars"
        out[nm] = float((vv > 1e-3).mean())
        if verbose:
            print(f"{nm:14s} mean mismatching px {vv.mean():8.3f}  max {vv.max():8.3f}  "
                  f"agent-obs with any mismatch {out[nm]:.3f}")
    return out


if __name__ == "__main__":
    import sys
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 60, int(sys.argv[2]) if len(sys.argv) > 2 else 0, verbose=True)
