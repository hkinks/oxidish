"""Entity observation: per agent, the nearest visible objects as a token list with exact numbers
(the grid encoder, sim/obs.py, paints them into 48x48 pixels instead).

    enc = EntityEncoder(world, slots)
    ent, scal, frame, alive, tokpos = enc.encode()
    # ent [B*NA, NT, FD]: tokens (own cells, enemy cells, pellets, viruses, ejected mass), zero rows = padding
    # tokpos [B, NA, NT, 2]: the tokens' world positions (pointer actions aim at them)

Visibility is the server's view box, as in sim/obs.py (what a client would be sent). Tokens per
agent: all own cells (16 slots), the `k_cell` nearest enemy cells, `k_food` nearest pellets,
`k_virus` viruses, `k_eject` ejected blobs; each kind sorted by distance, nearest first. Then one
token per region of the view (4 x 4): the food in it as one pellet-type token at its centre of
mass with mass = its pellet count (so far-away clusters, invisible among the 32 nearest pellets, can
be seen and pointed at: a split cell can be sent through a whole cluster).

Token features (FD = 20):
  0 valid | 1-5 type: pellet, enemy cell, own cell, virus, ejected | 6, 7 relative x, y / 1000 |
  8 distance / 1000 (centre to centre) | 9, 10 velocity per step / 40 (from the last encode, same
  object id) | 11 log(1 + mass) / 8 | 12 log(mass / our largest cell) / 3 | 13 we can eat it
  (ours > it x 1.3) | 14 it can eat us | 15 a split of ours could eat it (ours > it x 2.6) |
  16 it could split onto us | 17 merge readiness (own cells) | 18 log(owner's total / ours) / 3
  (enemy cells) | 19 virus fill (0 at 100 mass, 1 at 200: about to split off)
Scalars: the grid encoder's 16 (8 base + 8 zero hints), so both nets read the same.
No host syncs; static shapes (one CUDA graph when the world uses graphs).
"""

import math

import torch

from .engine import ES, MERGE_SIZE_COEF
from .obs import EAT_RATIO, MIN_SCALE, S, SPLIT_KILL_RATIO, VEL_SCALE, VIEW_X, VIEW_Y, VIRUS_MAX, VIRUS_MIN

FD = 20
T_FOOD, T_CELL, T_OWN, T_VIRUS, T_EJECT = range(5)
K_FOOD, K_CELL, K_VIRUS, K_EJECT = 32, 32, 8, 8
REGIONS = 4                                             # food clusters: the view split into 4 x 4 regions
N_TOK = 16 + K_FOOD + K_CELL + K_VIRUS + K_EJECT + REGIONS * REGIONS   # 112 tokens per agent


class EntityEncoder:
    def __init__(self, world, slots=None, k_food: int = K_FOOD, k_cell: int = K_CELL, k_virus: int = K_VIRUS, k_eject: int = K_EJECT,
                 call_steps: int = 1):
        self.w = world
        B, P = world.B, world.P
        self.slots = torch.arange(P, device=world.dev) if slots is None else \
            torch.as_tensor(slots, device=world.dev, dtype=torch.long)
        self.NA = len(self.slots)
        self.k = dict(food=k_food, cell=k_cell, virus=k_virus, eject=k_eject)   # fixed: padded if fewer objects
        self.NT = world.K + sum(self.k.values()) + REGIONS * REGIONS
        self.call_steps = call_steps
        d = world.dev
        # last encode's positions and ids per object slot: velocities
        self.prev = {n: (torch.zeros(B, m, device=d), torch.zeros(B, m, device=d), torch.full((B, m), -1, dtype=torch.long, device=d))
                     for n, m in (("c", world.C), ("e", world.E), ("v", world.V))}
        self._g = None
        self._out = None

    def reset_agents(self, mask):
        pass   # no per-agent history (velocities are per object)

    # ------------------------------------------------------------------
    def encode(self, dtype=torch.float32):
        w = self.w
        if not w.use_graphs:
            out = self._encode()
        else:
            if self._g is None:
                state = [t for p in self.prev.values() for t in p]
                backup = [t.clone() for t in state]
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    self._encode()
                torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                for t, b in zip(state, backup):
                    t.copy_(b)
                self._g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._g):
                    self._out = self._encode()
                torch.cuda.synchronize()
                for t, b in zip(state, backup):
                    t.copy_(b)
            self._g.replay()
            out = self._out
        ent, scal, cx, cy, ext, alive, tokpos, mass = out
        return ent.to(dtype), scal.to(dtype), (cx.clone(), cy.clone(), ext.clone()), alive.clone(), tokpos.clone(), mass.clone()

    def _encode(self):
        w = self.w
        B, P, K, NA = w.B, w.P, w.K, self.NA
        dev = w.dev
        sl = self.slots
        own = lambda t: t.view(B, P, K)[:, sl]                        # [B,NA,K]
        al = w.calive
        size = w.cr2.sqrt()
        fsize = torch.floor(size)
        xi, yi = torch.trunc(w.cx), torch.trunc(w.cy)
        o_al = own(al)
        o_fs = own(fsize) * o_al
        o_mass = o_fs * o_fs / 100
        alive = o_al.any(-1)
        total = o_mass.sum(-1).clamp(min=1.0)
        cx = (own(xi) * o_mass).sum(-1) / total
        cy = (own(yi) * o_mass).sum(-1) / total
        sum_size = o_fs.sum(-1)
        ss = torch.where(alive, sum_size, torch.ones_like(sum_size)).clamp(min=1.0)
        ext = 960.0 / torch.clamp(64.0 / ss, max=1.0).pow(0.4)
        me_mass = o_mass.amax(-1).clamp(min=1.0)
        n_cells = o_al.sum(-1)
        # server view box (as sim/obs.py)
        o_size = own(size) * o_al
        sscale = torch.clamp(64.0 / o_size.sum(-1).clamp(min=1.0), max=1.0).pow(0.4).clamp(min=MIN_SCALE)
        ccx = (own(w.cx) * o_al).sum(-1) / n_cells.clamp(min=1)
        ccy = (own(w.cy) * o_al).sum(-1) / n_cells.clamp(min=1)
        hx, hy = (VIEW_X / sscale)[..., None], (VIEW_Y / sscale)[..., None]

        def visible(px, py, ps):   # [B,M] -> [B,NA,M]
            return ((px[:, None, :] - ccx[..., None]).abs() <= hx + ps[:, None, :]) & \
                   ((py[:, None, :] - ccy[..., None]).abs() <= hy + ps[:, None, :]) & alive[..., None]

        def nearest(px, py, ok, k):
            """k nearest of the ok objects per agent: (index [B,NA,k], valid, dx, dy, dist)."""
            dx = px[:, None, :] - cx[..., None]
            dy = py[:, None, :] - cy[..., None]
            d2 = torch.where(ok, dx * dx + dy * dy, torch.full_like(dx, 1e18))
            if d2.shape[-1] < k:       # fewer objects than token slots (e.g. a one-player arena)
                d2 = torch.cat([d2, torch.full((*d2.shape[:-1], k - d2.shape[-1]), 1e18, device=dev)], -1)
                dx = torch.cat([dx, torch.zeros_like(d2[..., dx.shape[-1]:])], -1)
                dy = torch.cat([dy, torch.zeros_like(d2[..., dy.shape[-1]:])], -1)
            v, i = torch.topk(d2, k, dim=-1, largest=False)
            return i, v < 1e17, dx.gather(-1, i), dy.gather(-1, i), v.clamp(max=1e12).sqrt()

        def vel(name, x, y, ids, i):
            i = i.clamp(max=x.shape[1] - 1)          # padded slots (invalid anyway)
            px, py, pid = self.prev[name]
            same = pid == ids
            vx = torch.where(same, x - px, torch.zeros_like(x)) / (VEL_SCALE * self.call_steps)
            vy = torch.where(same, y - py, torch.zeros_like(y)) / (VEL_SCALE * self.call_steps)
            g = lambda t: t[:, None, :].expand(B, NA, t.shape[1]).gather(-1, i)
            return g(vx).clamp(-2, 2), g(vy).clamp(-2, 2)

        def rows(kind, ok, dx, dy, dist, mass, vx, vy, extra):
            n = ok.shape[-1]
            f = torch.zeros(B, NA, n, FD, device=dev)
            f[..., 0] = ok.float()
            f[..., 1 + kind] = 1.0
            f[..., 6], f[..., 7], f[..., 8] = dx / 1000, dy / 1000, dist / 1000
            f[..., 9], f[..., 10] = vx, vy
            ratio = mass / me_mass[..., None]
            f[..., 11] = torch.log1p(mass) / 8
            f[..., 12] = (torch.log(ratio.clamp(min=1e-6)) / 3).clamp(-2, 2)
            f[..., 13] = (ratio * EAT_RATIO < 1).float()
            f[..., 14] = (ratio > EAT_RATIO).float()
            for j, v in extra.items():
                f[..., j] = v
            return f * ok[..., None].float()

        owner = torch.arange(w.C, device=dev) // K
        isown_c = owner[None, None, :] == sl[None, :, None]                         # [B?,NA,C]
        mass_c = fsize * fsize / 100                                                # [B,C]
        gm = (mass_c * al).view(B, P, K).sum(-1)                                    # owner totals [B,P]
        # own cells: the agent's 16 slots, in slot order
        oi = (sl[:, None] * K + torch.arange(K, device=dev)[None]).reshape(-1)      # [NA*K]
        oidx = oi.view(1, NA, K).expand(B, NA, K)
        odx = own(xi) - cx[..., None]
        ody = own(yi) - cy[..., None]
        ovx, ovy = vel("c", w.cx, w.cy, w.cid, oidx)
        ready = (own(w.cage) / (torch.clamp(MERGE_SIZE_COEF * own(size), min=30.0) * 25.0)).clamp(0, 1)
        f_own = rows(T_OWN, o_al, odx, ody, (odx * odx + ody * ody).sqrt(), o_mass, ovx, ovy, {17: ready})
        # enemy cells
        vis_c = visible(w.cx, w.cy, size) & al[:, None, :] & ~isown_c
        ci, cok, cdx, cdy, cd = nearest(xi, yi, vis_c, self.k["cell"])
        ci = ci.clamp(max=w.C - 1)
        cmass = mass_c[:, None, :].expand(B, NA, w.C).gather(-1, ci)
        cvx, cvy = vel("c", w.cx, w.cy, w.cid, ci)
        cratio = cmass / me_mass[..., None]
        omass = gm[:, None, :].expand(B, NA, P).gather(-1, ci // K)
        f_cell = rows(T_CELL, cok, cdx, cdy, cd, cmass, cvx, cvy,
                      {15: (cratio * SPLIT_KILL_RATIO < 1).float(), 16: (cratio > SPLIT_KILL_RATIO).float(),
                       18: (torch.log(omass.clamp(min=1.0) / total[..., None]) / 3).clamp(-2, 2)})
        # pellets
        vis_f = visible(w.fx, w.fy, torch.full_like(w.fx, 10.0)) & w.falive[:, None, :]
        fi, fok, fdx, fdy, fdd = nearest(torch.trunc(w.fx), torch.trunc(w.fy), vis_f, self.k["food"])
        z = torch.zeros_like(fdx)
        f_food = rows(T_FOOD, fok, fdx, fdy, fdd, torch.ones_like(fdx), z, z, {})
        # viruses
        vs = w.vr2.sqrt()
        vis_v = visible(w.vx, w.vy, vs) & w.valive[:, None, :]
        vi, vok, vdx, vdy, vdd = nearest(torch.trunc(w.vx), torch.trunc(w.vy), vis_v, self.k["virus"])
        vmass = (torch.floor(vs) ** 2 / 100)[:, None, :].expand(B, NA, w.V).gather(-1, vi.clamp(max=w.V - 1))
        vvx, vvy = vel("v", w.vx, w.vy, w.vid, vi)
        f_vir = rows(T_VIRUS, vok, vdx, vdy, vdd, vmass, vvx, vvy,
                     {19: ((vmass - VIRUS_MIN) / (VIRUS_MAX - VIRUS_MIN)).clamp(0, 1)})
        # ejected mass
        vis_e = visible(w.ex, w.ey, torch.full_like(w.ex, ES)) & w.ealive[:, None, :]
        ei, eok, edx, edy, edd = nearest(torch.trunc(w.ex), torch.trunc(w.ey), vis_e, self.k["eject"])
        evx, evy = vel("e", w.ex, w.ey, w.eid, ei)
        f_ej = rows(T_EJECT, eok, edx, edy, edd, torch.full_like(edx, math.floor(ES) ** 2 / 100), evx, evy, {})

        # food clusters: visible pellets binned into REGIONS x REGIONS regions of the view box
        R = REGIONS
        fdx_all = torch.trunc(w.fx)[:, None, :] - cx[..., None]
        fdy_all = torch.trunc(w.fy)[:, None, :] - cy[..., None]
        bx = ((fdx_all + hx) / (2 * hx) * R).floor().clamp(0, R - 1)
        by = ((fdy_all + hy) / (2 * hy) * R).floor().clamp(0, R - 1)
        reg = (by * R + bx).long()                                                  # [B,NA,F]
        wf = vis_f.float()
        cnt = torch.zeros(B, NA, R * R, device=dev).scatter_add_(-1, reg, wf)
        sx = torch.zeros(B, NA, R * R, device=dev).scatter_add_(-1, reg, wf * fdx_all)
        sy = torch.zeros(B, NA, R * R, device=dev).scatter_add_(-1, reg, wf * fdy_all)
        rok = cnt > 0
        rdx, rdy = sx / cnt.clamp(min=1), sy / cnt.clamp(min=1)
        zr = torch.zeros_like(rdx)
        f_reg = rows(T_FOOD, rok, rdx, rdy, (rdx * rdx + rdy * rdy).sqrt(), cnt, zr, zr, {})

        ent = torch.cat([f_own, f_cell, f_food, f_vir, f_ej, f_reg], 2)            # [B,NA,NT,FD]
        dxs = torch.cat([odx, cdx, fdx, vdx, edx, rdx], 2)
        dys = torch.cat([ody, cdy, fdy, vdy, edy, rdy], 2)
        tokpos = torch.stack([cx[..., None] + dxs, cy[..., None] + dys], -1)        # world positions
        # remember positions for the next call's velocities
        for name, (x, y, ids) in (("c", (w.cx, w.cy, w.cid)), ("e", (w.ex, w.ey, w.eid)), ("v", (w.vx, w.vy, w.vid))):
            px, py, pid = self.prev[name]
            px.copy_(x)
            py.copy_(y)
            pid.copy_(ids)
        # scalars: the grid encoder's
        half = w.map / 2
        scal = torch.zeros(B, NA, S, device=dev)
        scal[..., 0] = torch.log(total) / 8
        scal[..., 1] = n_cells.float() / 16
        scal[..., 2] = me_mass / total
        scal[..., 3] = (cx + half) / w.map * 2 - 1
        scal[..., 4] = (cy + half) / w.map * 2 - 1
        scal[..., 5] = ext / 3000
        scal[..., 6] = (me_mass >= 36).float()
        scal[..., 7] = torch.log(me_mass) / 8
        scal = scal * alive[..., None]
        A = B * NA
        return (ent.reshape(A, self.NT, FD), scal.reshape(A, S), cx, cy, ext, alive, tokpos,
                o_mass.sum(-1))
