"""Batched observation encoder: World -> the guided net's [13, 48, 48] grid + 16 scalars.

Reproduces rl/obs.py::encode (G=48, server_view=True) and rl/guided.py::GuidedEncoder.encode
for every agent at once. What the real encoders see is what the *server sends the client*, so
the sim mimics that too: objects are visible only inside the server's view box (centred on the
unweighted mean of our cells, 1920x1080 / scale), sizes arrive floored (uint16), positions
truncated (int32). Per-agent state the Python GuidedEncoder keeps (velocity history, first-seen
steps) is kept in tensors. The 8 champion-hint scalars are zeros (no champion in the sim).

Two stages. Stage 1 (static shapes, one CUDA graph): frame, view-box visibility of every object,
history / first-seen bookkeeping, food, outside mask, scalars. Stage 2 (ragged, eager): every
(agent, visible object, channel) is an "op"; ops are grouped by disc size into square windows and
two scatters write them (max-blend, or overwrite with the newest object winning, as the Python
encoders do). Host syncs: 2 (visible-object count, op counts per size class).
"""

import math

import torch

from .engine import ES

G = 48
BASE_CH = ("food", "virus", "mine", "prey", "threat", "split_threat", "ejected", "outside")
EXTRA_CH = ("vx", "vy", "own_merge", "enemy_merge", "owner_mass", "killable", "density")
C = len(BASE_CH) + len(EXTRA_CH)          # 15
S = 8 + 8                                  # base scalars + hint scalars (zeros)
(CH_FOOD, CH_VIRUS, CH_MINE, CH_PREY, CH_THREAT, CH_SPLIT, CH_EJECT, CH_OUT,
 CH_VX, CH_VY, CH_OWN_MERGE, CH_ENEMY_MERGE, CH_OWNER, CH_KILL, CH_DEN) = range(C)
# killable: enemy cells a split half of our largest cell could eat (mass < ours / 2.6).
# density: log(1 + enemy mass whose centre falls in the pixel) / 9, summed, not blended, so a
# cluster of small pieces reads as what it is worth. Both are sim-only so far (rl/guided.py: 13).
EAT_RATIO, SPLIT_KILL_RATIO = 1.3, 2.6
VEL_STEPS, VEL_SCALE = 3, 40.0
SYNC_TICKS = 2
MERGE_SECONDS = 30
VIRUS_MIN, VIRUS_MAX = 100.0, 200.0
VIEW_X, VIEW_Y, MIN_SCALE = 960.0, 540.0, 0.15   # half of serverViewBaseX/Y, serverMinScale
MINE_PRIO = 1 << 30                               # own cells are painted first (lowest priority)
_RAD_CLASSES = (3, 8)                             # window half-widths; larger discs use the full grid
# painted rows: (channel, overwrite?) -- vx/vy/owner/virus overwrite, the rest max-blend
_ROWS = ((CH_MINE, False), (CH_OWN_MERGE, False), (CH_VX, True), (CH_VY, True), (CH_PREY, False),
         (CH_THREAT, False), (CH_SPLIT, False), (CH_ENEMY_MERGE, False), (CH_OWNER, True),
         (CH_VIRUS, True), (CH_EJECT, False), (CH_KILL, False))


class Encoder:
    """Per-agent encoder state for `slots` (player indices) of every arena."""

    def __init__(self, world, slots=None, call_steps: int = 1):
        """call_steps: env steps between encode calls (frame skip); velocities and merge timers
        are scaled by it, so they keep their per-step meaning."""
        self.w = world
        self.call_steps = call_steps
        B, P = world.B, world.P
        self.slots = torch.arange(P, device=world.dev) if slots is None else \
            torch.as_tensor(slots, device=world.dev, dtype=torch.long)
        NA = self.NA = len(self.slots)
        d = world.dev
        N = self.N = world.C + world.E + world.V      # object columns: cells | ejected | viruses
        self.n_obs = torch.zeros(B, NA, dtype=torch.long, device=d)
        self.fs_step = torch.full((B, NA, world.C), -1, dtype=torch.long, device=d)
        self.fs_id = torch.zeros(B, NA, world.C, dtype=torch.long, device=d)
        self._calls = torch.zeros(1, dtype=torch.long, device=d)
        R = VEL_STEPS + 1
        self.h = (torch.zeros(R, B, N, device=d), torch.zeros(R, B, N, device=d),
                  torch.zeros(R, B, N, dtype=torch.long, device=d),
                  torch.zeros(R, B, NA, N, dtype=torch.bool, device=d))
        self._g1 = None
        self._outs1 = None
        self._prio = torch.empty(B * NA * 4 * G * G, dtype=torch.int32, device=d)
        self._och = torch.full((C,), -1, dtype=torch.long, device=d)   # overwrite-channel slots
        self._och[CH_VX], self._och[CH_VY], self._och[CH_OWNER], self._och[CH_VIRUS] = 0, 1, 2, 3
        self._rows = torch.tensor([r[0] for r in _ROWS], device=d)
        self._rows_over = torch.tensor([r[1] for r in _ROWS], device=d)

    def reset_agents(self, mask: torch.Tensor):
        """Forget the history of agents ([B,NA] bool), e.g. on death (GuidedEncoder.reset)."""
        self.n_obs.masked_fill_(mask, 0)
        self.fs_step.masked_fill_(mask[:, :, None], -1)

    # ------------------------------------------------------------------ painting
    def _paint_ops(self, buf, a, ch, gx, gy, rr, val, prio, over, valid):
        """Paint discs (pixel centres within rr of (gx, gy), grid units) for a list of ops.
        over False: max-blend into channel ch of agent a. over True: overwrite, the op with the
        highest prio wins per (agent, channel, pixel). Invalid ops are skipped. Only one host
        sync (the per-size-class op counts); misses write harmless zeros instead of masking."""
        n = a.shape[0]
        if n == 0:
            return
        dev = a.device
        cls = torch.where(valid, (rr > 2.5).long() + (rr > 7.5).long(), torch.full_like(a, 3))
        order = torch.argsort(cls)
        counts = torch.zeros(4, dtype=torch.long, device=dev).scatter_add_(0, cls, torch.ones_like(cls)).tolist()
        parts, s = [], 0
        for c in range(3):
            cnt = counts[c]
            sel = order[s:s + cnt]
            s += cnt
            if cnt == 0:
                continue
            if c < 2:
                R = _RAD_CLASSES[c]
                dd = torch.arange(-R, R + 1, device=dev)
                X = torch.floor(gx[sel]).long()[:, None, None] + dd[None, None, :]
                Y = torch.floor(gy[sel]).long()[:, None, None] + dd[None, :, None]
            else:
                ar = torch.arange(G, device=dev)
                X = ar[None, None, :].expand(cnt, 1, G)
                Y = ar[None, :, None].expand(cnt, G, 1)
            ddx = X + 0.5 - gx[sel][:, None, None]
            ddy = Y + 0.5 - gy[sel][:, None, None]
            ok = (ddx * ddx + ddy * ddy <= (rr[sel] ** 2)[:, None, None]) & \
                 (X >= 0) & (X < G) & (Y >= 0) & (Y < G)
            pix = (Y.clamp(0, G - 1) * G + X.clamp(0, G - 1)).expand_as(ok)
            shape = ok.shape
            ex = lambda t, dt=None: (t[sel] if dt is None else t[sel].to(dt))[:, None, None].expand(shape).reshape(-1)
            parts.append((pix.reshape(-1), ok.reshape(-1), ex(a), ex(ch), ex(val, torch.float32), ex(prio), ex(over)))
        pix, ok, ag, chh, v, pr, ov = (torch.cat([p[i] for p in parts]) for i in range(7))
        idx = (ag * C + chh) * (G * G) + pix
        zero = torch.zeros((), device=dev)
        buf.scatter_reduce_(0, idx, torch.where(ok & ~ov, v, zero), "amax", include_self=True)
        key = (ag * 4 + self._och[chh].clamp(min=0)) * (G * G) + pix
        p_eff = torch.where(ok & ov, pr, torch.full_like(pr, -(1 << 30) - 1))
        self._prio.fill_(-(1 << 30) - 1)
        self._prio.scatter_reduce_(0, key, p_eff, "amax", include_self=True)
        win = (p_eff == self._prio[key]) & ok & ov
        buf.scatter_add_(0, idx, torch.where(win, v, zero))

    # ------------------------------------------------------------------ stage 1 (static shapes)
    def _state(self):
        return [self.n_obs, self.fs_step, self.fs_id, self._calls, *self.h]

    def _run_stage1(self):
        if not self.w.use_graphs:
            return self._stage1()
        if self._g1 is None:
            backup = [t.clone() for t in self._state()]
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._stage1()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            for t, b in zip(self._state(), backup):
                t.copy_(b)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._outs1 = self._stage1()
            torch.cuda.synchronize()
            for t, b in zip(self._state(), backup):
                t.copy_(b)
            self._g1 = g
        self._g1.replay()
        return self._outs1

    def _stage1(self):
        w = self.w
        B, P, K, NA, Cc, E, V, N = w.B, w.P, w.K, self.NA, w.C, w.E, w.V, self.N
        dev, A = w.dev, w.B * self.NA
        sl = self.slots
        x_all, y_all = w.cx, w.cy
        al_all = w.calive
        size_all = w.cr2.sqrt()
        fsize_all = torch.floor(size_all)
        xi_all, yi_all = torch.trunc(x_all), torch.trunc(y_all)
        own = lambda t: t.view(B, P, K)[:, sl]                        # [B,NA,K]
        o_al = own(al_all)
        o_fs = (own(fsize_all) * o_al).double()                      # client masses in float64, like Python
        o_mass = o_fs * o_fs / 100
        alive = o_al.any(-1)
        total = o_mass.sum(-1).clamp(min=1.0)
        sum_size = o_fs.sum(-1)
        ss = torch.where(alive, sum_size, torch.ones_like(sum_size)).clamp(min=1.0)
        cx = (own(xi_all).double() * o_mass).sum(-1) / total
        cy = (own(yi_all).double() * o_mass).sum(-1) / total
        ext = 960.0 / torch.clamp(64.0 / ss, max=1.0).pow(0.4)
        k = G / (2 * ext)
        me_idx = o_fs.argmax(-1, keepdim=True)
        me_mass = o_mass.gather(-1, me_idx)[..., 0]
        n_cells = o_al.sum(-1)
        # server view box (float sizes / positions, centre = unweighted mean of own cells)
        o_size = own(size_all) * o_al
        sscale = torch.clamp(64.0 / o_size.sum(-1).clamp(min=1.0), max=1.0).pow(0.4).clamp(min=MIN_SCALE)
        ccx = (own(x_all) * o_al).sum(-1) / n_cells.clamp(min=1)
        ccy = (own(y_all) * o_al).sum(-1) / n_cells.clamp(min=1)
        hx, hy = VIEW_X / sscale, VIEW_Y / sscale

        def visible(px, py, ps):   # [B,N] -> [B,NA,N]
            return ((px[:, None, :] - ccx[:, :, None]).abs() <= hx[:, :, None] + ps[:, None, :]) & \
                   ((py[:, None, :] - ccy[:, :, None]).abs() <= hy[:, :, None] + ps[:, None, :]) & \
                   alive[:, :, None]
        owner_c = torch.arange(Cc, device=dev) // K
        own_flag = (owner_c[None, None, :] == sl[None, :, None]) & al_all[:, None, :] & alive[:, :, None]
        vis_c_all = (visible(x_all, y_all, size_all) & al_all[:, None, :]) | own_flag
        vis_c = vis_c_all & ~own_flag
        vis_e = visible(w.ex, w.ey, torch.full_like(w.ex, ES)) & w.ealive[:, None, :]
        vis_v = visible(w.vx, w.vy, w.vr2.sqrt()) & w.valive[:, None, :]
        vis_f = visible(w.fx, w.fy, torch.full_like(w.fx, 10.0)) & w.falive[:, None, :]
        vis_all = torch.cat([vis_c_all, vis_e, vis_v], 2)                              # [B,NA,N]
        own_all = torch.cat([own_flag, torch.zeros(B, NA, E + V, dtype=torch.bool, device=dev)], 2)
        X = torch.cat([xi_all, torch.trunc(w.ex), torch.trunc(w.vx)], 1)               # [B,N] what the client sees
        Y = torch.cat([yi_all, torch.trunc(w.ey), torch.trunc(w.vy)], 1)
        FS = torch.cat([fsize_all, torch.full_like(w.ex, math.floor(ES)), torch.floor(w.vr2.sqrt())], 1)
        ID = torch.cat([w.cid, w.eid, w.vid], 1)

        # per-agent step counters and first-seen
        self.n_obs.copy_(torch.where(alive, self.n_obs + 1, torch.zeros_like(self.n_obs)))
        self.fs_step.masked_fill_(~alive[:, :, None], -1)
        newc = self.fs_id != w.cid[:, None, :]
        self.fs_step.copy_(torch.where(newc, torch.full_like(self.fs_step, -1), self.fs_step))
        self.fs_id.copy_(w.cid[:, None, :].expand(B, NA, Cc))
        self.fs_step.copy_(torch.where(vis_c_all & (self.fs_step < 0), self.n_obs[:, :, None].expand_as(self.fs_step),
                                       self.fs_step))
        steps = (self.n_obs - 1).clamp(min=0, max=VEL_STEPS)             # [B,NA]
        R = VEL_STEPS + 1
        r = self._calls % R                                              # device scalar: graph-safe ring index
        old_slot = (self._calls - steps) % R
        for tbl, v in zip(self.h, (X, Y, ID, vis_all)):
            tbl[r] = v[None]
        self._calls.add_(1)
        gm = (vis_c.view(B, NA, P, K) * (fsize_all.double() ** 2 / 100).view(B, 1, P, K)).sum(-1)   # [B,NA,P]

        # food: add 0.25 per visible pellet, clipped at 1
        buf = torch.zeros(A * C * G * G, device=dev)
        fxi, fyi = torch.trunc(w.fx), torch.trunc(w.fy)
        gxf = ((fxi[:, None, :] - cx[:, :, None]) * k[:, :, None] + G / 2).long()
        gyf = ((fyi[:, None, :] - cy[:, :, None]) * k[:, :, None] + G / 2).long()
        okf = vis_f & (gxf >= 0) & (gxf < G) & (gyf >= 0) & (gyf < G)
        agent = (torch.arange(B, device=dev)[:, None] * NA + torch.arange(NA, device=dev)[None, :])[:, :, None]
        spread = (torch.arange(w.F, device=dev) % (G * G))[None, None, :]       # misses add 0.0 to spread-out pixels
        fidx = torch.where(okf, gyf.clamp(0, G - 1) * G + gxf.clamp(0, G - 1), spread) + (agent * C + CH_FOOD) * (G * G)
        buf.index_add_(0, fidx.reshape(-1), okf.float().reshape(-1) * 0.25)
        buf.view(A, C, G * G)[:, CH_FOOD].clamp_(max=1.0)

        # outside-the-map mask and the scalars only depend on the frame
        kf, cxf, cyf = k.reshape(-1), cx.reshape(-1), cy.reshape(-1)
        me_f, tot_f = me_mass.reshape(-1), total.reshape(-1)
        half = w.map / 2
        pos = (torch.arange(G, device=dev)[None, :] + 0.5 - G / 2)
        xs = cxf[:, None] + pos / kf[:, None]
        ys = cyf[:, None] + pos / kf[:, None]
        out_x = (xs < -half) | (xs > half)
        out_y = (ys < -half) | (ys > half)
        buf.view(A, C, G, G)[:, CH_OUT] = (out_y[:, :, None] | out_x[:, None, :]).float()
        scal = torch.zeros(A, S, device=dev)
        scal[:, 0] = (torch.log(tot_f) / 8).float()
        scal[:, 1] = n_cells.reshape(-1).float() / 16
        scal[:, 2] = me_f / tot_f
        scal[:, 3] = (cxf + half) / w.map * 2 - 1
        scal[:, 4] = (cyf + half) / w.map * 2 - 1
        scal[:, 5] = ext.reshape(-1) / 3000
        scal[:, 6] = (me_f >= 36).float()
        scal[:, 7] = torch.log(me_f.clamp(min=1.0)) / 8
        scal.mul_(alive.reshape(-1)[:, None])
        cnt = vis_all.sum()[None]
        return (alive, cx, cy, ext, k, me_mass, total, steps, old_slot, vis_all, own_all, X, Y, FS, ID, gm,
                buf, scal, o_mass.sum(-1).float(), cnt)

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, dtype=torch.float32):
        """Returns grids [B*NA,13,G,G], scalars [B*NA,16], frame (cx, cy, ext) each [B,NA],
        alive [B,NA] bool, mass [B,NA] (sum of floor(size)^2/100, what the client sees)."""
        w = self.w
        B, P, K, NA, Cc, E, N = w.B, w.P, w.K, self.NA, w.C, w.E, self.N
        dev, A = w.dev, w.B * self.NA
        (alive, cx, cy, ext, k, me_mass, total, steps, old_slot, vis_all, own_all, X, Y, FS, ID, gm,
         buf, scal, mass_sum, cnt) = self._run_stage1()

        n = int(cnt.item())                                              # host sync 1
        ix = torch.nonzero_static(vis_all, size=n)                       # every (agent, visible object)
        b, na, j = ix[:, 0], ix[:, 1], ix[:, 2]
        a = b * NA + na
        fl = b * N + j
        kf, cxf, cyf = k.reshape(-1), cx.reshape(-1), cy.reshape(-1)
        x, y, fsz, oid = X.reshape(-1)[fl], Y.reshape(-1)[fl], FS.reshape(-1)[fl].double(), ID.reshape(-1)[fl]
        ka = kf[a]
        isown = own_all.reshape(-1)[a * N + j]
        is_cell, is_vir = j < Cc, j >= Cc + E
        is_ej = ~is_cell & ~is_vir
        oth = is_cell & ~isown
        gx = (x - cxf[a]) * ka + G / 2
        gy = (y - cyf[a]) * ka + G / 2
        gr = fsz * ka
        grb, grv, grs = gr.clamp(min=0.5), gr.clamp(min=0.7), gr + 650 * ka
        # velocity over up to 3 steps, by object id (needs the object visible in the old frame)
        hx_, hy_, hid_, hvis_ = (t.reshape(-1) for t in self.h)
        oi = old_slot.reshape(-1)[a]
        st = steps.reshape(-1)[a]
        hl = oi * (B * N) + fl
        stf = st.clamp(min=1).float() * self.call_steps
        vxv = (x - hx_[hl]) / stf / VEL_SCALE
        vyv = (y - hy_[hl]) / stf / VEL_SCALE
        velok = (st > 0) & (hid_[hl] == oid) & hvis_[(oi * A + a) * N + j] & (vxv.abs() + vyv.abs() > 0.02)
        # merge readiness (first seen by this agent) and size relations to our largest cell
        jc = j.clamp(max=Cc - 1)
        stepn = (self.n_obs.reshape(-1)[a] - self.fs_step.reshape(-1)[a * Cc + jc]).float() * (SYNC_TICKS * self.call_steps)
        ready = torch.clamp(stepn / (torch.clamp(0.2 * fsz, min=float(MERGE_SECONDS)) * 25), max=1.0).clamp(min=0.05)
        mass_o = fsz * fsz / 100
        tot_f = total.reshape(-1)
        ratio = mass_o / me_mass.reshape(-1)[a].clamp(min=1.0)
        threat = oth & (ratio > EAT_RATIO)
        owm = gm.reshape(-1)[a * P + jc // K]
        owner_v = (torch.log(owm.clamp(min=1.0) / tot_f[a].clamp(min=1.0)) / 3).clamp(-1, 1)
        fill = ((mass_o - VIRUS_MIN) / (VIRUS_MAX - VIRUS_MIN)).clamp(0.0, 1.0)
        one = torch.ones_like(gx)
        # rows in the order of _ROWS: mine, own_merge, vx, vy, prey, threat, split, enemy_merge, owner, virus, eject
        rr = torch.stack([grb, grv, grv, grv, grb, grb, grs, grv, grv, grv, grb, grb])
        val = torch.stack([one, ready.double(), vxv.clamp(-1, 1).double(), vyv.clamp(-1, 1).double(),
                           (0.3 + ratio).clamp(max=1.0), (torch.log(ratio.clamp(min=1e-9)) / 2).clamp(max=1.0),
                           one, ready.double(), owner_v, (0.3 + 0.7 * fill).double(), one, one])
        valid = torch.stack([isown, isown, velok, velok, oth & (ratio * EAT_RATIO < 1), threat,
                             threat & (ratio > SPLIT_KILL_RATIO), oth, oth, is_vir, is_ej,
                             oth & (ratio * SPLIT_KILL_RATIO < 1)])
        nrow = len(_ROWS)
        prio = torch.where(isown, oid - MINE_PRIO, oid).int()
        self._paint_ops(buf, a.repeat(nrow), self._rows[:, None].expand(nrow, n).reshape(-1), gx.repeat(nrow),
                        gy.repeat(nrow), rr.reshape(-1), val.reshape(-1), prio.repeat(nrow),
                        self._rows_over[:, None].expand(nrow, n).reshape(-1), valid.reshape(-1))

        ixd, iyd = gx.floor().long(), gy.floor().long()
        okd = oth & (ixd >= 0) & (ixd < G) & (iyd >= 0) & (iyd < G)
        didx = (a * C + CH_DEN) * (G * G) + iyd.clamp(0, G - 1) * G + ixd.clamp(0, G - 1)
        buf.index_add_(0, didx, (mass_o * okd).float())
        den = buf.view(A, C, G * G)[:, CH_DEN]
        den.copy_((torch.log1p(den) / 9).clamp(max=1.0))
        grid = (buf.view(A, C, G, G) * alive.reshape(-1)[:, None, None, None]).to(dtype)   # copy: buf is reused
        return grid, scal.to(dtype), (cx, cy, ext), alive, mass_sum
