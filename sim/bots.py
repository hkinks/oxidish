"""Vectorised scripted opponent for the simulator: flee threats, chase prey, else eat food.

Works directly on world state (no grid): per player it looks at the nearest threat (a cell
>1.3x our largest cell), nearest prey (<1/1.3x), nearest virus (when big enough to pop) and
nearest food, and returns the usual discrete action (16 headings / stay, split, eject)."""

import math

import torch

from .engine import EJECT_R2, SPLIT_SPEED, World

THREAT_RATIO, PREY_RATIO = 1.3, 1.3
SPLIT_KILL = 2.6


@torch.no_grad()
def scripted_actions(w: World, split_prob: float = 0.5, chase_range: float = 1200.0) -> torch.Tensor:
    """Returns [B,P,3] long: (move 0..16, split, eject) for every player (dead players: stay)."""
    B, P, K, C = w.B, w.P, w.K, w.C
    dev = w.dev
    al = w.calive
    r2 = w.cr2 * al
    size = w.cr2.sqrt() * al
    tot_r2 = r2.view(B, P, K).sum(-1).clamp(min=1.0)
    cx = (w.cx * r2).view(B, P, K).sum(-1) / tot_r2
    cy = (w.cy * r2).view(B, P, K).sum(-1) / tot_r2
    main_r2 = r2.view(B, P, K).max(-1).values
    main_size = main_r2.sqrt()
    main_mass = (main_r2 / 100).clamp(min=1.0)
    n_cells = al.view(B, P, K).sum(-1)
    alive = n_cells > 0
    owner = (torch.arange(C, device=dev) // K)[None, None, :]
    other = (owner != torch.arange(P, device=dev)[None, :, None]) & al[:, None, :]
    dx = w.cx[:, None, :] - cx[:, :, None]
    dy = w.cy[:, None, :] - cy[:, :, None]
    d = (dx * dx + dy * dy).sqrt()
    sd = d - size[:, None, :]                                   # surface distance
    ratio = (w.cr2 / 100)[:, None, :] / main_mass[:, :, None]
    inf = torch.full_like(sd, 1e9)
    threat = other & (ratio > THREAT_RATIO)
    prey = other & (ratio * PREY_RATIO < 1)
    reach = torch.where(ratio > SPLIT_KILL, torch.full_like(sd, 900.0), torch.full_like(sd, 350.0))
    td, ti = torch.where(threat & (sd < reach), sd, inf).min(-1)
    pd, pi = torch.where(prey & (d < chase_range), d, inf).min(-1)
    gather = lambda t, i: t.gather(-1, i[..., None])[..., 0]
    # food
    fdx = w.fx[:, None, :] - cx[:, :, None]
    fdy = w.fy[:, None, :] - cy[:, :, None]
    fd, fi = torch.where(w.falive[:, None, :], fdx * fdx + fdy * fdy, torch.full_like(fdx, 1e12)).min(-1)
    # viruses: dodge when big enough to pop
    vdx = w.vx[:, None, :] - cx[:, :, None]
    vdy = w.vy[:, None, :] - cy[:, :, None]
    vd = (vdx * vdx + vdy * vdy).sqrt() - w.vr2.sqrt()[:, None, :]
    vd, vi = torch.where(w.valive[:, None, :], vd, torch.full_like(vd, 1e9)).min(-1)
    dodge = (main_mass > 135) & (vd < main_size + 60)

    tx, ty = gather(dx, ti), gather(dy, ti)
    px, py = gather(dx, pi), gather(dy, pi)
    fxv, fyv = gather(fdx, fi), gather(fdy, fi)
    vxv, vyv = gather(vdx, vi), gather(vdy, vi)
    use_flee = td < 1e8
    use_prey = pd < 1e8
    dirx = torch.where(use_flee, -tx, torch.where(dodge, -vxv, torch.where(use_prey, px, fxv)))
    diry = torch.where(use_flee, -ty, torch.where(dodge, -vyv, torch.where(use_prey, py, fyv)))
    # keep off the border: bend toward the middle when close
    m = w.map / 2 - 400
    dirx = dirx + torch.where(cx.abs() > m, -torch.sign(cx) * 2000, torch.zeros_like(cx))
    diry = diry + torch.where(cy.abs() > m, -torch.sign(cy) * 2000, torch.zeros_like(cy))
    ang = torch.atan2(diry, dirx)
    move = torch.round(ang / (2 * math.pi) * 16).long() % 16
    move = torch.where(alive, move, torch.full_like(move, 16))
    prey_mass = (w.cr2 / 100)[:, None, :].expand(B, P, C).gather(-1, pi[..., None])[..., 0]
    split = (use_prey & ~use_flee & (main_mass > SPLIT_KILL * prey_mass)
             & (main_mass >= 80) & (pd < 650) & (n_cells <= 3)
             & (torch.rand(B, P, generator=w.gen, device=dev) < split_prob))
    out = torch.stack([move, split.long(), torch.zeros_like(move)], -1)
    return out


@torch.no_grad()
def farmer_actions(env) -> torch.Tensor:
    """The nearest-pellet farmer: per learner slot of `env` (a SimEnv), the heading (of 16) from
    the last observation's view centre to the nearest pellet; never splits or ejects. [B*NL, 3].
    Alone it reaches ~300 mass in 2 min (skip 2): the teacher for farming (rl.ppo_sim --bc-farmer-iters)."""
    w = env.w
    cx, cy, _ = env.frame                                                   # [B, NL]
    d2 = torch.where(w.falive[:, None, :], (w.fx[:, None, :] - cx[..., None]) ** 2 + (w.fy[:, None, :] - cy[..., None]) ** 2,
                     torch.full((w.B, cx.shape[1], w.F), 1e12, device=w.dev))
    i = d2.argmin(-1)
    fx, fy = w.fx.gather(1, i), w.fy.gather(1, i)
    ang = torch.atan2(fy - cy, fx - cx)
    move = (torch.round(ang / (2 * math.pi) * 16).long() % 16).reshape(-1)
    z = torch.zeros_like(move)
    return torch.stack([move, z, z], -1)


# ---- MultiOgar's own bot (vendor/MultiOgar/src/ai/BotPlayer.js), vectorised -------------------------
MO_VIEW_X, MO_VIEW_Y, MO_MIN_SCALE = 960.0, 540.0, 0.15     # serverViewBaseX/Y / 2, serverMinScale
MO_PURSUIT, MO_COOLDOWN = 20, 15                            # targetPursuit / splitCooldown (ticks, merge on)


class MultiOgarBots:
    """The real server's built-in bot, per player slot of a World: from its largest cell, a potential
    field over everything in its view box (pellets +1, ejected +2, smaller players +size/log(n), bigger
    ones -log(ratio) over the surface distance, same-size ones -ratio, viruses -1 while it can pop on
    them), and a deterministic split-kill: when not cooling down and under 6 cells, it splits on the
    biggest player cell it out-sizes by 1.69x (radius), that is over 0.4x its radius and within split
    reach, then chases that cell for 20 ticks (15-tick cooldown). These bots split-kill the sim nets'
    small split pieces on the real server; scripted_actions only splits at random on the nearest prey.

    Timers count down by the world ticks since the previous call (2 per env step; 4 when a policy
    decides every 2 steps). Returns [B,P,3] like scripted_actions."""

    def __init__(self, w: World):
        self.w = w
        self.cooldown = torch.zeros(w.B, w.P, device=w.dev)
        self.pursuit = torch.zeros(w.B, w.P, device=w.dev)
        self.target = torch.full((w.B, w.P), -1, dtype=torch.long, device=w.dev)   # cell index in [0, C)
        self.target_id = torch.full((w.B, w.P), -1, dtype=torch.long, device=w.dev)
        self.last_t = None

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        w = self.w
        ticks = 2 if self.last_t is None else max(w.t - self.last_t, 1)
        self.last_t = w.t
        B, P, K, C = w.B, w.P, w.K, w.C
        dev = w.dev
        al = w.calive.view(B, C)
        size = (w.cr2.view(B, C) * al).sqrt()
        sv = size.view(B, P, K)
        n_cells = al.view(B, P, K).sum(-1)
        alive = n_cells > 0
        big, bi = sv.max(-1)                                          # largest cell (the bot steers it)
        ci = bi + torch.arange(P, device=dev)[None, :] * K
        bx, by = w.cx.view(B, C).gather(1, ci), w.cy.view(B, C).gather(1, ci)
        # view box: centre = mean cell position, scale from the summed radius
        nc = n_cells.clamp(min=1)
        mx = (w.cx.view(B, P, K) * al.view(B, P, K)).sum(-1) / nc
        my = (w.cy.view(B, P, K) * al.view(B, P, K)).sum(-1) / nc
        scale = torch.clamp(64.0 / sv.sum(-1).clamp(min=1.0), max=1.0).pow(0.4).clamp(min=MO_MIN_SCALE)
        hx, hy = (MO_VIEW_X / scale)[..., None], (MO_VIEW_Y / scale)[..., None]

        def seen(px, py, ps, ok):                                    # [B,M] -> [B,P,M]
            return (((px[:, None, :] - mx[..., None]).abs() <= hx + ps[:, None, :])
                    & ((py[:, None, :] - my[..., None]).abs() <= hy + ps[:, None, :]) & ok[:, None, :])

        def field(px, py, ps, infl, vis):
            """Sum of unit vectors (from our largest cell) * infl / dist; negative infl uses the surface distance."""
            dx, dy = px[:, None, :] - bx[..., None], py[:, None, :] - by[..., None]
            d = (dx * dx + dy * dy).sqrt()
            dist = torch.where(infl < 0, d - big[..., None] - ps[:, None, :], d).clamp(min=1.0)
            f = torch.where(vis, infl / dist, torch.zeros_like(dist)) / d.clamp(min=1e-3)
            return (f * dx).sum(-1), (f * dy).sum(-1), d

        owner = torch.arange(C, device=dev) // K
        other = owner[None, None, :] != torch.arange(P, device=dev)[None, :, None]
        vis_c = seen(w.cx.view(B, C), w.cy.view(B, C), size, al) & other
        fs = torch.full_like(w.fx, 10.0)
        vis_f = seen(w.fx, w.fy, fs, w.falive)
        vs = w.vr2.sqrt()
        vis_v = seen(w.vx, w.vy, vs, w.valive)
        es = torch.full_like(w.ex, math.sqrt(EJECT_R2))
        vis_e = seen(w.ex, w.ey, es, w.ealive)
        n_view = (vis_c.sum(-1) + vis_f.sum(-1) + vis_v.sum(-1) + vis_e.sum(-1)).float().clamp(min=2.0)

        cell, chk = big[..., None], size[:, None, :].expand(B, P, C)
        infl_c = torch.where(cell > chk * 1.3, chk / n_view.log()[..., None],
                             torch.where(chk > cell * 1.3, -(chk / cell.clamp(min=1.0)).log(),
                                         -chk / cell.clamp(min=1.0)))
        rx, ry, dc = field(w.cx.view(B, C), w.cy.view(B, C), size, infl_c, vis_c)
        fx_, fy_, _ = field(w.fx, w.fy, fs, torch.ones_like(vis_f, dtype=torch.float), vis_f)
        full = (n_cells >= K)[..., None]
        infl_v = torch.where(cell > vs[:, None, :] * 1.3, torch.where(full, 2.0, -1.0), 0.0).expand(B, P, w.V)
        vx_, vy_, _ = field(w.vx, w.vy, vs, infl_v, vis_v)
        infl_e = torch.where(cell > es[:, None, :] * 1.3, 2.0, 0.0).expand(B, P, w.E)
        ex_, ey_, _ = field(w.ex, w.ey, es, infl_e, vis_e)
        dirx, diry = rx + fx_ + vx_ + ex_, ry + fy_ + vy_ + ey_

        # an ongoing pursuit (from an earlier split) overrides everything, as in BotPlayer.decide
        cid = w.cid.view(B, C)
        t = self.target.clamp(min=0)
        chase = ((self.target >= 0) & al.gather(1, t) & (cid.gather(1, t) == self.target_id)
                 & (self.pursuit > 0) & alive)
        self.cooldown = (self.cooldown - ticks).clamp(min=0)

        # split-kill: the biggest qualifying prey cell
        can = (1.5 * n_cells < 9) & (self.cooldown <= 0) & alive & ~chase
        reach = torch.clamp(big / 1.4142 * 4.5, min=1.3 * SPLIT_SPEED)[..., None]
        prey_ok = (vis_c & (cell / 1.3 > 1.3 * chk) & (cell * 0.4 < chk) & (dc.clamp(min=1.0) <= reach)
                   & can[..., None])
        psize, pi = torch.where(prey_ok, chk, torch.full_like(chk, -1.0)).max(-1)
        split = psize > 0

        tx, ty = w.cx.view(B, C).gather(1, t) - bx, w.cy.view(B, C).gather(1, t) - by
        dirx, diry = torch.where(chase, tx, dirx), torch.where(chase, ty, diry)
        self.pursuit = torch.where(chase, self.pursuit - ticks, torch.zeros_like(self.pursuit))
        self.target = torch.where(chase, self.target, torch.full_like(self.target, -1))
        # a new split: chase that cell from now on
        px, py = w.cx.view(B, C).gather(1, pi) - bx, w.cy.view(B, C).gather(1, pi) - by
        dirx, diry = torch.where(split, px, dirx), torch.where(split, py, diry)
        self.target = torch.where(split, pi, self.target)
        self.target_id = torch.where(split, cid.gather(1, pi), self.target_id)
        self.pursuit = torch.where(split, torch.full_like(self.pursuit, MO_PURSUIT), self.pursuit)
        self.cooldown = torch.where(split, torch.full_like(self.cooldown, MO_COOLDOWN), self.cooldown)

        ang = torch.atan2(diry, dirx)
        move = torch.round(ang / (2 * math.pi) * 16).long() % 16
        still = (dirx.abs() + diry.abs()) < 1e-9
        move = torch.where(alive & ~still, move, torch.full_like(move, 16))
        return torch.stack([move, split.long(), torch.zeros_like(move)], -1)


# ---- play-time guards (rl/policy.py virus_guard / threat_guard), vectorised for the learners -------
G_DANGER_MARGIN, G_SPLIT_RANGE, G_FLEE_WEIGHT, G_VIRUS_AVOID = 177.0, 842.0, 3.0, 227.5
G_EAT_RATIO, G_SPLIT_KILL_RATIO, G_VIRUS_EAT_MULT = 1.3, 2.6, 1.1576


@torch.no_grad()
def guard_mouse(w: World, mouse: torch.Tensor, level: int) -> torch.Tensor:
    """mouse [B,N,2] targets of players 0..N-1 -> guarded targets, as rl.policy does on the real server:
    level >= 2 adds the champion's threat push (bigger cells inside their danger radius), level >= 1
    steers around the nearest virus our largest cell would pop on that lies across the path."""
    B, P, K, C = w.B, w.P, w.K, w.C
    N, dev = mouse.shape[1], w.dev
    al = w.calive.view(B, P, K)[:, :N]
    r2 = w.cr2.view(B, P, K)[:, :N] * al
    big_r2, bi = r2.max(-1)
    ms = big_r2.sqrt()                                          # our largest cell: radius, mass, position
    mm = big_r2 / 100
    ci = bi + torch.arange(N, device=dev)[None, :] * K
    mx, my = w.cx.view(B, C).gather(1, ci), w.cy.view(B, C).gather(1, ci)
    alive = al.any(-1)
    tx, ty = mouse[..., 0], mouse[..., 1]
    dx, dy = tx - mx, ty - my
    n = (dx * dx + dy * dy).sqrt()
    ux, uy = dx / n.clamp(min=1e-6), dy / n.clamp(min=1e-6)
    ux, uy = torch.where(n > 1e-6, ux, 0.0), torch.where(n > 1e-6, uy, 0.0)
    if level >= 2:
        owner = torch.arange(C, device=dev) // K
        other = (owner[None, None, :] != torch.arange(N, device=dev)[None, :, None]) & w.calive.view(B, C)[:, None, :]
        cs = w.cr2.view(B, C).sqrt()[:, None, :]
        cm = (w.cr2.view(B, C) / 100)[:, None, :]
        ox, oy = mx[..., None] - w.cx.view(B, C)[:, None, :], my[..., None] - w.cy.view(B, C)[:, None, :]
        dist = (ox * ox + oy * oy).sqrt().clamp(min=1.0)
        d = dist - cs - ms[..., None]
        r = cs + ms[..., None] + torch.where(cm > mm[..., None] * G_SPLIT_KILL_RATIO, G_SPLIT_RANGE, 0.0) + G_DANGER_MARGIN
        act = other & (cm > mm[..., None] * G_EAT_RATIO) & (d < r)
        wgt = torch.where(act, (r - d) / r, torch.zeros_like(d))
        fx, fy = (ox / dist * wgt).sum(-1), (oy / dist * wgt).sum(-1)
        push = act.any(-1)
        gx, gy = ux + G_FLEE_WEIGHT * fx, uy + G_FLEE_WEIGHT * fy
        gn = (gx * gx + gy * gy).sqrt().clamp(min=1e-6)
        ux, uy = torch.where(push, gx / gn, ux), torch.where(push, gy / gn, uy)
        n = torch.where(push, n.clamp(min=600.0), n)
    # virus detour
    vs = w.vr2.sqrt()[:, None, :]
    vx, vy = w.vx[:, None, :] - mx[..., None], w.vy[:, None, :] - my[..., None]
    t = vx * ux[..., None] + vy * uy[..., None]
    h = vx * uy[..., None] - vy * ux[..., None]
    clear = vs + ms[..., None] + G_VIRUS_AVOID * 0.2
    block = (w.valive[:, None, :] & (ms[..., None] >= G_VIRUS_EAT_MULT * vs) & (t > 0)
             & (t < torch.minimum(clear + G_VIRUS_AVOID + 200, n[..., None] + clear)) & (h.abs() < clear))
    tb, vi = torch.where(block, t, torch.full_like(t, 1e9)).min(-1)
    hit = (tb < 1e8) & (n > 1e-6)
    g = lambda z: z.gather(-1, vi[..., None])[..., 0]
    side = torch.where(g(h) > 0, 1.0, -1.0)
    cl = g(clear)
    px = g(vx) - uy * side * cl * 1.15
    py = g(vy) + ux * side * cl * 1.15
    pn = (px * px + py * py).sqrt().clamp(min=1.0)
    ux, uy = torch.where(hit, px / pn, ux), torch.where(hit, py / pn, uy)
    out = torch.stack([mx + ux * n, my + uy * n], -1)
    return torch.where((alive & (n > 1e-6))[..., None], out, mouse)
