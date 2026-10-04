"""Vectorised scripted opponent for the simulator: flee threats, chase prey, else eat food.

Works directly on world state (no grid): per player it looks at the nearest threat (a cell
>1.3x our largest cell), nearest prey (<1/1.3x), nearest virus (when big enough to pop) and
nearest food, and returns the usual discrete action (16 headings / stay, split, eject)."""

import math

import torch

from .engine import World

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
