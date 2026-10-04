"""Batched GPU agar.io world replicating the MultiOgar rules (docs/sim.md lists every constant
with its source line and every deliberate simplification).

B independent arenas step in lockstep with fixed-size tensors:
  cells      [B, P*K]  (P players x K=16 slots; slot order = owner-major)
  food       [B, F]
  viruses    [B, V]
  ejected    [B, E]   (oldest blob is recycled when the pool is full)

Units follow MultiOgar: a cell's "size" is its radius, r2 = size**2, mass = r2 / 100.
One `step()` = 2 server ticks (sync=2): physics tick, input phase (split / eject use the
mouse of the *previous* step, then the mouse is updated, as PacketHandler.process does),
physics tick.
"""

import math

import torch

# --- constants (source in docs/sim.md) ---
REF_MAP = 14142                  # arena.js: food/virus density of the default map
SPEED_COEF = 2.2 * 40            # PlayerCell.getSpeed
SPEED_EXP = -0.439
STEER_DIST = 32.0                # speed *= min(dist, 32) / 32
SPLIT_SPEED = 780.0              # playerSplitSpeed; boost = 780 * size ** 0.0122
SPLIT_SPEED_EXP = 0.0122
BOOST_DECAY = 0.1                # moveCell: boostDistance -= boostDistance / 10 per tick
def mts(mass):                   # modules/ini.js:62-64: mts(m) = sqrt(m * 100) + 0.5 (note the +0.5)
    return math.sqrt(mass * 100.0) + 0.5


START_R2 = mts(10) ** 2          # playerStartSize
MIN_DECAY_SIZE = max(32.0, mts(10))  # loadConfig: max(32, playerMinDecay)
DECAY_RATE = 0.002
MAX_SIZE = mts(22500)            # playerMaxSize
MIN_SPLIT_R2 = mts(35) ** 2        # playerMinSplit / playerMinEject
MAX_CELLS = 16
EAT_MULT = 1.15                  # playerEatMult
VIRUS_EAT_MULT = 1.1576
OVERLAP_DIV = 3.0                # resolveCollision: eat when d < big - small / 3
MERGE_TICKS = 30 * 25            # playerMergeTime (s) * 25
MERGE_SIZE_COEF = 0.2            # updateMerge: max(30, 0.2 * size)
RESTORE_TICKS = 13               # splitRestoreTicks
FOOD_R2 = 100.0                  # foodMinSize 10
FOOD_SPAWN = 30
SPAWN_INTERVAL = 20
EJECT_R2 = mts(13) ** 2            # ejectMinSize = ejectMaxSize
EJECT_LOSS_R2 = mts(17) ** 2       # ejectSizeLoss
EJECT_SPEED = 780.0
EJECT_JITTER = 0.3               # ejectRandomAngle: +-0.3 rad
VIRUS_MIN_R2 = mts(100) ** 2         # virusMinSize
VIRUS_MAX_R2 = mts(200) ** 2         # virusMaxSize
VIRUS_SHOOT_SPEED = 780.0
VIRUS_SPLIT_MIN = 36.0           # virusSplitDiv (mass)
BIG_EAT_SIZE, SMALL_NO_GAIN_SIZE = 250.0, 41.23  # Cell.onEat playerBotGrow rule
ES = math.sqrt(EJECT_R2)
POP_MAX = 17


def virus_splits(M: torch.Tensor, L: torch.Tensor, nmax: int = POP_MAX):
    """Vectorised port of Virus.onEaten: masses of the pieces a cell of mass M breaks into
    when `L` cell slots are left. Returns (masses [N, nmax], count [N])."""
    N = M.shape[0]
    dev = M.device
    ar = torch.arange(nmax, device=dev)[None]
    Lc = L.clamp(min=1).float()
    # case A: mass too small for 36-mass pieces
    case_a = (L > 0) & (M / Lc < VIRUS_SPLIT_MIN)
    sc = torch.full((N,), 2.0, device=dev)
    sm = M / 2
    for _ in range(5):
        cond = case_a & (sm > VIRUS_SPLIT_MIN) & (2 * sc < L)
        sc = torch.where(cond, sc * 2, sc)
        sm = torch.where(cond, M / sc, sm)
    pa = M / (sc + 1)
    out_a = torch.where(ar < sc[:, None], pa[:, None], torch.zeros((), device=dev))
    # case B
    cl = L.clone()
    mass_left = M / 2
    smb = M / 2
    cnt = torch.zeros(N, dtype=torch.long, device=dev)
    done = ~((L > 0) & ~case_a)
    out_b = torch.zeros(N, nmax, device=dev)
    for _ in range(MAX_CELLS):
        run = ~done & (cl > 0)
        done = done | ~run
        cl = torch.where(run, cl - 1, cl)
        c = cl
        cf = c.clamp(min=1).float()
        small = run & (c > 0) & (mass_left / cf < VIRUS_SPLIT_MIN)
        smb = torch.where(small, mass_left / cf, smb)
        nsm = run & ~small
        for _ in range(12):
            cond = nsm & (smb >= mass_left) & (c > 0)
            smb = torch.where(cond, smb / 2, smb)
        npush = torch.where(small, c + 1, torch.where(nsm, torch.ones_like(c), torch.zeros_like(c)))
        m = (ar >= cnt[:, None]) & (ar < (cnt + npush)[:, None])
        out_b = torch.where(m, smb[:, None], out_b)
        mass_left = torch.where(run, mass_left - smb, mass_left)
        cnt = cnt + npush
        done = done | small
    masses = torch.where(case_a[:, None], out_a, out_b)
    count = torch.where(case_a, sc.long(), cnt).clamp(max=nmax)
    count = torch.where(L > 0, count, torch.zeros_like(count))
    return masses, count


def _trunc(x):
    return torch.trunc(x)


class World:
    def __init__(self, B: int, P: int = 16, device: str = "cuda", map_size: float = 7000.0,
                 food_mult: float = 2.0, max_eject: int = 256, seed: int = 0,
                 prefill: bool = True, use_graphs: bool | None = None):
        self.B, self.P, self.K = B, P, MAX_CELLS
        self.C = P * self.K
        self.dev = torch.device(device)
        self.map = float(map_size)
        self.lo, self.hi = -self.map / 2, self.map / 2
        area = (map_size / REF_MAP) ** 2
        base_food = int(math.floor(1500 * area + 0.5))
        self.food_min = int(math.floor(base_food * food_mult + 0.5))
        self.virus_min = max(4, int(math.floor(50 * area + 0.5)))
        self.virus_max = max(self.virus_min + 2, int(math.floor(100 * area + 0.5)))
        self.F, self.E, self.V = self.food_min, max_eject, self.virus_max + 4
        self.prefill = prefill
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.t = 0
        d, f = self.dev, torch.float32
        Bc = (B, self.C)
        self.cx, self.cy, self.cr2 = (torch.zeros(Bc, device=d) for _ in range(3))
        self.cbd, self.cbx, self.cby, self.cage = (torch.zeros(Bc, device=d) for _ in range(4))
        self.cid = torch.zeros(Bc, dtype=torch.long, device=d)
        self.calive = torch.zeros(Bc, dtype=torch.bool, device=d)
        self.mx, self.my = torch.zeros(B, P, device=d), torch.zeros(B, P, device=d)
        self.fx, self.fy = torch.zeros(B, self.F, device=d), torch.zeros(B, self.F, device=d)
        self.falive = torch.zeros(B, self.F, dtype=torch.bool, device=d)
        z = lambda n: torch.zeros(B, n, device=d)
        self.ex, self.ey, self.ebd, self.ebx, self.eby, self.eage = (z(self.E) for _ in range(6))
        self.eid = torch.zeros(B, self.E, dtype=torch.long, device=d)
        self.ealive = torch.zeros(B, self.E, dtype=torch.bool, device=d)
        self.vx, self.vy, self.vr2, self.vbd, self.vbx, self.vby = (z(self.V) for _ in range(6))
        self.vid = torch.zeros(B, self.V, dtype=torch.long, device=d)
        self.valive = torch.zeros(B, self.V, dtype=torch.bool, device=d)
        self.next_id = torch.ones(B, dtype=torch.long, device=d)
        # virus attribution (arena.js "Virus attribution"): who ejected a blob, who last fed a virus,
        # who shot it; cumulative per-player stats [shots, pops, popped mass] (column P = trash)
        self.eown = torch.full((B, self.E), -1, dtype=torch.long, device=d)
        self.vlast = torch.full((B, self.V), -1, dtype=torch.long, device=d)
        self.vshooter = torch.full((B, self.V), -1, dtype=torch.long, device=d)
        self.vstat = torch.zeros(3, B, P + 1, device=d)
        # cumulative per-player kills: [enemy cells eaten, enemy mass eaten] (column P = trash)
        self.kstat = torch.zeros(2, B, P + 1, device=d)
        self.use_graphs = (self.dev.type == "cuda") if use_graphs is None else use_graphs
        self._graphs = {}
        self._pool = None
        self._cnt = None          # (max live cells, max live blobs, any big) after the last step
        self._popcell = torch.full((B, P), -1, dtype=torch.long, device=d)
        self._in_mouse = torch.zeros(B, P, 2, device=d)
        self._in_flags = torch.zeros(3, B, P, dtype=torch.bool, device=d)   # split, eject, act
        self._in_respawn = torch.zeros(B, P, dtype=torch.bool, device=d)
        self._in_size = torch.zeros(B, P, device=d)
        self.eject_overwrites = torch.zeros((), dtype=torch.long, device=d)
        self.virus_overflow = torch.zeros((), dtype=torch.long, device=d)
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.dev)

    def _new_ids(self, mask: torch.Tensor) -> torch.Tensor:
        """Unique ids for the True entries of mask [B, ...] (per-arena counter)."""
        flat = mask.reshape(self.B, -1).long()
        ids = self.next_id[:, None] + torch.cumsum(flat, 1) - 1
        self.next_id.add_(flat.sum(1))
        return ids.reshape(mask.shape)

    def _v(self, t):
        return t.view(self.B, self.P, self.K)

    @staticmethod
    def _compact(alive: torch.Tensor, m: int):
        order = torch.argsort((~alive).to(torch.int8), dim=1, stable=True)[:, :m]
        return order, alive.gather(1, order)

    # ------------------------------------------------------------------ reset / spawn
    def reset(self, arenas: torch.Tensor | None = None):
        """Reset the given arenas ([B] bool, default all): prefill food/viruses, spawn everyone."""
        B = self.B
        self._cnt = None
        m = torch.ones(B, dtype=torch.bool, device=self.dev) if arenas is None else arenas
        m1 = m[:, None]
        self.next_id.copy_(torch.where(m, torch.ones_like(self.next_id), self.next_id))
        for name in ("calive", "falive", "ealive", "valive"):
            t = getattr(self, name)
            t.copy_(t & ~m1)
        self.cr2.masked_fill_(m1, 0)
        self.cbd.masked_fill_(m1, 0)
        self.vlast.masked_fill_(m1, -1)
        self.vshooter.masked_fill_(m1, -1)
        self.eown.masked_fill_(m1, -1)
        self.vstat.masked_fill_(m1[None], 0)
        self.kstat.masked_fill_(m1[None], 0)
        if self.prefill:
            pos = self.lo + self.map * self._rand(B, self.F, 2)
            self.fx.copy_(torch.where(m1, pos[..., 0], self.fx))
            self.fy.copy_(torch.where(m1, pos[..., 1], self.fy))
            self.falive.copy_(self.falive | m1)
            vp = self.lo + self.map * self._rand(B, self.V, 2)
            vmask = m1 & (torch.arange(self.V, device=self.dev)[None] < self.virus_min)
            self.vx.copy_(torch.where(vmask, vp[..., 0], self.vx))
            self.vy.copy_(torch.where(vmask, vp[..., 1], self.vy))
            self.vr2.copy_(torch.where(vmask, torch.full_like(self.vr2, VIRUS_MIN_R2), self.vr2))
            self.vbd.masked_fill_(m1, 0)
            self.valive.copy_(self.valive | vmask)
            self.vid.copy_(torch.where(vmask, self._new_ids(vmask), self.vid))
        size = torch.full((B, self.P), math.sqrt(START_R2), device=self.dev)
        self.spawn(m[:, None].expand(B, self.P), size)

    def _overlaps(self, px, py, size):
        """MultiOgar willCollide: box overlap with any live player cell or virus. px,py,size [B,P]."""
        cs = self.cr2.sqrt()
        r = size[:, :, None] + cs[:, None, :]
        hit = ((px[:, :, None] - self.cx[:, None, :]).abs() <= r) & \
              ((py[:, :, None] - self.cy[:, None, :]).abs() <= r) & self.calive[:, None, :]
        vs = self.vr2.sqrt()
        rv = size[:, :, None] + vs[:, None, :]
        hitv = ((px[:, :, None] - self.vx[:, None, :]).abs() <= rv) & \
               ((py[:, :, None] - self.vy[:, None, :]).abs() <= rv) & self.valive[:, None, :]
        return hit.any(-1) | hitv.any(-1)

    def spawn(self, mask: torch.Tensor, size: torch.Tensor):
        """Spawn one cell of the given size for each masked (dead) player at a random free spot."""
        B, P, K = self.B, self.P, self.K
        pos = self.lo + self.map * self._rand(B, P, 2)
        for _ in range(10):
            hit = self._overlaps(pos[..., 0], pos[..., 1], size)
            new = self.lo + self.map * self._rand(B, P, 2)
            pos = torch.where(hit[..., None], new, pos)
        slot0 = torch.zeros(B, P, K, dtype=torch.bool, device=self.dev)
        slot0[:, :, 0] = True
        sel = (slot0 & mask[:, :, None])
        sel_f = sel.view(B, self.C)
        put = lambda dst, val: dst.copy_(torch.where(sel_f, val[:, :, None].expand(B, P, K).reshape(B, self.C), dst))
        put(self.cx, pos[..., 0])
        put(self.cy, pos[..., 1])
        put(self.cr2, size * size)
        self.cbd.masked_fill_(sel_f, 0)
        self.cage.masked_fill_(sel_f, 0)
        self.cid.copy_(torch.where(sel_f, self._new_ids(sel_f), self.cid))
        self.calive.copy_(self.calive | sel_f)
        self.mx.copy_(torch.where(mask, pos[..., 0], self.mx))
        self.my.copy_(torch.where(mask, pos[..., 1], self.my))

    # ------------------------------------------------------------------ queries
    def alive_players(self) -> torch.Tensor:
        return self._v(self.calive).any(-1)

    def total_r2(self) -> torch.Tensor:
        return (self.cr2 * self.calive).view(self.B, self.P, self.K).sum(-1)

    def total_mass(self) -> torch.Tensor:
        return self.total_r2() / 100

    def n_cells(self) -> torch.Tensor:
        return self._v(self.calive).sum(-1)

    # ------------------------------------------------------------------ physics pieces
    def _boost(self, x, y, bd, bx, by, r2):
        """moveCell: boosted motion (10% of the remaining distance per tick), border bounce+clamp."""
        step = bd * BOOST_DECAY
        bd.sub_(step)
        bd.masked_fill_(bd < 0.05, 0)
        x.add_(bx * step)
        y.add_(by * step)
        r = r2.sqrt() * 0.5
        lo_, hi_ = self.lo + r, self.hi - r
        bx.copy_(torch.where((x < lo_) | (x > hi_), -bx, bx))
        by.copy_(torch.where((y < lo_) | (y > hi_), -by, by))
        x.copy_(torch.minimum(torch.maximum(x, lo_), hi_))
        y.copy_(torch.minimum(torch.maximum(y, lo_), hi_))

    def _steer(self):
        """movePlayer: every cell moves toward the player's mouse at getSpeed(dist)."""
        B, P, K = self.B, self.P, self.K
        mx = self.mx[:, :, None].expand(B, P, K).reshape(B, self.C)
        my = self.my[:, :, None].expand(B, P, K).reshape(B, self.C)
        dx, dy = _trunc(mx - self.cx), _trunc(my - self.cy)
        sq = dx * dx + dy * dy
        dist = sq.sqrt()
        size = self.cr2.sqrt().clamp(min=1e-3)
        speed = SPEED_COEF * size.pow(SPEED_EXP) * dist.clamp(max=STEER_DIST) / STEER_DIST
        go = self.calive & (sq >= 1)
        k = torch.where(go, speed / dist.clamp(min=1e-6), torch.zeros_like(speed))
        self.cx.add_(dx * k)
        self.cy.add_(dy * k)

    def _split_cells(self, elig: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor):
        """Halve every eligible cell [B,P,K], the new half is launched along (ux, uy) (unit)."""
        B, P, K = self.B, self.P, self.K
        alive = self.calive.view(B, P, K)
        free_order = torch.argsort(alive.to(torch.int8), dim=-1, stable=True)
        nfree = (K - alive.sum(-1, keepdim=True))
        rank = torch.cumsum(elig.long(), -1) - 1
        ok = elig & (rank < nfree)
        tgt = free_order.gather(-1, rank.clamp(min=0))
        tgt = torch.where(ok, tgt, torch.full_like(tgt, K))
        r2 = self._v(self.cr2)
        half = r2 / 2
        bd_new = SPLIT_SPEED * half.sqrt().pow(SPLIT_SPEED_EXP)

        def scat(vals, dtype=torch.float32):
            buf = torch.zeros(B, P, K + 1, dtype=dtype, device=self.dev)
            buf.scatter_(-1, tgt, vals.to(dtype))
            return buf[..., :K]
        newm = scat(ok, torch.bool)
        newm_f = newm.reshape(B, self.C)
        put = lambda dst, val: dst.copy_(torch.where(newm_f, val.reshape(B, self.C), dst))
        put(self.cx, scat(self._v(self.cx)))
        put(self.cy, scat(self._v(self.cy)))
        put(self.cr2, scat(half))
        put(self.cbd, scat(bd_new))
        put(self.cbx, scat(ux))
        put(self.cby, scat(uy))
        self.cage.masked_fill_(newm_f, 0)
        self.cid.copy_(torch.where(newm_f, self._new_ids(newm_f), self.cid))
        self.calive.copy_(self.calive | newm_f)
        self.cr2.copy_(torch.where(ok.reshape(B, self.C), half.reshape(B, self.C), self.cr2))

    def _aim(self):
        """Per-cell vector to the mouse, (dx, dy) float, [B,P,K]."""
        B, P, K = self.B, self.P, self.K
        return (self.mx[:, :, None] - self._v(self.cx), self.my[:, :, None] - self._v(self.cy))

    def _user_split(self, flag: torch.Tensor):
        """GameServer.splitCells: every cell with size > playerMinSplit splits toward the mouse."""
        dx, dy = self._aim()
        dx, dy = _trunc(dx), _trunc(dy)
        n = (dx * dx + dy * dy).sqrt()
        zero = n < 1
        ux = torch.where(zero, torch.zeros_like(dx), dx / n.clamp(min=1e-6))
        uy = torch.where(zero, torch.ones_like(dy), dy / n.clamp(min=1e-6))  # atan2(0,0)=0 -> (sin0, cos0)
        elig = self._v(self.calive) & (self._v(self.cr2) > MIN_SPLIT_R2) & flag[:, :, None]
        self._split_cells(elig, ux, uy)

    def _auto_split(self, big: torch.Tensor):
        """autoSplit: cells beyond playerMaxSize halve (random angle) or are clamped at 16 cells."""
        B, P, K = self.B, self.P, self.K
        full = self.n_cells()[:, :, None] >= K
        clamp = big & full
        self.cr2.copy_(torch.where(clamp.view(B, self.C), torch.full_like(self.cr2, MAX_SIZE ** 2), self.cr2))
        ang = self._rand(B, P, K) * (2 * math.pi)
        self._split_cells(big & ~full, torch.sin(ang), torch.cos(ang))

    def _eject(self, flag: torch.Tensor):
        """GameServer.ejectMass: each cell with size >= playerMinEject sheds 17 mass as a 13-mass blob."""
        B, P, K, C = self.B, self.P, self.K, self.C
        dx, dy = self._aim()
        sq = dx * dx + dy * dy
        aim = sq > 1
        n = sq.sqrt().clamp(min=1e-6)
        elig = self._v(self.calive) & (self._v(self.cr2) >= MIN_SPLIT_R2) & flag[:, :, None]
        r2n = self._v(self.cr2) - EJECT_LOSS_R2
        size_n = r2n.clamp(min=0).sqrt()
        ux = torch.where(aim, dx / n, torch.zeros_like(dx))
        uy = torch.where(aim, dy / n, torch.zeros_like(dy))
        ang = torch.where(aim, torch.atan2(dx, dy) + (0.6 * self._rand(B, P, K) - EJECT_JITTER),
                          torch.full_like(dx, math.pi / 2))
        bx_, by_ = torch.sin(ang), torch.cos(ang)
        px = self._v(self.cx) + ux * size_n
        py = self._v(self.cy) + uy * size_n
        self.cr2.copy_(torch.where(elig.view(B, C), r2n.reshape(B, C), self.cr2))
        el = elig.view(B, C)
        rank = torch.cumsum(el.long(), 1) - 1
        key = torch.where(self.ealive, -self.eage, torch.full_like(self.eage, -1e12))
        order = torch.argsort(key, dim=1)
        ok = el & (rank < self.E)
        tgt = order.gather(1, rank.clamp(min=0, max=self.E - 1))
        self.eject_overwrites += (ok & self.ealive.gather(1, tgt)).sum()
        tgt = torch.where(ok, tgt, torch.full_like(tgt, self.E))

        def scat(vals, dtype=torch.float32):
            buf = torch.zeros(B, self.E + 1, dtype=dtype, device=self.dev)
            buf.scatter_(1, tgt, vals.reshape(B, C).to(dtype))
            return buf[:, :self.E]
        newm = scat(ok, torch.bool)
        put = lambda dst, val: dst.copy_(torch.where(newm, val, dst))
        put(self.ex, scat(px))
        put(self.ey, scat(py))
        put(self.ebx, scat(bx_))
        put(self.eby, scat(by_))
        own_p = torch.arange(P, device=self.dev)[None, :, None].expand(B, P, K)
        put(self.eown, scat(own_p, torch.long))
        self.ebd.copy_(torch.where(newm, torch.full_like(self.ebd, EJECT_SPEED), self.ebd))
        self.eage.masked_fill_(newm, 0)
        self.eid.copy_(torch.where(newm, self._new_ids(newm), self.eid))
        self.ealive.copy_(self.ealive | newm)

    def _pop(self, popcell: torch.Tensor):
        """Virus.onEaten: popcell [B,P] flat cell index of the eater per player (-1 = none)."""
        B, P, K = self.B, self.P, self.K
        has = popcell >= 0
        ar_k = torch.arange(K, device=self.dev)
        oh = ((popcell.clamp(min=0) % K)[:, :, None] == ar_k) & has[:, :, None]
        r2v, cxv, cyv = self._v(self.cr2), self._v(self.cx), self._v(self.cy)
        mass = (r2v * oh).sum(-1) / 100
        nal = self._v(self.calive).sum(-1)
        masses, cnt = virus_splits(mass.reshape(-1), (K - nal).reshape(-1))
        masses = masses.view(B, P, -1)
        cnt = torch.where(has, cnt.view(B, P), torch.zeros_like(nal))
        nmax = POP_MAX
        px = (cxv * oh).sum(-1)
        py = (cyv * oh).sum(-1)
        for j in range(nmax):
            r2_1 = 100 * masses[:, :, j]
            parent = (self._v(self.cr2) * oh).sum(-1)
            newp = parent - r2_1
            alive = self._v(self.calive)
            free = ~alive
            ok = (cnt > j) & (newp >= MIN_DECAY_SIZE ** 2) & free.any(-1)
            tgt = free.int().argmax(-1)
            toh = (tgt[:, :, None] == ar_k) & ok[:, :, None]
            toh_f = toh.view(B, self.C)
            ang = self._rand(B, P) * (2 * math.pi)
            r2 = self._v(self.cr2)
            self.cr2.copy_(torch.where((oh & ok[:, :, None]).view(B, self.C),
                                       (r2 - r2_1[:, :, None]).view(B, self.C), self.cr2))
            ex_ = lambda t: t[:, :, None].expand(B, P, K).reshape(B, self.C)
            put = lambda dst, val: dst.copy_(torch.where(toh_f, ex_(val), dst))
            put(self.cx, px)
            put(self.cy, py)
            put(self.cr2, r2_1)
            put(self.cbd, SPLIT_SPEED * r2_1.clamp(min=1e-3).sqrt().pow(SPLIT_SPEED_EXP))
            put(self.cbx, torch.sin(ang))
            put(self.cby, torch.cos(ang))
            self.cage.masked_fill_(toh_f, 0)
            self.cid.copy_(torch.where(toh_f, self._new_ids(toh_f), self.cid))
            self.calive.copy_(self.calive | toh_f)

    # ------------------------------------------------------------------ collisions
    def _collide(self, ready, cm: int, em: int):
        B, P, K, C = self.B, self.P, self.K, self.C
        dev = self.dev
        cidx, valid = self._compact(self.calive, cm)
        g = lambda t: t.gather(1, cidx)
        X, Y, R2, AGE, RDY = g(self.cx), g(self.cy), g(self.cr2), g(self.cage), g(ready)
        OWN = cidx // K
        SZ = R2.sqrt()
        I = torch.arange(cm, device=dev)
        # ---- own cells: rigid push (before remerge) and cell-cell eating / merging
        dx = X[:, None, :] - X[:, :, None]
        dy = Y[:, None, :] - Y[:, :, None]
        d2 = dx * dx + dy * dy
        d = d2.sqrt()
        Si, Sj = SZ[:, :, None], SZ[:, None, :]
        pair = valid[:, :, None] & valid[:, None, :] & (I[:, None] != I[None, :])[None]
        same = OWN[:, :, None] == OWN[:, None, :]
        a13 = AGE >= RESTORE_TICKS
        both13 = a13[:, :, None] & a13[:, None, :]
        rdy2 = RDY[:, :, None] & RDY[:, None, :]
        rr = Si + Sj
        rigid = pair & same & both13 & ~rdy2 & (d <= rr)
        push = torch.minimum((rr - d) / d.clamp(min=1e-6), rr - d)
        w = R2[:, None, :] / (R2[:, :, None] + R2[:, None, :]).clamp(min=1e-6)
        mult = torch.where(rigid, push * w, torch.zeros_like(push))
        X2 = X - (mult * _trunc(dx)).sum(2)
        Y2 = Y - (mult * _trunc(dy)).sum(2)
        ov = d2 < (Si - Sj / OVERLAP_DIV) ** 2
        diff_eat = pair & ~same & (Si >= EAT_MULT * Sj) & ov
        merge = pair & same & both13 & rdy2 & ov & ((Si > Sj) | ((Si == Sj) & (I[:, None] < I[None, :])[None]))
        E_ = diff_eat | merge
        eater = torch.where(E_, Si.expand_as(E_) + 1, torch.zeros_like(d)).argmax(1)
        eaten = E_.any(1)
        eaten = eaten & ~eaten.gather(1, eater)
        esize = SZ.gather(1, eater)
        give = torch.where((esize >= BIG_EAT_SIZE) & (SZ <= SMALL_NO_GAIN_SIZE), torch.zeros_like(R2), R2)
        gain = torch.zeros_like(R2).scatter_add_(1, eater, give * eaten)
        # kills: cells of other players eaten, credited to the eater's owner
        eown = OWN.gather(1, eater)
        kill = (eaten & (eown != OWN)).float()
        self.kstat[0].scatter_add_(1, eown, kill)
        self.kstat[1].scatter_add_(1, eown, kill * give / 100)
        R2 = R2 + gain
        valid = valid & ~eaten
        SZ = R2.sqrt()
        X, Y = X2, Y2
        # ---- viruses: eaten by a big cell (pop), at most one per player per tick
        VS = self.vr2.sqrt()
        dxv = self.vx[:, None, :] - X[:, :, None]
        dyv = self.vy[:, None, :] - Y[:, :, None]
        Sc, Sv = SZ[:, :, None], VS[:, None, :]
        evt = valid[:, :, None] & self.valive[:, None, :] & (Sc >= VIRUS_EAT_MULT * Sv) & \
            (dxv * dxv + dyv * dyv < (Sc - Sv / OVERLAP_DIV) ** 2)
        ve = torch.where(evt, Sc.expand_as(evt) + 1, torch.zeros_like(dxv)).argmax(1)  # [B,V]
        vhas = evt.any(1)
        vown = OWN.gather(1, ve)
        lower = torch.tril(torch.ones(self.V, self.V, dtype=torch.bool, device=dev), -1)[None]
        vhas = vhas & ~((vown[:, :, None] == vown[:, None, :]) & vhas[:, None, :] & lower).any(2)
        vflat = cidx.gather(1, ve)
        vr2_eaten = torch.where(vhas, self.vr2, torch.zeros_like(self.vr2))
        # a player exploding on a virus another player shot pays that shooter (if there was room to split)
        ncell = self.calive.view(B, P, K).sum(-1)
        room = (K - ncell.gather(1, vown.clamp(max=P - 1))) > 0
        pop_ok = vhas & (self.vshooter >= 0) & (self.vshooter != vown) & room
        pidx = torch.where(pop_ok, self.vshooter, torch.full_like(vown, P))
        self.vstat[1].scatter_add_(1, pidx, pop_ok.float())
        self.vstat[2].scatter_add_(1, pidx, (R2.gather(1, ve) + vr2_eaten) / 100 * pop_ok.float())
        R2 = R2.scatter_add(1, ve, vr2_eaten)
        popcell = torch.full((B, P + 1), -1, dtype=torch.long, device=dev)
        popcell.scatter_(1, torch.where(vhas, vown, torch.full_like(vown, P)), vflat)
        popcell = popcell[:, :P]
        valive_v = self.valive & ~vhas
        SZ = R2.sqrt()
        # ---- ejected blobs: eaten by cells (age >= 1), else by viruses (feeding)
        eidx, evalid = self._compact(self.ealive, em)
        ge = lambda t: t.gather(1, eidx)
        EX, EY, EAGE = ge(self.ex), ge(self.ey), ge(self.eage)
        edible = evalid & (EAGE >= 1)
        # eject-eject rigid push (ejectCollisionType 0)
        edx = EX[:, None, :] - EX[:, :, None]
        edy = EY[:, None, :] - EY[:, :, None]
        ed = (edx * edx + edy * edy).sqrt()
        Ie = torch.arange(em, device=dev)
        epair = evalid[:, :, None] & evalid[:, None, :] & (Ie[:, None] != Ie[None, :])[None] & (ed <= 2 * ES)
        epush = torch.minimum((2 * ES - ed) / ed.clamp(min=1e-6), 2 * ES - ed) * 0.5
        epush = torch.where(epair, epush, torch.zeros_like(epush))
        EX2 = EX - (epush * _trunc(edx)).sum(2)
        EY2 = EY - (epush * _trunc(edy)).sum(2)
        cdx = EX[:, None, :] - X[:, :, None]
        cdy = EY[:, None, :] - Y[:, :, None]
        Sc = SZ[:, :, None]
        ec = valid[:, :, None] & edible[:, None, :] & (Sc >= ES) & \
            (cdx * cdx + cdy * cdy < (Sc - ES / OVERLAP_DIV) ** 2)
        eatenE = ec.any(1)
        eE = torch.where(ec, Sc.expand_as(ec) + 1, torch.zeros_like(cdx)).argmax(1)
        R2 = R2.scatter_add(1, eE, EJECT_R2 * eatenE.float())
        n_virus = valive_v.sum(1, keepdim=True)
        VS = self.vr2.sqrt()
        vdx = EX[:, None, :] - self.vx[:, :, None]
        vdy = EY[:, None, :] - self.vy[:, :, None]
        Svv = VS[:, :, None]
        evv = valive_v[:, :, None] & (edible & ~eatenE)[:, None, :] & (n_virus < self.virus_max)[:, :, None] & \
            (Svv >= ES) & (vdx * vdx + vdy * vdy < (Svv - ES / OVERLAP_DIV) ** 2)
        eatenV = evv.any(1)
        eV = evv.float().argmax(1)  # [B,Em]: virus that eats blob
        vgain = torch.zeros(B, self.V, device=dev).scatter_add_(1, eV, EJECT_R2 * eatenV.float())
        fed = vgain > 0
        feeder = torch.full((B, self.V + 1), -1, dtype=torch.long, device=dev).scatter_(
            1, torch.where(eatenV, eV, torch.full_like(eV, self.V)), ge(self.eown))
        self.vlast.copy_(torch.where(fed, feeder[:, :self.V], self.vlast))   # Virus.onEat: lastFeeder = blob.shooter
        edirx = EX.new_zeros(B, self.V + 1).scatter_(1, torch.where(eatenV, eV, torch.full_like(eV, self.V)), ge(self.ebx))
        ediry = EX.new_zeros(B, self.V + 1).scatter_(1, torch.where(eatenV, eV, torch.full_like(eV, self.V)), ge(self.eby))
        # ---- food
        FX, FY = self.fx, self.fy
        fdx = FX[:, None, :] - X[:, :, None]
        fdy = FY[:, None, :] - Y[:, :, None]
        fc = valid[:, :, None] & self.falive[:, None, :] & (fdx * fdx + fdy * fdy < (Sc - 10.0 / OVERLAP_DIV) ** 2)
        eatenF = fc.any(1)
        fe = fc.to(torch.int8).argmax(1)
        R2 = R2.scatter_add(1, fe, FOOD_R2 * eatenF.float())
        self.falive.copy_(self.falive & ~eatenF)
        # ---- write back
        self.cx.scatter_(1, cidx, X)
        self.cy.scatter_(1, cidx, Y)
        self.cr2.scatter_(1, cidx, R2)
        self.calive.copy_(torch.zeros_like(self.calive).scatter_(1, cidx, valid))
        self.ex.scatter_(1, eidx, EX2)
        self.ey.scatter_(1, eidx, EY2)
        self.ealive.copy_(torch.zeros_like(self.ealive).scatter_(1, eidx, evalid & ~eatenE & ~eatenV))
        # viruses: fed blobs grow them; at max size they shoot a new virus
        vr2 = torch.where(valive_v, self.vr2 + vgain, self.vr2)
        shoot = valive_v & (vr2 >= VIRUS_MAX_R2)
        self.vr2.copy_(torch.where(shoot, torch.full_like(vr2, VIRUS_MIN_R2), vr2))   # Virus.onEat: back to min size
        self.valive.copy_(valive_v)
        self._shoot_viruses(shoot, edirx[:, :self.V], ediry[:, :self.V])
        return popcell

    def _shoot_viruses(self, shoot, dirx, diry):
        B, V = self.B, self.V
        rank = torch.cumsum(shoot.long(), 1) - 1
        order = torch.argsort(self.valive.to(torch.int8), dim=1, stable=True)
        nfree = V - self.valive.sum(1, keepdim=True)
        ok = shoot & (rank < nfree)
        self.virus_overflow += (shoot & ~ok).sum()
        P = self.P
        self.vstat[0].scatter_add_(1, torch.where(shoot & (self.vlast >= 0), self.vlast, torch.full_like(self.vlast, P)),
                                   (shoot & (self.vlast >= 0)).float())
        tgt = order.gather(1, rank.clamp(min=0, max=V - 1))
        tgt = torch.where(ok, tgt, torch.full_like(tgt, V))

        def scat(vals, dtype=torch.float32):
            buf = torch.zeros(B, V + 1, dtype=dtype, device=self.dev)
            buf.scatter_(1, tgt, vals.to(dtype))
            return buf[:, :V]
        newm = scat(ok, torch.bool)
        put = lambda dst, val: dst.copy_(torch.where(newm, val, dst))
        put(self.vx, scat(self.vx))
        put(self.vy, scat(self.vy))
        put(self.vbx, scat(dirx))
        put(self.vby, scat(diry))
        put(self.vshooter, scat(self.vlast, torch.long))
        self.vlast.copy_(torch.where(newm, torch.full_like(self.vlast, -1), self.vlast))
        self.vr2.copy_(torch.where(newm, torch.full_like(self.vr2, VIRUS_MIN_R2), self.vr2))
        self.vbd.copy_(torch.where(newm, torch.full_like(self.vbd, VIRUS_SHOOT_SPEED), self.vbd))
        self.vid.copy_(torch.where(newm, self._new_ids(newm), self.vid))
        self.valive.copy_(self.valive | newm)

    # ------------------------------------------------------------------ periodic events
    def _spawn_food_virus(self):
        B = self.B
        dead = ~self.falive
        n = (self.food_min - self.falive.sum(1)).clamp(min=0, max=FOOD_SPAWN)
        new = dead & (torch.cumsum(dead.long(), 1) <= n[:, None])
        pos = self.lo + self.map * self._rand(B, self.F, 2)
        self.fx.copy_(torch.where(new, pos[..., 0], self.fx))
        self.fy.copy_(torch.where(new, pos[..., 1], self.fy))
        self.falive.copy_(self.falive | new)
        # one virus per spawn event if the random spot is clear (the 2nd would collide with the 1st)
        vp = self.lo + self.map * self._rand(B, 2)
        size = torch.full((B, 1), math.sqrt(VIRUS_MIN_R2), device=self.dev)
        hit = self._overlaps(vp[:, None, 0], vp[:, None, 1], size)[:, 0]
        want = (self.valive.sum(1) < self.virus_min) & ~hit
        free = ~self.valive
        first = free & (torch.cumsum(free.long(), 1) == 1) & want[:, None]
        self.vx.copy_(torch.where(first, vp[:, 0:1].expand_as(self.vx), self.vx))
        self.vy.copy_(torch.where(first, vp[:, 1:2].expand_as(self.vy), self.vy))
        self.vr2.copy_(torch.where(first, torch.full_like(self.vr2, VIRUS_MIN_R2), self.vr2))
        self.vbd.masked_fill_(first, 0)
        self.vlast.masked_fill_(first, -1)
        self.vshooter.masked_fill_(first, -1)
        self.vid.copy_(torch.where(first, self._new_ids(first), self.vid))
        self.valive.copy_(self.valive | first)

    def _decay(self):
        al = self._v(self.calive)
        r2 = self._v(self.cr2)
        small = al & (r2.sqrt() <= MIN_DECAY_SIZE)
        dec = al & (torch.cumsum(small.long(), -1) == 0)  # updateDecay `break`s at the first small cell
        new = torch.clamp(r2 * (1 - DECAY_RATE), min=MIN_DECAY_SIZE ** 2)
        self.cr2.copy_(torch.where(dec, new, r2).view(self.B, self.C))

    # ------------------------------------------------------------------ tick / step
    _STATE = ("cx", "cy", "cr2", "cbd", "cbx", "cby", "cage", "cid", "calive", "mx", "my", "fx", "fy",
              "falive", "ex", "ey", "ebd", "ebx", "eby", "eage", "eid", "ealive", "vx", "vy", "vr2",
              "vbd", "vbx", "vby", "vid", "valive", "next_id", "eown", "vlast", "vshooter", "vstat", "kstat", "eject_overwrites", "virus_overflow")

    def _capture(self, fn):
        """CUDA-graph `fn` (static shapes, no host sync). The world is restored after the warm-up run."""
        backup = {n: getattr(self, n).clone() for n in self._STATE}
        gen_state = self.gen.get_state()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        for n, v in backup.items():
            getattr(self, n).copy_(v)
        self.gen.set_state(gen_state)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.gen)
        with torch.cuda.graph(g, pool=self._pool):
            fn()
        torch.cuda.synchronize()
        for n, v in backup.items():
            getattr(self, n).copy_(v)
        self.gen.set_state(gen_state)
        return g

    def _run(self, key, fn):
        if not self.use_graphs:
            return fn()
        g = self._graphs.get(key)
        if g is None:
            g = self._graphs[key] = self._capture(fn)
        g.replay()

    @staticmethod
    def _bucket(n: int, lo: int, hi: int) -> int:
        b = lo
        while b < n and b < hi:
            b *= 2
        return min(b, hi)

    def _counts(self):
        """(max live cells per arena, max live ejected blobs, any cell beyond max size, any pending pop)"""
        big = (self.cr2 >= MAX_SIZE ** 2) & self.calive
        return torch.stack([self.calive.sum(1).max(), self.ealive.sum(1).max(), big.any().long(),
                            (self._popcell >= 0).any().long()]).tolist()

    def _tick_body(self, cm: int, em: int):
        size = self.cr2.sqrt()
        ready = self.calive & (self.cage >= torch.clamp(MERGE_SIZE_COEF * size, min=30.0) * 25.0)
        self._boost(self.cx, self.cy, self.cbd, self.cbx, self.cby, self.cr2)
        self._steer()
        self._boost(self.ex, self.ey, self.ebd, self.ebx, self.eby, self.ealive.float() * EJECT_R2)
        self._boost(self.vx, self.vy, self.vbd, self.vbx, self.vby, self.vr2)
        self._auto_split(self._v(self.calive) & (self._v(self.cr2) >= MAX_SIZE ** 2))
        self._popcell.copy_(self._collide(ready, cm, em))

    def _pop_body(self):
        self._pop(self._popcell)

    def _tick(self, n: int, e: int, big: int):
        """One physics tick given (live cells, live blobs, any big) counts from before it."""
        cm = self._bucket(max(n * (2 if big else 1), 1), 16, self.C)
        em = self._bucket(max(e, 1), 8, self.E)
        self._run(("tick", cm, em), lambda: self._tick_body(cm, em))
        n, e, big, pop = self._counts()
        if pop:
            self._run(("pop",), self._pop_body)
            n, e, big, _ = self._counts()
        if self.t % SPAWN_INTERVAL == 0:
            self._spawn_food_virus()
        if (self.t + 3) % 25 == 0:
            self._decay()
        self.t += 1
        self.cage.add_(1)
        self.eage.add_(1)
        return n, e, big

    def _input_body(self):
        split, eject, act = self._in_flags
        a = act & self.alive_players()
        self._user_split(split & a)   # splits/ejects use the previous mouse (PacketHandler.process order)
        self._eject(eject & a)
        m = _trunc(self._in_mouse)
        self.mx.copy_(torch.where(a, m[..., 0], self.mx))
        self.my.copy_(torch.where(a, m[..., 1], self.my))

    def _spawn_body(self):
        self.spawn(self._in_respawn & ~self.alive_players(), self._in_size)

    def step(self, mouse: torch.Tensor, split: torch.Tensor, eject: torch.Tensor,
             act: torch.Tensor, respawn: torch.Tensor | None = None,
             spawn_size: torch.Tensor | None = None):
        """One agent step (2 ticks). mouse [B,P,2] world coords, split/eject/act [B,P] bool,
        respawn [B,P] bool (dead players that re-enter now), spawn_size [B,P] radius."""
        B, P = self.B, self.P
        if respawn is not None:
            if spawn_size is None:
                spawn_size = torch.full((B, P), math.sqrt(START_R2), device=self.dev)
            self._in_respawn.copy_(respawn)
            self._in_size.copy_(spawn_size)
            self._run(("spawn",), self._spawn_body)
            act = act & ~respawn
        self._in_mouse.copy_(mouse)
        self._in_flags[0].copy_(split)
        self._in_flags[1].copy_(eject)
        self._in_flags[2].copy_(act)
        if self._cnt is None:
            n, e, big, _ = self._counts()
        else:
            n, e, big = self._cnt
            n = min(n + (P if respawn is not None else 0), self.C)   # a respawn adds <= 1 cell per player
        n, e, big = self._tick(n, e, big)
        self._run(("input",), self._input_body)
        self._cnt = self._tick(min(2 * n, self.C), min(e + n, self.E), big)   # exact counts, reused next step

    # ------------------------------------------------------------------ test helpers
    def clear(self, b: int = 0):
        self._cnt = None
        for name in ("calive", "falive", "ealive", "valive"):
            getattr(self, name)[b] = False

    def put_cell(self, b, p, x, y, mass, age=10_000.0, bd=0.0, ang=0.0, k=None):
        self._cnt = None
        k = int((~self.calive[b, p * self.K:(p + 1) * self.K]).nonzero()[0].item()) if k is None else k
        i = p * self.K + k
        self.cx[b, i], self.cy[b, i], self.cr2[b, i] = x, y, mass * 100
        self.cbd[b, i], self.cbx[b, i], self.cby[b, i] = bd, math.sin(ang), math.cos(ang)
        self.cage[b, i] = age
        self.cid[b, i] = int(self.next_id[b].item())
        self.next_id[b] += 1
        self.calive[b, i] = True
        return i

    def put_food(self, b, x, y):
        j = int((~self.falive[b]).nonzero()[0].item())
        self.fx[b, j], self.fy[b, j], self.falive[b, j] = x, y, True

    def put_virus(self, b, x, y, mass=100.0):
        j = int((~self.valive[b]).nonzero()[0].item())
        self.vx[b, j], self.vy[b, j], self.vr2[b, j] = x, y, mass * 100
        self.vbd[b, j] = 0
        self.vlast[b, j] = self.vshooter[b, j] = -1
        self.valive[b, j] = True
        return j

    def put_eject(self, b, x, y, bd=0.0, ang=0.0, age=5.0, owner=-1):
        self._cnt = None
        j = int((~self.ealive[b]).nonzero()[0].item())
        self.ex[b, j], self.ey[b, j], self.ebd[b, j] = x, y, bd
        self.ebx[b, j], self.eby[b, j] = math.sin(ang), math.cos(ang)
        self.eage[b, j] = age
        self.eown[b, j] = owner
        self.ealive[b, j] = True
        return j
