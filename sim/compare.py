"""Cross-check the simulator against the REAL MultiOgar arena server.

For each experiment the same scripted decision logic drives (a) real agents on an arena server
started here on a free port (no built-in bots, 1 food pellet; terminated afterwards) and (b) the
simulator started from the same initial cell sizes/positions. Quantities are measured with the same
function on both sides (what the real client sees: floored sizes, truncated positions).

    uv run python -m sim.compare [--quick]
"""

import argparse
import asyncio
import collections
import math
import subprocess
import sys
import threading

import numpy as np
import torch

from arena.client import StepConn
from arena.match import SERVER_JS, free_port

from .engine import World

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SYNC = 2


# ---------------------------------------------------------------------------- real server
def start_real(n_agents: int, seed: int, bots: int = 0, food: int = 1) -> subprocess.Popen:
    port = free_port()
    cmd = ["node", str(SERVER_JS), "--port", str(port), "--bots", str(bots), "--seed", str(seed),
           "--agents", str(n_agents), "--sync", str(SYNC), "--map", "7000", "--food", str(food)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if line.startswith("[arena] port="):
            break
    else:
        raise RuntimeError("arena server failed to start")
    proc.port = port
    proc.tail = collections.deque(maxlen=5)
    threading.Thread(target=lambda: proc.tail.extend(proc.stdout), daemon=True).start()
    return proc


class Obs:
    """What one agent sees at one step: own cells [(x, y, size)], others [(kind, x, y, size)]."""

    def __init__(self, mine, others):
        self.mine, self.others = mine, others

    @property
    def mass(self):
        return sum(s * s / 100 for _, _, s in self.mine)

    @property
    def centre(self):
        m = self.mass or 1.0
        return (sum(x * s * s / 100 for x, y, s in self.mine) / m, sum(y * s * s / 100 for x, y, s in self.mine) / m)


async def real_session(sizes, policy, steps, seed=1, bots=0, food=1, with_food=False):
    """Run `steps` steps with one agent per entry of `sizes` (spawn radius). policy(step, [Obs]) ->
    [(x, y, split, eject)]. Returns the list of per-step [Obs]."""
    proc = start_real(len(sizes), seed, bots, food)
    conns = [StepConn(f"ws://127.0.0.1:{proc.port}", f"cmp{i}", SYNC) for i in range(len(sizes))]
    log = []
    try:
        for c in conns:
            await c.open()
        for c, s in zip(conns, sizes):
            await c.set_spawn_size(s)
        await asyncio.gather(*(c.act(0, 0) for c in conns))
        for k in range(steps):
            await asyncio.gather(*(c.next_step() for c in conns))
            obs = []
            for c in conns:
                sn = c.snapshot()
                obs.append(Obs([(m.x, m.y, m.size) for m in sn.mine],
                               [(o.kind, o.x, o.y, o.size) for o in sn.others if with_food or o.kind != "food"]))
            log.append(obs)
            acts = policy(k, obs)
            await asyncio.gather(*(c.act(*a) if o.mine else c.act(0, 0) for c, a, o in zip(conns, acts, obs)))
    finally:
        for c in conns:
            try:
                await c.close()
            except Exception:
                pass
        proc.terminate()
        proc.wait(timeout=10)
    return log


def run_real(*a, **k):
    return asyncio.run(real_session(*a, **k))


# ---------------------------------------------------------------------------- simulator
def sim_session(initial, policy, steps, viruses=(), graphs=False):
    """initial: per agent [(x, y, size)] cells (age 2 ticks, like the first real observation)."""
    n = len(initial)
    w = World(1, n, device=DEV, prefill=False, use_graphs=graphs, max_eject=64)
    w.clear(0)
    w.food_min = 0
    w.virus_min = 0
    for p, cells in enumerate(initial):
        for x, y, s in cells:
            w.put_cell(0, p, x, y, s * s / 100, age=2.0)
        if cells:
            w.mx[0, p], w.my[0, p] = cells[0][0], cells[0][1]
    for x, y, m in viruses:
        w.put_virus(0, x, y, m)
    log = []
    for k in range(steps):
        obs = []
        al = w.calive[0].cpu().numpy()
        cx, cy, r2 = w.cx[0].cpu().numpy(), w.cy[0].cpu().numpy(), w.cr2[0].cpu().numpy()
        ea = w.ealive[0].cpu().numpy()
        ex, ey = w.ex[0].cpu().numpy(), w.ey[0].cpu().numpy()
        va = w.valive[0].cpu().numpy()
        vx, vy, vr = w.vx[0].cpu().numpy(), w.vy[0].cpu().numpy(), w.vr2[0].cpu().numpy()
        for p in range(n):
            mine = [(float(int(cx[i])), float(int(cy[i])), float(int(math.sqrt(r2[i]))))
                    for i in range(p * w.K, (p + 1) * w.K) if al[i]]
            others = [("ejected", float(int(ex[j])), float(int(ey[j])), 36.0) for j in range(w.E) if ea[j]]
            others += [("virus", float(int(vx[j])), float(int(vy[j])), float(int(math.sqrt(vr[j])))) for j in range(w.V) if va[j]]
            obs.append(Obs(mine, others))
        log.append(obs)
        acts = policy(k, obs)
        mouse = torch.tensor([[[a[0], a[1]] for a in acts]], device=w.dev, dtype=torch.float32)
        split = torch.tensor([[bool(a[2]) for a in acts]], device=w.dev)
        eject = torch.tensor([[bool(a[3]) for a in acts]], device=w.dev)
        act = torch.tensor([[bool(o.mine) for o in obs]], device=w.dev)
        w.step(mouse, split, eject, act)
    return log


# ---------------------------------------------------------------------------- experiments
def toward_centre(pos, dist=5000.0):
    r = math.hypot(*pos)
    ux, uy = (-pos[0] / r, -pos[1] / r) if r > 200 else (1.0, 0.0)
    return pos[0] + ux * dist, pos[1] + uy * dist


def exp_speed(sizes, quick):
    """Mean per-step displacement of a lone cell chasing a far mouse (steps 2..6)."""
    rows = []
    for size in sizes:
        steps = 8

        def pol(k, obs):
            x, y = obs[0].centre
            tx, ty = toward_centre((x, y))
            return [(tx, ty, False, False)]
        real = run_real([size], pol, steps)
        p0 = real[0][0].mine[0][:2]
        sim = sim_session([[(p0[0], p0[1], size)]], pol, steps)

        def speed(log):
            c = [o[0].centre for o in log]
            return float(np.mean([math.hypot(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]) for i in range(2, 6)]))
        rows.append((f"speed, size {size}", "units/step", speed(real), speed(sim)))
    return rows


def split_policy_factory():
    state = {}

    def pol(k, obs):
        if not obs[0].mine:
            return [(0, 0, False, False)]
        if "t" not in state:
            state["t"] = toward_centre(obs[0].centre)
        tx, ty = state["t"]
        return [(tx, ty, k == 2, False)]
    return pol


def exp_split(size, quick):
    pol_r, pol_s = split_policy_factory(), split_policy_factory()
    steps = 30
    real = run_real([size], pol_r, steps)
    p0 = real[0][0].mine[0][:2]
    sim = sim_session([[(p0[0], p0[1], size)]], pol_s, steps)
    rows = []
    for after in (1, 2, 5, 10, 25):
        def sep(log):
            cs = log[2 + after][0].mine
            return math.hypot(cs[0][0] - cs[1][0], cs[0][1] - cs[1][1]) if len(cs) == 2 else float("nan")
        rows.append((f"split, size {size}", f"separation {after} steps after", sep(real), sep(sim)))
    def masses(log):
        cs = log[3][0].mine
        return sorted(s * s / 100 for _, _, s in cs)
    rows.append((f"split, size {size}", "piece masses right after", str([round(m) for m in masses(real)]),
                 str([round(m) for m in masses(sim)])))
    return rows


def exp_merge(size, quick):
    def factory():
        st = {}

        def pol(k, obs):
            if not obs[0].mine:
                return [(0, 0, False, False)]
            st.setdefault("c", obs[0].centre)
            return [(st["c"][0], st["c"][1], k == 2, False)]
        return pol
    child = size / math.sqrt(2)
    need = max(30.0, 0.2 * child) * 25 / SYNC            # steps for the new cell to be ready (size at split)
    steps = int(need * 1.12) + 40
    real = run_real([size], factory(), steps)
    p0 = real[0][0].mine[0][:2]
    sim = sim_session([[(p0[0], p0[1], size)]], factory(), steps)

    def merge_step(log):
        for k in range(4, len(log)):
            if len(log[k][0].mine) == 1:
                return k - 3                  # split is applied during step index 3
        return float("nan")
    return [(f"merge, size {size}", f"steps from split to merge (formula {need:.0f})", merge_step(real), merge_step(sim))]


def exp_decay(size, quick):
    steps = 400 if quick else 800

    def pol(k, obs):
        c = obs[0].centre if obs[0].mine else (0, 0)
        return [(c[0], c[1], False, False)]
    real = run_real([size], pol, steps)
    p0 = real[0][0].mine[0][:2]
    sim = sim_session([[(p0[0], p0[1], size)]], pol, steps)

    def rate(log):
        m = np.array([o[0].mass for o in log if o[0].mine])
        t = np.arange(len(m)) * SYNC / 25.0
        slope = np.polyfit(t, np.log(m), 1)[0]
        return 100 * (1 - math.exp(slope))      # % mass lost per second
    return [(f"decay, size {size}", "% mass per second (0.2 expected)", rate(real), rate(sim))]


def exp_eat(quick):
    rows = []
    cases = [(86, 66), (86, 70), (86, 74), (86, 78), (88, 60), (86, 60)]
    for sb, dt in cases:
        steps = 330 if quick else 380

        def factory():
            st = {}

            def pol(k, obs):
                if not obs[0].mine:
                    return [(0, 0, False, False), (0, 0, False, False)]
                if "tgt" not in st and obs[1].mine:
                    a, b = obs[0].centre, obs[1].centre
                    d = math.hypot(a[0] - b[0], a[1] - b[1]) or 1.0
                    st["tgt"] = (b[0] + (a[0] - b[0]) / d * dt, b[1] + (a[1] - b[1]) / d * dt)
                tgt = st.get("tgt", obs[0].centre)
                bc = obs[1].centre if obs[1].mine else tgt
                return [(tgt[0], tgt[1], False, False), (bc[0], bc[1], False, False)]
            return pol
        real = run_real([100, sb], factory(), steps, seed=3)
        init = [[(x, y, s) for x, y, s in real[0][i].mine] for i in range(2)]
        sim = sim_session(init, factory(), steps)

        def outcome(log):
            for k, o in enumerate(log):
                if o[0].mine and not o[1].mine:
                    return f"eaten (step {k}, mass {o[0].mass:.0f})"
            return "survives"
        thr = 100 - sb / 3
        ratio = 100 / sb
        rows.append((f"eat, prey size {sb} (ratio {ratio:.3f})", f"A holds at d={dt} (overlap limit {thr:.1f})",
                     outcome(real), outcome(sim)))
    return rows


def exp_eject(quick):
    size = 100

    def factory():
        st = {}

        def pol(k, obs):
            if not obs[0].mine:
                return [(0, 0, False, False)]
            st.setdefault("t", toward_centre(obs[0].centre))
            return [(st["t"][0], st["t"][1], False, k == 2)]
        return pol
    steps = 24
    real = run_real([size], factory(), steps)
    p0 = real[0][0].mine[0][:2]
    sim = sim_session([[(p0[0], p0[1], size)]], factory(), steps)
    rows = []

    def mass_after(log):
        return log[4][0].mass
    rows.append(("eject, mass 100", "mass after one eject (83 expected)", mass_after(real), mass_after(sim)))

    def travel(log, k):
        first = [o for o in log[4][0].others if o[0] == "ejected"]
        later = [o for o in log[4 + k][0].others if o[0] == "ejected"]
        if not first or not later:
            return float("nan")
        return math.hypot(later[0][1] - first[0][1], later[0][2] - first[0][2])
    for k in (1, 3, 8, 15):
        rows.append(("eject, mass 100", f"blob travel {k} steps after first sight", travel(real, k), travel(sim, k)))
    return rows


def exp_view(quick):
    """Is the client's visible set what the sim assumes? Max |offset| / half-extent of everything
    the real server sent (1.0 = exactly at the view-box edge; >1 + size/half would mean a bigger box)."""
    rows = []
    for size in (32, 150, 400):
        waypoints = [(2000, 2000), (-2000, 2000), (-2000, -2000), (2000, -2000)]

        def pol(k, obs):
            if not obs[0].mine:
                return [(0, 0, False, False)]
            wp = waypoints[(k // 60) % 4]
            return [(wp[0], wp[1], False, False)]
        log = run_real([size], pol, 160 if quick else 260, seed=5, bots=8, food=1500, with_food=True)
        rx, ry, n = 0.0, 0.0, 0
        for o in (l[0] for l in log):
            if not o.mine:
                continue
            tot = sum(s for _, _, s in o.mine)
            sc = max(min(64 / tot, 1.0) ** 0.4, 0.15)
            cx = sum(x for x, _, _ in o.mine) / len(o.mine)
            cy = sum(y for _, y, _ in o.mine) / len(o.mine)
            hx, hy = 960 / sc, 540 / sc
            for kind, x, y, s in o.others:
                rx = max(rx, (abs(x - cx) - s) / hx)
                ry = max(ry, (abs(y - cy) - s) / hy)
                n += 1
        rows.append((f"view box, spawn size {size}", f"max (|dx| - size)/(960/s) and (|dy| - size)/(540/s), {n} objects",
                     f"{rx:.3f} / {ry:.3f}", "<= 1 expected"))
    return rows


def exp_virus(quick):
    """A mass-500 cell looks for a virus and runs into it; compare the pop."""
    mass, size = 500, int(math.sqrt(500 * 100))
    waypoints = [(2500, 2500), (-2500, 2500), (-2500, -2500), (2500, -2500), (0, 0)]

    def pol(k, obs):
        o = obs[0]
        if not o.mine:
            return [(0, 0, False, False)]
        if len(o.mine) > 1:
            return [(*o.centre, False, False)]
        vs = [v for v in o.others if v[0] == "virus"]
        c = o.centre
        if vs:
            v = min(vs, key=lambda v: math.hypot(v[1] - c[0], v[2] - c[1]))
            return [(v[1], v[2], False, False)]
        wp = waypoints[(k // 120) % len(waypoints)]
        return [(wp[0], wp[1], False, False)]
    real = run_real([size], pol, 700 if quick else 900, seed=7)
    k_pop = next((k for k, o in enumerate(real) if len(o[0].mine) > 1), None)
    if k_pop is None:
        return [("virus pop", "pieces / mass after pop", "no virus reached", "-")]
    before = real[k_pop - 1][0]
    after = real[k_pop][0]
    # sim: same cell eating a 100-mass virus right next to it
    c = before.centre
    sim = sim_session([[(c[0], c[1], before.mine[0][2])]], lambda k, obs: [(c[0], c[1], False, False)], 3,
                      viruses=[(c[0], c[1], 100.0)])
    simo = sim[2][0]
    fmt = lambda o: f"{len(o.mine)} pieces, {o.mass:.0f} mass, sizes {sorted(int(s) for _, _, s in o.mine)[::-1][:6]}"
    return [("virus pop", f"cell mass {before.mass:.0f} eats virus", fmt(after), fmt(simo))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    exps = {
        "speed": lambda: exp_speed((32, 60, 100, 200, 400, 700), args.quick),
        "split": lambda: exp_split(100, args.quick) + exp_split(400, args.quick),
        "merge": lambda: exp_merge(100, args.quick) + ([] if args.quick else exp_merge(400, args.quick)),
        "decay": lambda: exp_decay(316, args.quick),
        "eat": lambda: exp_eat(args.quick),
        "eject": lambda: exp_eject(args.quick),
        "virus": lambda: exp_virus(args.quick),
        "view": lambda: exp_view(args.quick),
    }
    rows = []
    for name, fn in exps.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}] ...", file=sys.stderr, flush=True)
        rows += fn()
    print()
    print(f"{'experiment':34} {'quantity':56} {'real server':>22} {'simulator':>22}")
    for e, q, r, s in rows:
        f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"{e:34} {q:56} {f(r):>22} {f(s):>22}")


if __name__ == "__main__":
    main()
