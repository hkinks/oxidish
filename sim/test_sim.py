"""Tests for the GPU simulator: deterministic mechanics checks (each run eagerly and under CUDA
graphs), the virus-split port against a literal port of the JS, and the observation encoder
against the real Python encoder (rl/guided.py) on live sim states.

    uv run python -m sim.test_sim            # or: pytest sim/test_sim.py
"""

import math
import sys
import traceback

import numpy as np
import torch

from . import engine as E
from .engine import World, virus_splits

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPEED = lambda size: 2.2 * size ** -0.439 * 40


def make(P=2, graphs=False, B=1):
    w = World(B, P, device=DEV, prefill=False, seed=1, use_graphs=graphs, max_eject=32)
    for b in range(B):
        w.clear(b)
    w.food_min = 0       # no spawning in unit tests
    w.virus_min = 0
    return w


def aim(w, b, p, x, y):
    w.mx[b, p], w.my[b, p] = x, y


def run_ticks(w, n):
    """n physics ticks without inputs (steps of 2; n must be even)."""
    B, P = w.B, w.P
    z = torch.zeros(B, P, dtype=torch.bool, device=w.dev)
    for _ in range(n // 2):
        w.step(torch.stack([w.mx, w.my], -1), z, z, z)


def one_step(w, p=0, split=False, eject=False, mouse=None):
    B, P = w.B, w.P
    z = torch.zeros(B, P, dtype=torch.bool, device=w.dev)
    s, e, a = z.clone(), z.clone(), z.clone()
    s[:, p], e[:, p], a[:, p] = split, eject, True
    m = torch.stack([w.mx, w.my], -1) if mouse is None else torch.tensor(mouse, device=w.dev).expand(B, P, 2).float()
    w.step(m, s, e, a)


def cells(w, b=0, p=None):
    al = w.calive[b].cpu().numpy()
    idx = [i for i in range(w.C) if al[i] and (p is None or i // w.K == p)]
    return [(i, float(w.cx[b, i]), float(w.cy[b, i]), float(w.cr2[b, i]) / 100) for i in idx]


def both(fn):
    def run():
        for g in ([False, True] if DEV == "cuda" else [False]):
            fn(g)
    run.__name__ = fn.__name__
    return run


@both
def test_speed_vs_size(graphs):
    for mass in (10, 100, 1000):
        w = make(graphs=graphs)
        w.put_cell(0, 0, 0.0, 0.0, mass)
        aim(w, 0, 0, 3000.0, 0.0)
        run_ticks(w, 2)
        x = cells(w)[0][1]
        want = 2 * SPEED(math.sqrt(mass * 100))
        assert abs(x - want) < 1e-2 * want, (mass, x, want)


@both
def test_steer_slows_near_mouse(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 10)
    aim(w, 0, 0, 16.0, 0.0)  # within 32: speed scales with distance / 32
    run_ticks(w, 2)
    s0 = SPEED(math.sqrt(1000))
    x1 = s0 * 16 / 32
    x2 = x1 + s0 * math.trunc(16 - x1) / 32   # the server truncates the mouse offset (~~)
    assert abs(cells(w)[0][1] - x2) < 0.05, (cells(w)[0][1], x2)


@both
def test_eat_ratio_and_overlap(graphs):
    # eater mass 100 (size 100); prey size 77.46 (ratio 1.29 >= 1.15): eaten only when d < 100 - 77.46/3
    for d, mass_b, eaten in [(70, 60, True), (80, 60, False), (70, 81, False)]:
        w = make(graphs=graphs)
        w.put_cell(0, 0, 0.0, 0.0, 100)
        w.put_cell(0, 1, float(d), 0.0, mass_b)
        aim(w, 0, 0, 0.0, 0.0)
        aim(w, 0, 1, float(d), 0.0)
        run_ticks(w, 2)
        got = len(cells(w, p=1)) == 0
        assert got == eaten, (d, mass_b, got, cells(w))
        if eaten:
            assert abs(sum(c[3] for c in cells(w, p=0)) - (100 + mass_b)) < 0.5


@both
def test_food_and_decay(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 1000)
    w.put_food(0, 5.0, 0.0)
    aim(w, 0, 0, 0.0, 0.0)
    run_ticks(w, 2)
    assert abs(cells(w)[0][3] - 1001) < 0.5 and not w.falive.any()
    m0 = cells(w)[0][3]
    n_decays = 0
    t0 = w.t
    run_ticks(w, 50)
    n_decays = sum(1 for t in range(t0, w.t) if (t + 3) % 25 == 0)
    want = m0 * (1 - E.DECAY_RATE) ** n_decays
    assert abs(cells(w)[0][3] - want) < 0.02 * 1.0, (cells(w)[0][3], want, n_decays)


@both
def test_split_distance_and_halving(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, -2000.0, 0.0, 100)
    aim(w, 0, 0, 1000.0, 0.0)
    one_step(w, split=True, mouse=(1000.0, 0.0))
    cs = cells(w)
    assert len(cs) == 2 and all(abs(c[3] - 50) < 0.2 for c in cs), cs
    bd0 = 780 * math.sqrt(5000) ** 0.0122
    off = abs(cs[0][1] - cs[1][1])
    assert abs(off - 0.1 * bd0) < 0.5, (off, 0.1 * bd0)       # first boost tick moves bd/10
    run_ticks(w, 60)
    cs = cells(w)
    far = abs(cs[0][1] - cs[1][1])
    assert abs(far - bd0 * (1 - 0.9 ** 61)) < 5, (far, bd0)   # ~820 units apart
    assert min(c[2] for c in cs) == max(c[2] for c in cs) == 0.0


@both
def test_split_uses_previous_mouse(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 100)
    aim(w, 0, 0, 1000.0, 0.0)                       # old mouse: +x
    one_step(w, split=True, mouse=(0.0, 1000.0))    # new mouse: +y -> the split still goes +x
    cs = sorted(cells(w), key=lambda c: c[1])
    assert abs(cs[1][1] - cs[0][1]) > 50 and abs(cs[1][2] - cs[0][2]) < 30


@both
def test_merge_timer_and_rigid_push(graphs):
    for age, merged in [(700, False), (760, True)]:
        w = make(graphs=graphs)
        w.put_cell(0, 0, 0.0, 0.0, 50, age=age)
        w.put_cell(0, 0, 10.0, 0.0, 50, age=age)
        aim(w, 0, 0, 5.0, 0.0)
        run_ticks(w, 4)
        cs = cells(w)
        if merged:
            assert len(cs) == 1 and abs(cs[0][3] - 100) < 0.5, cs
        else:
            assert len(cs) == 2
            d = abs(cs[0][1] - cs[1][1])
            assert d > 2 * math.sqrt(5000) - 10, d            # pushed apart to touching


@both
def test_virus_pop(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 500)
    w.put_virus(0, 0.0, 0.0, 100)
    aim(w, 0, 0, 0.0, 0.0)
    run_ticks(w, 2)
    cs = cells(w, p=0)
    assert len(cs) == 16, len(cs)
    assert abs(sum(c[3] for c in cs) - 600) < 1.0
    assert not w.valive.any()


@both
def test_virus_too_big_to_eat(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 110)             # size 104.9 < 1.1576 * 100
    w.put_virus(0, 0.0, 0.0, 100)
    run_ticks(w, 2)
    assert w.valive.any() and len(cells(w)) == 1


@both
def test_eject(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 0.0, 0.0, 100)
    aim(w, 0, 0, 1000.0, 0.0)
    one_step(w, eject=True, mouse=(1000.0, 0.0))
    assert abs(cells(w)[0][3] - (100 - E.EJECT_LOSS_R2 / 100)) < 0.1
    assert int(w.ealive.sum()) == 1
    j = int(w.ealive[0].nonzero()[0])
    assert float(w.ex[0, j]) > 10 and abs(float(w.ey[0, j])) < 0.31 * 600   # within the +-0.3 rad cone
    # a second cell in the path eats the blob (+13 mass)
    w2 = make(graphs=graphs)
    w2.put_cell(0, 0, 0.0, 0.0, 100)
    w2.put_cell(0, 1, 300.0, 0.0, 100)
    aim(w2, 0, 0, 1000.0, 0.0)
    aim(w2, 0, 1, 300.0, 0.0)
    one_step(w2, eject=True, mouse=(0.0, 0.0))     # the shot uses the previous mouse; then stand still
    run_ticks(w2, 30)
    assert int(w2.ealive.sum()) == 0
    assert abs(sum(c[3] for c in cells(w2, p=1)) - (100 + E.EJECT_R2 / 100)) < 0.5, cells(w2)


@both
def test_virus_feed_shoots(graphs):
    w = make(graphs=graphs)
    w.put_virus(0, 0.0, 0.0, 100)
    for i in range(8):                       # each blob adds 13 mass: 100 -> 204 > 200 on the 8th
        w.put_eject(0, -60.0, 0.0, bd=0.0, ang=math.pi / 2, age=5)
        run_ticks(w, 2)
    assert int(w.valive.sum()) == 2, int(w.valive.sum())
    assert abs(float(w.vr2[w.valive].min()) - E.VIRUS_MIN_R2) < 1.0
    run_ticks(w, 4)
    xs = sorted(w.vx[0][w.valive[0]].tolist())
    assert abs(xs[1] - xs[0]) > 100                  # the shot virus flies off


@both
def test_border_clamp(graphs):
    w = make(graphs=graphs)
    w.put_cell(0, 0, 3400.0, 0.0, 100)
    aim(w, 0, 0, 9000.0, 0.0)
    run_ticks(w, 40)
    x = cells(w)[0][1]
    size = 100.0
    assert 3500 - size / 2 - 0.5 <= x <= 3500 - size / 2 + SPEED(size) + 0.5, x
    w.put_cell(0, 1, -3499.0, 3499.0, 100)
    aim(w, 0, 1, -9000.0, 9000.0)
    run_ticks(w, 10)
    c = cells(w, p=1)[0]
    assert c[1] >= -3500 + 50 - 40 and c[2] <= 3500 - 50 + 40


@both
def test_virus_attribution(graphs):
    w = make(P=3, graphs=graphs)
    w.put_virus(0, 0.0, 0.0, 100)
    for i in range(8):                       # player 0's blobs feed the virus until it shoots
        w.put_eject(0, -60.0, 0.0, bd=0.0, ang=math.pi / 2, age=5, owner=0)
        run_ticks(w, 2)
    assert float(w.vstat[0, 0, 0]) == 1 and float(w.vstat[0, 0, 1]) == 0, w.vstat[:, 0]
    j = int(w.vshooter[0].argmax())
    assert int(w.vshooter[0, j]) == 0
    run_ticks(w, 40)
    xs = w.vx[0].clone()
    shot = [int(i) for i in w.valive[0].nonzero()[:, 0] if int(w.vshooter[0, i]) == 0][0]
    sx, sy = float(w.vx[0, shot]), float(w.vy[0, shot])
    # player 1 (big) explodes on the shot virus: player 0 gets a pop and the mass; the shooter eating its own does not
    w.put_cell(0, 1, sx, sy, 500)
    aim(w, 0, 1, sx, sy)
    run_ticks(w, 2)
    assert float(w.vstat[1, 0, 0]) == 1, w.vstat[:, 0]
    assert 500 <= float(w.vstat[2, 0, 0]) <= 700
    w.put_virus(0, 2000.0, 2000.0, 100)
    w.vshooter[0, int(w.valive[0].nonzero()[-1])] = 2
    w.put_cell(0, 2, 2000.0, 2000.0, 500)    # player 2 eats a virus it shot itself: no pop
    run_ticks(w, 2)
    assert float(w.vstat[1, 0, 2]) == 0


def test_virus_splits_matches_js():
    def js(M, L):
        splits = []
        cellsLeft = L
        if cellsLeft <= 0:
            return splits
        if M / cellsLeft < 36:
            sc = 2
            sm = M / sc
            while sm > 36 and 2 * sc < cellsLeft:
                sc *= 2
                sm = M / sc
            sm = M / (sc + 1)
            return [sm] * sc
        massLeft = M / 2
        sm = M / 2
        while True:
            t = cellsLeft > 0
            cellsLeft -= 1
            if not t:
                break
            if massLeft / cellsLeft < 36 if cellsLeft else False:
                sm = massLeft / cellsLeft
                while True:
                    t2 = cellsLeft > 0
                    cellsLeft -= 1
                    if not t2:
                        break
                    splits.append(sm)
            while sm >= massLeft and cellsLeft > 0:
                sm /= 2
            splits.append(sm)
            massLeft -= sm
        return splits
    rng = np.random.default_rng(0)
    Ms = np.concatenate([rng.uniform(40, 3000, 400), rng.uniform(1, 40, 100)]).astype(np.float32)
    Ls = rng.integers(1, 16, 500)
    m, c = virus_splits(torch.tensor(Ms, device=DEV), torch.tensor(Ls, device=DEV))
    m, c = m.cpu().numpy(), c.cpu().numpy()
    for i in range(500):
        ref = js(float(Ms[i]), int(Ls[i]))
        assert c[i] == len(ref), (Ms[i], Ls[i], c[i], len(ref))
        assert np.allclose(m[i, :c[i]], ref, rtol=1e-4), (Ms[i], Ls[i], m[i, :c[i]], ref)


@both
def test_random_world_invariants(graphs):
    """Many random steps: no NaNs, mass bounds, <= 16 cells."""
    from .env import SimEnv
    env = SimEnv(B=8, P=8, seed=3, curriculum_p=0.5)
    env.w.use_graphs = graphs
    env.reset()
    n = env.n_agents
    for _ in range(150):
        a = torch.stack([torch.randint(0, 17, (n,)), (torch.rand(n) < 0.1).long(), (torch.rand(n) < 0.1).long()], 1).to(env.dev)
        g, s, r, d, v = env.step(a)
        w = env.w
        assert torch.isfinite(g).all() and torch.isfinite(s).all() and torch.isfinite(r).all()
        assert (w.cr2[w.calive] >= 1000.0 - 1).all()
        assert (w.n_cells() <= 16).all()
        assert (w.cx[w.calive].abs() <= 3500 + 60).all()
        assert (g[~v].abs().sum() == 0)
    assert w.eject_overwrites >= 0


def test_obs_matches_python_encoder():
    try:
        from rl.guided import GuidedEncoder
    except Exception as ex:  # pragma: no cover
        print("   (skipped: cannot import rl.guided:", ex, ")")
        return
    from .compare_obs import compare
    res = compare(steps=60, seed=5)
    for name, frac in res.items():
        assert frac < 0.08, (name, frac)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print("PASS", t.__name__)
        except Exception:
            failed += 1
            print("FAIL", t.__name__)
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
