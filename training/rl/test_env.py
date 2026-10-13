"""Smoke tests for the oxidish_gym VecEnv (build: see docs/vision-bot.md, M1). Run from the repo root."""
import numpy as np
import pytest

oxidish_gym = pytest.importorskip("oxidish_gym")

PX = 96


def make(n_worlds=2, learners=2, threads=2, **kw):
    return oxidish_gym.VecEnv(
        n_worlds, learners, opponents=[("c34", "bots/specs/c3_4.json", 2)], teacher="bots/specs/c3_4.json",
        px=PX, threads=threads, seed=7, **kw)


def test_shapes_dtypes_and_info():
    env = make()
    n = env.n
    assert n == 4
    obs = env.reset()
    assert obs["view"].shape == (n, 2, PX, PX) and obs["view"].dtype == np.uint8
    assert obs["wide"].shape == (n, 2, PX // 2, PX // 2) and obs["wide"].dtype == np.uint8
    obs, reward, done, info = env.step(np.full((n, 4), [PX / 2, PX / 2, 0, 0], np.float32))
    assert reward.shape == (n,) and reward.dtype == np.float32
    assert done.shape == (n,) and done.dtype == np.uint8
    assert info["teacher"].shape == (n, 4) and info["teacher"].dtype == np.float32
    for k in ("mass", "rank", "leader", "ncells"):
        assert info[k].shape == (n,)
    assert (info["teacher"][:, :2] >= 0).all() and (info["teacher"][:, :2] <= PX).all()
    assert (obs["view"] > 0).any()
    with pytest.raises(ValueError):
        env.step(np.zeros((n, 3), np.float32))


def test_previous_frame_is_stacked():
    env = make(n_worlds=1, learners=1, threads=1)
    first = env.reset(copy=True)
    assert (first["view"][:, 0] == first["view"][:, 1]).all()  # no history yet
    now = env.step(np.zeros((1, 4), np.float32), copy=True)[0]
    assert (now["view"][:, 1] == first["view"][:, 0]).all()  # channel 1 = previous decision's frame


def test_teacher_driven_learners_grow():
    env = make(n_worlds=4, learners=2, threads=4)
    env.set_teacher(True)
    env.reset()
    zero = np.zeros((env.n, 4), np.float32)
    m0 = peak = None
    for t in range(1500):  # 1500 decisions x 2 ticks x 40 ms = 2 game minutes
        _, _, _, info = env.step(zero)
        if t == 0:
            m0 = info["mass"].copy()
            peak = m0.copy()
        peak = np.maximum(peak, info["mass"])
    assert peak.mean() > 3 * m0.mean(), (m0, peak)
    assert (peak > m0).mean() >= 0.75


def test_student_actions_are_applied_and_dones_respawn():
    env = make(n_worlds=1, learners=1, threads=1)
    env.reset()
    t = np.array([[PX - 1, PX / 2, 0, 0]], np.float32)
    for _ in range(50):
        obs, r, d, info = env.step(t)
    assert info["mass"][0] > 0 and info["ncells"][0] >= 1
