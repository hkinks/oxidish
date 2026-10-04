"""Behaviour cloning: imitate the latest evolved champion, to start RL from a sane policy.

    uv run python -m rl.bc --samples 300000 --out runs/rl/bc.pt

Collects (observation, teacher action) pairs from arenas where the champion drives
the learner slots, plus discounted returns for the value head, then trains the
Policy with cross-entropy (+ value regression).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import ROOT, env_cfgs, latest_champion, save
from .model import Policy
from .vec import VecEnv


def collect(args, teacher):
    env = VecEnv(env_cfgs(args.workers, args.n_learn, args.n_opp, args.episode_ticks,
                          args.seed, teacher=teacher))
    g, s, r, d, v, lab, _ = env.reset()
    N = len(v)
    grids, scals, labels, rews, dones, valids = [], [], [], [], [], []
    count, t0 = 0, time.time()
    try:
        while count < args.samples:
            grids.append((g * 255).astype(np.uint8)); scals.append(s); labels.append(lab)
            valids.append(v)
            g, s, r, d, v, lab, _ = env.step(np.zeros((N, 3), np.int64))
            rews.append(r); dones.append(d)
            count += int(valids[-1].sum())
            if len(grids) % 200 == 0:
                print(f"  collected {count}/{args.samples} ({count / (time.time() - t0):.0f}/s)",
                      flush=True)
    finally:
        env.close()
    # Discounted reward-to-go per slot (value targets); a done or invalid step cuts it.
    T = len(rews)
    ret = np.zeros((T, N), np.float32)
    acc = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        acc = rews[t] + args.gamma * acc * (~dones[t])
        ret[t] = acc
    valid = np.stack(valids)
    pick = valid.reshape(-1)
    return (np.stack(grids).reshape(T * N, *grids[0].shape[1:])[pick],
            np.stack(scals).reshape(T * N, -1)[pick],
            np.stack(labels).reshape(T * N, 3)[pick],
            ret.reshape(-1)[pick])


def train(args, data):
    grids, scals, labels, rets = data
    dev = "cuda"
    model = Policy().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(labels)
    idx = np.random.permutation(n)
    val, tr = idx[: n // 20], idx[n // 20:]
    # Split and eject are rare: weight their positive class up.
    w_split = torch.tensor([1.0, min(20.0, (labels[:, 1] == 0).sum() / max((labels[:, 1] == 1).sum(), 1))], device=dev)
    w_eject = torch.tensor([1.0, min(20.0, (labels[:, 2] == 0).sum() / max((labels[:, 2] == 1).sum(), 1))], device=dev)

    def batch(ix):
        return (torch.from_numpy(grids[ix]).to(dev).float() / 255, torch.from_numpy(scals[ix]).to(dev),
                torch.from_numpy(labels[ix]).to(dev), torch.from_numpy(rets[ix]).to(dev))

    for epoch in range(args.epochs):
        np.random.shuffle(tr)
        model.train()
        for k in range(0, len(tr), args.batch):
            g, s, lab, ret = batch(tr[k:k + args.batch])
            logits, value = model(g, s)
            loss = (F.cross_entropy(logits[0], lab[:, 0])
                    + F.cross_entropy(logits[1], lab[:, 1], weight=w_split)
                    + F.cross_entropy(logits[2], lab[:, 2], weight=w_eject)
                    + 0.5 * F.mse_loss(value, ret))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            g, s, lab, ret = batch(val)
            logits, value = model(g, s)
            acc = (logits[0].argmax(-1) == lab[:, 0]).float().mean().item()
            pred = logits[0].argmax(-1)
            diff = (pred - lab[:, 0]).abs()
            circ = torch.minimum(diff, 16 - diff)  # headings wrap around; 16 = stay
            near = torch.where((pred == 16) | (lab[:, 0] == 16), diff == 0, circ <= 1).float().mean().item()
            vloss = F.mse_loss(value, ret).item()
        print(f"epoch {epoch + 1}: move acc {acc:.3f} (within 1 dir {near:.3f}) value mse {vloss:.4f}",
              flush=True)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300_000)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--n-learn", type=int, default=4)
    p.add_argument("--n-opp", type=int, default=3)
    p.add_argument("--episode-ticks", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "runs" / "rl" / "bc.pt"))
    p.add_argument("--reuse", action="store_true", help="reuse the cached dataset")
    args = p.parse_args()
    teacher = latest_champion()
    cache = Path(args.out).with_suffix(".data.npz")
    if args.reuse and cache.exists():
        z = np.load(cache)
        data = (z["grids"], z["scals"], z["labels"], z["rets"])
        print(f"loaded cached dataset {cache}", flush=True)
    else:
        print(f"teacher: latest champion; collecting {args.samples} samples", flush=True)
        data = collect(args, teacher)
        np.savez(cache, grids=data[0], scals=data[1], labels=data[2], rets=data[3])
    print(f"dataset: {len(data[2])} samples; split rate {data[2][:, 1].mean():.4f} "
          f"eject rate {data[2][:, 2].mean():.4f}", flush=True)
    model = train(args, data)
    save(model, Path(args.out), teacher=teacher, kind="bc")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
