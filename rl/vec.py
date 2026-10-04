"""Run several ArenaEnv workers in subprocesses and step them together."""

import multiprocessing as mp

import numpy as np

from .env import worker


class VecEnv:
    def __init__(self, cfgs: list[dict]):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for cfg in cfgs:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=worker, args=(child, cfg), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.n_learn = cfgs[0]["n_learn"]

    def _gather(self, results):
        g, s, r, d, v, lab, fin = zip(*results)
        finished = [f for fs in fin for f in fs]
        return (np.concatenate(g), np.concatenate(s), np.concatenate(r), np.concatenate(d),
                np.concatenate(v), np.concatenate(lab), finished)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return self._gather([r.recv() for r in self.remotes])

    def step(self, acts: np.ndarray):
        k = self.n_learn
        for i, r in enumerate(self.remotes):
            r.send(("step", acts[i * k:(i + 1) * k]))
        return self._gather([r.recv() for r in self.remotes])

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
