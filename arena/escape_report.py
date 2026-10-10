"""Markdown tables from `ogar escape --out` files (docs/escape-drill.md).

    python -m arena.escape_report runs/esc/a.json runs/esc/b.json [--base c3_4] [--what pooled|cells|methods]

`pooled`: per agent over all conditions of the files: survival at 30 / 60 s (Wilson), survival time (mean of min(death, 60 s)),
the paired difference to `--base` (same condition and seed, mean +- 1.96 se), mass kept, time inside the chaser's split reach.
`cells`: survival at 60 s per condition and agent. `methods`: what the survivors did and where the others died.
Files are merged (the same condition and agent in several files are one cell).
"""

import argparse
import json
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mean_ci(x):
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(x) / n
    if n < 2:
        return m, float("nan")
    v = sum((a - m) ** 2 for a in x) / (n - 1)
    return m, 1.96 * (v / n) ** 0.5


def surv_t(r, horizon=60.0):
    return min(r["died_t"] if r.get("died_t") is not None else horizon, horizon)


def load(paths):
    rows = []
    for p in paths:
        rows += json.load(open(p))["reps"]
    return rows


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


def pooled(rows, base):
    by = defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    key = lambda r: (r["cond"], r["seed"])
    bt = {key(r): surv_t(r) for r in by.get(base, [])}
    print("| agent | n | survive 30 s | survive 60 s | survival time s | paired dt vs " + base + " s | mass kept 60 s | in split reach s | touched wall |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a, rs in by.items():
        n = len(rs)
        s30 = sum(1 for r in rs if r["alive30"])
        s60 = sum(1 for r in rs if r["alive60"])
        t = [surv_t(r) for r in rs]
        d = [surv_t(r) - bt[key(r)] for r in rs if key(r) in bt]
        tm, tc = mean_ci(t)
        dm, dc = mean_ci(d)
        mm, mc = mean_ci([r["mass60"] / r["m0"] for r in rs])
        rm, rc = mean_ci([r["t_in_reach"] for r in rs])
        wl = sum(1 for r in rs if r["touched_wall"])
        dtxt = "-" if a == base or not d else f"{dm:+.1f}+-{dc:.1f}"
        print(f"| {a} | {n} | {rate(s30, n)} | {rate(s60, n)} | {tm:.1f}+-{tc:.1f} | {dtxt} | {mm:.2f}+-{mc:.2f} | {rm:.1f}+-{rc:.1f} | {100 * wl / n:.0f}% |")


def cells(rows):
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    conds = list(dict.fromkeys(r["cond"] for r in rows))
    print("| condition | n | " + " | ".join(agents) + " |")
    print("|---|---|" + "---|" * len(agents))
    for c in conds:
        cells_ = []
        n = 0
        for a in agents:
            rs = [r for r in rows if r["cond"] == c and r["agent"] == a]
            n = max(n, len(rs))
            cells_.append(f"{100 * sum(1 for r in rs if r['alive60']) / max(len(rs), 1):.0f}% / {sum(surv_t(r) for r in rs) / max(len(rs), 1):.0f}s")
        print(f"| {c} | {n} | " + " | ".join(cells_) + " |")


def methods(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    print("| agent | survivors by method | deaths by place |")
    print("|---|---|---|")
    for a, rs in by.items():
        m = Counter(r["method"] for r in rs if r["alive_end"])
        w = Counter(r["died_where"] for r in rs if not r["alive_end"])
        print(f"| {a} | " + ", ".join(f"{k} {v}" for k, v in m.most_common()) + " | " + ", ".join(f"{k} {v}" for k, v in w.most_common()) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="c3_4")
    ap.add_argument("--what", default="pooled", choices=["pooled", "cells", "methods"])
    a = ap.parse_args()
    rows = load(a.files)
    {"pooled": lambda: pooled(rows, a.base), "cells": lambda: cells(rows), "methods": lambda: methods(rows)}[a.what]()


if __name__ == "__main__":
    main()
