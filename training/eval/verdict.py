"""Does a candidate consistently beat every other bot? Play fresh-seed batches until it is decided.

    uv run python -m training.eval.verdict --candidate ramp_r2 --agents ramp_r2=bots/specs/r2g9_ramp800.json:1 \
        champ=bots/specs/final_champion.json:1 ... --out runs/verdict_ramp_r2.json

Every match holds the whole lineup (free-for-all, 8 MultiOgar bots, long games); a match counts as a
win against an opponent when the candidate's log-mass beats that opponent's in it (training.eval.tournament's
head-to-head). After each batch, per opponent, a sequential probability ratio test (as in chess-engine
testing) of H0: win rate p0 against H1: p1 decides PASS (beats it), FAIL, or keeps playing; the 95%
Wilson interval (training.eval.pooled's "consistently beats": lower end above 50%) is printed alongside. Stops
when every opponent is decided or at --max matches. --out is appended per batch, so a stopped run resumes
(new batches take the next unused seeds).
"""

import argparse
import json
import math
import os
import statistics

from .util import limit_cpus
from .pooled import wilson
from .tournament import play_many


def per_match(rows: list[dict]) -> dict[str, dict[int, float]]:
    """policy -> seed -> mean log-mass of its copies in that match."""
    by: dict[tuple, list[float]] = {}
    for r in rows:
        by.setdefault((r["policy"], r["seed"]), []).append(r["log_mass"])
    team: dict[str, dict[int, float]] = {}
    for (pol, seed), v in by.items():
        team.setdefault(pol, {})[seed] = statistics.fmean(v)
    return team


def sprt(k: int, n: int, p0: float, p1: float) -> float:
    """Log-likelihood ratio of H1 (win rate p1) over H0 (p0) after k wins in n matches."""
    return k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))


def status(rows: list[dict], cand: str, p0: float, p1: float, alpha: float, beta: float) -> dict[str, str]:
    up, down = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    team = per_match(rows)
    out = {}
    print(f"\n{cand} vs   wins / matches   win rate  [95% Wilson]   SPRT llr (bounds {down:.2f}, {up:.2f})")
    for opp in sorted(t for t in team if t != cand):
        common = team[cand].keys() & team[opp].keys()
        k, n = sum(team[cand][s] > team[opp][s] for s in common), len(common)
        llr = sprt(k, n, p0, p1)
        lo, hi = wilson(k, n)
        out[opp] = "PASS" if llr >= up else "FAIL" if llr <= down else "open"
        print(f"  {opp:<10} {k:4d} / {n:<4d}        {k / max(n, 1):5.0%}  [{lo:4.0%}-{hi:4.0%}]   {llr:6.2f}  {out[opp]}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--max", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ticks", type=int, default=22500, help="22500 = 15 game-min")
    p.add_argument("--food-mult", type=float, default=2.0)
    p.add_argument("--seed-base", type=int, default=200000)
    p.add_argument("--p0", type=float, default=0.5)
    p.add_argument("--p1", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--cpus", type=int, default=11, help="pin to this many CPUs (0 = all): stay under 90% CPU")
    a = p.parse_args()
    limit_cpus(a.cpus)
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    while True:
        n = len({r["seed"] for r in rows})
        st = status(rows, a.candidate, a.p0, a.p1, a.alpha, a.beta) if rows else {}
        if st and all(v != "open" for v in st.values()):
            break
        if n >= a.max:
            print(f"\nstopped at --max {a.max} matches with undecided opponents")
            break
        seed0 = max((r["seed"] for r in rows), default=a.seed_base - 1) + 1
        batch = min(a.batch, a.max - n)
        print(f"\nplaying matches {n + 1}-{n + batch} (seeds {seed0}-{seed0 + batch - 1})", flush=True)
        rows += play_many(a.agents, batch, a.workers, a.ticks, seed_base=seed0, food_mult=a.food_mult)
        tmp = a.out + ".tmp"     # atomic: a reader (or a crash) never sees a half-written file
        with open(tmp, "w") as f:
            json.dump(rows, f)
        os.replace(tmp, a.out)
    st = status(rows, a.candidate, a.p0, a.p1, a.alpha, a.beta)
    verdict = "CONSISTENTLY BEATS ALL" if all(v == "PASS" for v in st.values()) else "does not beat all"
    print(f"\nverdict: {a.candidate} {verdict} ({len({r['seed'] for r in rows})} matches)")


if __name__ == "__main__":
    main()
