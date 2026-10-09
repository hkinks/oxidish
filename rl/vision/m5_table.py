"""M5 summary: leader share of `vis` per variant and seed from logs/m5/<name>_<seed>.txt, mean, wins vs the base.

    python rl/vision/m5_table.py vision_p6 vision_p6_rs vision_p6_rh vision_p6_rv vision_p6_ra
The first name is the base. 'wins' = seeds where the variant is strictly above the base (needs >= 3 of 5 and a better mean to keep a skill).
"""
import re
import sys

SEEDS = [3000, 11000, 21000, 31000, 41000]


def share(name, seed):
    try:
        t = open(f"logs/m5/{name}_{seed}.txt", encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = re.search(r"vis LEADER SHARE \(#1 on the leaderboard\): (\d+)%", t)
    return int(m.group(1)) if m else None


def main(names):
    base = names[0]
    rows = {n: [share(n, s) for s in SEEDS] for n in names}
    print(f"{'spec':<16}" + "".join(f"{s:>7}" for s in SEEDS) + f"{'mean':>8}{'n':>3}{'wins':>6}{'d_mean':>8}")
    for n in names:
        r = rows[n]
        done = [x for x in r if x is not None]
        mean = sum(done) / len(done) if done else float("nan")
        paired = [(a, b) for a, b in zip(r, rows[base]) if a is not None and b is not None]
        wins = sum(a > b for a, b in paired)
        dm = sum(a - b for a, b in paired) / len(paired) if paired else float("nan")
        print(f"{n:<16}" + "".join(f"{('-' if x is None else x):>7}" for x in r) + f"{mean:8.1f}{len(done):3d}{wins:6d}{dm:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
