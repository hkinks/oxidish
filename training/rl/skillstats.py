"""Per-family skill counters from `ogar endless --out` / `training.eval.endless --out` json (key stats.agents), see docs/vision-bot.md M5.

    python training/rl/skillstats.py runs/vision/evals/M5_*_e*.json   (the sum over the files, one row per family)
"""
import collections
import json
import sys

KEYS = ["ticks", "deaths", "kills", "splits", "split_kills", "split_kill_mass", "harvest_kills", "harvest_mass",
        "virus_shots", "virus_pops", "virus_pop_mass"]


def family_counts(doc):
    ag = collections.defaultdict(collections.Counter)
    for world in doc.get("stats", {}).get("agents", []):
        for r in world:
            c = ag[r["policy"]]
            for k in KEYS:
                c[k] += r.get(k, 0)
            c["n"] += 1
            for k, v in r.get("counters", {}).items():
                if k.startswith("reflex"):
                    c[k] += v
    return ag


def per_hour(c, k):
    hours = c["ticks"] / (25 * 3600)
    return c[k] / hours if hours else 0.0


def main(paths):
    tot = collections.defaultdict(collections.Counter)
    for p in paths:
        for f, c in family_counts(json.load(open(p))).items():
            tot[f].update(c)
    print(f"{'family':<8}{'agent-h':>9}{'deaths/h':>10}{'splits/h':>10}{'splitK/h':>10}{'splitKmass/h':>13}{'pops/h':>8}"
          f"{'popmass/h':>10}{'harvK/h':>9}{'harvmass/h':>11}{'shots/h':>9}")
    for f, c in sorted(tot.items()):
        print(f"{f:<8}{c['ticks'] / (25 * 3600):9.0f}{per_hour(c, 'deaths'):10.2f}{per_hour(c, 'splits'):10.1f}"
              f"{per_hour(c, 'split_kills'):10.2f}{per_hour(c, 'split_kill_mass'):13.0f}{per_hour(c, 'virus_pops'):8.2f}"
              f"{per_hour(c, 'virus_pop_mass'):10.0f}{per_hour(c, 'harvest_kills'):9.2f}{per_hour(c, 'harvest_mass'):11.0f}"
              f"{per_hour(c, 'virus_shots'):9.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
