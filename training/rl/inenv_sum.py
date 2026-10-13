"""Print leader/top3/mass/deaths per seed for eval_env jsonl files, and the mean."""
import json
import sys

for path in sys.argv[1:]:
    rs = [json.loads(l) for l in open(path)]
    for r in rs:
        print(path, r["seed"], round(r["leader_frac"], 3), round(r["top3_frac"], 3), round(r["mean_mass"]), round(r["deaths_per_hour"], 1))
    n = len(rs)
    print(path, "MEAN", round(sum(r["leader_frac"] for r in rs) / n, 3), round(sum(r["top3_frac"] for r in rs) / n, 3),
          round(sum(r["mean_mass"] for r in rs) / n), round(sum(r["deaths_per_hour"] for r in rs) / n, 1))
