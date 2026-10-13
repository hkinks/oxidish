"""Print the validation history of a run:  python training/rl/summ.py bc1"""
import json
import sys

for l in open(f"logs/vision/{sys.argv[1]}.jsonl"):
    d = json.loads(l)
    if d["kind"] == "train":
        continue
    s = d["split_best"]
    print(f"{d['kind']:5s} t={d['t']:5d}s samples={d['samples'] / 1e6:6.1f}M p_stu={d.get('p_student', 0):.2f} "
          f"agree15={d['target_agree15']:.3f} agree30={d['target_agree30']:.3f} med_px={d['median_px']:.1f} | "
          f"split thr={s['thr']:.2f} P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} "
          f"| events: R@thr={d.get('event_split_recall_at_best', float('nan')):.2f} agree15={d.get('event_target_agree15', float('nan')):.2f}")
