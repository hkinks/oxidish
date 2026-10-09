"""Why do the reflex variants die more? Breaks the `vis` family's recorded losses (AgentStats.losses: one entry per cell eaten
by another player) down by the state of the victim, per spec file set.

    python rl/vision/lossdiag.py vision_p6 vision_p6_rs vision_p6_rh vision_p6_rv     (reads runs/vision/evals/M5_<name>_e*.stats.json)
Columns: agent-hours, cells lost/h, mass lost/h, share of lost cells that were one of >=2 own cells (split state), mean
killer mass / own total, lost mass in cells smaller than 100 (food-sized pieces), deaths/h.
"""
import glob
import json
import sys


def main(names):
    print(f"{'spec':<16}{'agent-h':>8}{'deaths/h':>9}{'cells lost/h':>13}{'mass lost/h':>12}{'multi-cell %':>13}{'killer/own':>11}{'lost in 1-cell state/h':>24}")
    for n in names:
        ticks = deaths = 0
        lost = []
        for f in sorted(glob.glob(f"runs/vision/evals/M5_{n}_e*.stats.json")):
            for w in json.load(open(f))["stats"]["agents"]:
                for r in w:
                    if r["policy"] != "vis":
                        continue
                    ticks += r["ticks"]
                    deaths += r["deaths"]
                    lost += r["losses"]
        h = ticks / (25 * 3600)
        if not h:
            continue
        multi = [x for x in lost if x["my_cells"] >= 2]
        ratio = sum(x["killer_total"] / max(x["my_total"], 1) for x in lost) / max(len(lost), 1)
        single = sum(x["mass"] for x in lost if x["my_cells"] < 2)
        print(f"{n:<16}{h:8.0f}{deaths / h:9.2f}{len(lost) / h:13.1f}{sum(x['mass'] for x in lost) / h:12.0f}"
              f"{100 * len(multi) / max(len(lost), 1):13.0f}{ratio:11.2f}{single / h:24.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
