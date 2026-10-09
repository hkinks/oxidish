"""Markdown tables from `ogar drill --out` files (docs/harvest-drill.md).

    python -m arena.drill_report runs/drill/main_rule.json runs/drill/main_vision.json [--what main|diag|sweep|first]

`main`: per condition and agent the capture / overtake / death table; `diag`: the behaviour counters; files are merged
(same condition from several files = one table). Intervals are 95%: mean +- 1.96 se for means, Wilson for rates.
"""

import argparse
import json


def pct(x, nd=0):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def pm(m, h, nd=0):
    return "-" if m is None else (f"{100 * m:.{nd}f}+-{100 * h:.{nd}f}%" if h is not None else f"{100 * m:.{nd}f}%")


def load(paths):
    rows, order = {}, []
    for p in paths:
        for s in json.load(open(p))["summary"]:
            k = s["cond"]
            if k not in rows:
                rows[k] = []
                order.append(k)
            rows[k].append(s)
    return order, rows


def num(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main_table(cond, sums):
    print(f"\n**{cond}**\n")
    print("| agent | n | capture 10 s | 20 s | 30 s | ours/giant @30 s | overtake (any time) | median t_overtake | first piece (rate / mean t) | died | pieces |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sums:
        w = s["windows"]
        last = sorted(w, key=lambda k: int(k[:-1]))[-1]
        a, b, c = w["10s"], w["20s"], w["30s"]
        print(f"| {s['agent']} | {s['n']} | {pm(a['capture'], a['capture_ci'])} | {pm(b['capture'], b['capture_ci'])} | {pm(c['capture'], c['capture_ci'])} "
              f"| {num(c['ours_over_giant'], 2)}+-{num(c['ours_over_giant_ci'], 2)} | {pct(s['overtake_any'])} [{pct(s['overtake_any_lo'])}-{pct(s['overtake_any_hi'])}] "
              f"| {num(s['t_overtake_median'])} s | {pct(s['first_piece'])} / {num(s['t_first_mean'])} s | {pct(s['died'])} [{pct(s['died_lo'])}-{pct(s['died_hi'])}] | {num(s['pieces'])} |")
    if len(sums) and any(k not in ("10s", "20s", "30s") for k in sums[0]["windows"]):
        print()
        print("| agent | " + " | ".join(f"capture {k}" for k in sums[0]["windows"] if k not in ("10s", "20s", "30s")) + " |")
        print("|---|" + "---|" * len([k for k in sums[0]["windows"] if k not in ("10s", "20s", "30s")]))
        for s in sums:
            ks = [k for k in s["windows"] if k not in ("10s", "20s", "30s")]
            print(f"| {s['agent']} | " + " | ".join(pm(s["windows"][k]["capture"], s["windows"][k]["capture_ci"]) + f", overtake {pct(s['windows'][k]['overtake'])}" for k in ks) + " |")


def diag_table(cond, sums):
    print(f"\n**{cond}**\n")
    print("| agent | edible piece in view (ticks) | ...of those: moving towards it | standing still | threat piece near (ticks) | ...of those: moving away | splits / rep | max cells | peak mass | lost to 3rd | stolen by comp |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sums:
        d = s["diag"]
        print(f"| {s['agent']} | {pct(d['edible_vis'])} | {pct(d['closing_of_vis'])} | {pct(d['still_of_vis'])} | {pct(d['threat_near'])} | {pct(d['fleeing_of_threat'])} "
              f"| {num(d['splits'])} | {num(d['max_cells'])} | {num(s['peak_ours'], 0)} | {num(s['lost_third'], 0)} | {num(s['stolen_by_comp'], 0)} |")


def compact(order, rows):
    """one line per condition x agent: capture@30, overtake, died"""
    print("| condition | agent | n | capture @30 s | overtake | died |")
    print("|---|---|---|---|---|---|")
    for k in order:
        for s in rows[k]:
            c = s["windows"]["30s"]
            print(f"| {k} | {s['agent']} | {s['n']} | {pm(c['capture'], c['capture_ci'])} | {pct(s['overtake_any'])} | {pct(s['died'])} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--what", default="main", choices=["main", "diag", "compact"])
    a = ap.parse_args()
    order, rows = load(a.files)
    if a.what == "compact":
        compact(order, rows)
    else:
        for k in order:
            (main_table if a.what == "main" else diag_table)(k, rows[k])
