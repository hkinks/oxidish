"""Summarize human play recorded by `ogar serve --record DIR` into a markdown report of bot weaknesses.

    uv run python -m arena.human_report logs/human                  # all sessions in the dir
    uv run python -m arena.human_report logs/human --session 1791371135 --top 8 --out docs/human/2026-10-07.md

Sections: the human's leader share (#1 time) and mass stats, deaths; mass taken from each bot family; loss
scenes (an in-process bot lost >= 30% of its mass or died; humans' kills, or all kills if the server ran with
--record-all) classified by cause (popped / split_kill / pieces / unseen / fled / no_flee) with
geometry tags (wall, corner); the worst scenes with a text rendering.
"""

import argparse
import datetime
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .human_log import classify_scene, family, load, render_scene

TICKS_PER_S = 25


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def human_stats(events) -> list[str]:
    tops = [e for e in events if e["e"] == "top"]
    if not tops:
        return ["No `top` samples recorded."]
    span = (tops[-1]["t"] - tops[0]["t"]) / TICKS_PER_S
    per: dict[str, dict] = defaultdict(lambda: {"alive": 0, "lead": 0, "lead_all": 0, "mass": [], "mass_ub": [], "peak": 0, "peak_ub": 0})
    leaders = Counter()
    for e in tops:
        if e["top"]:
            leaders[family(e["top"][0][0], e["top"][0][1])] += 1
        for name, _, mass, _, boosted in e["h"]:
            d = per[name]
            d["alive"] += 1
            d["mass"].append(mass)
            d["peak"] = max(d["peak"], mass)
            if not boosted:
                d["mass_ub"].append(mass)
                d["peak_ub"] = max(d["peak_ub"], mass)
            if e["top"] and e["top"][0][0] == name:
                d["lead"] += 1
    out = [f"Session length {span / 60:.1f} min ({len(tops)} one-second samples)."]
    for name, d in sorted(per.items()):
        n = d["alive"]
        out.append(f"- **{name}**: alive {n}s ({100 * n / max(len(tops), 1):.0f}% of the session); "
                   f"**#1 for {100 * d['lead'] / max(n, 1):.0f}% of its alive time**; "
                   f"mass mean {statistics.mean(d['mass']):.0f}, median {statistics.median(d['mass']):.0f}, "
                   f"p90 {_pct(d['mass'], 0.9)}, peak {d['peak']}"
                   + (f"; without +500 boosted lives: mean {statistics.mean(d['mass_ub']):.0f}, peak {d['peak_ub']}"
                      if d["mass_ub"] and len(d["mass_ub"]) != n else
                      ("; every life was boosted (+500), numbers are inflated" if not d["mass_ub"] else "")))
    lead = ", ".join(f"{k} {100 * v / len(tops):.0f}%" for k, v in leaders.most_common(6))
    out.append(f"- Who was #1 over the whole session (all samples): {lead}")
    return out


def eat_tables(events, humans: set[str]) -> list[str]:
    eats = [e for e in events if e["e"] == "eat"]
    taken: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0])   # events, mass, last cells, popped
    lost: Counter = Counter()
    deaths = []
    for e in eats:
        ea, vi = e["eater"], e["victim"]
        if ea["n"] in humans and ea["k"] == "human":
            t = taken[family(vi["n"], vi["k"])]
            t[0] += 1
            t[1] += vi["cm"]
            t[2] += vi["last"]
            t[3] += e["popped"]
        if vi["n"] in humans and vi["k"] == "human":
            lost[family(ea["n"], ea["k"])] += vi["cm"]
            if vi["last"]:
                deaths.append(e)
    out = []
    out.append("| victim family | cells eaten | mass taken | bots killed (last cell) | victims that had just popped |")
    out.append("|---|---:|---:|---:|---:|")
    for fam, (n, m, last, pop) in sorted(taken.items(), key=lambda kv: -kv[1][1]):
        out.append(f"| {fam} | {n} | {m} | {last} | {pop} |")
    if not taken:
        out.append("| (the human ate nothing) | | | | |")
    out.append("")
    out.append(f"Human deaths: **{len(deaths)}**" + (": " + "; ".join(
        f"tick {e['t']} by {e['eater']['n']} ({e['eater']['k']}, cell {e['eater']['cm']}, total {e['eater']['tot']}, "
        f"x{e['eater']['cells']}) at {e['victim']['tot']} mass" for e in deaths[:8]) if deaths else ""))
    if lost:
        out.append("Human cells lost to: " + ", ".join(f"{k} {v}" for k, v in lost.most_common()))
    # all bot-vs-bot eating, for reference
    fam_eat: Counter = Counter()
    for e in eats:
        ea, vi = e["eater"], e["victim"]
        fam_eat[(family(ea["n"], ea["k"]), family(vi["n"], vi["k"]))] += vi["cm"]
    if fam_eat:
        out.append("")
        out.append("Biggest eater -> victim flows (mass), all players:")
        for (a, b), m in fam_eat.most_common(8):
            out.append(f"- {a} ate {b}: {m}")
    return out


def scene_tables(scenes, top_n: int) -> tuple[list[str], list[str]]:
    rows = []
    for sc in scenes:
        rows.append((sc, classify_scene(sc)))
    out = []
    for label, sel in (("eaten by a human", [r for r in rows if r[0].get("human_eater")]),
                       ("eaten by any other player", [r for r in rows if not r[0].get("human_eater")])):
        if not sel:
            continue
        by: dict[str, list] = defaultdict(list)
        for sc, cl in sel:
            by[cl["cause"]].append(sc["before"] - sc["after"])
        tot = sum(sum(v) for v in by.values()) or 1
        out += [f"**Loss scenes {label}**: {len(sel)} scenes, {tot} mass lost", "",
                "| cause | scenes | mass lost | share | at wall | corner | fled before |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for c, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
            rs = [r for r in sel if r[1]["cause"] == c]
            out.append(f"| {c} | {len(v)} | {sum(v)} | {100 * sum(v) / tot:.0f}% | "
                       f"{sum('wall' in r[1]['tags'] for r in rs)} | {sum('corner' in r[1]['tags'] for r in rs)} | "
                       f"{sum(bool(r[1].get('fled')) for r in rs)} |")
        by_agent: dict[str, list] = defaultdict(list)
        for sc, cl in sel:
            by_agent[family(sc["agent"], "agent")].append(sc["before"] - sc["after"])
        out += ["", "By bot family: " + ", ".join(f"{k} {len(v)} scenes / {sum(v)} mass"
                                                    for k, v in sorted(by_agent.items(), key=lambda kv: -sum(kv[1]))), ""]
    worst = sorted(rows, key=lambda r: (not r[0].get("human_eater"), -(r[0]["before"] - r[0]["after"])))[:top_n]
    return out, [render_scene(sc) for sc, _ in worst]


def build_report(rec_dir: str, session: str | None, top_n: int) -> str:
    data = load(rec_dir, session)
    events, scenes = data["events"], data["scenes"]
    humans = {n for e in events if e["e"] == "top" for n, *_ in e["h"]}
    md = [f"# Human play report ({', '.join(sorted(humans)) or 'no human'})", "",
          f"Recording: `{rec_dir}`, sessions {', '.join(data['sessions']) or '-'}; generated {datetime.date.today()}. "
          f"{len(scenes)} loss scenes, {sum(e['e'] == 'eat' for e in events)} eat events.", ""]
    md += ["## The human", ""] + human_stats(events) + [""]
    md += ["## Mass taken from each bot family", ""] + eat_tables(events, humans) + [""]
    tbl, rendered = scene_tables(scenes, top_n)
    md += ["## Loss scenes", ""]
    md += tbl if tbl else ["No loss scenes recorded (the server needs `--record`; scenes are dumped when a bot loses >= 30% mass "
                           "to a human, or to anyone with `--record-all`).", ""]
    if rendered:
        md += [f"## Worst {len(rendered)} scenes (human eaters first, then by mass lost)", ""]
        for r in rendered:
            md += ["```", r, "```", ""]
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="recording dir (ogar serve --record)")
    p.add_argument("--session", help="one session stamp (the number in events-<stamp>.jsonl.gz); default all")
    p.add_argument("--top", type=int, default=6, help="how many worst scenes to render")
    p.add_argument("--out", help="write markdown here (e.g. docs/human/<date>.md); default stdout")
    a = p.parse_args()
    md = build_report(a.dir, a.session, a.top)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
