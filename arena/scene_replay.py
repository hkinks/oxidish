"""Replay a recorded loss scene through the Python policy, or turn it into a pytest regression stub.

    uv run python -m arena.scene_replay logs/human --list                      # scenes: id, bot, loss, cause
    uv run python -m arena.scene_replay logs/human 1791371135-0                # decisions step by step
    uv run python -m arena.scene_replay logs/human 1791371135-0 --spec bot/r2g9_ramp800.json
    uv run python -m arena.scene_replay logs/human 1791371135-0 --pytest --back 1.0 [--write bot/test_scene_x.py]

The default policy is the genome the server's bot was built from (stored in the scene), so the replay
shows what the bot decided (the Rust policy recorded its own aim and mode: they are printed next to the
Python ones; the two implementations agree). `--spec` replays with another genome to see whether a fix
changes the decision. Food is only recorded for the last 3 steps of a scene, so earlier farm decisions
differ (hunt / flee do not depend on food). The pytest stub rebuilds the Snapshot `--back` seconds before
the loss, in bot/test_steer.py style; the assertion is left to fill in.
"""

import argparse
import json
import math
from pathlib import Path

from bot.game import Cell, Snapshot
from bot.modes import heuristic_plan

from .human_log import KINDS, classify_scene, load
from .policies import build


def snapshot_of(step, border, food_ids=True) -> Snapshot:
    mine = [Cell(c[0], c[1], c[2], c[3], "mine", "") for c in step["mine"]]
    others = []
    for o in step["o"]:
        others.append(Cell(o[0], o[1], o[2], o[3], KINDS.get(o[4], "player"), o[5]))
    for i, f in enumerate(step.get("food") or []):
        others.append(Cell(-1 - i, f[0], f[1], f[2], "food", ""))
    l, r, t, b = border
    return Snapshot(mine, others, {}, 0, 0, {"left": l, "right": r, "top": t, "bottom": b}, not mine)


def find(data, sid):
    for sc in data["scenes"]:
        if sc["id"] == sid:
            return sc
    raise SystemExit(f"no scene {sid}; --list shows them")


def replay(sc, spec=None) -> list[str]:
    policy = build(spec or sc["spec"])
    out = [f"scene {sc['id']}: {sc['agent']} {sc['kind']} {sc['before']} -> {sc['after']} (tick {sc['t']}), "
           f"policy: {'--spec' if spec else 'recorded genome'}",
           "  t-s   mine(py)  rust: mode aim(dx,dy) split    python: mode aim(dx,dy) split   differs"]
    last_t = sc["steps"][-1]["t"]
    for st in sc["steps"]:
        s = snapshot_of(st, sc["border"])
        me = s.main
        p = getattr(policy, "_p", None)
        mode = heuristic_plan(s, p).mode if p is not None else "-"
        x, y, split, *_ = policy.act(s)
        ra = (st["aim"][0] - me.x, st["aim"][1] - me.y)
        pa = (x - me.x, y - me.y)
        ang = abs((math.atan2(ra[1], ra[0]) - math.atan2(pa[1], pa[0]) + math.pi) % (2 * math.pi) - math.pi)
        diff = "" if (mode == st["mode"] or mode == "-") and ang < 0.2 and bool(split) == st["split"] else "<<<"
        out.append(f"  {(st['t'] - last_t) / 25:+5.1f} {len(s.mine):3d} cells   {st['mode']:>5} ({ra[0]:+5.0f},{ra[1]:+5.0f}) "
                   f"{'SPLIT' if st['split'] else '     '}    {mode:>5} ({pa[0]:+5.0f},{pa[1]:+5.0f}) "
                   f"{'SPLIT' if split else '     '}  {diff}")
    return out


def pytest_stub(sc, back_s: float, spec=None) -> str:
    steps = sc["steps"]
    k = max(0, len(steps) - 1 - round(back_s * 25 / sc.get("sync", 2)))
    st = steps[k]
    cl = classify_scene(sc)
    l, r, t, b = sc["border"]
    name = f"scene_{sc['id'].replace('-', '_')}"
    lines = [f"def test_{name}():",
             f'    """Human loss scene {sc["id"]}: {sc["agent"]} {sc["kind"]} {sc["before"]} -> {sc["after"]} '
             f'(cause {cl["cause"]}, tags {",".join(cl["tags"]) or "-"}), {back_s:g} s before the loss.',
             f"    The bot then did: mode {st['mode']}, aim ({st['aim'][0]:.0f}, {st['aim'][1]:.0f}), split {st['split']}.",
             '    TODO: say what it should have done and assert it."""',
             f"    border = {{\"left\": {l:.0f}, \"right\": {r:.0f}, \"top\": {t:.0f}, \"bottom\": {b:.0f}}}"]
    cells = []
    for i, c in enumerate(st["mine"]):
        cells.append(f'Cell({c[0]}, {c[1]}, {c[2]}, {c[3]}, "mine", "")')
    lines.append("    mine = [" + ", ".join(cells) + "]")
    oth = []
    for o in st["o"]:
        oth.append(f'Cell({o[0]}, {o[1]}, {o[2]}, {o[3]}, "{KINDS.get(o[4], "player")}", {o[5]!r})')
    lines.append("    others = [" + (",\n              ".join(oth)) + "]")
    if st.get("food"):
        lines.append("    # food in view was recorded for this step: " + str(len(st["food"])) + " pellets (omitted)")
    lines.append("    s = Snapshot(mine, others, {}, 0, 0, border, False)")
    spec_json = json.dumps(spec or sc["spec"])
    lines.append(f"    spec = {spec_json}")
    lines.append("    x, y, split, *_ = build(spec).act(s)")
    lines.append("    me = s.main")
    lines.append("    dx, dy = x - me.x, y - me.y")
    lines.append("    # TODO: assert on the decision, e.g. the aim points away from the threat / the split is withheld")
    lines.append("    pytest.skip('stub: assertion not written yet')")
    head = ("import pytest\n\nfrom arena.policies import build\nfrom bot.game import Cell, Snapshot\n\n\n")
    return head + "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="recording dir")
    p.add_argument("scene", nargs="?", help="scene id (<session>-<n>), see --list")
    p.add_argument("--session")
    p.add_argument("--list", action="store_true")
    p.add_argument("--spec", help="genome json to replay with instead of the recorded one")
    p.add_argument("--pytest", action="store_true", help="print a regression test stub instead of replaying")
    p.add_argument("--back", type=float, default=1.0, help="stub: seconds before the loss to take the snapshot from")
    p.add_argument("--write", help="stub: write to this file instead of stdout")
    a = p.parse_args()
    data = load(a.dir, a.session)
    if a.list or not a.scene:
        for sc in data["scenes"]:
            cl = classify_scene(sc)
            print(f"{sc['id']:>16}  {sc['agent']:<10} {sc['kind']:<5} {sc['before']:>6} -> {sc['after']:<6} "
                  f"{cl['cause']:<10} {','.join(cl['tags']):<18} eaters: "
                  + ", ".join(f"{e['eater']['n']}({e['eater']['k']})" for e in sc["eaters"][:2]))
        return
    sc = find(data, a.scene)
    spec = json.load(open(a.spec)) if a.spec else None
    if a.pytest:
        txt = pytest_stub(sc, a.back, spec)
        if a.write:
            Path(a.write).write_text(txt, encoding="utf-8")
            print(f"wrote {a.write}")
        else:
            print(txt)
        return
    print("\n".join(replay(sc, spec)))


if __name__ == "__main__":
    main()
