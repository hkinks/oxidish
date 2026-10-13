"""arena.human_report on a synthetic recording (no server).

    uv run pytest arena/test_human_report.py
"""

import gzip
import json

from .human_log import classify_scene, load
from .human_report import build_report

BORDER = [-3500, 3500, -3500, 3500]


def _step(t, me_x, mode, threat_gap_x):
    return {"t": t, "mine": [[1, me_x, 0, 100]], "o": [[9, me_x + threat_gap_x, 0, 200, 0, "human"]], "food": [],
            "aim": [me_x - 500, 0], "split": False, "eject": False, "mode": mode, "stage": "small", "target": None,
            "split_ok": False}


def scene(mode="flee"):
    steps = [_step(100 + 2 * i, 3300 + i, mode, -600 + 40 * i) for i in range(16)]
    eat = {"e": "eat", "t": 132, "popped": False, "pop_ago": None,
           "eater": {"n": "human", "k": "human", "cm": 400, "tot": 400, "cells": 1, "x": 3300, "y": 0},
           "victim": {"n": "safe-0", "k": "agent", "cm": 100, "tot": 100, "cells": 1, "x": 3310, "y": 0, "last": True}}
    return {"e": "scene", "t": 132, "agent": "safe-0", "kind": "death", "before": 100, "after": 0, "eaters": [eat],
            "human_eater": True, "sync": 2, "border": BORDER, "steps": steps,
            "spec": {"farm": "route", "params": {}}}, eat


def write(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_dir(tmp_path, mode="flee"):
    sc, eat = scene(mode)
    tops = [{"e": "top", "t": 25 * i, "n": 3, "top": [["human", "human", 900 + i, 1, 0], ["safe-0", "agent", 100, 1, 0]],
             "h": [["human", "human", 900 + i, 1, 0]]} for i in range(1, 6)]
    write(tmp_path / "events-1.jsonl.gz", [{"e": "meta"}] + tops + [eat])
    write(tmp_path / "scenes-1.jsonl.gz", [sc])
    return tmp_path


def test_classify_flee_at_wall(tmp_path):
    sc = load(make_dir(tmp_path))["scenes"][0]
    assert sc["id"] == "1-0"
    cl = classify_scene(sc)
    assert cl["cause"] == "fled" and "wall" in cl["tags"] and "human" in cl["tags"]


def test_classify_no_flee(tmp_path):
    sc = load(make_dir(tmp_path, "farm"))["scenes"][0]
    assert classify_scene(sc)["cause"] == "no_flee"


def test_report(tmp_path):
    d = make_dir(tmp_path)
    md = build_report(str(d), None, 3)
    assert "#1 for 100%" in md and "safe" in md and "scene 1-0" in md


def test_truncated_gzip_is_readable(tmp_path):
    d = make_dir(tmp_path)
    raw = (d / "scenes-1.jsonl.gz").read_bytes()
    (d / "scenes-1.jsonl.gz").write_bytes(raw[:-8])
    assert load(d)["events"]  # events still read; the cut scene file does not raise
