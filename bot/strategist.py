"""Jev picks the mode (farm / hunt X / flee / hide) and whether split attacks are OK.

Runs on a worker thread; steering keeps using the last plan meanwhile. Asks when
the situation changes (new threats or prey in view), at most every
MIN_INTERVAL seconds and at least every MAX_INTERVAL seconds.
"""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .game import Snapshot
from .steer import SPLIT_KILL_RATIO, Plan, classify
from .systemone import SystemOne

MIN_INTERVAL = 1.0
MAX_INTERVAL = 5.0
MAX_LISTED = 4  # nearest prey/threats described to Jev

GOAL = ("You control a cell in agar.io. Goal: grow as large as possible without being "
        "eaten. A cell can eat another cell about 1.3x lighter; a cell 2.6x heavier "
        "can split and lunge across a gap to eat you. Viruses (spiky green cells) "
        "burst cells larger than themselves, but small cells can hide behind them.")


def _situation(s: Snapshot):
    me = s.main
    food, prey, threats, viruses = classify(s)
    near = lambda cells: sorted(cells, key=me.dist)[:MAX_LISTED]
    prey, threats = near(prey), near(threats)
    state = {
        "my_mass": round(s.total_mass),
        "my_cells": len(s.mine),
        "visible_food": len(food),
        "nearest_virus_distance": round(min((me.dist(v) - v.size - me.size for v in viruses),
                                            default=-1)),
        "threats": [{"mass": round(t.mass), "distance": round(me.dist(t) - t.size - me.size),
                     "can_split_on_me": t.mass > me.mass * SPLIT_KILL_RATIO} for t in threats],
        "prey": [{"id": f"p{i}", "mass": round(p.mass),
                  "distance": round(me.dist(p) - p.size - me.size),
                  "split_kill_possible": me.mass / 2 > p.mass * 1.3} for i, p in enumerate(prey)],
    }
    key = (tuple(t.id for t in threats), tuple(p.id for p in prey), len(s.mine))
    return state, key, prey


class Strategist:
    def __init__(self, jev: SystemOne, log, show, log_dir: Path):
        self._jev = jev
        self._log = log
        self._show = show
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None
        self._key = None
        self._asked_at = 0.0
        self.plan = Plan()
        self.calls = 0
        self._log_file = log_dir / f"decisions-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def reset(self) -> None:
        self.plan = Plan()
        self._key = None

    def step(self, s: Snapshot) -> None:
        if self._pending is not None:
            if not self._pending.done():
                return
            result = self._pending.result()
            self._pending = None
            if result:
                self._apply(*result)

        state, key, prey = _situation(s)
        since = time.monotonic() - self._asked_at
        if since < MIN_INTERVAL or (key == self._key and since < MAX_INTERVAL):
            return
        if not state["threats"] and not state["prey"]:
            if self.plan.mode != "farm":
                self.plan = Plan()
                self._show("farm (nothing nearby)", {})
            self._key = key
            return

        self._key = key
        self._asked_at = time.monotonic()
        criteria = {"farm": "Ignore other players and eat food pellets",
                    "flee": "Run directly away from the threats"}
        if state["nearest_virus_distance"] >= 0:
            criteria["hide"] = "Move next to the nearest virus so bigger cells cannot reach me"
        for p in state["prey"]:
            criteria[p["id"]] = (f"Chase prey {p['id']} (mass {p['mass']}, "
                                 f"{p['distance']} units away)")
        questions = {
            "mode": {"type": "choice",
                     "instructions": f"{GOAL} Given the situation in the state, what should "
                                     "my cell do for the next few seconds?",
                     "criteria": criteria},
        }
        if any(p["split_kill_possible"] for p in state["prey"]):
            questions["split"] = {
                "type": "noul",
                "instructions": f"{GOAL} If I chase the prey with `split_kill_possible`, "
                                "is it safe to split onto it (halving my cell and lunging) "
                                "given the threats nearby?",
            }
        ids = {f"p{i}": p.id for i, p in enumerate(prey)}
        self._pending = self._pool.submit(self._ask, state, questions, ids)

    def _ask(self, state, questions, ids):
        started = time.monotonic()
        try:
            answers = self._jev.ask(state, questions)
        except Exception as exc:  # keep steering with the old plan on API errors
            self._log(f"Jev error: {exc}")
            return None
        self.calls += 1
        ms = int((time.monotonic() - started) * 1000)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "ms": ms, "state": state,
                                "questions": questions, "answers": answers}) + "\n")
        return answers, ids, ms

    def _apply(self, answers, ids, ms) -> None:
        a = answers["mode"]
        choice = a["choice"]
        split_p = answers.get("split", {}).get("noul", 0.0)
        if choice in ids:
            self.plan = Plan("hunt", ids[choice], split_p > 0.6)
        else:
            self.plan = Plan(choice)
        label = f"hunt {choice}" if choice in ids else choice
        if "split" in answers:
            label += f" (split p={split_p:.2f})"
        self._show(f"{label} [{ms} ms]", a["probabilities"])
