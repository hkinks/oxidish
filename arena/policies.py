"""Policies usable by arena agents: `act(snapshot) -> (x, y, split)`, optional `reset()`.

Policies are built from picklable specs so tournament workers can construct them:
    {"farm": "route" | "field", "params": {...Params fields...}}
"""

from bot.farm import FoodRouter
from bot.game import Snapshot
from bot.modes import heuristic_plan
from bot.params import Params
from bot.steer import steer
from bot.tactics import Tactics


class HeuristicPolicy:
    """The browser bot's steering, with rule-based modes instead of Jev."""

    def __init__(self, farm: str = "route", params: dict | None = None):
        self._router = FoodRouter() if farm == "route" else None
        self._p = Params.from_dict(params or {})
        self._tactics = Tactics()
        self.counters = self._tactics.counters
        self._prev: dict[int, tuple[float, float]] = {}  # last positions, for velocities

    def act(self, s: Snapshot):
        """Return (x, y, split, eject)."""
        prev, self._prev = self._prev, {}
        for c in s.others:
            if c.kind == "food":
                continue
            if c.id in prev:
                c.vx, c.vy = c.x - prev[c.id][0], c.y - prev[c.id][1]
            self._prev[c.id] = (c.x, c.y)
        plan = heuristic_plan(s, self._p)
        out = self._tactics.act(s, plan, self._p)
        if out is None:
            out = (*steer(s, plan, self._router, self._p), False)
        # Playstyle counters (quality-diversity niches): how it plays, not how well.
        c = self.counters
        c["steps"] = c.get("steps", 0) + 1
        c[f"mode_{plan.mode}"] = c.get(f"mode_{plan.mode}", 0) + 1
        c["splits"] = c.get("splits", 0) + bool(out[2])
        return out

    def reset(self) -> None:
        self._prev = {}
        self._tactics.reset()
        if self._router:
            self._router.reset()


class JevPolicy(HeuristicPolicy):
    """Jev (TypeSafe System One, bot/strategist.py) picks the strategy - farm / hunt prey X (and
    whether a split-kill is safe) / flee / hide behind a virus - on a worker thread every 1-5 s; the
    champion's routines (steering, tactics: virus shots, double splits, with its evolved Params)
    execute it every step. Jev's answers take a few hundred ms, so the rule-based threat check
    overrides: when the champion's own plan says flee, the bot flees at once."""

    def __init__(self, params: dict | None = None, farm: str = "route", name: str = "jev"):
        super().__init__(farm, params)
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from bot.strategist import Strategist
        from bot.systemone import SystemOne
        root = Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
        key = os.environ.get("JEV_API_KEY")
        if not key:
            raise SystemExit("JEV_API_KEY missing (expected in .env)")
        (root / "logs").mkdir(exist_ok=True)
        say = lambda msg, *_: print(f"[{name}] {msg}", flush=True)
        self._strategist = Strategist(SystemOne(key), say, say, root / "logs")

    def act(self, s: Snapshot):
        prev, self._prev = self._prev, {}
        for c in s.others:
            if c.kind == "food":
                continue
            if c.id in prev:
                c.vx, c.vy = c.x - prev[c.id][0], c.y - prev[c.id][1]
            self._prev[c.id] = (c.x, c.y)
        self._strategist.step(s)
        rule = heuristic_plan(s, self._p)
        plan = rule if rule.mode == "flee" else self._strategist.plan     # safety first, Jev otherwise
        out = self._tactics.act(s, plan, self._p)
        if out is None:
            out = (*steer(s, plan, self._router, self._p), False)
        c = self.counters
        c["steps"] = c.get("steps", 0) + 1
        c[f"mode_{plan.mode}"] = c.get(f"mode_{plan.mode}", 0) + 1
        c["jev_calls"] = self._strategist.calls
        return out

    def reset(self) -> None:
        super().reset()
        self._strategist.reset()


POLICIES = {
    "route": {"farm": "route"},
    "field": {"farm": "field"},
}


def hunter_spec(base: dict) -> dict:
    """Aggressive sparring partner built on a champion: chases anything edible however small,
    hunts far, splits readily (also while already split) and flees late, like an
    aggressive human rather than our cautious evolved bots."""
    params = dict(base.get("params", {}))
    params.update(min_prey_frac=0.0, hunt_range=2500, farm_prey=2.5, target_bonus=15,
                  split_safety=0.8, split_reach=1.1, split_max_cells=4, flee_margin=0.6)
    return {"farm": "route", "params": params}


def build(spec: dict):
    if spec.get("kind") == "jev":  # {"kind": "jev", "params": {...champion Params...}}
        return JevPolicy(spec.get("params"), spec.get("farm", "route"))
    if spec.get("kind") == "nn":  # {"kind": "nn", "ckpt": "runs/rl/p1/latest.pt"}
        from rl.policy import NNPolicy
        return NNPolicy(spec["ckpt"], spec.get("greedy"), nosplit=spec.get("nosplit", False),
                         guard=spec.get("guard", False))
    return HeuristicPolicy(spec.get("farm", "route"), spec.get("params"))
