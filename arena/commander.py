"""Autonomous commander: every ~1.5 game-seconds per commanded agent, read the agent's `scene` from the ogar
control port, ask a backend for one order, apply it. Log every decision to logs/commander-*.jsonl.

    uv run python -m arena.commander --port 9811 --agents plan --backend heuristic
    uv run python -m arena.commander --port 9811 --agents plan --backend jev        # needs JEV_API_KEY
    uv run python -m arena.commander --port 9811 --agents plan --backend anthropic   # needs ANTHROPIC_API_KEY

Backends (credentials are only ever read from the environment, never stored):
  heuristic  rule-based stand-in (no latency, no cost), for tests and as a baseline
  jev        bot/systemone.py: the hosted System One API, same choice/noul questions as bot/strategist.py
             (JEV_API_KEY; JEV_BASE_URL / JEV_MODEL optional)
  anthropic  Messages API (ANTHROPIC_API_KEY; COMMANDER_MODEL, default claude-haiku-4-5-20251001)
"""

import argparse
import json
import os
import re
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTERVAL_S = 1.5       # game seconds between decisions per agent
DEFAULT_TTL_S = 4.0            # an order outlives one interval, so a slow/failed call leaves the last order active
VERBS = ("hunt", "goto", "farm", "flee", "hide", "virus-split", "harvest", "free")


class Control:
    """Line protocol client of `ogar serve --control-port` (one request, one reply line)."""

    def __init__(self, host="127.0.0.1", port=9601, token=None, timeout=30.0):
        self.addr, self.token, self.timeout = (host, port), token, timeout
        self._lock = threading.Lock()
        self._sock = None
        self._buf = b""

    def _connect(self):
        self._sock = socket.create_connection(self.addr, timeout=self.timeout)
        self._buf = b""
        if self.token:
            self._sock.sendall(f"auth {self.token}\n".encode())
            self._readline()

    def _readline(self) -> str:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("control port closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", "replace").rstrip("\r")

    def cmd(self, line: str) -> str:
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._sock is None:
                        self._connect()
                    self._sock.sendall(line.encode() + b"\n")
                    return self._readline()
                except (OSError, ConnectionError):
                    self._sock = None
                    if attempt:
                        raise
        raise RuntimeError("unreachable")

    def json(self, line: str):
        r = self.cmd(line)
        if r.startswith("error"):
            raise RuntimeError(r)
        return json.loads(r)

    def agents(self) -> list[dict]:
        return self.json("agents")["agents"]

    def scene(self, agent: str) -> dict:
        return self.json(f"scene {agent}")


# ---------------------------------------------------------------------------------------------------------
# options: the orders worth considering in a scene (shared by every backend)

def options(scene: dict) -> list[dict]:
    """[{id, order, desc}]; `order` is the control-port verb string (without agent and ttl)."""
    if not scene.get("alive"):
        return []
    me = scene["me"]
    opts = [{"id": "farm", "order": "farm", "desc": "Ignore other players and eat food pellets"},
            {"id": "flee", "order": "flee", "desc": "Run directly away from the bigger cells"}]
    if any(v["hides_us"] for v in scene["viruses"]):
        opts.append({"id": "hide", "order": "hide", "desc": "Move to the nearest virus that is bigger than me so bigger cells cannot reach me"})
    for i, p in enumerate(scene["players"]):
        if p["we_can_eat"]:
            kind = "Harvest the pieces of" if p["popped"] else "Chase"
            opts.append({"id": f"p{i}", "order": f"{'harvest' if p['popped'] else 'hunt'} #{p['cell_id']}",
                         "desc": f"{kind} prey p{i} (mass {p['mass']}, {p['cells_visible']} cells, {p['gap']} units away)",
                         "player": p})
        elif p["can_eat_us"] and p["virus_pop_possible"] and scene["viruses"] and me["mass"] >= 150:
            opts.append({"id": f"v{i}", "order": f"virus-split #{p['cell_id']}",
                         "desc": f"Shoot viruses at the bigger player v{i} (mass {p['mass']}) so a virus pops it",
                         "player": p})
    return opts


def pick_heuristic(scene: dict) -> dict:
    """Rule-based stand-in: avoid trouble first, otherwise chase the juiciest edible player."""
    me = scene["me"]
    opts = {o["id"]: o for o in options(scene)}
    close_eaters = [p for p in scene["players"] if p["can_eat_us"] and p["gap"] < (700 if p["split_threat"] else 400)]
    if close_eaters:
        if "hide" in opts and min(v["gap"] for v in scene["viruses"] if v["hides_us"]) < 500:
            return {"order": "hide", "ttl": 3.0, "split": "off", "why": "eater close, cover near"}
        return {"order": "flee", "ttl": 3.0, "split": "off", "why": "eater close"}
    prey = [o for o in opts.values() if o["id"].startswith("p") and o["player"]["gap"] < 1600
            and o["player"]["mass"] >= 0.2 * me["mass"]]
    if prey:
        best = max(prey, key=lambda o: o["player"]["mass"] / (max(o["player"]["gap"], 0) + 100))
        p = best["player"]
        safe_split = not any(q["can_eat_us"] and q["gap"] < 900 for q in scene["players"])
        return {"order": best["order"], "ttl": DEFAULT_TTL_S, "split": "on" if p["split_kill"] and safe_split else "off",
                "why": f"prey {p['mass']}"}
    shot = [o for o in opts.values() if o["id"].startswith("v") and o["player"]["gap"] > 500]
    if shot and me["mass"] >= 300 and not any(p["can_eat_us"] and p["gap"] < 700 for p in scene["players"]):
        best = max(shot, key=lambda o: o["player"]["mass"])
        return {"order": best["order"], "ttl": DEFAULT_TTL_S, "split": "off", "why": "virus-split a big one"}
    return {"order": "free", "ttl": DEFAULT_TTL_S, "split": None, "why": "nothing to do"}


GOAL = ("You command a cell in agar.io. Goal: grow as large as possible without being eaten, and stay on top of the "
        "leaderboard. A cell can eat another cell about 1.3x lighter; a cell 2.6x heavier can split and lunge across a gap "
        "to eat you. Viruses (spiky green cells) burst cells larger than themselves; small cells can hide behind big viruses.")


class JevBackend:
    """bot/strategist.py's questions against the System One API."""

    name = "jev"

    def __init__(self):
        key = os.environ.get("JEV_API_KEY")
        if not key:
            raise SystemExit("JEV_API_KEY is not set: no Jev credentials available")
        from .systemone import SystemOne
        kw = {}
        if os.environ.get("JEV_BASE_URL"):
            kw["base_url"] = os.environ["JEV_BASE_URL"]
        if os.environ.get("JEV_MODEL"):
            kw["model"] = os.environ["JEV_MODEL"]
        self.jev = SystemOne(key, **kw)
        self.calls = 0

    def questions(self, scene: dict, opts: list[dict]):
        me = scene["me"]
        state = {"my_mass": me["mass"], "my_cells": me["cells"], "in_danger": me["in_danger"],
                 "nearest_virus_gap": min((v["gap"] for v in scene["viruses"]), default=-1),
                 "leaderboard": [f"{e['name']} {e['mass']}" + (" (me)" if e["you"] else "") for e in scene["leaderboard"][:5]],
                 "players": [{"id": o["id"], "mass": o["player"]["mass"], "gap": o["player"]["gap"],
                              "can_eat_me": o["player"]["can_eat_us"]} for o in opts if "player" in o],
                 "options": {o["id"]: o["desc"] for o in opts}}
        qs = {"mode": {"type": "choice", "instructions": f"{GOAL} Given the state, what should my cell do for the next few seconds?",
                       "criteria": {o["id"]: o["desc"] for o in opts}}}
        if any(o.get("player", {}).get("split_kill") for o in opts):
            qs["split"] = {"type": "noul", "instructions": f"{GOAL} Is it safe to split onto the prey with `split_kill` "
                           "(halving my cell and lunging) given the other players nearby?"}
        return state, qs

    def decide(self, scene: dict) -> dict:
        opts = options(scene)
        by_id = {o["id"]: o for o in opts}
        state, qs = self.questions(scene, opts)
        ans = self.jev.ask(state, qs)
        self.calls += 1
        choice = ans["mode"]["choice"]
        o = by_id.get(choice, by_id["farm"])
        split = None
        if "split" in ans:
            split = "on" if ans["split"].get("noul", 0.0) > 0.6 else "off"
        return {"order": o["order"], "ttl": DEFAULT_TTL_S, "split": split, "why": f"jev {choice}", "probs": ans["mode"].get("probabilities")}


ORDER_RE = re.compile(r"^(hunt|virus-split|harvest)\s+\S.*$|^goto\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?$|^(farm|flee|hide|free)$")


class AnthropicBackend:
    """One Messages API call per decision; the model answers with one JSON line."""

    name = "anthropic"
    SYSTEM = (GOAL + "\n\nYou get the agent's scene as JSON (positions relative to the agent, y down). Reply with ONE JSON line "
              '{"order": "<verb>", "ttl": <seconds 2-8>, "split": "on"|"off"|null}. Verbs: farm | flee | hide | hunt #<cell_id> | '
              "harvest #<cell_id> (eat a popped player's pieces) | virus-split #<cell_id> (shoot viruses at a bigger player to pop it) | "
              "goto <x> <y> (world coordinates) | free (let the bot decide). The bot keeps its own reflexes (it will not walk into a "
              "bigger cell), so orders are suggestions about strategy. Prefer `free` when nothing is going on. No other text.")

    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set: no Anthropic credentials available")
        import httpx
        self.model = os.environ.get("COMMANDER_MODEL", "claude-haiku-4-5-20251001")
        self.http = httpx.Client(base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"), timeout=30,
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        self.calls = 0

    def decide(self, scene: dict) -> dict:
        r = self.http.post("/v1/messages", json={"model": self.model, "max_tokens": 120, "system": self.SYSTEM,
                                                  "messages": [{"role": "user", "content": json.dumps(scene, separators=(",", ":"))}]})
        r.raise_for_status()
        self.calls += 1
        return parse_reply("".join(b.get("text", "") for b in r.json()["content"]))


def parse_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0)) if m else {}
    order = str(d.get("order", "free")).strip()
    if not ORDER_RE.match(order):
        order = "free"
    try:
        ttl = min(max(float(d.get("ttl", DEFAULT_TTL_S)), 1.0), 30.0)
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_S
    split = d.get("split") if d.get("split") in ("on", "off") else None
    return {"order": order, "ttl": ttl, "split": split, "why": "llm"}


class HeuristicBackend:
    name = "heuristic"
    calls = 0

    def decide(self, scene: dict) -> dict:
        self.calls += 1
        return pick_heuristic(scene)


def make_backend(name: str):
    return {"heuristic": HeuristicBackend, "jev": JevBackend, "anthropic": AnthropicBackend}[name]()


def apply(ctl: Control, agent: str, d: dict) -> list[str]:
    replies = []
    if d["order"] == "free":
        replies.append(ctl.cmd(f"order {agent} free"))
        return replies
    replies.append(ctl.cmd(f"order {agent} {d['order']} ttl={d['ttl']:g}"))
    if d.get("split"):
        replies.append(ctl.cmd(f"order {agent} split {d['split']} ttl={d['ttl']:g}"))
    return replies


class Commander:
    def __init__(self, ctl: Control, backend, agents: list[str], interval_s=DEFAULT_INTERVAL_S, log_path=None):
        self.ctl, self.backend, self.agents, self.interval = ctl, backend, agents, interval_s
        self.log_path = log_path
        self.stop = threading.Event()
        self.decisions = 0
        self.errors = 0
        self._log_lock = threading.Lock()

    def log(self, rec: dict):
        if self.log_path:
            with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def step(self, agent: str):
        scene = self.ctl.scene(agent)
        if not scene.get("alive"):
            return
        t0 = time.monotonic()
        try:
            d = self.backend.decide(scene)
        except Exception as e:  # keep the last order running on backend errors
            self.errors += 1
            self.log({"wall": time.time(), "agent": agent, "t": scene["t"], "error": repr(e)})
            return
        ms = int((time.monotonic() - t0) * 1000)
        replies = apply(self.ctl, agent, d)
        self.decisions += 1
        self.log({"wall": time.time(), "agent": agent, "t": scene["t"], "ms": ms, "backend": self.backend.name,
                  "mass": scene["me"]["mass"], "decision": d, "replies": replies})

    def _agent_loop(self, agent: str):
        last = None
        while not self.stop.is_set():
            try:
                t = self.ctl.scene(agent)["t"]
                if last is None or t - last >= self.interval:
                    last = t
                    self.step(agent)
            except (OSError, ConnectionError, RuntimeError, ValueError):
                self.errors += 1
            self.stop.wait(0.2)

    def run(self):
        threads = [threading.Thread(target=self._agent_loop, args=(a,), daemon=True) for a in self.agents]
        for t in threads:
            t.start()
        return threads


def resolve_agents(ctl: Control, prefixes: list[str]) -> list[str]:
    names = [a["name"] for a in ctl.agents()]
    return [n for n in names if any(n.startswith(p) for p in prefixes)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9601, help="ogar control port")
    ap.add_argument("--token", default=os.environ.get("OGAR_CONTROL_TOKEN"))
    ap.add_argument("--agents", nargs="+", required=True, help="agent name prefixes to command (e.g. plan)")
    ap.add_argument("--backend", choices=["heuristic", "jev", "anthropic"], default="heuristic")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="game seconds between decisions per agent")
    ap.add_argument("--minutes", type=float, default=0, help="stop after N wall minutes (0 = until Ctrl-C)")
    a = ap.parse_args()
    ctl = Control(a.host, a.port, a.token)
    agents = resolve_agents(ctl, a.agents)
    if not agents:
        raise SystemExit(f"no agent matches {a.agents}")
    (ROOT / "logs").mkdir(exist_ok=True)
    log = ROOT / "logs" / f"commander-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    c = Commander(ctl, make_backend(a.backend), agents, a.interval, log)
    print(f"commanding {agents} with {a.backend}; log {log}")
    c.run()
    try:
        end = time.time() + a.minutes * 60 if a.minutes else None
        while end is None or time.time() < end:
            time.sleep(5)
            print(f"decisions={c.decisions} errors={c.errors} calls={c.backend.calls}")
    except KeyboardInterrupt:
        pass
    c.stop.set()
    for ag in agents:
        ctl.cmd(f"order {ag} free")


if __name__ == "__main__":
    main()
