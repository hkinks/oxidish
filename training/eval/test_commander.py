import json

from training.eval import commander as c


def player(name="Bob", mass=300, gap=200, eat_us=False, eat=True, popped=False, cid=7, split_kill=False):
    return {"name": name, "cell_id": cid, "mass": mass, "biggest_cell_mass": mass, "cells_visible": 3 if popped else 1,
            "cells": 1, "x": 100, "y": 0, "dist": gap + 100, "gap": gap, "can_eat_us": eat_us, "we_can_eat": eat,
            "split_threat": False, "split_kill": split_kill, "virus_pop_possible": True, "popped": popped}


def scene(players, viruses=(), mass=1000):
    return {"agent": "a-0", "t": 10.0, "alive": True,
            "me": {"mass": mass, "cells": 1, "x": 0, "y": 0, "biggest_cell_mass": mass, "kills": 0, "deaths": 0, "mode": "farm", "in_danger": False},
            "leaderboard": [{"name": "a-0", "mass": mass, "you": True}], "players": list(players), "viruses": list(viruses),
            "walls": {"left": 1, "right": 1, "top": 1, "bottom": 1}, "order": None}


def test_heuristic_hunts_prey_and_flees_eaters():
    d = c.pick_heuristic(scene([player(mass=400, gap=300)]))
    assert d["order"] == "hunt #7"
    d = c.pick_heuristic(scene([player(mass=3000, gap=100, eat_us=True, eat=False)]))
    assert d["order"] == "flee" and d["split"] == "off"


def test_heuristic_harvests_popped_players_and_uses_cover():
    assert c.pick_heuristic(scene([player(mass=400, gap=300, popped=True)]))["order"] == "harvest #7"
    v = {"x": 0, "y": 0, "mass": 400, "gap": 200, "pops_us": False, "hides_us": True}
    d = c.pick_heuristic(scene([player(mass=3000, gap=100, eat_us=True, eat=False)], [v]))
    assert d["order"] == "hide"


def test_options_offer_virus_split_on_big_players():
    v = {"x": 0, "y": 0, "mass": 100, "gap": 300, "pops_us": True, "hides_us": False}
    o = c.options(scene([player(mass=3000, gap=900, eat_us=True, eat=False)], [v]))
    assert any(x["order"].startswith("virus-split") for x in o)


def test_parse_reply_validates():
    assert c.parse_reply('sure: {"order": "hunt #5", "ttl": 6, "split": "off"}')["order"] == "hunt #5"
    assert c.parse_reply('{"order": "rm -rf /"}')["order"] == "free"
    assert c.parse_reply("no json")["order"] == "free"
    assert c.parse_reply('{"order": "goto 10 -20", "ttl": 999}')["ttl"] == 30.0


class FakeCtl:
    def __init__(self, s):
        self.s, self.sent = s, []

    def scene(self, agent):
        return self.s

    def cmd(self, line):
        self.sent.append(line)
        return "ok"


def test_jev_backend_maps_choice(tmp_path):
    b = c.JevBackend.__new__(c.JevBackend)
    b.calls = 0

    class Jev:
        def ask(self, state, qs):
            assert "mode" in qs and "p0" in qs["mode"]["criteria"]
            return {"mode": {"choice": "p0", "probabilities": {"p0": 0.9}}, "split": {"noul": 0.8}}

    b.jev = Jev()
    d = b.decide(scene([player(mass=400, gap=300, split_kill=True)]))
    assert d["order"] == "hunt #7" and d["split"] == "on"


def test_commander_step_applies_and_logs(tmp_path):
    ctl = FakeCtl(scene([player(mass=400, gap=300)]))
    log = tmp_path / "c.jsonl"
    cm = c.Commander(ctl, c.HeuristicBackend(), ["a-0"], log_path=log)
    cm.step("a-0")
    assert ctl.sent[0].startswith("order a-0 hunt #7 ttl=")
    assert json.loads(log.read_text().splitlines()[0])["decision"]["order"] == "hunt #7"
