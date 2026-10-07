"""Regenerate the hunter sparring genomes in bot/sparring/ from the c3_4 farming genes.

    uv run python -m arena.make_sparring

Hunters keep the evolved farming/flee genes of bot/c3_4.json (so they grow instead of dying early) and turn
the aggression genes up: they chase anything edible from far away, split-kill and double-split, and the virus
variants stage and shoot viruses at big players (new genes shoot_stage / shoot_walk / shoot_standoff /
shoot_safety / shoot_max_cells in bot/params.py, bot/tactics.py).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bot" / "sparring"

CHASE = dict(hunt_range=1800, min_prey_frac=0.05, farm_prey=1.5, target_bonus=12, split_max_cells=2,
             split_reach=1.0, split_safety=1.0, double_split_enable=1.0)
VIRUS = dict(shoot_enable=1.0, hide_enable=1.0, shoot_stage=1.0, shoot_target_mult=1.6, shoot_min_mass=200.0,
             shoot_range=700.0, shoot_align=0.94, shoot_standoff=35.0, shoot_reach=1000.0, shoot_walk=1400.0,
             shoot_safety=0.4, shoot_max_cells=3)
RECKLESS = dict(min_prey_frac=0.0, split_max_cells=3, hunt_range=2500)

VARIANTS = {
    "hunter_hunt": CHASE,                                   # chases / split-kills / double-splits, no virus play
    "hunter_virus": {**CHASE, **VIRUS},                     # + feeds viruses at big players (staged shots), hides
    "hunter_reckless": {**CHASE, **VIRUS, **RECKLESS},      # + chases everything, up to 4 pieces
}


def main() -> None:
    base = json.loads((ROOT / "bot" / "c3_4.json").read_text())
    OUT.mkdir(exist_ok=True)
    for name, genes in VARIANTS.items():
        spec = json.loads(json.dumps(base))
        spec["params"].update(genes)
        (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print(f"bot/sparring/{name}.json")


if __name__ == "__main__":
    main()
