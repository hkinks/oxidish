"""Tunable behaviour parameters (the genome evolved by arena/evolve.py).

Defaults reproduce the original hand-tuned bot. BOUNDS gives the search range
for each field; evolution works in the normalised [0, 1] box over them.
"""

from dataclasses import asdict, dataclass, fields


@dataclass
class Params:
    # threat handling
    danger_margin: float = 250.0     # extra gap kept from threats, world units
    split_range: float = 650.0       # assumed reach of a split lunge (ours and theirs)
    threat_force: float = 30.0       # repulsion strength inside a danger radius
    flee_margin: float = 1.0         # flee when a threat is within this x danger radius
    # weights per mode: food, prey, threat
    farm_prey: float = 0.5
    farm_threat: float = 1.5
    hunt_food: float = 0.3
    hunt_threat: float = 1.5
    flee_threat: float = 4.0
    target_bonus: float = 6.0        # extra pull toward the hunt target
    food_pull: float = 3.0           # pull toward the next pellet waypoint
    # hunting and splitting
    hunt_range: float = 900.0
    min_prey_frac: float = 0.1       # ignore prey lighter than this fraction of our mass
    split_safety: float = 2.0        # no split if a threat is within this x danger radius
    split_reach: float = 0.8         # split only when target gap < this x split_range
    split_max_cells: float = 1.0     # split only while we have at most this many cells
    # viruses and borders
    virus_avoid: float = 150.0
    border_margin: float = 200.0
    # food routing
    switch_margin: float = 1.25
    # virus skills (bot/tactics.py); *_enable genes switch a skill on above 0.5
    shoot_enable: float = 0.45       # off by default: evolution decides
    shoot_range: float = 600.0       # max gap from us to the virus we feed
    shoot_align: float = 0.9         # cos(angle) between us->virus and virus->target
    shoot_reach: float = 900.0       # max virus->target distance
    shoot_min_mass: float = 300.0    # only shoot when we can spare the ~136 mass
    shoot_target_mult: float = 2.0   # target must be this x the mass a virus pops
    hide_enable: float = 0.45
    hide_range: float = 500.0        # max gap to a virus worth hiding behind
    # combat
    lead: float = 0.0                # steps of target motion to lead when chasing / split-aiming
    virus_detour: float = 1.0        # >0.5: steer around viruses we'd pop on (else old push-away)
    double_split_enable: float = 0.45  # tactic switches start off: evolution decides
    double_split_reach: float = 1.8  # max target gap as a multiple of single-split reach
    selffeed_enable: float = 0.45
    selffeed_min_mass: float = 300.0 # only worth it with real mass spread over pieces
    selffeed_frac: float = 0.7       # feed while our biggest piece holds less than this share
    # big cells: pellets stop paying (decay eats more than food brings), only eating players does
    big_hunt: float = 0.0            # off: lost 33-67% to plain farming in long 2x-food matches. >0.5: above big_mass, hunt range grows and the prey floor stops rising
    big_mass: float = 1500.0
    # split pieces (from the long-game loss analysis: about half the big losses happen while split, and a
    # quarter of champ's to players smaller than it overall, i.e. its pieces)
    half_split_safety: float = 0.0   # >0.5: judge split safety against the halves we split into
    piece_threats: float = 0.0       # >0.5: while split, every piece's threats push (by its mass share)
    # caution ramp: above caution_mass total mass (0 = off) the bot plays the cautious genes below (and
    # the half-split check): reckless growth while there is little to lose, safety once there is
    caution_mass: float = 0.0
    flee_margin_big: float = 1.0
    split_safety_big: float = 2.0
    # flee by planning (bot/steer.flee_heading): try headings, keep the one that stays out of every
    # threat's reach longest, with walls clamping our path. Pushing straight away from a threat pinned
    # us on walls: 82% of the rule bots' lost mass was lost at a wall (2026-10-06 loss mining).
    flee_plan: float = 0.0
    # >0.5: threats to any piece holding >= PIECE_SHARE of our mass trigger the flee, and the flee planner
    # protects every such piece (only the biggest cell counted before: 56% of safe_fp's lost mass was lost
    # while split)
    flee_pieces: float = 0.0
    # contenders: a player too small to eat us (mass > contender x ours) is one meal away from it; within
    # contender_gap it counts as a threat (0 = off). Hunting next to one handed it the meal and then us:
    # 30% of r300_fp's lost mass (2026-10-06 mining)
    # >0.5: the mouse goes AIM_DISTANCE beyond our farthest cell along the heading (else from the main cell
    # only: split pieces further ahead turned back into the main cell and jammed; ramp300 parked so)
    aim_all_cells: float = 1.0
    contender: float = 0.0
    contender_gap: float = 200.0
    # wall_danger: extra danger gap (world units) at a wall, fading to 0 at 700 from it. Threats count as
    # closer while we are pinned on a wall, so we flee, and stop farming near them, earlier (74% of
    # the rule bots' lost mass is lost fleeing within 400 of a wall, 2026-10-06 mining). 0 = off.
    wall_danger: float = 0.0
    # wall_wary: extra wall push-back margin while a player heavier than wall_wary_ratio x ours is within 1500
    # (a wider margin all the time was much worse: it costs the wall pellets when nobody is about). 0 = off.
    wall_wary: float = 0.0
    wall_wary_ratio: float = 0.8
    # >0.5: the virus detour also covers our split pieces heavier than a virus (only the main cell's path was
    # checked: 42 of 53 own virus pops happened while split, mostly by trailing pieces; 2026-10-07)
    piece_detour: float = 0.0
    # >0: split by expected value instead of the fixed split_reach / split_safety checklist: within the full split
    # range, split when the mass our half can eat around the landing spot > split_ev x the mass at risk (a half that
    # some cell could eat there, or the half we keep). The fixed rules shied away from split enemies' pieces.
    split_ev: float = 0.0
    # >0.5: in flee mode, when every heading is caught within the flee look-ahead (flee_heading), split away along
    # the best one (an escape split: half the mass gets out instead of all of it being eaten; the user's suggestion)
    escape_split: float = 0.0
    # >0: while fleeing, a new heading 60-100 deg off the last flee heading is blended toward it with this
    # weight (commit to one escape). A live c3_4 flipped between two headings 80 deg apart for 2 s at a wall and
    # was eaten (human session 2026-10-07)
    flee_smooth: float = 0.0
    # >0.5: a threat no longer vetoes virus shooting when it is itself a shot target behind a virus (closer to us than
    # it, lined up, within shoot_reach): the virus shields us while we feed it. Mined from the human's play
    # (2026-10-07): his shots were at rivals his own size or bigger (median 1.06x), 640-1180 beyond the virus
    shoot_threats: float = 0.0
    # giant stage: above giant_mass total mass (0 = off) the bot plays these giant_* genes (on top of the caution
    # ramp), so late-game play (an endless world's giants) can be tuned without touching the early game
    giant_mass: float = 0.0
    giant_hunt_range: float = 900.0
    giant_min_prey_frac: float = 0.1
    giant_split_safety: float = 2.0
    giant_split_reach: float = 0.8
    giant_farm_threat: float = 1.5
    giant_flee_margin: float = 1.0
    giant_danger_margin: float = 250.0
    giant_target_bonus: float = 6.0
    giant_food_pull: float = 3.0
    # virus staging (bot/tactics.py): >0.5: with the shoot skill on and no shot lined up, walk round a virus
    # (within shoot_walk) to its far side from a big player, shoot_standoff
    # from its edge, so the next tick's shot goes through the virus at them. Off by default; not evolved.
    shoot_stage: float = 0.0
    shoot_walk: float = 1200.0
    shoot_standoff: float = 90.0
    shoot_safety: float = 0.7        # the shot is called off while a threat is within this x its danger radius
    shoot_max_cells: float = 1.0     # shoot only while we have at most this many cells

    def cautious(self) -> "Params":
        """These genes with the big-cell caution applied (caution ramp)."""
        return Params(**{**asdict(self), "flee_margin": self.flee_margin_big,
                         "split_safety": self.split_safety_big, "half_split_safety": 1.0})

    def giant(self) -> "Params":
        """The genes for the giant stage: the cautious genes (if the caution ramp is on) with the giant_* overrides."""
        base = self.cautious() if self.caution_mass > 0 else self
        return Params(**{**asdict(base), **{k[len("giant_"):]: v for k, v in asdict(self).items()
                                            if k.startswith("giant_") and k != "giant_mass"}})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


BOUNDS = {
    "danger_margin": (0, 900), "split_range": (300, 1000), "threat_force": (5, 120),
    "flee_margin": (0.2, 2.5), "farm_prey": (0, 3), "farm_threat": (0.2, 6),
    "hunt_food": (0, 2), "hunt_threat": (0.2, 6), "flee_threat": (0.5, 12),
    "target_bonus": (1, 20), "food_pull": (0.3, 12), "hunt_range": (200, 2500),
    "min_prey_frac": (0, 0.6), "split_safety": (0.5, 4), "split_reach": (0.3, 1.2),
    "split_max_cells": (1, 6), "virus_avoid": (20, 500), "border_margin": (30, 800),
    "switch_margin": (1.0, 2.5),
    "shoot_enable": (0, 1), "shoot_range": (150, 1200), "shoot_align": (0.6, 0.995),
    "shoot_reach": (300, 1600), "shoot_min_mass": (150, 1500),
    "shoot_target_mult": (1, 4),
    "hide_enable": (0, 1), "hide_range": (100, 1200),
    "lead": (0, 30), "virus_detour": (0, 1),
    "double_split_enable": (0, 1), "double_split_reach": (1.1, 2.5),
    "selffeed_enable": (0, 1), "selffeed_min_mass": (50, 3000), "selffeed_frac": (0.3, 0.95),
    "big_hunt": (0, 1), "big_mass": (300, 6000),
    "half_split_safety": (0, 1), "piece_threats": (0, 1),
    "caution_mass": (0, 3000), "flee_margin_big": (0.2, 2.5), "split_safety_big": (0.5, 4),
    "flee_plan": (0, 1), "flee_pieces": (0, 1), "contender": (0, 1), "contender_gap": (0, 800),
    "aim_all_cells": (0, 1), "wall_danger": (0, 800), "wall_wary": (0, 800), "wall_wary_ratio": (0.3, 1.3), "piece_detour": (0, 1), "split_ev": (0, 4), "escape_split": (0, 1), "flee_smooth": (0, 0.95), "shoot_threats": (0, 1),
    "giant_mass": (0, 12000), "giant_hunt_range": (200, 3000), "giant_min_prey_frac": (0, 0.6),
    "giant_split_safety": (0.5, 4), "giant_split_reach": (0.3, 1.2), "giant_farm_threat": (0.2, 6),
    "giant_flee_margin": (0.2, 2.5), "giant_danger_margin": (0, 900), "giant_target_bonus": (1, 20),
    "giant_food_pull": (0.3, 12),
}

DEFAULT = Params()


def to_unit(p: Params) -> list[float]:
    return [(getattr(p, k) - lo) / (hi - lo) for k, (lo, hi) in BOUNDS.items()]


def from_unit(x) -> Params:
    vals = {}
    for (k, (lo, hi)), v in zip(BOUNDS.items(), x):
        vals[k] = lo + min(max(float(v), 0.0), 1.0) * (hi - lo)
    return Params(**vals)
