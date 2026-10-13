# Sparring opponents: hunters (2026-10-07)

Why: every opponent in the field (champ, r2g9, safe, ramp_r2, ramp300) is the same potential-field rule bot with
different genes: cautious farmers that rarely hunt and never use viruses. c3_4 is evolved against them, so they
give little pressure. This adds opponents that hunt and use viruses.

## What there is

| Name | Spec | What |
|---|---|---|
| `hunter_hunt` | `bots/specs/sparring/hunter_hunt.json` | c3_4 farming genes + chase everything from 1800 away, split-kill, double split |
| `hunter_virus` | `bots/specs/sparring/hunter_virus.json` | + stages and feeds viruses at big players, hides behind viruses |
| `hunter_reckless` | `bots/specs/sparring/hunter_reckless.json` | + chases everything (no prey floor), up to 4 pieces |

Use them like any genome: `training.eval.tournament --engine oxidish-server --agents
hhunt=bots/specs/sparring/hunter_hunt.json:1 ...`, `training.eval.endless --agents ...`, `oxidish-server match|serve --agents ...`.
`training.eval.make_sparring` regenerates the hunter files.

## Hunter genes (`bots/brain/src/policy.rs`, all default off)

- `shoot_stage` > 0.5: with the shoot skill on and no shot lined up, walk round a virus (within `shoot_walk`) to the
  side away from a big player, `shoot_standoff` from its edge, and hold position while
  shooting (aim just past our edge instead of at the virus).
- `shoot_safety` (0.7): shot called off while a threat is within this x its danger radius.
- `shoot_max_cells` (1): shoot while we have at most this many cells.
Not in `BOUNDS`, so evolution is unchanged; existing genomes behave identically (parity re-run on c3_4 / champ).

## Results

**Parity (Python vs Rust, `training.eval.rust_tools dump` on real node-server matches + `oxidish-bot parity`):**
26996 decisions (hunter_virus, c3_4, champ and a heuristic bot since removed), 0 aim mismatches, 0 split/eject
mismatches, max |aim diff| 5e-12; 16000 fuzz decisions (hunter genes with staging, c3_4, champ) also 0 mismatches.

**15-min matches** (`training.eval.tournament --engine oxidish-server --matches 24 --ticks 22500 --food-mult 2`, one copy each, 8 server bots, seeds 5000+):

| bot | log mass | avg mass | deaths | kills (cells) | virus shots | virus pops |
|---|---|---|---|---|---|---|
| c3_4 | 6.95 | 1339 | 3.0 | 62 | 0 | 0 |
| ramp300 | 6.89 | 1199 | 4.0 | 61 | 0 | 0 |
| rampr2 | 6.76 | 1184 | 4.3 | 63 | 0 | 0 |
| champ | 6.71 | 1133 | 4.7 | 45 | 0 | 0 |
| r2g9 | 6.69 | 1058 | 4.8 | 69 | 0 | 0 |
| safe | 6.58 | 1061 | 3.8 | 41 | 0 | 0 |
| hunter_hunt | 6.53 | 946 | 4.8 | 98 | 0 | 0 |
| hunter_virus | 5.94 | 452 | 6.1 | 87 | 1.8 | 1.5 |
| hunter_reckless | 5.75 | 356 | 8.5 | 89 | 1.4 | 1.3 |

c3_4 still beats every one of them head to head (71-96% of matches vs the hunters). hunter_hunt is the top
killer of the other bots (13-18% of c3_4's, champ's, ramp's lost mass). Virus shots are rare (a staged feed needs ~8 aligned ejects while a
big player stays put): ~2 per hunter_virus per 15 min, nearly every shot popped someone.

**Endless worlds** (`training.eval.endless`, 4 worlds x 240 game-min, windows of 10 min, first 30 min skipped, seeds 800+;
field = c3_4 x2, champ, r2g9, safe, rampr2, ramp300; "+ sparring" adds one each of hunter_hunt,
hunter_virus, hunter_reckless and a heuristic bot since removed):

| | c3_4 leader share | c3_4 dominance (heaviest family) | c3_4 mean mass |
|---|---|---|---|
| field only | 32% | 22% (19/88) | 3411 |
| + sparring | 38% | 31% (27/88) | 3404 |

Family mean mass with sparring: c3_4 3404, ramp300 2770, champ 2463, rampr2 2122, r2g9 2038, safe 1783, hunter_hunt 1202,
hunter_virus 471, hunter_reckless 424 (leader shares: c3_4 38%, ramp300 16%, champ 14%, r2g9 12%, rampr2 8%,
safe 7%, hunter_hunt 2%). So the sparring bots do not lower c3_4 (its mass is unchanged); they take mass
from the other farmers, which is why its leader share goes up. They are pressure on everybody, not a stronger
bot to beat: none is heavy enough to lead. A hunter that is also a good farmer would be the next step.
