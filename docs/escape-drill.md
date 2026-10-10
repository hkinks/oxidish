# Escape drill

How do our bots die when a bigger player chases them, and does a better flee fix it? The recorded games of the user
(`docs/human/2026-10-09-50bots.md`) say: of the mass the human took from bots, 48% was lost after fleeing (170 of 226 scenes
at a wall, 101 in a corner), 29% while split into pieces, 10% without fleeing at all; against all players 50% / 33%. The typical
scene: a 6.5-7k bot flees a 10k human in a straight line with the gap near 1000 for 3 s, is herded into the wall and killed.

The drill is a seeded, repeatable version of that scene: `ogar escape` (`rust/ogar/src/escape.rs`), tables from
`arena/escape_report.py`. It is separate from `ogar drill` (own subcommand and file) to keep the merge with the harvest/hunter work easy.

## Running it

    # models are gitignored: copy runs/vision/export/*.onnx and runs/vision/ort from the main checkout; nets need
    # OGAR_DLL_DIRS (torch wheel lib) and OGAR_VISION_EP=cuda_plain, --threads 2 (the faster "cuda" backend died in cuDNN with several threads)
    cargo build --release -j 2 --manifest-path rust/ogar/Cargo.toml        # your own CARGO_TARGET_DIR
    ogar escape --agents c3_4=bot/c3_4.json evade=bot/escape/evade.json oracle=oracle \
                --our-mass 1000,3000,7000 --ratio 1.4,2,4 --gap 1200,2000 --place open,wall,corner [--virus 0,1] [--pieces 1,2,4] \
                --reps 30 --seed 1 --threads 3 --horizon 60 --out runs/esc/x.json
    python -m arena.escape_report runs/esc/x.json [--base c3_4] [--what pooled|cells|methods]

Every list option is a grid (one cell per combination); rep `i` of every agent and cell uses seed `seed + i` (paired). `OGAR_ESC_TRACE=1`
prints a line per second. Tests: `cargo test --release escape` (ogar) and `cargo test --release evade` (arena_client).

## The scene

Map 7000, 2x food, no server bots, no stray viruses. Our agent has mass M (1000 / 3000 / 7000) and a scripted chaser of `ratio` x M
(1.4 / 2 / 4, capped at the maximum cell mass 22500; at 7000 x 4 the chaser is 22.5k = 3.2x) stands `gap` away (edge to edge, 1200 / 2000;
shrunk and rotated when it would not fit in the map; the real gap is `start_gap` in the rows).

* `open`: around the map centre, random direction. `wall`: our edge 300 from a wall, the chaser on the inner side (+-50 deg). `corner`: 300 from two walls, the chaser towards the middle.
* `--virus 1`: one virus about 700 from us at roughly a right angle to the line chaser-us (the only virus of the world).
* `--pieces 2|4`: our mass starts split in 2 or 4 cells (split commands thrown sideways, then 1 s to settle) before the chaser appears; the gap is measured from the whole-mass cell.

**The chaser** (every 2 ticks, sees everything; `escape.rs::Chaser`):
* pursues the biggest of our cells it can eat with a lead (intercept time solved against the target velocity of the last decision, capped at 60 ticks);
* herds: when the target is within 2000 of a wall and the chaser is not yet on the inner side, the aim is pushed `min(0.3 d, 600)` inwards, so it comes in from the open side;
* split-kills when the target is inside its split reach and a half cell can eat it; double-splits (second command 4 ticks later) when only that reaches (quarter cell still eats, distance up to 1.8 x reach);
  the real reach: a split child (size / sqrt 2) is thrown `780 * size^0.0122` (the server boost, travelled in full), and eats a cell whose centre is closer than `child size - target size / 3`;
  10 ticks of the target's flight are taken off;
* recombines: when it has pieces and the target is farther than 2 reaches, the aim is the centroid of its pieces;
* never pops on a virus (goes round it at 0.9 x its size + the virus).

Consequence of the rules: a split child (half the mass) must still eat us (size ratio 1.15, mass ratio 1.32), so a chaser needs about **2.65x our mass** before a split-kill exists
(`x4` only); at 1.4x and 2x the split reach is geometry without a lethal effect, and the chaser kills by catching.

**Metrics** (per rep, 60 s): survival at 30 / 60 s (any cell alive), survival time (min(death, 60 s)), mass kept (total mass / start, 0 when dead),
seconds with the chaser's split reach on our biggest cell (`in_reach`; `in_lethal` when the half can eat) and in plain eating range, touched a wall (an edge within 100) or a corner,
where it died (`wall`/`corner`/`open`; `pieces` if it had several cells in the last 3 s), and the escape method of a survivor (first that applies:
`outgrew` the chaser can no longer eat us, `virus` the chaser popped on a virus, `split` we split, `wall_slide`, `outrun`). 95% intervals: Wilson for rates, +-1.96 se for means,
**paired** differences (same condition and seed) for the time comparison.

## Agents measured

c3_4 (`bot/c3_4.json`), `plan` (`bot/plan_c34.json`), the wall genes (`nowall` = c3_4 with `wall_danger = wall_wary = 0`; c3_4 itself has 370 / 428),
other existing flee genes on c3_4 (`flee_plan`, `flee_pp` = + `flee_pieces`, `esc_split`; specs in `bot/escape/`), the new `evade*` genes, the scripted `oracle` of the
harvest drill (flees along the clearest of 16 directions, walls counted as threats), and the vision nets `p6`, `p3` (`bot/vision_p6.json`, `vision_p3.json`).
The variants `runs/ab/c34_*` of the earlier wall-gene searches were not run separately (c3_4 already carries the wall genes of that search; `nowall` is the ablation).

## Results

### Main grid: M 1000 / 3000 / 7000 x ratio 1.4 / 2 / 4 x gap 1200 / 2000 x open / wall, 30 reps per cell (1080 per agent)

| agent | survive 30 s | survive 60 s | survival time s | paired dt vs c3_4 s | mass kept 60 s | in split reach s | touched wall |
|---|---|---|---|---|---|---|---|
| c3_4 | 86% [84-88] | 26% [23-28] | 46.1+-0.8 | - | 0.24+-0.02 | 5.8+-0.5 | 88% |
| plan | 86% [84-88] | 26% [23-29] | 46.2+-0.7 | +0.1+-0.1 | 0.24+-0.02 | 6.0+-0.5 | 89% |
| nowall | 92% [90-93] | **40% [37-43]** | 49.7+-0.7 | **+3.6+-0.5** | 0.37+-0.03 | 8.0+-0.6 | 100% |
| evade | 90% [88-92] | 33% [30-36] | 48.9+-0.7 | +2.8+-0.5 | 0.31+-0.03 | 5.8+-0.4 | 89% |
| flee_plan | 93% [91-94] | 29% [27-32] | 48.6+-0.7 | +2.5+-0.4 | 0.27+-0.02 | 3.1+-0.2 | 100% |
| esc_split | 87% [85-89] | 25% [23-28] | 46.2+-0.7 | +0.1+-0.3 | 0.21+-0.02 | 5.7+-0.4 | 88% |
| oracle | 91% [89-92] | 32% [29-34] | 48.2+-0.7 | +2.1+-0.5 | 0.29+-0.03 | 9.0+-0.7 | 100% |

(`plan` equals c3_4 here: the planner's trigger never fires in these scenes. `evade` = c3_4 + `evade 1, evade_wall 1`.)

Survive 60 s (Wilson) / mean survival time, by factor:

| mass | c3_4 | nowall | evade | flee_plan | esc_split | oracle |
|---|---|---|---|---|---|---|
| 1000 | 8% [6-11] / 38s | **31% [26-36] / 45s** | 7% [5-10] / 38s | 2% [1-4] / 40s | 13% [10-17] / 39s | 9% / 39s |
| 3000 | 23% [19-27] / 47s | **36% [31-41] / 50s** | 31% [26-36] / 52s | 26% [22-31] / 50s | 22% / 47s | 23% / 49s |
| 7000 | 46% [41-52] / 53s | 52% [47-57] / 54s | 62% [57-67] / 56s | 60% [55-65] / 56s | 39% / 52s | 63% [58-68] / 56s |

| ratio | c3_4 | nowall | evade | flee_plan | esc_split | oracle |
|---|---|---|---|---|---|---|
| 1.4 | 18% [14-22] / 44s | **36% [31-41] / 48s** | 24% [20-29] / 45s | 22% / 47s | 16% / 44s | 23% / 46s |
| 2 | 29% [24-34] / 47s | **48% [43-53] / 52s** | 37% [32-42] / 50s | 30% / 49s | 29% / 47s | 30% / 48s |
| 4 | 31% [26-36] / 48s | 35% [31-40] / 49s | 38% [33-43] / 51s | 37% / 50s | 31% / 48s | **42% [37-47] / 51s** |

| place | c3_4 | nowall | evade | flee_plan | esc_split | oracle |
|---|---|---|---|---|---|---|
| open | 39% [35-43] / 53s | **51% [47-55] / 55s** | 44% [40-48] / 53s | 50% [46-54] / 55s | 36% / 53s | 50% / 54s |
| wall | 13% [10-16] / 40s | **28% [24-32] / 44s** | 23% [19-26] / 45s | 9% [7-11] / 42s | 14% / 39s | 13% / 42s |
| gap 1200 | 21% [18-24] / 44s | **34% [31-39] / 48s** | 27% [24-31] / 47s | 23% / 47s | 20% / 44s | 26% / 46s |
| gap 2000 | 31% [27-35] / 48s | **45% [41-49] / 51s** | 39% [35-43] / 51s | 36% / 50s | 30% / 48s | 38% / 50s |

Corner (M 3000, ratio 1.4 / 2 / 4, gap 1200 / 2000, 180 reps per agent): everything dies. Survive 30 s: c3_4 9% [6-14], nowall 19% [14-26], evade 31% [24-38], flee_plan 28%, oracle 2%;
survive 60 s 0-2% for all; survival time c3_4 24.7+-1.3 s, nowall 29.8+-1.8, evade 27.9+-1.6, flee_plan 26.1+-0.5. A bot that starts in a corner with the chaser in front of it cannot get out.

Virus (M 3000, ratio 2 / 4, gap 1200 / 2000, open + wall, 240 reps per agent):

| agent | survive 30 s | survive 60 s | survival time s | dt vs c3_4 | mass kept | survivors by method | deaths at |
|---|---|---|---|---|---|---|---|
| c3_4 | 92% [88-95] | 40% [34-47] | 50.7+-1.4 | - | 0.54+-0.11 | wall_slide 63, outgrew 18, virus 13, split 3 | corner 67, wall 65, pieces 9, open 2 |
| nowall | 94% [90-96] | 42% [36-48] | 51.2+-1.4 | +0.5+-0.9 | 0.49+-0.10 | wall_slide 72, outgrew 12, virus 12, split 5 | corner 125, pieces 8, wall 6 |
| evade | 94% [90-96] | 50% [44-56] | 52.5+-1.4 | +1.8+-1.1 | 0.62+-0.11 | wall_slide 83, virus 19, outgrew 13 | wall 101, open 18, corner 1 |
| evade_v | 94% [90-96] | 50% [44-56] | 52.5+-1.4 | +1.8+-1.1 | 0.69+-0.12 | wall_slide 81, outgrew 21, virus 15 | wall 100, open 19, corner 1 |
| oracle | 71% [65-76] | 23% [18-29] | 41.5+-2.0 | -9.3+-2.1 | 0.25+-0.08 | wall_slide 22, split 20, outgrew 8 | pieces 130, corner 43, wall 9 |

In 5 to 8% of the virus reps a survivor owes its life to the chaser popping on the virus ("virus" method; the scripted chaser avoids viruses but not perfectly); `evade_virus`
(a chaser model that goes round viruses, so a virus on the line works as a shield) did not move the survival rate (identical 50%), only the mass kept (0.62 -> 0.69, inside the interval).

Pieces (M 3000, ratio 2 / 4, gap 1200, open + wall, our mass starts in 2 or 4 cells, 240 reps per agent):

| agent | survive 30 s | survive 60 s | survival time s | dt vs c3_4 | mass kept | deaths at |
|---|---|---|---|---|---|---|
| c3_4 | 78% [73-83] | 42% [36-48] | 44.2+-2.1 | - | 0.38+-0.07 | pieces 135 |
| nowall | 80% [74-84] | 44% [38-50] | 45.2+-2.2 | +1.0+-2.1 | 0.38+-0.07 | pieces 130 |
| evade (falls back to the old flee in pieces) | 77% [71-82] | 39% [33-45] | 43.3+-2.2 | -0.9+-1.1 | 0.35+-0.07 | pieces 144 |
| flee_plan | 80% [74-84] | 30% [25-36] | 44.6+-2.0 | +0.4+-2.2 | 0.29+-0.06 | pieces 166 |
| flee_pp (+ flee_pieces) | 78% [73-83] | 34% [28-40] | 45.0+-2.0 | +0.7+-2.2 | 0.32+-0.06 | pieces 156 |
| oracle | 57% [50-63] | 28% [23-34] | 35.1+-2.9 | -9.1+-3.2 | 0.29+-0.08 | pieces 167 |

In an earlier run (40 reps per cell, 320 per agent) the search with every big piece guarded (`evade_pieces 1`) was worse (survive 30 s 63% vs c3_4's 78%, dt -5.4+-2.5 s) and aiming for a closer group
(`evade_pieces 2`, "the direction that merges") much worse (45% at 30 s, dt -12.2+-2.7 s; I did not find out why, my guess is that the pieces behind run into the chaser). Pieces cannot merge for ~30 s anyway (the merge timer), so closing up
only brings them together as food. **No flee-in-pieces rule helped; the shipped `evade` leaves a split bot to the old flee.**

### Vision nets (18 cells: M 1000 / 3000 / 7000 x 1.4 / 2 / 4 x gap 1200 x open / wall, 20 reps, CUDA `cuda_plain`)

| agent | survive 30 s | survive 60 s | survival time s | paired dt vs c3_4 s | mass kept | in split reach s |
|---|---|---|---|---|---|---|
| c3_4 | 86% [82-89] | 21% [17-26] | 44.8+-1.3 | - | 0.20+-0.04 | 6.4+-0.8 |
| p6 | 85% [81-88] | 26% [22-31] | 45.8+-1.3 | +1.0+-1.0 | 0.25+-0.04 | 8.6+-1.0 |
| p3 | 84% [80-87] | 21% [17-26] | 44.2+-1.3 | -0.7+-0.9 | 0.20+-0.04 | 9.0+-1.0 |

The nets flee about like c3_4 (neither differs by more than the noise) and spend more time inside the chaser's split reach (8.6-9.0 s vs 6.4). Their weak cells are the ones where
c3_4 is also weak (walls, ratio 4 at M 7000 where p3 falls to 45% in the open against c3_4's 100%: 0.45 [p3] / 0.80 [p6] / 1.00 [c3_4] at 60 s in `our7000/x4/gap1200/open`).
Not run: the nets with `reflex` layers, or the nets on the corner / virus / pieces grids.
**This is a training scenario candidate for the vision agent**: a cheap, seeded loss that matches the human losses (deaths in the walls/corners, survival time as dense reward, `start_gap`/`ratio` as curriculum).

## Diagnosis

1. **The chaser wins by geometry, not speed.** A bigger cell is slower (`88 * size^-0.439` per tick): a 1.4x chaser moves at 0.93 of our speed, a 2x chaser at 0.86. In open space the bot
   holds the gap (`in_reach` only 6-9 s of 60), but the arena is 7000 wide: a straight run lasts ~50 s, and every turn at a wall lets a lead-pursuing chaser cut the corner. Survival at 30 s is 86%,
   at 60 s 26%; at a wall start 13% (open 39%). Deaths of c3_4 (M 1000 / 3000, ratio 1.4 / 2 / 4, gap 1200, open + wall, 360 reps, 320 deaths): wall 163, open 80, corner 76, i.e. 75% at a wall or corner (in the recorded human scenes of fleeing bots 170 of 226 were at a wall).
2. **A split-kill needs about 2.65x our mass; in 2/3 of these cells it does not exist** (`in_lethal` is 0.0 s at 1.4x and 2x, 0.2-0.8 s at 4x). The human scenes (10k vs 6.5-7k, 1.45x) are of this kind:
   the loss is the catch after the wall, not a split. "Keep out of the split reach" matters only against 4x chasers; there the bots spend 0.7-1.0 s in reach and `evade` is about as good as the best (38% at 60 s vs 31%).
3. **The wall genes of c3_4 hurt in this scene.** `wall_danger = 370, wall_wary = 428` (found by a gene search on endless leader share) widen the danger radius and the wall margin near walls. With the chaser on the
   inner side the margin pushes the bot **away from the wall towards the chaser**. Without them (`nowall`): survival 60 s 26% -> 40%, +3.6+-0.5 s, and the bot lives longer in every
   cell group except M 7000 x 4 (equal). The effect is largest for small bots (M 1000: 8% -> 31%) and at wall starts (13% -> 28%). The explanation is a hypothesis (I saw the aim vectors of a few traces, not an ablation of each gene alone);
   `nowall` deaths move to the corners (corner 227, wall 30, vs c3_4 wall 163 / open 80 / corner 76).
4. **The look-ahead flee (`evade`) helps against big chasers and big bots, not against small bots.** M 7000: 46% -> 62% at 60 s; M 3000: 23% -> 31%; M 1000: 8% -> 7% (no change).
   At 1000 the c3_4 / `nowall` bot gets its time by something the search ignores (it grows: `outgrew` for a few reps, and it runs along the walls; I did not isolate it). A food term (`evade_food`, tie-break towards pellets)
   changed nothing (identical rows).
5. **Existing flee genes**: `flee_plan` (the old straight-line projection, 32 headings) is about as good as `evade` at M 7000 and in the open (+2.5 s) but is the worst at walls (9%); `esc_split` (split away as a last resort) changes nothing
   in the main grid (+0.1 s; it spends 0.8 s in lethal reach at 4x against 0.2-0.4 for the others); `flee_pieces`/`flee_plan` in pieces: no gain or worse.
   The scripted `oracle` (flee along the clearest of 16 directions) is as good as `evade` in the open and at M 7000, and the worst agent in the virus and pieces cells (it splits at the wrong time).
6. **The lookahead planner is not involved.** `plan` is identical to c3_4 in every cell (its trigger needs a predicted loss; the chaser stays outside its horizon until the last second). It was not tuned for this scene.

## The 2026-10-09 hunters session (`docs/human/2026-10-09-hunters.md`): what the `no_flee` losses are

Of the mass the bots lost to the human: pieces 35% (22 scenes), no_flee 31% (17 scenes, 20990 mass), fled 27%. I looked at the 17 `no_flee` scenes of `scenes-1791576644.jsonl.gz`:

* **All 17 are vision nets**: p6 6, p3 2, p6h 9 (the rule bots c34h, planhv, plan, r2g9, ramp300, champ, c3_4 are in `fled` / `pieces` only: none of their ~40 human scenes). The classifier calls a scene `no_flee` when the rule mode of the
  last second is not `flee`; the nets record no mode (`-`), so every net loss with a visible eater is `no_flee` by construction. The number is therefore "nets that lost to the human", not "bots that decided not to flee".
* **They do not run.** The centre distance to the human falls steadily over the last 0.8 s in 12 of 17 (about 10-15 per decision step, e.g. 486 -> 289, 259 -> 190, 562 -> 476), in 4 it holds for half a second and then falls (625, 1250, 1372, 1392), in 1 it grows at the end (1452).
  The human is 1.4x to 300x the bot's mass (median about 2x; p6h scenes: 3080 vs 4316 at the wall; 1421 vs 5655 in a corner; 615 vs 6642). 12 of 17 happen with the bot's edge within 150 of a wall (tag `wall`).
* **The hunter's commit phase does not explain it by itself**: p6 and p3 (no hunter) show the same picture (8 of 17), and in the drill `p6h` (hunter + guard) is identical to `p6` because no burst is on the field. In the recorded scenes I could not see the commit
  phase (it is not in the scene steps), so whether the p6h losses were commits into the human is **not known**. What the code does say (`reflex.rs`): the hunter's commit override was ahead of every other reflex and nothing vetoed it for a near giant except `Hunter::step`'s own checks.
* The rule-based check "would a flee fire here?" (lethal reach + 600, as `evade_trigger`) is true in the first half of the scene (about 1.5 s before the loss) in **all 17** `no_flee`, 25 of 25 `fled`, 18 of 22 `pieces` scenes.

Fix built (`rust/arena_client/src/reflex.rs`, spec key `reflex_flee`, off by default): when the rule bots' flee check says Flee (genes `bot/c3_4.json` with `flee_margin` = `reflex_flee_margin`, default 1.0, and the evade trigger
`reflex_flee_trigger`, default 600), the net's aim is replaced by the look-ahead flee (`evade.rs`; in pieces the rule bots' own flee steering) and **nothing overrides it, not even the hunter's commit**. Specs: `bot/escape/vision_p6f.json`,
`vision_p3f.json`, `vision_p6hf.json` (= p6 / p3 / p6h + `"reflex_flee": true`). Unit test `flee_reflex_takes_the_aim_from_a_net_that_does_not_run`.

Drill (M 3000 / 7000 x ratio 1.4 / 2 / 4 x gap 1200 x open / wall, 12 reps per cell, 144 per agent, CUDA `cuda_plain`):

| agent | survive 30 s | survive 60 s | survival time s | paired dt vs p6 s | mass kept 60 s | in split reach s |
|---|---|---|---|---|---|---|
| p6 | 90% [84-94] | 27% [20-35] | 47.7+-1.9 | - | 0.25+-0.07 | 6.0+-1.1 |
| p6 + `reflex_flee` (p6f) | 98% [94-99] | 42% [35-51] | 52.7+-1.5 | **+5.0+-1.2** | 0.39+-0.07 | 6.1+-1.1 |
| p6h (hunter + guard) | = p6 (no burst on the field) | | | | | |
| p6h + `reflex_flee` | = p6f | | | | | |

The gain is at the walls and for the 2x chaser (p6: 0-25% at 60 s there, p6f: 25-58%); in the open at M 3000 x 1.4 and in `our7000/x4/wall` (0% -> 0%) there is none.
This is the same pattern as the rule bots (the nets had no flee at all; the look-ahead flee is as good as the rule bots' best on this scene).

Endless, vision candidate p6 (held-out field of `endlessH.sh`: plan x2, plan2, bc3 x2, apos, hunter_virus), `--engine inproc`, 8 worlds x 240 min:

' + vp6f + '

What the evasion would fix of the human losses, as far as I can say: the `no_flee` losses (31% of the mass in this session, 20990) are nets with no flee, a gap `reflex_flee` closes in the drill (+5 s, 90% -> 98% at 30 s); whether it keeps them alive against the *human* I cannot show,
the replay of a recorded scene needs the Python policy, not the Rust reflex. `pieces` (35%): no flee-in-pieces rule helped in the drill (four variants), so the evasion fixes none of it; the guard (`reflex_guard`, M5) is the remaining tool. `fled` (27%): the `evade` search
is as good as the best rule bots in the drill and not better in endless: not a fix.

## The `evade` genes (`rust/arena_client/src/evade.rs`, all off by default)

When the rule bot decides to flee and `evade = 1`, the aim is the first move of the best path of a search tree: 4 moves (1 + 1 + 2 + 3 s) over 8 headings (4096 paths, 7 s), cell speed `88 * size^-0.439`, walls clamp the cell (a
heading into a wall becomes a slide, a corner a dead end), poppable viruses on the way score minus infinity, every chaser (the nearest four) follows a lead pursuit at its own speed (optionally round viruses).
Score of a path: `min(min margin over the path, cap) + 0.3 * min(final margin, cap) + evade_wall * 0.5 * (- distance of the final position from the map centre)`; the margin is the centre distance minus the chaser's lethal reach
(eating distance `size - our size / 3`; if the chaser's half can still eat us, the real split throw `780 * child^0.0122 + child size - our size / 3`, `lethal_reach` in the file). Past the cap all margins count the same and the
centrality (the walls and corners are the dead ends) decides. In several pieces the old flee runs (`evade_pieces 0`).

| gene | default | meaning |
|---|---|---|
| `evade` | 0 | 1 = use the search for the flee aim (rule bots and the planner's base) |
| `evade_wall` | 0 | weight of the centrality term (1 is the shipped value; 0.3 -> 19% survival at 60 s, 2 -> 0%, a cliff, see the sweep) |
| `evade_cap` | 0 (= 800) | margins above this are equal |
| `evade_buffer` | 0 | added to every lethal reach |
| `evade_trigger` | 0 | > 0: also enter the flee mode when a chaser is within its lethal reach + this (no effect measured: the bots flee early enough in these scenes) |
| `evade_virus` | 0 | chasers go round viruses in the model |
| `evade_pieces` | 0 | 0 old flee in pieces; 1 / 2 guard the pieces / close up (both measured worse) |
| `evade_food` | 0 | bonus for a first move towards pellets (no effect) |

Unit tests (`cargo test --release evade`): open field runs away, slides along the wall instead of into it, avoids the corner dead end, nothing to flee from gives none, lethal reach (split throw only when the half still eats),
pieces aim at closing up. `cargo test --release escape`: scene determinism per seed (3 placements x 1 / 4 pieces, with virus), scene geometry, split reach formula, chaser splits only when the half can eat, chaser leads, rep determinism.

Parameter sweep on M 3000 x 1.4 / 2 / 4 x gap 1200 x open + wall (N = 40 per cell, 240 per agent, in-sample, so optimistic): survival time dt vs c3_4: `evade` +5.8+-1.1 s (30 s survival 98% vs 84%, 60 s 23% vs 13%), cap 500 +4.8, cap 1200 +4.2,
`evade_wall` 0.3 +3.4, 0 -0.2 (the centrality term is what keeps it off the walls), 2 -14.9 and 3 -30.5 (the bot runs into the chaser). `nowall` +3.7 on the same cells (60 s survival 28%).

## Endless gate

`arena.endless --engine inproc`, 8 worlds x 240 game-min, candidate x2 against the field of `endless5.sh` (champ, r2g9, safe, ramp_r2, ramp300, each x2), leader share of the candidate per seed base, rule: a gene is kept only if
the mean improves **and** it wins >= 3 of the 5 sets.

Leader share (dominance in brackets), binary of this branch, `OGAR_BIN` copy, 8 worlds x 240 min per cell, field of `endless5.sh`. `c3_4` lineup (candidate = `bot/c3_4.json` + genes):

| set | c3_4 | + `nowall` (wall genes off) | + `evade` | + `evade` at total mass >= 1500 (`evade_big`) |
|---|---|---|---|---|
| 3000 | 31% | 32% | 33% | 30% |
| 11000 | 28% (30%) | 31% (34%) | 22% (22%) | 38% (38%) |
| 21000 | 31% (30%) | 33% (35%) | 34% (34%) | 29% (31%) |
| 31000 | 32% (32%) | 30% (32%) | 32% (31%) | 31% (30%) |
| 41000 | 36% (38%) | 28% (30%) | 28% (30%) | 41% (43%) |
| **mean** | **31.6** | 30.8 | 29.8 | 33.8 |
| sets won vs c3_4 | - | 3 of 5 | 2 of 5 (one tie) | 2 of 5 |

(the 3000 baseline is from the console output of the first run, 31%, dominance not recorded.) **No candidate passes the gate** (mean up *and* >= 3 of 5 sets): `nowall` wins 3 sets but the mean is lower; `evade_big` has the higher mean (+2.2) but
from two sets (+10 and +5) and loses three; `evade` loses. All three stay off. The wall genes of c3_4 were found on this very metric, and the drill is the scene in which they hurt: the drill result is not a result on leader share.

What `evade` does in endless (deaths of the two candidate agents per world, summed over the 5 sets x 8 worlds, by total mass at death; `logs/gate/*.stats.json`):

| candidate | < 150 | 150-600 | 600-2000 | >= 2000 | losses >= 1500 mass (count / mass) |
|---|---|---|---|---|---|
| c3_4 | 100 | 23.6 | 18.1 | 6.7 | 24.0 / 62243 |
| evade | 250 | 39.8 | 19.3 | 8.0 | 21.6 / 60919 |
| nowall | 103 | 25.0 | 17.5 | 6.4 | 23.8 / 61310 |
| evade_big | (not split by bucket) | | | | 26.4 / 72226 |

`evade` makes the small bots die 2.5 times as often (the centrality term pulls a small flee into the middle of the map, where the field is; the drill had no field), and its big-bot losses
are 10% fewer only by an amount inside the noise. The mass gate (`evade_min_mass 1500`) removes the small deaths (140 per world vs 128) but not the big-bot losses (26.4 vs 24.0 per world). I did not find a version
that lowers the big-bot losses in endless; the drill gain (M 7000: 46% -> 62% at 60 s) does not transfer.

`plan_c34` lineup: **not run** (planner + `evade` / `nowall`): I stopped it after the `c3_4` lineup failed (the planner calls the same base bot and was identical to c3_4 in the drill).


## Not done / caveats

* The chaser is one scripted policy (plus the rule bots in the endless gate). It is stronger than a bot at herding and weaker than the human at feints; the human's 10k vs 7k scenes have a ratio of 1.45 and a gap near 1000 inside the split reach, which is the 1.4x / gap 1200 cells.
* The search tree, its weights (cap 800, centrality 0.5, headings 8) and the chaser model were tuned on the drill itself (in-sample); the sweep above is the only tuning. `evade_wall` has a cliff between 1 and 2.
* The optional last-resort split / eject was not implemented as a new gene: the existing `escape_split` was measured instead (no effect; it splits only in the 4x cells).
* Merge-direction flee in pieces: three variants measured, none helped; the pieces scenario is only 2 / 4 pieces thrown sideways at the start, not pieces that come from a split near a chaser.
* Nets: only the gap-1200 grids (18 cells x 20 reps for p6/p3; 12 cells x 12 reps for the `reflex_flee` variants), open + wall, no corner / virus / pieces. The vision endless gate was run on one seed set (3000) for p6 and two `reflex_flee` variants.
* The `plan_c34` endless lineup was not run (see above). The planner (`plan_c34`) was not tuned for or against these scenes; the drill does not exercise it.
* `touched wall` is ~90-100% for everyone within 60 s (it is a map of 7000 with cells of 300-1500); `outgrew` is rare; the survivors' method labels are dominated by `wall_slide` because the first applicable label wins.
* Mass kept above 1 (food) is included in the means.
