# Lookahead planner (`"kind": "plan"`)

Rust-only policy (`rust/arena_client/src/plan.rs`) that wraps a rule bot and, when it predicts a loss for the
rule bot's action, simulates a ~2 game-second horizon and may replace the action. Runs in `ogar match|serve`,
`arena_client run`, `arena.endless`, `arena.tournament --engine ogar`.

```json
{"kind": "plan", "base": "bot/c3_4.json", "need_loss": 0.1}
```
`base` is a rule-bot spec path (or an inline spec object). Example: `bot/plan_c34.json`.

## How it decides
1. The base rule bot always decides first (its state - food router, tactics, velocity tracking - stays coherent).
2. Gate: if no rival player cell is within `gate` (900) of one of our cells, the base action stands.
3. A light sim is built from the Snapshot only: our cells, the nearest `max_rivals` rival players (grouped by colour),
   viruses and the nearest `max_food` pellets (static). Physics follow rust/ogar: speed by size, eat rule
   (size ratio 1.15, overlap rule), split boost, virus pop, border. No merging or decay inside the horizon.
4. Rivals move greedily; the opening is priced under three rival scenarios (0 adversarial: what can eat us chases us,
   what we can eat runs; 1 inertial: keeps its observed heading; 2 greedy: food / prey / flee) and the score is
   `min_w * worst + (1 - min_w) * mean`.
5. Candidates: the base action, `headings` (16) headings, and splits at up to `split_cands` prey (only when the base
   genes allow a split: cells <= split_max_cells). Each opening is held `hold` ticks, then the base rule bot's steering
   continues as the rollout policy (re-planned every `replan` ticks).
6. Score = our end mass - `loss_w` x mass lost + `eat_w` x mass eaten - `near_w` x mass within `near_range` of an eater
   - `frag_w` x mass outside our largest cell - `pop_w` x popped mass; death = `-death_w` x start mass.
7. Safety net: the base opening is priced first. Only if it loses >= `need_loss` x our mass in some scenario are the
   alternatives priced, and one replaces the base action only if it scores `margin` x mass higher.

## Findings (what mattered)
- Overriding on every small score difference hurts: the best of 18 noisy outcomes is optimistic (winner's curse;
  predicted mass at override steps was +35 % over real at 2 s, +1 % at non-override steps) and it breaks the
  base bot's farming routes. Gating on a predicted loss fixed both the quality and the cost.
- Rewarding virus pops made the planner steer into viruses (the sim shows +virus mass, reality is fragmentation).
- Free multi-splits (cells > 1) were fatal; the planner now splits only when the base genes would.
- A single rival model is too exploitable; min/mean over three scenarios is much more robust.

## Debugging
`AGAR_PLAN_LOG=1` (every gated step: scores per opening), `AGAR_PLAN_TRACE=<step>` (eat events of the chosen
opening), `AGAR_PLAN_VAL=1` (prediction vs reality of rival positions and our mass at 0.4/1/2 s).
Counters (in the per-agent stats): `plan_steps`, `plan_gated`, `plan_deep`, `plan_override`, `plan_split`, `plan_us`.

## Rival model v2 (rule-bot rivals) - 2026-10-07 experiment, result: no gain
`"rival_bots": ["runs/final_champion.json"]` + scenario 3 (4, 5, ... = further genomes): each of the `rule_rivals` (4)
nearest rivals decides with `heuristic_plan` + `steer` of that genome from its own view (`rule_view` 1100 px around it,
built from the sim world), heading blended with its old one (`rule_inert` 0.3); a farming rival that knows < `rule_food`
pellets keeps its heading. Extra scenarios for diagnosis: 5 stationary, 6 inertial at observed speed, 7 inertial + turn
rate (`turn_damp`). Spec: `bot/plan2_c34.json` (scenarios [3, 0, 1]).

Rival position error (px, `uv run python -m arena.plan_val --scen 0 1 2 3`, 15 game-min, planner among 10 rule bots +
8 MultiOgar bots; "near" = rivals within 600 px gap of us at decision time; 0.4 / 1 / 2 s):
| model | all rivals | near rivals |
|---|---|---|
| stationary (5) | 122 / 266 / 484 | |
| 0 adversarial | 87 / 214 / 411 | 81 / 193 / 351 |
| 2 greedy | 85 / 208 / 404 | 78 / 186 / 350 |
| 3 rule bot (v2) | 87 / 201 / 382 | 82 / 179 / 325 |
| **1 inertial** | **49 / 130 / 296** | **45 / 116 / 256** |
| 7 inertial + turn rate | 55 / 152 / 347 | 51 / 137 / 310 |
The rule-bot model does NOT predict better than "keep the observed heading" (also not against rule bots only: 60 / 137 /
250 vs 29 / 80 / 197). Straight segments are predicted well by every model; the error is in turns, which neither rule
logic (state: food router, flee smoothing, hysteresis; unseen pellets) nor turn-rate extrapolation anticipates.
Decision cost per deep evaluation (19 openings, 6 rivals, `cargo test --release deep_eval_cost -- --ignored`): v1 3.9 ms,
v2 8.5 ms. Endless wall time (8 worlds x 240 game-min): v1 316-333 s, v2 366-381 s.

Leader share (endless, 8 worlds x 240 min, field of the task) per seed base:
| spec | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|
| plan v1 (`plan_c34`) | 46 | 35 | 38 | 38 | 44 | 40.2 |
| plan v2 (`plan2_c34`, [3,0,1]) | 42 | 42 | 49 | 32 | 40 | 41.0 |
Within noise. Tried on seeds 3000 / 11000 only: scenarios [3,1] 36 / 38, [1,0,2,1] 25 / 39, [0,1,2,3] 36 / 39,
v1 with horizon 75 35 / 26 (longer horizons hurt: the model is not good enough), v1 with need_loss 0.05 44 / 39.
