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
