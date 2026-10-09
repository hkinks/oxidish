# Learnings

What we learned building agar.io bots, distilled from the experiments; details live in the linked docs. Goal throughout:
be #1 on the leaderboard ~80% of the time in endless worlds on the Rust server (`arena.endless`, leader share).

## Measuring

- **Leader share in endless worlds is the metric, not 15-min match mass.** Short-match winners (c3_4 beat every bot
  on Node, 348 matches) only lead ~31% of the time in endless worlds: leads are lost late, to rivals that grow ~3.5x
  in 5 minutes by eating the leader.
- **Noise is ~±5 points per run** (8 worlds × 240 game-min). Decide on ≥ 2 seed sets, preferably all 5
  (3000/11000/21000/31000/41000), and require a win on ≥ 3 of 5 sets.
- **Evaluate on a held-out field.** PPO trained against the evaluation field scored 78.8% there and 43% on a disjoint
  field (and 3% live against a human). Any opponent set used in training must not be the evaluation set.
- **Live play is the final test.** The user led 81% of the time on 9600 against all bots; human sessions are recorded
  (`--record`) and are the best regression set (`arena.human_report`, `arena.scene_replay`).
- **Use the fast evaluator** (`arena.endless --engine inproc`): bit-identical for rule bots, ~1.3-1.5x faster for vision.

## Rule bots and genes

- Hand-written genes plateau around 30-35% leader share; most new genes measured as noise or worse on endless
  (split_ev, virus staging, big_hunt, flee_smooth, escape split, shoot_threats, harvest). A gene that looks good in
  matches often fails on endless; test there first.
- Lookahead (the planner, `plan_c34`) was the one qualitative step for rule bots: 40% vs 31%.
- Port Python genes to Rust with an exact parity check; new bot logic is Rust-first.

## Vision bot

- Plain grayscale frames (256 px view + 128 px wide view) are enough for planner-level play; no feature channels.
- **Imitation caps at the teacher.** BC/DAgger from the planner reached 42% (planner 40%).
- **Rare actions don't imitate.** Split is 0.07% of planner decisions and depends on sub-pixel size ratios and hidden
  state (merge cooldown). The BC split head had recall ~0.2 and its splits cost mass, so splitting was shipped off -
  and then stayed off through RL. That was a mistake: RL is where split timing can be learned from outcomes.
- **No memory hides key state.** Two stacked frames cannot show our own merge timer or rivals that left the view; a
  recurrent core is needed for split and defence timing.
- **PPO near the teacher gives safe, local gains.** With a small lr and a KL anchor to the BC net, deaths/h fell
  15.6 → 5.5 but held-out leader share only moved 28 → ~42%.
- **Fixed opponents → overfitting to them.** League pools and frozen self-play closed the train/held-out gap mostly by
  dropping training-field scores. Real self-play (current + past policies) is the next step.
- **Reward must match the goal.** Δlog(mass) + a tiny #1 bonus does not teach "don't get split-killed while big" or
  "take the leader down", which is how leads are actually won and lost.

- **Bolted-on skills hurt (M5).** Split-kill, harvest and virus-split reflexes on top of p6 work mechanically
  (38 split kills/h, 11.5 virus pops/h) but cost 11-31 points of leader share: after a split our halves get eaten by
  mid-size players, harvesting chases pieces into the crowd. A skill is only worth it together with the defence to
  survive its aftermath.
- **PPO that executes splits collapses** (deaths 30-50/h, leader ~1%) every time; random splits are too costly to
  explore. Splitting needs a safe teacher or curriculum (drills), not exploration from scratch.
- **Engines differ for vision runs**: `--engine inproc` vs `serve` diverge chaotically (p6 58% vs ~45%); compare
  candidates only within one engine.

## Drills (harvest, escape)

- The hunter module (commit to popped giants with a kiting lookahead, safe split-harvest) is the first skill that raised leader share on every seed set: c3_4 31.6 -> 39.8%, planner 40.2 -> 49.6%. Safety lookahead before committing is what made it work where reflexes failed.
- A 500-mass hunter vs a popped 5k giant captures only 11-17% (bound 51%) and rarely overtakes: bots back away from
  giants instead of lurking, and competitors take the pieces ([harvest-drill.md](harvest-drill.md)).
- The human beats the bots mostly by herding fleeing bots into walls and split-killing them (48% of mass lost; 29% lost
  while in pieces) ([human/2026-10-09-50bots.md](human/2026-10-09-50bots.md)). Escape is the weakest skill.

## Commander (LLM / heuristic orders)

- Orders over the control port work (`hunt`, `harvest`, `virus-split`, ...), but a simple heuristic commander did not
  help the planner (28 → 25%, noise) and hurt the vision bot (22 → 2.5%): overriding the net's aim point throws away
  what it sees. LLM backends are untested (no keys). See [commander.md](commander.md).

## Process

- Merge agent branches promptly; long-lived worktrees conflict on docs and `main.rs`.
- Keep runs within resource caps (VRAM ≤ ~26 GB, CPU/RAM ≤ 90%) and give each agent its own `CARGO_TARGET_DIR` and
  ports; never touch the live servers' ports.
- Stop processes by PID after checking the command line, never by image name (other projects share the machine).

See also: [training-plan.md](training-plan.md), [vision-bot.md](vision-bot.md),
[lookahead-planner.md](lookahead-planner.md), [commander.md](commander.md), [ogar.md](ogar.md).
