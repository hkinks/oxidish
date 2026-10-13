# Status (2026-10-10)

Where the project stands after the Rust-server push toward "one bot that is #1 on the leaderboard ~80% of the time,
and can split-eat, pop others with viruses and eat exploded enemies efficiently". Work stopped here; the 80%
goal was **not** reached. Lessons are in [learnings.md](learnings.md); this page is the summary and the how-to.

## Results

Metric: endless leader share (share of 15-s samples where the candidate family is #1), 8 worlds × 240 game-min per
seed set, seed sets 3000/11000/21000/31000/41000, `training.eval.endless --engine inproc`. Noise ~±5 per set; a change is kept
only if the mean improves and it wins ≥ 3 of 5 sets.

Leaderboard names of the four reference bots in `bots/specs/best/`: **vision** (m7a), **vision-old** (p6_hunt),
**planner** (planhv), **rules** (c34h); older results use the names in parentheses.

**Vision bot** (grayscale frames rendered in the engine, ONNX net in `ogar`), held-out field (opponents never trained
against: plan ×2, plan2, bc3 ×2, apos, hunter_virus + 8 server bots; `apos` was a heuristic bot since removed from the repo):

| spec | what | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|---|
| `bots/specs/vision_bc3.json` | imitation of the planner (M2/M3) | 26 | 31 | | | | ~28 * |
| `bots/specs/vision_p6.json` | league PPO (M4) | 55 | 59 | 66 | 53 | 55 | 57.6 |
| `bots/specs/best/vision-old.json` | p6 + hunter reflex (M6) | 57 | 62 | 66 | 64 | 57 | 61.2 |
| **`bots/specs/best/vision.json`** | PPO with the hunter in the loop, stronger league (M7) | 64 | 68 | 69 | 58 | 63 | **64.4** |

\* bc3 numbers are from the serve engine (M4); the engines diverge for vision runs, so compare rows only within inproc.
M8 (9 h continuation of m7a, 10% human-like chaser, m7a as a frozen league member) plateaued after ~1 h: checkpoints scored
57-67 on sets 3000/11000 vs m7a 64/68, no gain; no checkpoint promoted (see the M8 section of [vision-bot.md](vision-bot.md)).

**Champions field** (candidate ×2 vs p6_hunt ×2, planhv ×2, c34h ×2, plan + 8 server bots): m7a 40/44/48 on seed sets
3000/11000/21000; p6_hunt in the same slot 40/40 (3000/11000). Among near-equal top bots every family sits near 40%.

**Rule bots** (usual field: champ, r2g9, safe, ramp_r2, ramp300 ×2 each + 8 server bots):

| spec | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|
| `bots/specs/c3_4.json` | 31 | 28 | 31 | 32 | 36 | 31.6 |
| `bots/specs/best/rules.json` (c3_4 + hunter) | 43 | 44 | 38 | 37 | 37 | 39.8 |
| `bots/specs/plan_c34.json` (lookahead planner) | 46 | 35 | 38 | 38 | 44 | 40.2 |
| **`bots/specs/best/planner.json`** (planner + hunter) | 49 | 46 | 49 | 47 | 57 | **49.6** |

**Live server, 2026-10-10 (11 h, 50 players, [report](human/2026-10-10-m7.md))**: #1 share over the whole session m7 30%, p6h 25%, planhv 11%, p6 10%, plan 8%, human 7%; the user (no boost this time) was #1 for 63% of their 77 min alive. m7a is the best bot overall.

**Against the human** (live server, recorded): the user was #1 for 66-70% of the session against the hunter bots
(partly with the +500 boost). Bots lost mass to the human mostly while in pieces (35%), by not fleeing at all (31%,
all vision nets) and after fleeing into walls (27%) ([human/2026-10-09-hunters.md](human/2026-10-09-hunters.md)).

## The requested skills

| skill | state |
|---|---|
| eat exploded enemies | **works**: the hunter module (lookahead commit to a popped giant's pieces, safe split-harvest, race) doubles drill capture (500 vs popped 5k: 12% → 24-33%, overtake 6% → 27-52%) and is the one skill that raised leader share for every bot type ([harvest-drill.md](harvest-drill.md)) |
| split-eat | the rule split-kill exists; as a vision reflex it costs 11-31 points (halves get eaten); PPO that executes splits collapses. Only the hunter's safe splits are on |
| pop others with viruses | virus-split tactic exists (11.5 pops/h as a reflex) but costs leader share; off |
| escape a hunting human | `ogar escape` drill + `evade` / `reflex_flee` genes help in the drill (+5 s survival) but lose in the full game; off ([escape-drill.md](escape-drill.md)) |

## What was built in this stretch

- **Vision bot pipeline** (M0-M8): `server/src/render.rs` (frames), `vision.rs` + `training/env` (VecEnv for
  training, league pool, giants, scripted chaser, hunter in the loop), `visionbot.rs` + `nn.rs` (ONNX Runtime with CUDA
  graphs, batched, GRU state), `training/rl/` (BC/DAgger, PPO v1-v5, export, eval scripts). [vision-bot.md](vision-bot.md)
- **Fast evaluator**: `ogar endless` / `training.eval.endless --engine inproc [--stats-out]` with per-skill counters.
- **Drills**: `ogar drill` (giant pop harvest, oracle, mass-chain bound) and `ogar escape` (human-like chaser).
- **Skills in `bots/brain`**: `hunter.rs` (lurk/commit/split-harvest/race), `reflex.rs` (vision reflex layer:
  hunter, split-kill, harvest, virus, flee, guard), `evade.rs`, `order.rs`.
- **Commander**: control-port orders (`hunt`, `harvest`, `virus-split`, `goto`, `flee`, ...), `scene`/`frame`, an MCP
  server (`training/eval/commander_mcp.py`) and an LLM/heuristic loop; the heuristic did not help, LLM backends untested (no
  keys). [commander.md](commander.md)
- Cleanup: dead code removed, `parse_lineup` deduplicated, `.gitignore` for raw match rows; later the pre-Rust code
  was removed ([history/](history/README.md)).

## How to run

```bash
cargo build --release      # target/release/ogar (server), target/release/oxidish (client); .exe on Windows

# live server with the best bots and recording; vision bots need ONNX Runtime (see training.md)
target/release/ogar serve --port 9600 --bots 12 --food-mult 2 --map 10000 --control-port 9601 \
  --record logs/human --record-all --agents vision=bots/specs/best/vision.json:6 planner=bots/specs/best/planner.json:6 \
  rules=bots/specs/best/rules.json:4 vision-old=bots/specs/best/vision-old.json:4
target/release/oxidish --name you    # or scripts/client.sh; server address is saved in the client settings
uv run python -m training.eval.human_report logs/human --session <stamp> --out docs/human/<date>.md   # after playing

# leader share of a candidate: scripts/eval.sh, or directly (held-out lineup in vision-bot.md)
uv run python -m training.eval.endless --candidate vis --agents vis=<spec>:2 <field> --worlds 8 --game-min 240 \
  --seed-base 3000 --engine inproc --stats-out out.json
target/release/ogar drill --agents me=<spec> --scenario giant_popped   # harvest drill; ogar escape for the escape drill
```

The vision models are in `bots/models/`. With CUDA 12 the nets run on the GPU (on Windows the CUDA DLL folders go in
`OGAR_DLL_DIRS`, see [harvest-drill.md](harvest-drill.md)); otherwise they fall back to the CPU.

## If work resumes

1. Defence first: bots lose most mass while in pieces or when hunted. A learned response is needed (rules failed in
   the field): train against the scripted chaser at a low rate from the start, and add the human recordings as data.
2. Splitting only through a safe teacher (the hunter's split-harvest, drills as curriculum), never free exploration.
3. Judge in the champions field and live against the human, not only the held-out field.
4. Try a real LLM commander once a key is available (`training.eval.commander --backend anthropic|jev`).
