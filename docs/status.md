# Status (2026-10-10)

Where the project stands after the Rust-server push toward "one bot that is #1 on the leaderboard ~80% of the time,
and can split-eat, pop others with viruses and eat exploded enemies efficiently". Paused here by the user; the 80%
goal was **not** reached. Lessons are in [learnings.md](learnings.md); this page is the summary and the how-to.

## Results

Metric: endless leader share (share of 15-s samples where the candidate family is #1), 8 worlds × 240 game-min per
seed set, seed sets 3000/11000/21000/31000/41000, `arena.endless --engine inproc`. Noise ~±5 per set; a change is kept
only if the mean improves and it wins ≥ 3 of 5 sets.

**Vision bot** (grayscale frames rendered in the engine, ONNX net in `ogar`), held-out field (opponents never trained
against: plan ×2, plan2, bc3 ×2, apos, hunter_virus + 8 server bots):

| spec | what | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|---|
| `bot/vision_bc3.json` | imitation of the planner (M2/M3) | 26 | 31 | | | | ~28 * |
| `bot/vision_p6.json` | league PPO (M4) | 55 | 59 | 66 | 53 | 55 | 57.6 |
| `bot/vision_p6_hunt.json` | p6 + hunter reflex (M6) | 57 | 62 | 66 | 64 | 57 | 61.2 |
| **`bot/vision_m7a_final.json`** | PPO with the hunter in the loop, stronger league (M7) | 64 | 68 | 69 | 58 | 63 | **64.4** |

\* bc3 numbers are from the serve engine (M4); the engines diverge for vision runs, so compare rows only within inproc.
M8 (8-10 h continuation of m7a with a 10% human-like chaser) was stopped during evaluation: checkpoints scored
57-66 on sets 3000/11000, none clearly above m7a (see the M8 section of [vision-bot.md](vision-bot.md)).

**Champions field** (candidate ×2 vs p6_hunt ×2, planhv ×2, c34h ×2, plan + 8 server bots): m7a 40/44 vs p6_hunt 40/40
in the same slot. Among near-equal top bots every family sits near 40%.

**Rule bots** (usual field: champ, r2g9, safe, ramp_r2, ramp300 ×2 each + 8 server bots):

| spec | 3000 | 11000 | 21000 | 31000 | 41000 | mean |
|---|---|---|---|---|---|---|
| `bot/c3_4.json` | 31 | 28 | 31 | 32 | 36 | 31.6 |
| `bot/drill/hv_commit_only.json` (c3_4 + hunter) | 43 | 44 | 38 | 37 | 37 | 39.8 |
| `bot/plan_c34.json` (lookahead planner) | 46 | 35 | 38 | 38 | 44 | 40.2 |
| **`bot/drill/plan_hv_commit_only.json`** (planner + hunter) | 49 | 46 | 49 | 47 | 57 | **49.6** |

**Live 9600, 2026-10-10 (11 h, 50 players, [report](human/2026-10-10-m7.md))**: #1 share over the whole session m7 30%, p6h 25%, planhv 11%, p6 10%, plan 8%, human 7%; the user (no boost this time) was #1 for 63% of their 77 min alive. m7a is the best bot overall.

**Against the human** (live 9600, recorded): the user was #1 for 66-70% of the session against the hunter bots
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

- **Vision bot pipeline** (M0-M8): `rust/ogar/src/render.rs` (frames), `vision.rs` + `rust/ogar_py` (VecEnv for
  training, league pool, giants, scripted chaser, hunter in the loop), `visionbot.rs` + `nn.rs` (ONNX Runtime with CUDA
  graphs, batched, GRU state), `rl/vision/` (BC/DAgger, PPO v1-v5, export, eval scripts). [vision-bot.md](vision-bot.md)
- **Fast evaluator**: `ogar endless` / `arena.endless --engine inproc [--stats-out]` with per-skill counters.
- **Drills**: `ogar drill` (giant pop harvest, oracle, mass-chain bound) and `ogar escape` (human-like chaser).
- **Skills in `rust/arena_client`**: `hunter.rs` (lurk/commit/split-harvest/race), `reflex.rs` (vision reflex layer:
  hunter, split-kill, harvest, virus, flee, guard), `evade.rs`, `order.rs`.
- **Commander**: control-port orders (`hunt`, `harvest`, `virus-split`, `goto`, `flee`, ...), `scene`/`frame`, an MCP
  server (`arena/commander_mcp.py`) and an LLM/heuristic loop; the heuristic did not help, LLM backends untested (no
  keys). [commander.md](commander.md)
- Cleanup: dead code removed, `apos` split, `parse_lineup` deduplicated, `.gitignore` for raw match rows.

## How to run

```bash
# live server (as running now on 9600, control port 9601), best bots, 50 players
rust/ogar/target-live4/release/ogar.exe serve --port 9600 --bots 12 --food-mult 2 --map 10000 --control-port 9601 \
  --record logs/human --record-all --agents m7=bot/vision_m7a_final.json:6 planhv=bot/drill/plan_hv_commit_only.json:6 \
  c34h=bot/drill/hv_commit_only.json:4 p6h=bot/vision_p6_hunt.json:4 ...
rust/game_client/target/release/game_client.exe          # client, server from %APPDATA%\agario-client\settings.json
uv run python -m arena.human_report logs/human --session <stamp> --out docs/human/<date>.md   # after playing

# leader share of a candidate (see docs/vision-bot.md for the held-out lineup)
uv run python -m arena.endless --candidate vis --agents vis=<spec>:2 ... --worlds 8 --game-min 240 \
  --seed-base 3000 --engine inproc --stats-out out.json
ogar drill --agents ... --scenario giant_popped      # harvest drill;  ogar escape ... for the escape drill
```

Vision models are gitignored: `runs/vision/export/vision_*.onnx` (+ `runs/vision/ort` DLLs; CUDA needs
`OGAR_DLL_DIRS`, see harvest-drill.md).

## If work resumes

1. Defence first: bots lose most mass while in pieces or when hunted. A learned response is needed (rules failed in
   the field): train against the scripted chaser at a low rate from the start, and add the human recordings as data.
2. Splitting only through a safe teacher (the hunter's split-harvest, drills as curriculum), never free exploration.
3. Judge in the champions field and live against the human, not only the held-out field.
4. Try a real LLM commander once a key is available (`arena.commander --backend anthropic|jev`).
