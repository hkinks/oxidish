# Vision bot: plan (2026-10-07)

A bot that decides from **rendered frames**: the Rust server (`rust/ogar`) draws each controlled player's camera
view as a small image, and a CNN policy picks the move from the pixels. Actions still go through the normal agent
interface (heading, split, eject). There is no screen capture: frames are rendered headless inside the engine,
so training scales to many worlds.

**Goal.** Beat the planner (≈40% leader share; c3_4 ≈30%, mean of 5 seed sets) on the endless metric, toward the
80% goal. The frames carry only what the client shows: the visible cells, their colours and sizes, and the border.
Hidden state and other players' internals are not included.

**Honest prior.** Our earlier neural bots (grid CNN in `rl/model.py`, trained with PPO in `sim/`) never matched
the rule bots, and never learned splitting (see [architecture-research.md](architecture-research.md); AgarCL hit
the same wall). Pixels are a harder input than our hand-made grid. So this plan does not start from RL. It starts
by **imitating the planner on frames**, and uses RL only to improve on that base. Every stage has a gate, and the
project stops at the first gate it misses.

## Architecture

```
rust/ogar (world, rule-bot agents)
  └─ render.rs   Snapshot -> uint8 frame [C,H,W]      (same data the client gets; no privileged info)
rust/ogar_py (new, PyO3 + maturin)   VecEnv: N worlds in-process, threads over worlds
  reset() / step(actions) -> frames [N,C,H,W] uint8 (numpy, no copy), reward, done, info
  teacher actions: the planner's choice for the same step (for imitation / DAgger)
Python (ML only, per rust-first)   rl/vision/: model, BC, PPO, export to ONNX
rust/ogar agent kind "vision"   runs the ONNX net in-process (crate `ort` or `tract`)
  -> arena.endless / verdict / live servers work unchanged
```

**Why render from `agent::snapshot`.** It is the exact view the protocol sends a player, so a frame shows no
more than a human sees. The renderer is a pure function of the Snapshot, which makes it unit-testable and lets it
reuse the client's camera maths (`rust/game_client`: centre, zoom = server's `scale`).

## Design choices (recommendations; easy to change later)

- **Frame.** 2 views stacked as channels:
  - a **96×96 local view** of the server's camera rectangle;
  - a **48×48 wide view** at 2× that extent.
  - **Why two:** at big mass, pellets are under one pixel in a single view, and threats arrive from off-screen.
- **Colour vs semantics.** Draw **RGB as the client does**: cell colour, our cells outlined, viruses green and
  spiky, pellets small and coloured. Add **one size channel** (cell radius encoded as intensity), because size
  ratios decide everything and RGB at 96 px blurs them. This is still "visual": it is the cell's on-screen size.
  As an ablation, a pure-RGB variant tests whether the size channel matters.
- **Motion.** Stack **2 frames** (now and 4 ticks ago). Velocities matter, and a stack is simpler than a recurrent
  network (GRU).
- **Action.** A **spatial action map**: the net outputs a heat map over the local view, and the argmax pixel is
  the cursor target ("go here"; AlphaStar-style). It also outputs split and eject logits. A target pixel matches
  the visual input better than 16 fixed headings, and the planner's action converts to a target pixel exactly.
  Fallback: the 16 headings plus stay that we already proved work (303 vs 266 farming).
- **Decision rate.** Every 2 server ticks (80 ms), as the rule bots do.
- **Network.** IMPALA-ResNet (3 stages, no early stride on the local view), a heat-map head via 1×1 convs, and a
  pooled MLP for the value, split and eject heads. Small enough to run in-process on CPU for evaluation
  (aim ≤ 1 ms per decision with `ort`).

## Stages and gates

| # | stage | output | gate (stop if missed) |
|---|---|---|---|
| M0 | `render.rs` plus PNG dump (`ogar render --seed S --tick T`) and unit tests (circle sizes, camera, border) | frames that look like the client | a side-by-side PNG next to a client screenshot looks right to you |
| M1 | `ogar_py` VecEnv (PyO3), threaded worlds, rule-bot opponents in-process, teacher actions | `pip install -e rust/ogar_py` | ≥ 20k frames/s on ≤ 10 cores, CPU ≤ 90% |
| M2 | **Imitation**: behaviour cloning on planner actions, then DAgger (the student drives, the teacher labels) | `runs/vision/bc_*.pt` | target-pixel agreement ≥ 80%, split agreement ≥ 70%; ≥ 45% vs c3_4 in 15-min matches |
| M3 | ONNX export plus agent kind `"vision"` in ogar | `bot/vision_bc.json` | identical actions Python vs Rust on 1k frames; endless leader share ≥ c3_4's (~30%) |
| M4 | **PPO fine-tune** from the BC net, with a KL anchor to the teacher (decaying), against a league of rule bots, the planner and past selves | `runs/vision/ppo_*.pt` | endless leader share > planner (40%) on the mean of 5 seed sets and on ≥ 3 of 5 |
| M5 | Scale and tune (resolution, frame stack, reward shaping toward leader share) | | each change kept only if it wins on ≥ 2 seed sets |

**Reward for M4.** Δ log(own mass), plus a bonus while we are #1 on the leaderboard (the goal metric), minus a
death penalty. The value head sees the leaderboard rank as a scalar.

**Training data size.** The teacher costs ~0.3 ms per decision. With 10 cores, BC data collection gives about
10⁷ labelled frames per hour. That is enough for M2; PPO in M4 is limited by the GPU, not by the engine.

## Resources and rules

- Rust for the engine, renderer, bindings and evaluation agent; Python only for the network and training loops.
- GPU (RTX 5090): training only, VRAM ≤ 26 GB, power 400 W. The existing ffa1 RL run stays paused; this project
  is separate and starts only when approved.
- CPU ≤ 90%, so the VecEnv thread count is capped (`--cpus`). Live servers (9600/9700) stay pinned to cores 11-15.

## Open questions

1. Is a size channel acceptable as "visual", or must frames be **pure RGB** like the screen? (The plan runs both
   as an ablation in M2.)
2. Teacher: the **planner** (best, ~0.3 ms) or c3_4 (simpler, faster labels)? The plan uses the planner, with c3_4
   as fallback.
3. Should it later run against human-style clients (screen capture)? Not planned now; the renderer could be made
   to match the client's look later if wanted.

## Effort estimate

| milestone | effort |
|---|---|
| M0 | half a day |
| M1 | 1 day |
| M2 | 1-2 days (mostly training time) |
| M3 | half a day |
| M4 | multi-day training with checkpoints in between |

Each gate is a natural point to decide whether to continue.
