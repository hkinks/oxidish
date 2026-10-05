# Code map

How the pieces fit, from the simulator up to the real-server showcase.

```
sim/engine.py        batched GPU world (MultiOgar rules)
   |  sim/obs.py (grid encoder), sim/entities.py (token encoder), sim/bots.py (scripted opponents)
   v
sim/env.py  SimEnv   reset/step, rewards, respawn, curriculum, per-life stats
   v
rl/ppo_sim.py  Learner   PPO for one network on a slice of SimEnv arenas (+ league, bots, drills)
   v
rl/pbt.py (one process, population)      rl/evo.py (one trainer process per member)
   |  both: genomes + mutation (rl/evo.py), evaluation via sim/tourney.py, results in runs/rl/<name>/evo.json
   v
sim/tourney.py, sim/progress.py (evaluation)      sim/watch.py (+ watch.html, live browser viewer)
   v
checkpoints (rl/common.save) -> rl/policy.py NNPolicy -> arena/ (showcase, autojoin) on a real MultiOgar server
```

## sim/ - the GPU simulator
- `engine.py`: `World`, B arenas x P players x K cells, all tensors. Physics; do not change without re-running `compare.py`.
- `obs.py`: grid encoder ([13,48,48] + 16 scalars), batched copy of `rl/obs.py` + `rl/guided.py`.
- `entities.py`: entity-token encoder (nearest visible objects) for `rl/entity_model.py`.
- `env.py`: `SimEnv`, the RL environment (same actions/reward/respawn as the real-server `rl/env.py`).
- `bots.py`: vectorised scripted opponent (`scripted_actions`) and nearest-pellet farmer (`farmer_actions`).
- `tourney.py`: shared-arena tournament (`run`) and farming-alone check (`farm`); fitness = mean log of average mass.
- `progress.py`: learning curve of a run (newest checkpoint vs. fixed past ones), every N minutes.
- `watch.py` + `watch.html`: live viewer of runs / populations in the browser.
- `compare.py`, `compare_obs.py`, `test_sim.py`, `bench.py`: fidelity checks against the real server / Python encoder, tests, throughput.

## rl/ - networks and trainers
- `actions.py`: discrete action space (16 headings + stay, split, eject).
- `model.py` (`Policy`, CNN) and `entity_model.py` (transformer over entity tokens): same interface (`dists`, `logp_entropy`, `act`, `binary_eps`).
- `ppo_sim.py`: `Learner` (one network, its arenas, league, PPO state), `collect` (rollout over several learners), `main` (CLI: `build_parser`, `build_start_model`, `build_hparams`).
- `pbt.py`: `Population`: all members train in one process; a side process evaluates; `select` ranks, promotes to the hall of fame, replaces the bottom half by mutated copies.
- `evo.py`: genome (architecture, input masks, hyper-parameters), `mutate`, and the older one-process-per-member evolution loop.
- `common.py`: checkpoint IO (`save`/`load`), opponent pool, env configs. `policy.py`: `NNPolicy`, plays a checkpoint on the real server (`nohint.py`, `obs.py`, `guided.py` build its observations).
- Real-server PPO (history, see training-plan.md): `ppo.py`, `bc.py`, `vec.py`, `env.py` (ArenaEnv), `guided.py` (residual net over the champion).

## bot/ - rule-based bot
`params.py` (genome of the bot), `steer.py` (potential-field steering), `farm.py` (food routes), `tactics.py` (virus skills), `modes.py` (rule-based mode choice), `game.py` (state `Snapshot`/`Cell`), `main.py` + `strategist.py` + `systemone.py` + `hud.py` (browser bot with the Jev strategist).

## arena/ - real MultiOgar server
`server/arena.js` (server wrapper), `client.py` (lockstep client), `policies.py` (arena policies from picklable specs), `match.py` / `tournament.py` (seeded matches), `evolve.py` (CMA-ES over bot params), `qd.py` (MAP-Elites), `stuck.py`, `food.py`, `apos.py`, `watch.py`; showcase: `showcase.py`, `join.py`, `autojoin.py`, `showcase_stats.py` (see showcase.md).
