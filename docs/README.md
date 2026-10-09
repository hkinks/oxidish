# agario-bot docs

| Doc | What |
|---|---|
| [plan.md](plan.md) | the original browser bot (Playwright + Jev strategist) |
| [training-plan.md](training-plan.md) | history of the arena, evolution, QD and real-server PPO work, with results |
| [sim.md](sim.md) | the GPU simulator: mechanics, encoder, fidelity checks |
| [sim-training.md](sim-training.md) | **current work:** PPO self-play in the simulator, the evolution loop, viewer, evaluation |
| [code-map.md](code-map.md) | module roles and how the pieces fit |
| [showcase.md](showcase.md) | the real-server showcase for playing against the bots |
| [sparring.md](sparring.md) | sparring opponents: AposBot port, hunters (virus play), strength and endless results |
| [learnings.md](learnings.md) | **what we learned**: metrics, genes, vision bot, RL pitfalls, commander |

Layout:

```
bot/     rule-based bot: game state, steering, food routing, tactics, params (genes)
arena/   real-server harness: MultiOgar wrapper (server/arena.js), clients, tournaments, evolution, QD, showcase
rl/      networks and trainers: real-server PPO (ppo.py), sim PPO (ppo_sim.py), evolution (evo.py)
sim/     GPU simulator, encoder, tournaments, viewer
rust/    Rust crates: arena_client (bot policies), ogar (server, engine, vision bot), ogar_py, game_client, emerge
runs/    checkpoints and genomes (not in git)
logs/    logs and TensorBoard (not in git)
vendor/  MultiOgar server and web client (not in git; see plan.md and client-hook.patch)
```
