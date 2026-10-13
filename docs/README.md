# oxidish docs

oxidish is a Rust agar.io server (`server/`, a port of MultiOgar) plus the bots, drills and training tools built
around it. Start with [training.md](training.md) to run things and [status.md](status.md) for where the bots stand.

## Start here
| Doc | What |
|---|---|
| [training.md](training.md) | write, test, evaluate and train bots: specs, the external-bot protocol, `oxidish-server match/endless/drill`, `scripts/*.sh`, vision PPO |
| [status.md](status.md) | results tables, the best bots (`bots/specs/best/`: vision, vision-old, planner, rules), next steps |
| [playing.md](playing.md) | play against the bots: `oxidish-server serve`, the control port, the native client, recording human games |
| [learnings.md](learnings.md) | what we learned: metrics, genes, vision bot, drills, pitfalls |

## Bots and skills
| Doc | What |
|---|---|
| [vision-bot.md](vision-bot.md) | the CNN bot that plays from rendered frames; milestones M1-M8 with results |
| [lookahead-planner.md](lookahead-planner.md) | the planner policy (`"kind": "plan"`) and failed variants |
| [sparring.md](sparring.md) | hunter opponents used for evaluation |
| [commander.md](commander.md) | high-level orders over the control port, MCP server, LLM loop |

## Measurements
| Doc | What |
|---|---|
| [harvest-drill.md](harvest-drill.md) | `oxidish-server drill`: eating a giant that just popped on a virus |
| [escape-drill.md](escape-drill.md) | `oxidish-server escape`: fleeing a bigger human-like chaser |
| [human/](human/) | generated reports of recorded human games (scenes, who ate whom) |

## Server
| Doc | What |
|---|---|
| [server.md](server.md) | the Rust server: validation against MultiOgar, known gaps, how to run |

## Layout
```
server/            oxidish-server: server, engine, frame renderer, drills, endless evaluator, vision inference (Apache-2.0)
client/            oxidish: the native client
bots/brain/        bot policies in Rust: rules, planner, hunter, reflexes, evade (crate oxidish-bot)
bots/specs/        JSON bot specs; best/ holds the four reference bots
bots/models/       ONNX models of the vision bots
training/gym/      oxidish_gym: the engine as a Python module (PyO3) for RL
training/rl/       vision-bot training (imitation, DAgger, PPO) and ONNX export
training/eval/     evaluation and report tools (endless, tournament, verdict, drill / escape / human reports, commander)
scripts/           client.sh, train.sh, eval.sh, export.sh
runs/, logs/       local checkpoints and logs (not in git)
```

## History
[history/](history/) holds the pre-Rust work: the browser bot, the Node MultiOgar harness, the GPU simulator and its
PPO and evolution experiments. The code was removed (git tag `python-era`); the write-ups stay because their lessons
shaped the Rust-era work.
