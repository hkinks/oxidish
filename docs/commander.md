# Commander layer (Phase 4: a strategist on top of the bots)

A commander (a human, a Claude session over MCP, Jev, or a heuristic) gives a running bot high-level orders: who to
hunt, where to go, when to hide, whether splitting is allowed, when to shoot viruses at a big player or eat the pieces of
a popped one. The bot keeps its reflexes: **an order never walks a bot into a bigger cell's danger radius.**

Everything lives in the Rust server (`server`, control port) and works for rule bots, the planner
(`bots/specs/plan_c34.json`) and vision agents (`"kind":"vision"`). The Python side (`training/eval/commander*.py`) is only glue:
the autonomous loop, the LLM backends and the MCP server.

## Orders

Control-port commands (same line protocol and optional token as `add` / `speed`; reply is one line):

```
order <agent> hunt <player name | #cell_id> [ttl=S]
order <agent> harvest <target> [ttl=S]        eat the pieces of a popped player (best cloud of its edible pieces)
order <agent> virus-split <target> [ttl=S]    shoot viruses at a bigger player so they pop it
order <agent> goto <x> <y> [ttl=S]
order <agent> farm | flee | hide [ttl=S]
order <agent> split on|off [ttl=S]            go / no-go for split attacks (independent of the verb)
order <agent> free                            clear everything
orders                                        active orders
```

`<agent>` is an exact agent name (`plan-0`) or a unique prefix. `ttl` is game seconds (default 10); afterwards the
verb (and, separately, the split flag) reverts to free. A name target matches exactly (case-insensitive), else a unique
substring; `#123` is a cell id (resolved to its owner, so it follows the player when it splits). The target is
re-resolved every decision, so an order on a player who died does nothing until it respawns.

### What each verb does

| verb | rule bots / planner | vision agents |
|---|---|---|
| `hunt X` | plan = Hunt on X's best edible cell, any distance (target must be >1.3x lighter, else ignored) | the net's aim point is replaced by X's cell |
| `harvest X` | like hunt, picking the cell whose neighbourhood (250 u) holds most edible mass | like hunt |
| `goto x y` | farm plan plus a strong pull to the point; threat repulsion and walls still apply | aim point replaced |
| `farm` | plan = Farm (no hunting) | net acts (no override) |
| `flee` | plan = Flee | net acts |
| `hide` | plan = Hide if a virus bigger than 1.3x our mass is in view (else Flee if threatened, else Farm) | net acts |
| `virus-split X` | the virus-shooting tactic is forced on, for X only (needs a virus within 700 of us and X within 900 of the virus); we walk round the virus to line up | net acts |
| `split off` | no split attacks (also the double split) | net's split vetoed |
| `split on` | split attacks allowed as the bot's own safety rule decides (it cannot force an unsafe split) | net's split enabled even if the spec has `split:false` |

**Veto.** If the bot's own plan this step is Flee (a bigger cell inside its danger radius) the order is skipped
(`order_ignored` counter), and for vision agents `hunt`/`harvest`/`goto` are skipped while a threat is in reach. Steering
keeps its threat repulsion under every verb. Consequence: `virus-split` only fires while the target is outside the
danger radius (~1000-1400 u gap), which is a narrow window; it is a real tactic, not a trick to get near a giant.

The planner (`kind: plan`) evaluates its candidate openings from the ordered base action, so the order replaces the base
action it evaluates; its danger search can still override it.

Counters in the agent stats: `order_steps` (order acted on) and `order_ignored`.

## Scene (observation)

`scene <agent>` returns one JSON line (relative to the agent's biggest cell, x right, y down, world units):

```json
{"agent":"plan-0","t":289.6,"alive":true,
 "me":{"mass":2443,"cells":2,"x":-496,"y":2463,"biggest_cell_mass":1697,"kills":32,"deaths":0,"mode":"hunt","in_danger":false},
 "leaderboard":[{"name":"c34-0","mass":2451,"you":true}, ...10 entries],
 "players":[{"name":"Moon","cell_id":6723,"mass":561,"biggest_cell_mass":422,"cells_visible":2,"cells":2,
             "x":-510,"y":566,"dist":762,"gap":144,
             "can_eat_us":false,"we_can_eat":true,"split_threat":false,"split_kill":true,
             "virus_pop_possible":true,"popped":false}],
 "viruses":[{"x":-1942,"y":..,"mass":101,"gap":1557,"pops_us":true,"hides_us":false}],
 "walls":{"left":..,"right":..,"top":..,"bottom":..},
 "order":{"verb":"hunt Moon","ttl_left_s":6.2,"split":"off"}}
```

`players` are the 12 nearest visible players (grouped per owner; `x,y,dist,gap` refer to the nearest cell, `gap` = edge to
edge). `split_kill`: we could split onto them; `split_threat`: they could split onto us; `popped`: 3+ cells visible.
`viruses` are the 6 nearest. Also: `agents` (JSON list of agents with kind/mass/order, game time, speed),
`leaderboard` (top 10 JSON), `frame <agent> [path]` (256 px grayscale PNG of the agent's camera, the vision input; without a
path the reply is `png-base64:...`).

## MCP server

`training/eval/commander_mcp.py` (Python, official `mcp` SDK, FastMCP; pinned `mcp<2`). Python rather than Rust `rmcp` because the
server is a thin stdio-to-TCP adapter (about 100 lines), `uv run --with` needs no build step or repo dependency, and the
SDK's `Image` type returns frames properly.

Tools: `list_agents()`, `scene(agent)`, `leaderboard()`, `order(agent, verb, arg, ttl)`, `orders()`,
`add_agent(spec, count)`, `set_speed(x)`, `pause()`, `resume()`, `frame(agent) -> image`.

Register it yourself (nothing in this repo touches your config):

```
claude mcp add agar-commander -- uv run --no-project --with "mcp<2" python training/eval/commander_mcp.py --port 9601
```

Run it from the repo root (or give the absolute script path). Host/port/token: `--host --port --token` or
`OGAR_CONTROL_HOST / OGAR_CONTROL_PORT / OGAR_CONTROL_TOKEN`. Then ask the session e.g. "list the agents, look at the
planner's scene, and have it hunt the leader for 15 s". Tested end to end with an MCP client (initialize, list tools,
`list_agents`, `order`, `frame`, `scene`, `set_speed`).

## Autonomous loop

`training/eval/commander.py`: every 1.5 game seconds per commanded agent it reads `scene`, asks a backend for one order, applies it
(verb with a 4 s ttl, plus a split on/off), and logs to `logs/commander-*.jsonl` (time, latency, mass, decision, replies).
A failing backend leaves the last order running until it expires.

```
uv run python -m training.eval.commander --port 9601 --agents plan --backend heuristic
```

Backends (credentials are only read from the environment, never from or into files):

| backend | needs | what it does |
|---|---|---|
| `heuristic` | nothing | rules over the scene: hide/flee when an eater is close, otherwise hunt (or harvest) the best prey, split only when safe, virus-split a big player when nothing else is on |
| `jev` | `JEV_API_KEY` (optional `JEV_BASE_URL`, `JEV_MODEL`) | `training/eval/systemone.py` `/v1/systemone`: a `choice` question over the options (farm, flee, hide, hunt pN, harvest, virus-split) plus a `noul` split question, like `bots/specs/strategist.py` |
| `anthropic` | `ANTHROPIC_API_KEY` (optional `COMMANDER_MODEL`, default `claude-haiku-4-5-20251001`) | one Messages API call per decision, the model answers one JSON order line; replies are validated before they are sent |

On the machine where this was built neither `JEV_API_KEY` nor `ANTHROPIC_API_KEY` was set, so only the heuristic backend
was run live; the Jev and Anthropic paths are unit-tested with fakes (`training/eval/test_commander.py`) but have not talked to the
real APIs.

## Running it on a live server

The live servers' control ports are 9601 (for 9600) and, if started with one, 9701. Start the commander against it:

```
uv run python -m training.eval.commander --port 9601 --agents c3_4 plan --backend heuristic
```

(commands go to every agent whose name starts with a given prefix; agent names are `<spec name>-<i>`, see `agents`). Orders
are plain control-port lines, so `echo "order plan-0 hunt Moon ttl=20" | nc 127.0.0.1 9601` works too. Remember that LLM
latency is real time: at server speed 1 a 400 ms Jev call is 10 game ticks; at higher speed the commander lags
proportionally more.

## Measurement

`training/eval/commander_eval.py`: per world an `ogar serve` (own ports) with the candidate, the usual rule-bot field (champ, r2g9,
safe, ramp_r2, ramp300) and 8 server bots; leader share = share of 5 s samples (after a 2 game-minute warm-up) where the heaviest
player of the whole world is the candidate; mean mass counts death as 0.

Heuristic backend, 20 game-min per world, server at speed 4 (the heuristic has no latency; LLM backends must use
`--speed 1`), same seeds on and off, `plan` = `bots/specs/plan_c34.json`, `vis` = `bots/specs/vision_bc3.json`
(`runs/cmd_{plan,plan12,vis}_{off,heuristic}.json`):

| base bot | worlds (seeds) | commander | leader share % | mean mass | orders |
|---|---|---|---|---|---|
| planner | 4 (51000+) | off | 13.0 | 1966 | 0 |
| planner | 4 (51000+) | heuristic | 34.1 | 3006 | 2637 |
| planner | 12 (61000+) | off | 33.3 | 3202 | 0 |
| planner | 12 (61000+) | heuristic | 22.3 | 2247 | 7880 |
| vision | 4 (51000+) | off | 22.4 | 2219 | 0 |
| vision | 4 (51000+) | heuristic | 2.5 | 973 | 2714 |

Reading: **the heuristic commander does not help.** The two planner sets disagree in sign (+21 and -11 points): per-world
leader share swings from 0 to 90 %, so 4 worlds say nothing; pooled over the 16 worlds it is 28.2 -> 25.2 %, within noise but
not positive. For the vision bot it clearly hurts (22.4 -> 2.5 %, every world worse): there only `hunt`/`harvest`/`goto`
change anything (the net keeps `flee`/`farm`/`hide`), and pointing the net straight at prey drops what the net knew about
the rest of the frame. The likely cause is unverified (no ablation was run). The loop, orders and logging work as designed;
the heuristic decision rules are a stand-in, and a stronger judge (Jev or an LLM) was not measured because no credentials
were available (0 LLM calls). Interpretation: as the training plan predicted, once the policy handles reflexes a strategy
layer adds little unless it knows something the bot does not.

## Tests

- `bots/brain/src/order.rs`: hunt retargets the policy, the Flee reflex vetoes orders, a bigger target is refused,
  `split off` vetoes the split, harvest prefers the cloud, goto, virus-split shoots a virus (and does not without the order).
- `server/src/commander.rs`: order parsing (accepted and rejected forms), base64.
- `training/eval/test_commander.py`: heuristic choices, option building, reply validation, Jev mapping with a fake client, loop step.
