# EMERGE: an agar.io that grows an economy

Crate: `rust/emerge` (macroquad, single process: simulation + client, no network).
Run: `cd rust/emerge && cargo run --release -j 2`

## Concept

Agar.io is already a tiny ecology: pellets are a faucet, growth is compounding, bigger eats smaller,
hunting and fleeing are the whole game. EMERGE keeps that as stage 0 and lets new rules appear only
when the *world state* reaches a measurable condition (shown as a progress bar in the HUD), never
from a menu. By stage 5 the same cells are miners, haulers, pact members and traders in a priced
market with faucets, sinks, inflation and inequality.

### Substrate ideas used (from the Substrate notes)

| Idea | Where it shows up |
|---|---|
| Complexity from minimal primitives (distinction / constraint / coupling) | Stage 0 has one primitive (a pellet is a distinction, a cell is a persistent structure). Stage 1 adds a constraint (lossy eating), stage 3 adds coupling (pacts bind cells). Each layer exists only once the layer below is stable. |
| Layered emergence, detected by metrics not by fiat | Stage gates are measured conditions: largest persistent structure, pellet scarcity, ore mined, pact mass, ore delivered. HUD shows the live metric. |
| Reversibility / conservation, "no hard-coded constants, no engineered conservation laws" in spirit | One closed mass ledger: pellets + cells + cargo + plates + armor + station structure + veins + heat. Only explicit faucets add mass (sunlight, newcomers). The HUD shows the conservation error (it stays at ~0). |
| Landauer: irreversible operations cost heat | Every lossy act (20% of a meal, dash, mass decay of big cells, refining, armor wear-off, destroyed loot) dumps mass into a heat lattice. Heat diffuses and blooms back into pellets and ore, so fights create farmland. |
| Fakeon / ancillary state | Heat is bookkeeping that is never an asset: it cannot be owned or traded, but it is essential to the dynamics. |
| Weak emergence / computational irreducibility | Nothing about specialization, inequality or prices is scripted; it only appears by running the rules. |
| Time as process, observers see layers | The tick is the only clock; the "stage" is purely what the observers (HUD) report from measured state. (Wolfram-style causal graphs, bounded-observer fog and asynchronous updates are future work.) |

### EVE Online lessons used

| Lesson | Mechanic |
|---|---|
| Faucets and sinks | Credits are minted only by drifter bounties and station grants; burned by a 1% broker fee, 40% of looted credits lost on a kill, abandoned treasuries. Faucets/sinks are shown live. |
| Destruction as a sink | Eating is 80% efficient; a dead cell's plates, armor and part of its ore and credits turn to heat. Armor wears off, so plate demand never ends. |
| Veteran capital compounds | Mining yield scales with sqrt(mass) and with learned skill; big cells keep growing. Newcomers respawn at mass 10 with nothing. Gini and top-10% share are on screen. |
| Specialization | Bots are forager / hunter / miner / trader. Per-role skill grows by doing, feeds back into role choice (with crowding), so niches lock in. |
| Empires, welfare, N+1 | Pacts give speed when close, found stations (havens), and pay a share of the station treasury to members as dividends. Outsiders pay higher fees. |
| Money printing and stagflation | At stage 5 a dev patch halves mining yield and multiplies drifter bounties, so miners flip to hunting, money supply rises while ore falls. The patch later reverts, the extra credits do not. |
| Market with price discovery | Each station quotes ore bid and plate ask from its own stock (scarce = dearer). Prices differ between stations, so traders arbitrage plates between them. Station treasury limits what it can buy. |
| Scams / betrayal | Hunters inside a pact may defect when a richer, smaller pact-mate is close. |

## Stages (all emerge from play, see HUD bar)

0. SOUP: pellets, eat, grow. Gate: a cell reaches mass 140 (or 50 s).
1. HEAT: eating is lossy, big cells decay, heat blooms into food, SPACE dashes. Gate: pellets below 45% of capacity (min 35 s in stage).
2. SCARCITY: sunlight dims, ore veins appear (hold still inside to mine), R burns ore into mass, red drifters roam. Gate: 250 ore mined.
3. PACTS: F proposes a pact, G leaves; no friendly eating, speed bonus near mates. Gate: a pact of 2+ cells with 400 mass.
4. STATIONS: B founds a station (80 ore + 40 mass). Havens, refinery (4 ore -> 1 plate + heat), barter 5 ore : 1 plate (key 1), plates become armor (key 3). Gate: 150 ore delivered or 50 s after the first station.
5. MARKET: credits, priced trades (keys 1-4), bounties, fees, dividends, dev patch, Gini / money / price panels.

## Controls

Mouse steers (distance = throttle). Wheel zoom. SPACE dash, R burn ore, F pact, G leave, B build station, 1 sell ore, 2 buy plate, 3 wear plate, 4 sell plate, H help, M ledger panel, `]` skip to next stage (debug), `+`/`-` sim speed, Esc quit.
Mining needs a low throttle: keep the cursor near your cell while inside a vein.

## Command line

- `emerge --sim N` headless: N ticks (30/s), prints stage transitions, economy stats, conservation check, top agents. `--seed`, `--bots`, `--dump`.
- `--screenshot out.png [--shot-at SECS] [--stage N] [--warm SECS] [--autoplay] [--zoom Z] [--help-open]` renders one frame and exits. `--stage N` pre-runs the world (forcing a stage every 25 s) so structures exist.
- `--autoplay` lets the bot AI drive the player cell.

Screenshots from the build: `docs/emerge/stage0_help.png`, `stage2.png`, `stage4.png`, `stage5.png`.

## Implemented vs future

Implemented: everything above, 27 bots + drifters, one-process play, headless self-test (`--sim`) with an exact mass ledger.
Not yet: splitting/ejecting, fog of war (bounded observers), asynchronous / causal-invariant updates, player-posted limit orders (stations are automated market makers), inheritance when players quit, contracts and loans, price manipulation and scams beyond defection, regulation/taxes as policy, multiplayer (the sim is deterministic per seed and could be moved behind the `ogar` server), tuned pacing for human players (stages take about 3 minutes with bots alone).
