# EMERGE: an agar.io that grows an economy

Crate: `rust/emerge` (macroquad, single process: simulation + client, no network).
Run: `cd rust/emerge && cargo run --release -j 2`

## Concept

Agar.io is already a tiny ecology: food specks are a faucet, growth compounds, bigger eats smaller,
hunting and fleeing are the whole game. Real economies mirror it (big firms absorb small ones, wealth
compounds, resources are finite). EMERGE keeps plain agar.io as stage 0 and lets new rules appear only
when the *world state* reaches a measurable condition (a progress bar in the HUD), never from a menu.
By stage 5 the same cells are gatherers, hunters, colony members and traders, trading nutrients and
chitin for spores (stored energy used as money) at nests whose prices follow local scarcity.
The setting is a living pond: cells, specks, nutrient patches, detritus, spiked parasites, colonies and
their nests. (The economic lessons come from the EVE Online notes; the setting is not EVE.)

## Stages (each emerges from play)

0. SOUP: food specks, eat, grow. Gate: a cell reaches mass 140.
1. DECAY: eating is lossy, big cells shrink, detritus feeds new growth, SPACE lunges. Gate: food below 45% of capacity.
2. SCARCITY: sunlight dims, nutrient patches appear (hold still inside to gather), R digests nutrients into mass, spiked parasites roam. Gate: 250 nutrients gathered.
3. SYMBIOSIS: F bonds with a nearby cell into a colony (no friendly eating, faster when close), G leaves. Gate: a colony of 2+ cells with 400 mass.
4. COLONIES: B builds a nest (80 nutrients + 40 mass): a refuge nobody can be eaten in, with a digester (4 nutrients -> 1 chitin + detritus). Nutrients are traded for chitin at 5:1 (key 1), chitin hardens the shell (key 3). Gate: 150 nutrients delivered or 50 s after the first nest.
5. EXCHANGE: spores exist. Nests quote prices from local stock (keys 1-4). Parasite harvests and first-bloom grants make spores; tolls and lost loot destroy them. A drought halves patch yield while parasite blooms inflate harvests. Big colonies take over small ones.

## Mechanic to Substrate idea to economic principle

| Mechanic | Substrate idea | Economic principle |
|---|---|---|
| One closed mass ledger (specks, cells, cargo, chitin, shells, nests, patches, detritus); HUD shows conservation error (~0) | Conservation / reversibility: nothing is created or lost except through explicit processes | Faucets and sinks must be accounted for |
| Lossy eating, lunging, decay of big cells, digesting, shell wear, destroyed loot all produce detritus; detritus diffuses and regrows specks and nutrients | Landauer: irreversible operations cost heat; heat is ancillary state that is never an asset | Destruction as a sink that fuels renewed production (fights make farmland) |
| Stage gates are measured world states (largest cell, food density, nutrients gathered, colony mass, nutrients delivered) | Layers of structure from simple primitives, detected by metrics, not fiat | Economies bootstrap in layers: subsistence, extraction, organisation, then markets |
| Specialization, inequality, prices are not scripted | Weak emergence: only visible by running the rules | Specialization by learning-by-doing with crowding; Gini and top-10% share shown |
| Distinction (specks), constraint (lossy eating), coupling (bonds) added in that order | Primitives distinction / constraint / coupling | Alliances reduce risk but invite free riding |
| Gathering yield scales with sqrt(mass) and skill; newcomers start at mass 10 | Compounding structure persists longer | Veteran capital advantage |
| Colonies share a nest, pay members dividends from its treasury; outsiders pay higher tolls; hunters may defect | Coupling between structures | Alliances that can betray, welfare to hold labour |
| Spores minted only by parasite harvests and grants, burned by a 1% toll and 40% of loot lost on kills | Explicit accounting of what enters/leaves the closed system | Faucets and sinks of a currency, inflation |
| Nest prices from local stock; traders carry chitin between nests | Local rules, global pattern | Price discovery and arbitrage |
| Drought halves yield and boosts harvests, then reverts; the extra spores stay | A perturbation of a rule, not of the state | Shock leading to stagflation-like prices |
| A dominant colony absorbs a much smaller one (members, treasury, nest) | Larger structures absorb smaller | Mergers and acquisitions |

## Controls

Mouse steers (distance = throttle). Wheel zoom. SPACE lunge, R digest nutrients, F bond, G leave colony, B build nest, 1 sell nutrients (hold to keep selling), 2 buy chitin, 3 harden shell, 4 sell chitin, H help, M ledger panel, `]` skip to next stage (debug), `+`/`-` sim speed, Esc quit.
Gathering needs a low throttle: keep the cursor near your cell while inside a patch.

## Command line

- `emerge --sim N` headless: N ticks (30/s), prints stage transitions, economy stats, conservation check, top agents. `--seed`, `--bots`, `--dump`.
- `--screenshot out.png [--shot-at SECS] [--stage N] [--warm SECS] [--autoplay] [--zoom Z] [--help-open]` renders one frame and exits. `--stage N` pre-runs the world so structures exist; `--autoplay` lets the bot AI drive your cell.

Screenshots: `docs/emerge/stage0_help.png`, `stage2.png`, `stage4.png`, `stage5.png`.

## Implemented vs future

Implemented: all of the above with 27 bot organisms plus parasites, an exact mass ledger, headless self-test.
Not yet: splitting, fog of war (bounded observers), asynchronous updates, player-posted limit orders (nests are automated market makers), inheritance, loans, richer takeovers, multiplayer, pacing tuned for humans (bots alone reach stage 5 in about 3 minutes).
