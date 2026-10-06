# ogar: the Rust MultiOgar server (rust/ogar)

A port of vendor/MultiOgar (FFA mode) plus our arena.js launcher. Two uses: a headless match runner for
evaluation throughput (`ogar match`) and a real-time websocket server (`ogar serve`, see docs/showcase.md).
It reuses the rule-bot policy code of `rust/arena_client` (now also a library crate), so in-process agents make
the same decisions as the network client; they see the same Snapshot (integer positions/sizes, cells in the
server's view box for that player).

Parity is statistical, not bit-for-bit: different RNG consumption, a uniform grid instead of the quadtree, own
iteration orders. Constants come from vendor config.ini (`mts(x)` = sqrt(100x)+.5, so start size 32.12, split
min 59.66, virus 100.5 / 141.9, eject 36.56 / loss 41.73, eject spawn chance .5, ...) and arena.js overrides
(map 7000 → food 367 x `--food-mult`, 12 viruses min, scrambleLevel 0).

## Validation (15 game-min, food x2, 8 server bots, 5 rule bots in each match)
Node = `arena.tournament --rust` (real server + Rust client), 48 matches, seeds 1-48. ogar = 144 matches
(seeds 1-48 and 1000-1095). Cells show Node | ogar (relative difference). 95% CIs over matches are about
±0.2 on log-mass, ±20% on avg/peak mass, ±0.5-0.8 on deaths.

| bot | log-mass | avg mass | peak | deaths | kills |
|---|---|---|---|---|---|
| ramp300 | 7.14 / 7.14 (0%) | 1607 / 1562 (-3%) | 4602 / 4771 (+4%) | 2.02 / 2.04 (+1%) | 63.5 / 67.6 (+6%) |
| ramp_r2 | 7.34 / 7.12 (-3%) | 1870 / 1591 (-15%) | 5261 / 4761 (-10%) | 1.88 / 2.42 (+29%) | 66.2 / 66.8 (+1%) |
| r2g9 | 6.85 / 6.91 (+1%) | 1234 / 1281 (+4%) | 4120 / 4115 (0%) | 3.54 / 3.20 (-10%) | 69.3 / 71.8 (+3%) |
| safe | 6.89 / 6.95 (+1%) | 1210 / 1335 (+10%) | 3594 / 3692 (+3%) | 2.00 / 1.99 (0%) | 41.0 / 40.3 (-2%) |
| champ | 6.63 / 6.81 (+3%) | 963 / 1153 (+20%) | 3005 / 3693 (+23%) | 4.08 / 3.45 (-15%) | 47.2 / 49.3 (+5%) |

Head-to-head (row beats column, share of matches by mean log-mass): the orderings agree up to noise.
Node: ramp_r2 > ramp300 (58%) > safe ~ r2g9 (50%) > champ; ogar: ramp300 ~ ramp_r2 (50%) > r2g9 (62% vs safe) ~ safe > champ
(champ loses to every other bot in both, 29-38% on ogar vs 27-46% on Node). Every pair difference is inside the
Wilson intervals (±14%). Systematic tendencies worth knowing: in ogar the spread between bots is slightly
compressed (the weakest bot, champ, looks a bit better; ramp_r2 a bit worse), the loss shares by killer type match
(server bots 24% / 23% of all lost mass). Re-validate a candidate on Node before trusting a close call.

Speed (one 22500-tick match, 5 rule bots + 8 server bots, machine at ~70% load): Node + Rust client 24 s wall
(two busy processes), ogar 5.4 s on one core (7 s/match when 2 threads share the machine) = about 4.5x faster
per match and ~7-9x per core: ~500-660 long matches per hour per core vs ~75-150 for Node. Serve mode runs
unthrottled at ~100x real time with 8 in-process agents.

## Known differences / gaps
- Only FFA (no Teams/Experimental/mother cells), no chat (disabled like arena.js), no user roles / OP keys, no
  minions (removed), no bad-word filter, IP limits, stats HTTP server, scrambling. Protocol 6 is what is sent
  (the web client and game_client use it); other protocol numbers are accepted but get protocol-6 packets.
- Policies: rule bots only (route / field / name=genome.json); NN / Jev policies still need the Node path.
- Loss records of in-process agents use the killer's whole cell count/mass rather than the visible part.
- Eject cells stay "moving" with an exponentially vanishing boost forever, as in the Node build (ported on purpose).
- Disconnected humans' cells stay 60 s, like Node.

## Run
```bash
cargo build --release -j 2 --manifest-path rust/ogar/Cargo.toml
rust/ogar/target/release/ogar match --agents <lineup> --ticks 22500 --food-mult 2 --seed 1 --matches 48 --threads 2 --out rows.json
uv run python -m arena.tournament --engine ogar --agents <lineup> --matches 48 --workers 2 --ticks 22500 --food-mult 2
uv run python -m arena.verdict ... --engine ogar        # and arena.long_search ... --engine ogar
```
Lineup syntax as in arena.tournament (`name=genome.json:N`, `route:N`, `field:N`). `OGAR_BIN` overrides the binary path.
