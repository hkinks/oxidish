# Earlier phases

Before the Rust server, the project went through four phases whose code has since been removed:

- **Browser bot** ([browser-bot.md](browser-bot.md)): a Python rule bot that played in a real Chromium window through
  Playwright, with Jev making mode choices, against a local Node.js MultiOgar server.
- **Node MultiOgar harness** ([python-era.md](python-era.md)): a headless Python client and match runner around the
  Node server, used for gene evolution (CMA-ES), MAP-Elites and the first real-server PPO.
- **GPU simulator** ([sim.md](sim.md)): a batched PyTorch re-implementation of the arena, built to feed PPO faster
  than the Node server could.
- **Sim-era RL** ([sim-training.md](sim-training.md), [architecture-research.md](architecture-research.md)): PPO,
  PBT and evolution in the simulator; the research that led to the vision bot.

The Rust server (`server/`) replaced the Node harness and the simulator, and the vision bot replaced the sim-trained
nets. The lessons that still matter are in [../learnings.md](../learnings.md). The code is in git history at tag
`python-era` (commit `436fb38`): `git checkout python-era` to browse it. Paths in these files refer to that layout.
