# Network architectures for the agar.io bots: research shortlist (2026-10-05)

Our current net: a 48x48 grid with 15 channels plus 16 scalars, then 3 convs (32 s1, 64 s2, 64 s2), then an MLP 256x2, with a separate critic and heads for 17 moves, split and eject. Its problems:
- it learns precise geometry poorly (imitating "nearest pellet": ~60% exact heading offline);
- it almost never splits;
- it farms ~250 vs a routine's 300-318;
- continued PPO sometimes regresses.

Unverified items are marked. The GoBigger winners' write-ups could not be retrieved.

**What other agar.io work used:**
- **AgarCL** ([arXiv 2505.18347](https://arxiv.org/abs/2505.18347), verified):
  - a pixel CNN (128x128x4, then 8x8 / 4x4 / 3x3 convs with LayerNorm), trained with PPO, DQN and SAC;
  - a continuous cursor plus a discrete split/eject choice, with frame skip 4;
  - no agent learned split or virus sequences, the same wall we hit.
- **GoBigger** ([ICLR'23](https://openreview.net/forum?id=NnOZT_CR26Z), [challenge repo](https://github.com/opendilab/GoBigger-Challenge-2021)):
  - structured scalar / spatial / team / ball observations, with MAPPO as the strongest baseline;
  - its baseline is probably an entity encoder plus a spatial branch plus discretized direction x action (unverified).

## Ranked shortlist

1. **Entity encoder plus pointer (target-selection) head.** Highest expected impact, on geometry, splitting and farming.
   - **Sources:** AlphaStar (Nature 2019), OpenAI Five (arXiv 1912.06680), Relational deep RL (1806.01830), Pointer Networks (1506.03134), Set Transformer (1810.00825).
   - **Tokens:** the top-K pellets by distance (K 32-64), enemy cells, viruses, ejected mass and our own cells (up to 16), each with a type embedding.
   - **Per-entity features:** relative x/y, relative velocity, log mass, mass ratio, merge timer, and can-eat / killable-by-split flags.
   - **Network:** a per-entity MLP (d 64-128), then 2-3 pre-LN transformer layers (4-8 heads, d 128, a self token), then pooling.
   - **Pointer head:** a query against the entity keys, plus a learned null entity for "stay". The chosen entity's relative position becomes the heading, and the split/eject heads are conditioned on the target's embedding.
   - **Why it fits:** exact float geometry instead of grid quantization ("nearest pellet" becomes an argmin over a feature), and "split at this target" gets clean credit.
   - **Cost:** 1-2 days. The sim already holds object tensors, so this needs a top-k gather plus padding masks.
   - **Risks:** PPO with transformers can be unstable (use pre-LN, a low lr and a small head init); the K cutoff leaves blind spots (add a far-density token).
2. **Hybrid: grid CNN plus an entity branch.** A low-risk path, usable as a PBT gene.
   - **Sources:** AlphaStar's entity plus spatial encoders; Transformers as Policies for Variable Action Envs (2301.03679, details unverified).
   - **Design:** concatenate the pooled vectors, or let entity tokens cross-attend to a 12x12 CNN map.
   - **Risk:** the CNN branch may dominate (mitigate with branch dropout, or by training the entity branch first).
3. **Imitation pretraining plus a decaying KL / imitation anchor during PPO.**
   - **Sources:** AlphaStar (supervised init plus KL to the teacher), Jump-Start RL (2204.02372), PPG (2009.04416).
   - **Design:** imitate the scripted farmer and a scripted split-kill rule, then run PPO with a decaying imitation loss.
   - **Why it fits:** it seeds splitting (AgarCL shows exploration won't find it) and reduces regression.
   - **Cost:** low, since the teachers exist.
   - **Risk:** staying stuck near the teacher, so decay the anchor.
4. **No-stride IMPALA-ResNet plus spatial softmax:** a cheap patch if we stay with the grid.
   - **Sources:** Levine et al. (1504.00702), IMPALA (1802.01561).
   - **Design:** remove the stride-2 convs, use ResNet blocks at 48x48, and replace flatten with spatial softmax (32 channels to 64 keypoint floats).
   - **Cost:** 3-5x the FLOPs.
   - **Risk:** spatial softmax suits one object per channel and is weak with many pellets.
5. **Action parameterization.** Best is the pointer from #1. The cheap alternative is 16 headings plus a learned continuous angle residual. A full continuous cursor is not recommended (Gaussian heads handle stay and discrete choices badly).
6. **Recurrence** (a GRU of 256 after pooling): lowest priority, since the inputs already carry velocity channels.
   - **Sources:** OpenAI Five, AlphaStar, GTrXL (1910.06764).
   - **Cheaper first step:** feed in the previous action plus a 2-frame history.

Also cited, from memory and unverified: UPDeT (2101.08001, entity-token transformer for SMAC), Neural MMO (2001.12004), and Lux AI winners using CNN/UNet.

**Suggested order:** #1 with #3's pretraining, then PPO with the KL anchor. #2 is the fallback, and #4 a side experiment on the existing net.

**Our own evidence so far:**
- 16 headings deciding every 160 ms farm as well as continuous aim (303 vs 266).
- CoordConv did not help the grid net imitate (62% vs 61%).

## Forgetting across modes (farm / hunt / evade): research shortlist (2026-10-05)

**The observation:** specialists trained from one seed farmer forget the skills they don't practise.

| Net | Farming from 10 | Gain from 1000 |
|---|---|---|
| seed farmer | 288 | +889 |
| evader | 228 | +313 |
| hunter | 102 | +251 |

**Sources:** from the agent's memory, not fetched, so the IDs and claims are unverified: kickstarting (arXiv 1803.03835), Policy Distillation (1511.06295), Actor-Mimic (1511.06342), Distral (1707.04175), multi-task PopArt (1809.04474), PCGrad (2001.06782), Soft Modularization (2003.13661), CARE (2102.06177), Progress & Compress (1805.06370), AlphaStar (Nature 2019).

**Recommendation:** no architecture change first. The modes can be told apart from the observation, so one net can switch implicitly as long as every mode keeps appearing in its training batches. Sequential phases alone would reproduce the forgetting.

Ranked:
1. **Interleaved multi-task PPO.**
   - Every learner gets every arena type in every batch, through `layout`: about 1/3 solo farming (half from 10 mass, half from 1000), 1/3 prey arenas, 1/3 arenas with big hunters.
   - An optional drill one-hot input is worth testing both with and without.
   - Target: at least 90% of each specialist on its own test, confirmed on fresh seeds.
2. **Kickstarting from the frozen specialists** on top of 1.
   - Loss: `L = L_ppo + lambda * KL(pi_expert(arena type) || pi)` on each arena type's states, with lambda annealed from 1 to 0.1. It covers all heads, plus value regression to the expert critic.
   - It is a dense anti-forgetting signal, at about 1.5x compute.
3. **DAgger distillation into one generalist** as a fallback if 1 is unstable: label mixed-arena states with the matching expert, where evade overrides hunt; then fine-tune with PPO.
4. **MoE**, only if 1 and 2 fall short: expert FFNs in the last transformer block with a top-1/2 router, or 3 head sets with a soft gate; each expert initialized from a specialist; a load-balancing loss. Risks: router collapse, and noisy routing under PPO.
5. **Per-task advantage normalization or PopArt, plus PCGrad if gradients conflict.** A pellet is worth ~1e-3 per step while a kill is large, so an unnormalized sum favours hunting.
6. **A manager over frozen specialists**, as a last resort: no forgetting, but 3x inference, and the frozen skills can't improve.

**What large agents did:** AlphaStar and OpenAI Five each ended with one network trained on a broad distribution with a distillation-like anchor (AlphaStar's KL to its human prior). The exploiters were training opponents and were not merged in.
