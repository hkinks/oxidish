# Human play report (human)

Recording: `logs/human_test`, sessions 1791371135; generated 2026-10-07. 3 loss scenes, 245 eat events.

## The human

Session length 4.2 min (251 one-second samples).
- **human**: alive 187s (75% of the session); **#1 for 36% of its alive time**; mass mean 812, median 888, p90 1627, peak 1795; every life was boosted (+500), numbers are inflated
- Who was #1 over the whole session (all samples): c3_4 45%, human 27%, safe 21%, server-bot 7%

## Mass taken from each bot family

| victim family | cells eaten | mass taken | bots killed (last cell) | victims that had just popped |
|---|---:|---:|---:|---:|
| server-bot | 12 | 660 | 2 | 4 |
| ramp | 2 | 291 | 1 | 0 |

Human deaths: **1**: tick 5167 by Moon (bot, cell 302, total 684, x6) at 65 mass
Human cells lost to: c3_4 1754, safe 739, server-bot 278, ramp 40

Biggest eater -> victim flows (mass), all players:
- server-bot ate server-bot: 5147
- c3_4 ate human: 1754
- safe ate server-bot: 1124
- c3_4 ate server-bot: 999
- safe ate human: 739
- human ate server-bot: 660
- server-bot ate safe: 331
- human ate ramp: 291

## Loss scenes

**Loss scenes eaten by a human**: 2 scenes, 288 mass lost

| cause | scenes | mass lost | share | at wall | corner | fled before |
|---|---:|---:|---:|---:|---:|---:|
| pieces | 1 | 149 | 52% | 1 | 1 | 1 |
| fled | 1 | 139 | 48% | 1 | 1 | 1 |

By bot family: ramp 2 scenes / 288 mass

**Loss scenes eaten by any other player**: 1 scenes, 106 mass lost

| cause | scenes | mass lost | share | at wall | corner | fled before |
|---|---:|---:|---:|---:|---:|---:|
| pieces | 1 | 106 | 100% | 0 | 0 | 1 |

By bot family: c3_4 1 scenes / 106 mass

## Worst 2 scenes (human eaters first, then by mass lost)

```
scene 1791371135-0  ramp-0  tick 2928  drop 288 -> 139  cause=pieces tags=wall,corner,human wall_dist=-68
  eaten by: human(human) cell 204 total 1379 x14 ate 151
  t-s  mode(stage) total cells  aim(dx,dy)    nearest players (mass, gap, bearing dx/dy)
   -2.9  flee(sma)    247   2  ( +500, +332)       human:346m g65 (-361,-76)x4 I Am Λzтє:28m g167 (+337,+31)x2 I Am Λzтє:29m g224 (+390,-71)x2
   -2.4  flee(sma)    249   2  ( +495, +340)       human:346m g75 (-362,-113)x4 I Am Λzтє:28m g171 (+320,+120)x2 I Am Λzтє:31m g192 (+366,+13)x2
   -1.9  flee(sma)    249   2  ( +536, +270)       human:346m g113 (-403,-106)x4 I Am Λzтє:28m g210 (+370,+91)x2 I Am Λzтє:32m g239 (+414,-11)x2
   -1.4  hunt(sma)    249   2  ( +232, -553)       I Am Λzтє:28m g123 (+293,+23)x2 I Am Λzтє:32m g129 (+291,-88)x2 human:202m g189 (-443,-75)x16
   -1.0  flee(sma)    288   2  ( +728, +384)       human:202m g-69 (-186,-60)x14 human:256m g172 (-179,-417)x14 human:20m g201 (+82,-359)x14
   -0.5  flee(sma)    288   2  ( +801, +255)       human:202m g-99 (-165,-8)x14 human:20m g112 (+12,-279)x14 human:256m g134 (-221,-352)x14
   +0.0  flee(sma)    139   1  ( +598,  +49)       human:353m g-8 (-297,-20)x14 I Am Λzтє:28m g30 (+35,-198)x2 human:20m g103 (-164,-210)x14
  map at loss (N up, +-1000 units, one char = 33 x 66):
  |                                 ############################|
  |                                 ############################|
  |                                 ############################|
  |                                 ############################|
  |         E                       ############################|
  |                                 ############################|
  |    p                            ############################|
  |p                E     p         ############################|
  |                                p############################|
  |      E                p p      p############################|
  |             p                   ############################|
  |               p                 ############################|
  | p                   E        @  ############################|
  | p  p                            ############################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
```

```
scene 1791371135-1  ramp-0  tick 2944  death 139 -> 0  cause=fled tags=wall,corner,human wall_dist=-69
  eaten by: human(human) cell 354 total 1531 x14 ate 140
  t-s  mode(stage) total cells  aim(dx,dy)    nearest players (mass, gap, bearing dx/dy)
   -2.9  flee(sma)    249   2  ( +494, +341)       human:346m g77 (-362,-119)x4 I Am Λzтє:28m g180 (+330,+119)x2 I Am Λzтє:31m g201 (+375,+15)x2
   -2.4  flee(sma)    249   2  ( +547, +247)       human:346m g120 (-411,-104)x4 I Am Λzтє:28m g216 (+378,+82)x2 I Am Λzтє:32m g219 (+393,-28)x2
   -1.9  flee(sma)    291   2  ( +667, +425)       human:202m g-52 (-200,-73)x14 human:20m g43 (-53,-204)x14 human:20m g44 (-212,-9)x14
   -1.4  flee(sma)    288   2  ( +744, +359)       human:202m g-74 (-183,-50)x14 human:256m g168 (-185,-410)x14 human:20m g168 (-262,-208)x14
   -1.0  flee(sma)    288   2  ( +804, +248)       human:202m g-104 (-160,-9)x14 human:20m g87 (-3,-254)x14 human:256m g128 (-228,-341)x14
   -0.5  flee(sma)    139   1  ( +595,  +78)       human:353m g-29 (-275,-30)x14 I Am Λzтє:28m g43 (+34,-211)x2 human:20m g92 (-135,-216)x14
   +0.0  flee(sma)    139   1  ( +579, +157)       human:353m g-149 (-154,-33)x14 human:20m g5 (+40,-163)x14 human:20m g75 (-37,-235)x14
  map at loss (N up, +-1000 units, one char = 33 x 66):
  |                                #############################|
  |                                #############################|
  |                                #############################|
  |                                #############################|
  |            E                   #############################|
  |                                #############################|
  |       p                       p#############################|
  |    p                          p#############################|
  |                    E     p     #############################|
  |                             p  #############################|
  |          E                     p############################|
  |                   p            #############################|
  |    p              p      E   @ #############################|
  |      p                         #############################|
  |#########p###################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
  |#############################################################|
```

Replay: `uv run python -m arena.scene_replay <dir> <scene id> [--spec bot/x.json] [--pytest]`.
