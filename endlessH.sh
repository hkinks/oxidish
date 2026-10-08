#!/bin/bash
# held-out field (PPO never trains against bc3/vision copies or this exact mix): endlessH.sh name [seeds...]
name=$1; shift
seeds=${@:-3000 11000 21000 31000 41000}
export OGAR_BIN=$PWD/logs/bin/ogar.exe
export OGAR_DLL_DIRS=$PWD/.venv/Lib/site-packages/torch/lib
for S in $seeds; do
  .venv/Scripts/python.exe -m arena.endless --candidate vis --agents vis=bot/$name.json:2 plan=bot/plan_c34.json:2 plan2=bot/plan2_c34.json:1 bc3=bot/vision_bc3.json:2 apos=bot/sparring/apos.json:1 hvirus=bot/sparring/hunter_virus.json:1 --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/H_${name}_e$S.json > logs/endlessH_${name}_$S.txt 2>&1
done
