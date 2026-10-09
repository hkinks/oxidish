#!/bin/bash
# usage: endless5.sh name [seeds...]   (spec bot/<name>.json)
name=$1; shift
seeds=${@:-3000 11000 21000 31000 41000}
export OGAR_BIN=$PWD/logs/bin/ogar.exe
export OGAR_DLL_DIRS=$PWD/.venv/Lib/site-packages/torch/lib
for S in $seeds; do
  .venv/Scripts/python.exe -m arena.endless --candidate vis --agents vis=bot/$name.json:2 champ=runs/final_champion.json:2 r2g9=runs/r2/champion_009.json:2 safe=bot/champ_safe.json:2 ramp_r2=bot/r2g9_ramp800.json:2 ramp300=runs/r2g9_ramp300.json:2 --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/${name}_e$S.json > logs/endless_${name}_$S.txt 2>&1
done
