#!/bin/bash
# M7 evals. usage: endlessM7.sh field spec-name seed [threads]
#   field 1 = held-out (M5/M6 lineup), field 2 = "champions" (candidate x2, p6_hunt x2, planhv x2, c34h x2, plan x1 + 8 server bots)
# results runs/vision/evals/M7_f<field>_<name>_e<seed>.json (+ .stats.json), log logs/m7/
F=$1; name=$2; S=$3; T=${4:-3}
export OGAR_BIN=$PWD/logs/bin/ogar.exe
export OGAR_DLL_DIRS=/c/Users/hkink/Workspace/agario-bot/.venv/Lib/site-packages/torch/lib
PY=/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe
mkdir -p logs/m7 runs/vision/evals
if [ "$F" = "1" ]; then
  AG="vis=bot/$name.json:2 plan=bot/plan_c34.json:2 plan2=bot/plan2_c34.json:1 bc3=bot/vision_bc3.json:2 apos=bot/sparring/apos.json:1 hvirus=bot/sparring/hunter_virus.json:1"
else
  AG="vis=bot/$name.json:2 p6h=bot/vision_p6_hunt.json:2 planhv=bot/drill/plan_hv_commit_only.json:2 c34h=bot/drill/hv_commit_only.json:2 plan=bot/plan_c34.json:1"
fi
$PY -m arena.endless --engine inproc --threads $T --cpus 12 --candidate vis --agents $AG --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/M7_f${F}_${name}_e$S.json --stats-out runs/vision/evals/M7_f${F}_${name}_e$S.stats.json > logs/m7/f${F}_${name}_$S.txt 2>&1
