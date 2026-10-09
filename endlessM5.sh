#!/bin/bash
# M5 held-out field with the fast evaluator + skill counters: endlessM5.sh spec-name seed [threads]
# (spec-name = bot/<name>.json; results runs/vision/evals/M5_<name>_e<seed>.json (+ .stats.json), log logs/m5/)
name=$1; S=$2; T=${3:-3}
export OGAR_BIN=$PWD/logs/bin/ogar.exe
export OGAR_DLL_DIRS=/c/Users/hkink/Workspace/agario-bot/.venv/Lib/site-packages/torch/lib
PY=/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe
mkdir -p logs/m5 runs/vision/evals
$PY -m arena.endless --engine inproc --threads $T --cpus 12 --candidate vis --agents vis=bot/$name.json:2 plan=bot/plan_c34.json:2 plan2=bot/plan2_c34.json:1 bc3=bot/vision_bc3.json:2 apos=bot/sparring/apos.json:1 hvirus=bot/sparring/hunter_virus.json:1 --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/M5_${name}_e$S.json --stats-out runs/vision/evals/M5_${name}_e$S.stats.json > logs/m5/${name}_$S.txt 2>&1
