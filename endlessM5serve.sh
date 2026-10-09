#!/bin/bash
# baseline check: the same held-out field with the OLD engine (--engine serve, one `ogar serve` per world): endlessM5serve.sh name seed
name=$1; S=$2
export OGAR_BIN=$PWD/logs/bin/ogar.exe
export OGAR_DLL_DIRS=/c/Users/hkink/Workspace/agario-bot/.venv/Lib/site-packages/torch/lib
PY=/c/Users/hkink/Workspace/agario-bot/.venv/Scripts/python.exe
$PY -m arena.endless --engine serve --cpus 12 --candidate vis --agents vis=bot/$name.json:2 plan=bot/plan_c34.json:2 plan2=bot/plan2_c34.json:1 bc3=bot/vision_bc3.json:2 apos=bot/sparring/apos.json:1 hvirus=bot/sparring/hunter_virus.json:1 --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/M5serve_${name}_e$S.json > logs/m5/serve_${name}_$S.txt 2>&1
