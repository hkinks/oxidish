#!/bin/bash
# Leader share of a bot spec. usage: scripts/eval.sh field spec-name seed [threads]
#   field 1 = held-out (plan x2, plan2, bc3 x2, hunter_virus), field 2 = "champions" (candidate x2, p6_hunt x2, planhv x2, c34h x2, plan x1)
#   both + 8 server bots. Results runs/vision/evals/f<field>_<name>_e<seed>.json (+ .stats.json), log logs/eval/
# needs the vision-net build of ogar (OGAR_BIN) and onnxruntime (ORT_DYLIB_PATH, see docs/vision-bot.md)
F=$1; name=$2; S=$3; T=${4:-3}
export OGAR_BIN=${OGAR_BIN:-$PWD/rust/ogar/target/release/ogar}
mkdir -p logs/eval runs/vision/evals
if [ "$F" = "1" ]; then
  AG="vis=bot/$name.json:2 plan=bot/plan_c34.json:2 plan2=bot/plan2_c34.json:1 bc3=bot/vision_bc3.json:2 hvirus=bot/sparring/hunter_virus.json:1"
else
  AG="vis=bot/$name.json:2 p6h=bot/vision_p6_hunt.json:2 planhv=bot/drill/plan_hv_commit_only.json:2 c34h=bot/drill/hv_commit_only.json:2 plan=bot/plan_c34.json:1"
fi
${PYTHON:-python} -m arena.endless --engine inproc --threads $T --cpus 12 --candidate vis --agents $AG --worlds 8 --game-min 240 --seed-base $S --out runs/vision/evals/f${F}_${name}_e$S.json --stats-out runs/vision/evals/f${F}_${name}_e$S.stats.json > logs/eval/f${F}_${name}_$S.txt 2>&1
