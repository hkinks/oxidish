#!/bin/bash
# League PPO with the hunter reflex in the loop (how m7a was trained). usage: scripts/train.sh name [extra args]
# needs the oxidish_gym module (uv run maturin develop --release --uv in training/gym) and torch
# INIT = the checkpoint to start from (m7a started from p6; train your own first, see docs/training.md)
n=$1; shift
init=${INIT:-runs/vision/p6/final.pt}
[ -f "$init" ] || { echo "no checkpoint $init: set INIT=path/to/final.pt (see docs/training.md)" >&2; exit 1; }
mkdir -p logs/vision
${PYTHON:-uv run python} training/rl/train_ppo5.py --name $n --init $init --ref $init --threads 4 "$@" > logs/vision/$n.out 2>&1
