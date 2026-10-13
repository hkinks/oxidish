#!/bin/bash
# League PPO with the hunter reflex in the loop (how m7a was trained). usage: scripts/train.sh name [extra args]
# needs the oxidish_env module (maturin develop --release in training/env) and torch in $PYTHON's environment
n=$1; shift
mkdir -p logs/vision
${PYTHON:-python} training/rl/train_ppo5.py --name $n --init runs/vision/p6/final.pt --ref runs/vision/p6/final.pt --threads 4 "$@" > logs/vision/$n.out 2>&1
