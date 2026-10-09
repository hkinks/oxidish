#!/bin/bash
# held-out runs after the p3 3000 run: p6 and the baselines on seeds 3000, 11000, 21000, then the rest
while pgrep -f "endlessH.sh vision_p3 3000" > /dev/null 2>&1 || [ ! -f runs/vision/evals/H_vision_p3_e3000.json ]; do sleep 20; done
until [ -f runs/vision/export/vision_p6.onnx ]; do sleep 20; done
for S in 3000 11000 21000 31000 41000; do
  for n in vision_p6 vision_bc3 vision_p2 vision_p3; do
    [ -f runs/vision/evals/H_${n}_e$S.json ] || ./endlessH.sh $n $S
  done
done
