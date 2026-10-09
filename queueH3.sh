#!/bin/bash
# after the running p3 11000 held-out job: p7 first, then bc3 / p6 / p7 on the other seeds
until [ -f runs/vision/evals/H_vision_p3_e11000.json ]; do sleep 20; done
for S in 3000 11000 21000 31000 41000; do
  for n in vision_p7 vision_p6 vision_bc3; do
    [ -f runs/vision/evals/H_${n}_e$S.json ] || ./endlessH.sh $n $S
  done
done
