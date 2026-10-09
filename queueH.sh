#!/bin/bash
# held-out runs in priority order: baseline (bc3) and the two candidates on 2 seeds first, then the other seeds
for S in 3000 11000 21000 31000 41000; do
  for n in vision_bc3 vision_p3 vision_p2; do
    [ -f runs/vision/evals/H_${n}_e$S.json ] || ./endlessH.sh $n $S
  done
done
