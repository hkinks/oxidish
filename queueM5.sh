#!/bin/bash
# queueM5.sh [parallel] : every (variant, seed) of the M5 skill A/B, seeds-major, `parallel` runs at a time
P=${1:-4}
run() { # wait for a free slot, then start one run in the background
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 20; done
  if [ ! -s "runs/vision/evals/M5_$1_e$2.stats.json" ]; then ./endlessM5.sh $1 $2 3 & fi
}
for S in 3000 11000 21000 31000 41000; do
  for n in vision_p6 vision_p6_rs vision_p6_rh vision_p6_rv vision_p6_ra; do run $n $S; done
done
wait
echo ALL DONE
