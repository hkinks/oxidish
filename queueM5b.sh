#!/bin/bash
# second M5 matrix (safer reflex variants, ogar2.exe): waits for queueM5.sh to finish, then 4 runs at a time
P=${1:-4}
until grep -q "ALL DONE" logs/m5/queue.log; do sleep 60; done
run() {
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 20; done
  if [ ! -s "runs/vision/evals/M5_$1_e$2.stats.json" ]; then ./endlessM5b.sh $1 $2 3 & fi
}
for S in 3000 11000 21000 31000 41000; do
  for n in vision_p6_rs2 vision_p6_rh2 vision_p6_rv2; do run $n $S; done
done
wait
echo ALL DONE B
