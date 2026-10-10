#!/bin/bash
# M8: while m8a trains, take the latest checkpoint, export it and run the held-out sets 3000 and 11000 (3 threads).
while [ ! -f runs/vision/m8a/final.pt ]; do
  ck=$(ls runs/vision/m8a/ck_*.pt | tail -n 1); t=$(basename $ck .pt)
  echo "$(date) eval $t" >> logs/m8/queue.txt
  ./exportM8.sh m8a $t >> logs/m8/queue.txt 2>&1
  for s in 3000 11000; do ./endlessM8.sh 1 vision_m8a_$t $s 3; done
done
