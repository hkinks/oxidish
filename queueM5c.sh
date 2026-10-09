#!/bin/bash
# queueM5c.sh "<name> <seed> <script>" ... : run the listed jobs, keeping at most 4 `ogar.exe endless` of ours running
# (the foreign ogar.exe, pid 4180, is counted in the 5). Jobs whose stats file exists are skipped.
while read -r name seed script; do
  [ -z "$name" ] && continue
  [ -s "runs/vision/evals/M5_${name}_e${seed}.stats.json" ] && continue
  while [ "$(tasklist | grep -c 'ogar.*exe')" -ge 5 ]; do sleep 30; done
  nohup ./$script $name $seed 3 > /dev/null 2>&1 &
  sleep 45
done < "$1"
echo QUEUE C LAUNCHED
