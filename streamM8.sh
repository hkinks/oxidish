#!/bin/bash
# usage: streamM8.sh "field:spec:seed field:spec:seed ..."  runs the evals one after the other (3 threads each)
for j in $1; do IFS=: read f s e <<< "$j"; ./endlessM8.sh $f $s $e 3; done
