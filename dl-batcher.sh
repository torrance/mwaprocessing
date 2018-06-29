#! /bin/bash

set -e
set -x

grep -v '^#' obsid.txt | while read obsid; do
  sbatch dl.sh $obsid
done
