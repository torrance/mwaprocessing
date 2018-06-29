#! /bin/bash

set -e
set -x

grep -v '^#' obsid.txt | while read obsid; do
  sbatch preprocess.sh $obsid
done
