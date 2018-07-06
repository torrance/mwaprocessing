#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ -f $obsid/download_started || -f $obsid/download_complete ]]; then
    echo "Skipping $obsid"
    continue
  fi
  sbatch -J "dl.sh $obsid" dl.sh $obsid
done
