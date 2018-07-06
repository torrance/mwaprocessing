#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ -f $obsid/download_complete && ! -f $obsid/preprocess_scheduled && ! -f $obsid/preprocess_started && ! -f $obsid/preprocess_complete ]]; then
    touch $obsid/preprocess_scheduled
    sbatch -J "preprocess.sh $obsid" --workdir $obsid preprocess.sh $obsid
  fi
done
