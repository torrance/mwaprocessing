#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ -f $obsid/preprocess_complete && ! -f $obsid/recal_scheduled && ! -f $obsid/cal_started && ! -f $obsid/cal_complete ]]; then
    touch $obsid/recal_scheduled
    sbatch -J "cal.sh $obsid" --workdir $obsid cal.sh $obsid
  fi
done
