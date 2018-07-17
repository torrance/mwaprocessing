#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ ! -f $obsid/badantennae || ! -f  $obsid/preprocess_complete || -f $obsid/postprocess_scheduled || -f $obsid/cal_started || -f $obsid/cal_complete ]]; then
    continue
  fi

  touch $obsid/postprocess_scheduled
  jobid=$(sbatch -J "cal.sh $obsid" --workdir $obsid cal.sh $obsid | cut -d ' ' -f 4)
  jobid=$(sbatch -J "selfcal.sh $obsid 1" --workdir $obsid -d afterok:$jobid selfcal.sh $obsid 1 | cut -d ' ' -f 4)
  jobid=$(sbatch -J "selfcal.sh $obsid 2" --workdir $obsid -d afterok:$jobid selfcal.sh $obsid 2 | cut -d ' ' -f 4)
  jobid=$(sbatch -J "image.sh $obsid briggs 0" --workdir $obsid -d afterok:$jobid image.sh $obsid 'briggs 0' | cut -d ' ' -f 4)
  sbatch -J "image.sh $obsid natural" --workdir $obsid -d afterok:$jobid image.sh $obsid 'natural'
done
