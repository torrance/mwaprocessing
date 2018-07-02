#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ -f $obsid/cal_started || -f $obsid/cal_complete ]]; then
    continue
  fi

  jobid=$(sbatch --workdir $obsid cal.sh $obsid | cut -d ' ' -f 4)
  jobid=$(sbatch --workdir $obsid -d afterok:$jobid selfcal.sh $obsid 1 | cut -d ' ' -f 4)
  jobid=$(sbatch --workdir $obsid -d afterok:$jobid selfcal.sh $obsid 2 | cut -d ' ' -f 4)
  jobid=$(sbatch --workdir $obsid -d afterok:$jobid image.sh $obsid)
done
