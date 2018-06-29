#! /bin/bash

set -e
set -x

grep -v '^#' obsid.txt | while read obsid; do
  jobid=$(sbatch --workdir $obsid cal.sh $obsid | cut -d ' ' -f 4)
  jobid=$(sbatch --workdir $obsid -d afterok:$jobid selfcal.sh $obsid 2 | cut -d ' ' -f 4)
  jobid=$(sbatch --workdir $obsid -d afterok:$jobid image.sh $obsid)
done
