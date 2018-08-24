#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  # Test for whether the observation has been downloaded yet
  if [[ ! -d $obsid || ! -d ${obsid}/${obsid}.ms ]]; then
    continue
  fi

  if [[ ! -f $obsid/cal_pre_scheduled && ! -f $obsid/cal_pre_started && ! -f $obsid/cal_pre_complete ]]; then
    jobid=$(sbatch -J "cal.sh $obsid pre" --workdir $obsid cal.sh $obsid pre | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/cal_pre_scheduled
    fi
  fi
done
