#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  # Test for whether the observation has been downloaded yet
  if [[ ! -f $obsid/download_complete ]]; then
    continue
  fi

  if [[ ! -f $obsid/preprocess_scheduled && ! -f $obsid/preprocess_started && ! -f $obsid/preprocess_complete ]]; then
    jobid=$(sbatch -J "preprocess.sh $obsid" --workdir $obsid preprocess.sh $obsid | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/preprocess_scheduled
    fi
  fi

  jobid=$(cat $obsid/preprocess_scheduled 2>/dev/null || cat $obsid/preprocess_started 2>/dev/null || cat $obsid/preprocess_complete 2>/dev/null || echo '')
  if [[ ! -z $jobid && ! -f $obsid/cal_pre_scheduled && ! -f $obsid/cal_pre_started && ! -f $obsid/cal_pre_complete ]]; then
    jobid=$(sbatch -J "cal.sh $obsid pre" --workdir $obsid -d afterok:$jobid cal.sh $obsid pre | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/cal_pre_scheduled
    fi
  fi
done
