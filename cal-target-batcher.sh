#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  # Test for whether the observation has been downloaded yet
  if [[ ! -d $obsid || ! -f ${obsid}/download_complete ]]; then
    continue
  fi

  if [[ ! -f $obsid/cal_target_scheduled && ! -f $obsid/cal_target_started && ! -f $obsid/cal_target_complete ]]; then
    read calid calibrator <<< $(find_calibrator.py ${obsid}/${obsid}.metafits)

    if [[ $obsid != $calid ]]; then
      echo "ERROR: No matching calibrator"
      continue
    fi

    echo $obsid $calibrator
    jobid=$(sbatch -J "cal-target.sh $obsid $calibrator" --workdir $obsid cal-target.sh $obsid $calibrator | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/cal_target_scheduled
    fi
  fi

  jobid=$(cat $obsid/cal_target_scheduled 2>/dev/null || cat $obsid/cal_target_started 2>/dev/null || cat $obsid/cal_target_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/selfcal_scheduled && ! -f $obsid/selfcal_started && ! -f $obsid/selfcal_complete ]]; then
    jobid=$(sbatch -J "selfcal.sh $obsid" --workdir $obsid -d afterok:$jobid selfcal.sh $obsid selfcal target | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/selfcal_scheduled
    fi
  fi
done
