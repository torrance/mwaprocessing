#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  if [[ ! -f $obsid/badantennae ]]; then
    continue
  fi

  jobid=$(cat $obsid/cal_pre_scheduled 2>/dev/null || cat $obsid/cal_pre_started 2>/dev/null || cat $obsid/cal_pre_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/cal_post_scheduled && ! -f $obsid/cal_post_started && ! -f $obsid/cal_post_complete ]]; then
    jobid=$(sbatch -J "cal.sh $obsid post" --workdir $obsid -d afterok:$jobid cal.sh $obsid post pre | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/cal_post_scheduled
    fi
  fi

  jobid=$(cat $obsid/cal_post_scheduled 2>/dev/null || cat $obsid/cal_post_started 2>/dev/null || cat $obsid/cal_post_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/selfcal_scheduled && ! -f $obsid/selfcal_started && ! -f $obsid/selfcal_complete ]]; then
    jobid=$(sbatch -J "selfcal.sh $obsid 1" --workdir $obsid -d afterok:$jobid selfcal.sh $obsid selfcal post | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/selfcal_scheduled
    fi
  fi

  jobid=$(cat $obsid/selfcal_scheduled 2>/dev/null || cat $obsid/selfcal_started 2>/dev/null || cat $obsid/selfcal_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/image_uniform_scheduled && ! -f $obsid/image_uniform_started && ! -f $obsid/image_uniform_complete ]]; then
    jobid=$(sbatch -J "image.sh $obsid uniform" --workdir $obsid -d afterok:$jobid image.sh $obsid uniform | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/image_uniform_scheduled
    fi
  fi

  jobid=$(cat $obsid/image_uniform_scheduled 2>/dev/null || cat $obsid/image_uniform_started 2>/dev/null || cat $obsid/image_uniform_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/image_briggs0_scheduled && ! -f $obsid/image_briggs0_started && ! -f $obsid/image_briggs0_complete ]]; then
    jobid=$(sbatch -J "image.sh $obsid briggs 0" --workdir $obsid -d afterok:$jobid image.sh $obsid 'briggs 0' | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/image_briggs0_scheduled
    fi
  fi

  jobid=$(cat $obsid/image_briggs0_scheduled 2>/dev/null || cat $obsid/image_briggs0_started 2>/dev/null || cat $obsid/image_briggs0_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/image_briggs+1_scheduled && ! -f $obsid/image_briggs+1_started && ! -f $obsid/image_briggs+1_complete ]]; then
    jobid=$(sbatch -J "image.sh $obsid briggs +1" --workdir $obsid -d afterok:$jobid image.sh $obsid 'briggs +1' | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/image_briggs+1_scheduled
    fi
  fi

  jobid=$(cat $obsid/image_briggs+1_scheduled 2>/dev/null || cat $obsid/image_briggs+1_started 2>/dev/null || cat $obsid/image_briggs+1_complete 2>/dev/null || echo '')
  if [[ -n $jobid && ! -f $obsid/image_natural_scheduled && ! -f $obsid/image_natural_started && ! -f $obsid/image_natural_complete ]]; then
    jobid=$(sbatch -J "image.sh $obsid natural" --workdir $obsid -d afterok:$jobid image.sh $obsid 'natural' | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/image_natural_scheduled
    fi
  fi
done
