#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

if [[ -z $2 ]]; then
  echo "No image suffix specified"
  exit 1
fi

grep -v '^#' $1 | while read obsid; do
  # Test for whether the image exists
  if [[ ! -f ${1}-${2}.fits ]]; then
    continue
  fi

  if [[ ! -f $obsid/ionkludge_scheduled && ! -f $obsid/ionkludge_started && ! -f $obsid/ionkludge_complete ]]; then
    jobid=$(sbatch -J "ionkludge.sh $obsid $2" --workdir $obsid ionkludge.sh $obsid $2 | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/ionkludge_scheduled
    fi
  fi
done
