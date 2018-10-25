#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

if [[ ! -f $2 ]]; then
  echo "Model is not a file"
  exit 1
fi
name=$(basename $2 .txt)

grep -v '^#' $1 | while read obsid; do
  # Test for whether the observation has been downloaded yet
  if [[ ! -d $obsid || ! -d ${obsid}/${obsid}.ms ]]; then
    continue
  fi

  if [[ ! -f $obsid/peel_${name}_scheduled && ! -f $obsid/peel_${name}_started && ! -f $obsid/peel_${name}_complete ]]; then
    jobid=$(sbatch -J "peel.sh $obsid $name" --workdir $obsid peel.sh $obsid $2 | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/peel_${name}_scheduled
    fi
  fi
done
