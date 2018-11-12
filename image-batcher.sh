#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid file specified"
  exit 1
fi

if [[ -z $2 ]]; then
  echo "No image weight specified"
  exit 1
fi

if [[ -z $3 ]]; then
  echo "No image name specified"
  exit 1
fi

if [[ -n $4 ]]; then
  extras="-o '$4'"
fi

grep -v '^#' $1 | while read obsid; do
  if [[ ! -f $obsid/badantennae ]]; then
    continue
  fi

  if [[ ! -f $obsid/image_${3}_scheduled && ! -f $obsid/image_${3}_started && ! -f $obsid/image_${3}_complete ]]; then
    jobid=$(sbatch -J "image.sh $obsid $name" --workdir $obsid image.sh -n $3 -o " $4" $obsid "$2" | cut -d ' ' -f 4)
    if [[ -n $jobid ]]; then
      echo $jobid > $obsid/image_${3}_scheduled
    fi
  fi
done
