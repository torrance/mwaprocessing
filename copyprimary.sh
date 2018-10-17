#! /bin/bash

set -x
set -e

if [[ ! -f $1 ]]; then
  echo "No obsid.txt file provided"
fi

for obsid in $(cat $1); do
  read calid _ <<< $(find_calibrator.py ${obsid}/${obsid}.metafits)
  if [[ -f calibrators/${calid}/solutions-target.bin && -f calibrators/${calid}/badantennae ]]; then
    cp calibrators/${calid}/solutions-target.bin ${obsid}/solutions-primary.bin
    cat calibrators/${calid}/badantennae >> ${obsid}/badantennae
  else
    echo "No calibration for $obsid"
  fi
done
