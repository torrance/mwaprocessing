#! /bin/bash

if command -v module < /dev/null; then
  set +x
  module load manta-ray-client
  set -x
fi

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid list supplied"
  exit 1
fi

rm obsid.csv || true
touch obsid.csv

for obsid in $(cat $1); do
  echo "obs_id=${obsid}, job_type=c, timeres=4, freqres=40, edgewidth=80, conversion=ms, allowmissing=true, flagdcchannels=true" >> obsid.csv
done

mwa_client -c obsid.csv -d .

for obsid in $(cat $1); do
  if [[ -f ${obsid}_ms.zip ]]; then
    if [[ ! -d ${obsid} ]]; then
      mkdir $obsid
    fi
    unzip -u -d ${obsid} ${obsid}_ms.zip
  else
    echo "No zip file found for $obsid"
  fi
done
