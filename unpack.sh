#! /bin/bash

set -e
set -x

if [[ -z $1 ]]; then
  echo "No obsid supplied"
  exit 1
fi
obsid=$1

if unzip -z ${obsid}_ms.zip && [[ ! -f ${obsid}/download_started && ! -f ${obsid}/download_complete ]]; then
  if [[ ! -d ${obsid} ]]; then
    mkdir $obsid
  fi
  touch $obsid/download_started

  unzip -u -d ${obsid} ${obsid}_ms.zip

  mv ${obsid}/download_started ${obsid}/download_complete
else
  echo "No zip file found for $obsid"
fi
