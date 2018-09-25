#! /bin/bash
#SBATCH -M zeus
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM

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
  if [[ ! -f ${obsid}_ms.zip ]]; then
    echo "obs_id=${obsid}, job_type=c, timeres=4, freqres=40, edgewidth=80, conversion=ms, allowmissing=true, flagdcchannels=true" >> obsid.csv
  fi
done

mwa_client -c obsid.csv -d .


