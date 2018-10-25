#! /bin/bash
#SBATCH -M magnus
#SBATCH --account pawsey0272
#SBATCH --time=01:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM

obsid=$1
model=$2

if [[ -z $obsid ]]; then
  echo "No obsid provided"
  exit 1
fi

if [[ ! -f $model ]]; then
  echo "Model does not exist"
  exit 1
fi

name=$(basename $model .txt)

set -e
set -x

mv peel_${name}_scheduled peel_${name}_started || touch peel_${name}_started

# To do: calculate time steps
peel -t 999 -minuv 60 -m $model ${obsid}.ms

mv peel_${name}_started peel_${name}_complete
