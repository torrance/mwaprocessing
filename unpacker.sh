#! /bin/bash
#SBATCH -M zeus
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH -c 1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM

set -x
set -e

if [[ ! -f $1 ]]; then
  echo "obsid is not a file"
  exit 1
fi

cat $1 | parallel -j 20 unpack.sh {1}
