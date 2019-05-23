#! /bin/bash
#SBATCH -M zeus
#SBATCH -A pawsey0293
#SBATCH --time=2:00:00
#SBATCH --partition copyq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
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
