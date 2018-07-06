#! /bin/bash
#SBATCH -M zeus
#SBATCH --partition copyq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE
#SBATCH --time=02:00:00

set -e
set -x
if [[ ! -d $1 ]]; then
  mkdir $1
fi
touch $1/download_started
obsdownload.py -o $1
rm $1/download_started && touch $1/download_complete
