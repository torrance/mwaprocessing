#! /bin/bash
#SBATCH -M zeus
#SBATCH --partition copyq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE
#SBATCH --time=00:01:00

set -e
set -x
obsdownload.py -o $1
touch $1/download_complete
