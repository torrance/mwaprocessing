#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=04:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE

set -e
set -x

if command -v module < /dev/null; then
	module load pyfits
fi

obsid=$1
touch preprocess_started
make_metafits.py --gps=${obsid}

cotter -m ${obsid}.metafits -freqres 40 -timeres 4 -m ${obsid}.metafits -allowmissing -o ${obsid}.ms *gpubox*.fits

rm *gpubox*.fits
rm preprocess_started && touch preprocess_complete
