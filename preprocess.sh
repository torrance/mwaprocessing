#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=04:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type ALL
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE

set -e

if command -v module < /dev/null; then
	module load pyfits
fi

obsid=$1
make_metafits.py --gps=${obsid}

cotter -m ${obsid}.metafits -freqres 40 -timeres 4 -m ${obsid}.metafits -allowmissing -o ${obsid}.ms *gpubox*.fits

/home/torrance/srclist_by_beam.py --aocalibrate -x -m ${obsid}.metafits -s /home/torrance/srclist_pumav3_EoR0aegean_EoR1pietro+ForA.txt -n 1000

calibrate -minuv 60 -m srclist_pumav3_EoR0aegean_EoR1pietro+ForA_${obsid}_aocal1000.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions.bin
