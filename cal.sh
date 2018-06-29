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

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem $ABSMEM"
else
  absmem=""
fi

obsid=$1

touch cal_started

/home/torrance/srclist_by_beam.py --aocalibrate -x -m ${obsid}.metafits -s /home/torrance/srclist_pumav3_EoR0aegean_EoR1pietro+ForA.txt -n 1000

calibrate $absmem -minuv 60 -maxuv 2600 -m srclist_pumav3_EoR0aegean_EoR1pietro+ForA_${obsid}_aocal1000.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions.bin

applysolutions ${obsid}.ms solutions.bin

aoflagger ${obsid}.ms

aocal_plot.py solutions.bin

rm cal_started && touch cal_complete
