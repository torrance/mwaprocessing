#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=04:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM

set -e
set -x

if command -v module < /dev/null; then
  set +x
  module load pyfits
  set -x
fi

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem $ABSMEM"
else
  absmem=""
fi

obsid=$1

rm postprocess_scheduled || true
rm recal_scheduled || true
touch cal_started

if [[ -f badantennae ]]; then
  cat badantennae | xargs flagantennae ${obsid}.ms
fi

srclist_by_beam.py -x -m ${obsid}.metafits -s /home/torrance/srclist_pumav3_EoR0aegean_EoR1pietro+ForA.txt -n 1000
cat srclist_pumav3_EoR0aegean_EoR1pietro+ForA_${obsid}_peel1000.txt | rts_to_skymodel.py > model.txt

calibrate $absmem -minuv 60 -maxuv 2600 -m model.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions.bin

applysolutions ${obsid}.ms solutions.bin

aoflagger ${obsid}.ms

aocal_plot.py solutions.bin

rm cal_started && touch cal_complete
