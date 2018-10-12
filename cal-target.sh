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
calibrator=$2

mv cal_target_scheduled cal_target_started || touch cal_target_started

# Clean up from any previous runs
rm solutions-target.bin || true

# Flag any tiles that MWA ops have flagged
# This should be handled automatically but is a bug in Cotter at the moment
flagged=$(getflaggedtiles.py ${obsid}.metafits)
if [[ ! -z $flagged ]]; then
  echo $flagged | xargs flagantennae ${obsid}.ms
fi

# Get calibrator model
model=$(dirname "$0")
model=${model}/models/${calibrator}.txt

for _ in {1..2}; do
  calibrate $absmem -m $model -minuv 60 -maxuv 2600 -applybeam -j 20 -i 500 ${obsid}.ms solutions-target.bin
  applysolutions ${obsid}.ms solutions-target.bin
  aoflagger ${obsid}.ms
done

aocal_plot.py solutions-target.bin

mv cal_target_started cal_target_complete
