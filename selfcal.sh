#! /bin/bash
#SBATCH -M magnus
#SBATCH -A pawsey0286
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task 20
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM,BASEDIR

set -e
set -x

obsid=$1
label=$2
prior=$3

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem ${ABSMEM}"
else
  absmem=""
fi

if [[ -z "$label" ]]; then
  echo "Label not set!"
  exit 1
fi

mv ${label}_scheduled  ${label}_started || touch ${label}_started

# Clean up from any previous run
rm solutions-${label}.bin || true
rm ${obsid}-wsclean-${label}*.fits || true
rm ${obsid}-${label}-stokes*.fits || true
rm ${obsid}-beam*.fits || true
rm *.tmp || true

# Flag any tiles that MWA ops have flagged
# This should be handled automatically but may have been undone by running aoflagger
getflaggedtiles.py ${obsid}.metafits | xargs -r --verbose flagantennae ${obsid}.ms

# Flag tiles if badantennae file is present
cat badantennae | xargs -r --verbose flagantennae ${obsid}.ms

# Apply previous calibration solution, if one is present
# This is necessary when recovering from a failed job
if [[ ! -z $prior ]]; then
  applysolutions ${obsid}.ms solutions-${prior}.bin
fi

# Change to pointing direction, then to minw
pointing=$(pointing.py ${obsid}.metafits)
chgcentre ${obsid}.ms $pointing
chgcentre -minw -shiftback ${obsid}.ms

# Do a shallow clean, to be used for selfcal
scale=$(echo "scale=6; 0.6 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean \
  -name ${obsid}-wsclean-${label} \
  -mgain 0.8 \
  -weight briggs 0 \
  -size 8500 8500 \
  -scale $scale \
  -niter 9999999 \
  -auto-threshold 8 \
  -mwa-path $BASEDIR \
  -channels-out 8 \
  -fit-spectral-pol 2 \
  -join-channels \
  -pol xx,yy \
  -nmiter 12 \
  -minuv-l 40 \
  -padding 1.8 \
  -nwlayers-factor 2 \
  -deconvolution-channels 2 \
  $absmem \
  ${obsid}.ms

# Selfcal
calibrate -minuv 100 -j 20 -i 500 $absmem -mwa-path $BASEDIR -ch 4 ${obsid}.ms solutions-${label}.bin

applysolutions ${obsid}.ms solutions-${label}.bin

# Flag data
# Order matters here! aoflagger will UNFLAG any flagged tiles
# so we must repply our own flags again after
aoflagger -indirect-read ${obsid}.ms
getflaggedtiles.py ${obsid}.metafits | xargs -r --verbose flagantennae ${obsid}.ms
cat badantennae | xargs -r --verbose flagantennae ${obsid}.ms

aocal_plot.py --amp_max 2 solutions-${label}.bin

mv ${label}_started ${label}_complete
