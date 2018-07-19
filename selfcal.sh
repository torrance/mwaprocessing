#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM

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
rm wsclean-${label}*.fits || true
rm stokes-${label}*.fits || true
rm *.tmp || true

# Apply previous calibration solution, if one is present
# This is necessary when recovering from a failed job
if [[ ! -z $prior ]]; then
  applysolutions ${obsid}.ms solutions-${prior}.bin
fi

# First chg phase centre to increase wsclean speed
if [[ ! -f chgcentred ]]; then
  chgcentre -minw -shiftback ${obsid}.ms
  touch chgcentred
fi

# Do a shallow clean, to be used for selfcal
# scale = 0.5 / chan
wsclean -name wsclean-${label} -j 20 -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 300000 -auto-threshold 5 -auto-mask 8 $absmem ${obsid}.ms

# Create a beam if it doesn't already exist
if [[ ! -f beam-xxi.fits ]]; then
  beam -2016 -proto wsclean-${label}-XX-image.fits -ms ${obsid}.ms -m ${obsid}.metafits
fi

# Output image of selfcal
pbcorrect wsclean-${label} image.fits beam stokes-${label}

# Selfcal
calibrate -minuv 60 -j 20 -i 500 $absmem ${obsid}.ms solutions-${label}.bin

applysolutions ${obsid}.ms solutions-${label}.bin

aoflagger ${obsid}.ms

mv ${label}_started ${label}_complete
