#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE

set -e
set -x

obsid=$1
i=$2

if [[ ! -z $ABSMEM ]]; then
	absmem="-absmem ${ABSMEM}"
else
	absmem=""
fi

if [ -z "$i" ]; then
	echo "Iterations not set!"
	exit 1
fi

touch selfcal_started

# First chg phase centre to increase wsclean speed
if [[ ! -f chgcentred ]]; then
	chgcentre -minw -shiftback ${obsid}.ms
	touch chgcentred
fi

# Do a shallow clean, to be used for selfcal
# scale = 0.5 / chan
wsclean -name wsclean-${i} -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 1000000 -auto-threshold 5 -auto-mask 8 $absmem ${obsid}.ms

# Create a beam if it doesn't already exist
if [[ ! -f beam-xxi.fits ]]; then
	beam -2016 -proto wsclean-${i}-XX-image.fits -ms ${obsid}.ms -m ${obsid}.metafits
fi

# Output image of selfcal
pbcorrect wsclean-${i} image.fits beam stokes-${i}

# Selfcal

calibrate -minuv 60 -j 20 -i 500 $absmem ${obsid}.ms selfcal-solutions-${i}.bin

applysolutions ${obsid}.ms selfcal-solutions-${i}.bin

aoflagger ${obsid}.ms

rm selfcal_started && touch selfcal_complete
