#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=08:00:00
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
suffix=$2

if [ -z "$suffix" ]; then
	echo "Suffix not set!"
	exit 1
fi

# Do a shallow clean, to be used for selfcal

# scale = 0.5 / chan
wsclean -name wsclean-${suffix} -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 1000000 -auto-threshold 5 -auto-mask 8 ${obsid}.ms

beam -2016 -proto wsclean-${suffix}-XX-image.fits -ms ${obsid}.ms -m ${obsid}.metafits

pbcorrect wsclean-${suffix} image.fits beam stokes-${suffix}

# Selfcal

calibrate -minuv 60 -j 20 -i 500 ${obsid}.ms selfcal-solutions-${suffix}.bin

applysolutions ${obsid}.ms selfcal-solutions-${suffix}.bin

aoflagger ${obsid}.ms

rm selfcal${suffix}_started && touch selfcal${suffix}_complete
