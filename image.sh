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

touch image_started

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem ${ABSMEM}"
else
  absmem=""
fi

if [[ ! -f chgcentred ]]; then
  chgcentre -minw -shiftback ${obsid}.ms
  touch chgcentred
fi

wsclean $absmem -name wsclean-final -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 1000000 -auto-threshold 1 -auto-mask 3 ${obsid}.ms

pbcorrect wsclean-final image.fits beam stokes-final

rm image_started && touch image_complete
