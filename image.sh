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
weight=$2

if [[ -z $weight ]]; then
  echo "No CLEAN weighting specified"
  exit 1
fi
name=$(echo $weight | tr -d '[:space:]')

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem ${ABSMEM}"
else
  absmem=""
fi

mv image_${name}_scheduled  image_${name}_started || touch image_${name}_started

# Clean up any files from previous job
rm ${obsid}-wsclean-${name}*.fits || true
rm ${obsid}-${name}-stokes*.fits || true
rm *.tmp || true

if [[ ! -f chgcentred ]]; then
  chgcentre -minw -shiftback ${obsid}.ms
  touch chgcentred
fi

scale=$(echo "scale=6; 0.5 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean $absmem -j 20 -name ${obsid}-wsclean-${name} -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight $weight -minuv-l 15 -size 8000 8000 -scale $scale -niter 300000 -auto-threshold 1 -auto-mask 3 ${obsid}.ms

# Create a beam if it doesn't already exist
if [[ ! -f ${obsid}-beam-xxi.fits ]]; then
  beam -2016 -proto ${obsid}-wsclean-${name}-XX-image.fits -ms ${obsid}.ms -m ${obsid}.metafits -name ${obsid}-beam
fi

pbcorrect ${obsid}-wsclean-${name} image.fits ${obsid}-beam ${obsid}-${name}-stokes

mv image_${name}_started image_${name}_complete
