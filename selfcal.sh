#! /bin/bash
#SBATCH -M magnus
#SBATCH --account pawsey0272
#SBATCH --time=12:00:00
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

# Flag tiles if badantennae file is present
if [[ -f badantennae ]]; then
  cat badantennae | xargs flagantennae ${obsid}.ms
fi

# Apply previous calibration solution, if one is present
# This is necessary when recovering from a failed job
if [[ ! -z $prior ]]; then
  applysolutions ${obsid}.ms solutions-${prior}.bin
fi

# Flag data
aoflagger ${obsid}.ms

# Change to pointing direction, then to minw
pointing=$(pointing.py ${obsid}.metafits)
chgcentre ${obsid}.ms $pointing
chgcentre -minw -shiftback ${obsid}.ms

# Do a shallow clean, to be used for selfcal
scale=$(echo "scale=6; 0.6 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean -name ${obsid}-wsclean-${label} -apply-primary-beam -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs -2 -size 7500 7500 -scale $scale -niter 300000 -auto-threshold 5 -auto-mask 8 -mwa-path $BASEDIR $absmem ${obsid}.ms

# Selfcal
calibrate -minuv 60 -j 20 -i 500 $absmem -mwa-path $BASEDIR -ch 4 ${obsid}.ms solutions-${label}.bin

applysolutions ${obsid}.ms solutions-${label}.bin

aoflagger ${obsid}.ms

aocal_plot.py solutions-${label}.bin

mv ${label}_started ${label}_complete
