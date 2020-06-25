#! /bin/bash
#SBATCH -M zeus
#SBATCH --partition workq
#SBATCH --account pawsey0293
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task 14
#SBATCH --mem 60G

set -x
set -e

obsid=$1
ra=$2
dec=$3

mv peel_${ra}_${dec}_scheduled peel_${ra}_${dec}_started || touch peel_${ra}_${dec}_started

chgcentre ${obsid}.ms $ra $dec
# chgcentre -minw -shiftback ${obsid}.ms

# Save main model
# columnabacus.py ${obsid}.ms::MODEL_SAVED = ${obsid}.ms::MODEL_DATA

# Remove primary beam model
# columnabacus.py ${obsid}.ms::CORRECTED_DATA = ${obsid}.ms::CORRECTED_DATA - ${obsid}.ms::MODEL_DATA

scale=$(echo "scale=6; 0.6 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean \
  -name ${ra}-${dec}-before \
  -mgain 0.8 \
  -pol i \
  -weight briggs 0 \
  -minuv-l 15 \
  -size 250 250 \
  -scale $scale \
  -niter 999999 \
  -auto-threshold 1.5 \
  -auto-mask 3 \
  -nmiter 12 \
  -channels-out 8 \
  -fit-spectral-pol 3 \
  -join-channels \
  ${obsid}.ms

# Subtract out model
columnabacus.py ${obsid}.ms::CORRECTED_DATA = ${obsid}.ms::CORRECTED_DATA - ${obsid}.ms::MODEL_DATA

# Restore model data
# columnabacus.py ${obsid}.ms::MODEL_DATA = ${obsid}.ms::MODEL_SAVED

# Restore primary beam model
# columnabacus.py ${obsid}.ms::CORRECTED_DATA = ${obsid}.ms::CORRECTED_DATA + ${obsid}.ms::MODEL_DATA

mv peel_${ra}_${dec}_started peel_${ra}_${dec}_complete
