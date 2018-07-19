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
  absmem="-absmem ${ABSMEM}"
else
  absmem=""
fi

obsid=$1
mv preprocess_scheduled preprocess_started || touch preprocess_started

# Clean up previous run
rm -r ${obsid}.ms || true
rm ${obsid}*.mwaf || true

# Unpack flags and create command line option
if [[ -f ${obsid}_flags.zip ]]; then
  unzip ${obsid}_flags.zip
  flags="-flagfiles ${obsid}_%%.mwaf"
else
  flags=""
fi

# Make metafits and measurement set from raw gpubox files

make_metafits.py --gps=${obsid}

cotter $absmem $flags -m ${obsid}.metafits -freqres 40 -timeres 4 -m ${obsid}.metafits -allowmissing -o ${obsid}.ms *gpubox*.fits

chgcentre ${obsid}.ms | grep -A 1 'Current phase direction' | tail -n 1 > phase_centre

rm *gpubox*.fits

mv preprocess_started preprocess_complete
