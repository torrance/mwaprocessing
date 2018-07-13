#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=06:00:00
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
rm preprocess_scheduled || true
touch preprocess_started

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

rm *gpubox*.fits

# Run two rounds of calibration, flagging in between

/home/torrance/srclist_by_beam.py --aocalibrate -x -m ${obsid}.metafits -s /home/torrance/srclist_pumav3_EoR0aegean_EoR1pietro+ForA.txt -n 1000

for i in {1..2}; do
  calibrate $absmem -minuv 60 -maxuv 2600 -m srclist_pumav3_EoR0aegean_EoR1pietro+ForA_${obsid}_aocal1000.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions.bin

  applysolutions ${obsid}.ms solutions.bin

  aoflagger ${obsid}.ms
done

aocal_plot.py solutions.bin

rm preprocess_started && touch preprocess_complete
