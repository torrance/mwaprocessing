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
  absmem="-absmem $ABSMEM"
else
  absmem=""
fi

obsid=$1
label=$2
prior=$3

if [[ -z $label ]]; then
  echo "No calibration label provided"
  exit 1
fi

mv cal_${label}_scheduled cal_${label}_started || touch cal_${label}_started

# Clean up from any previous runs
rm solutions-${label}.bin || true
rm srclist_pumav3_EoR0aegean_EoR1pietro+ForA_TGSSgalactic_${obsid}_peel200.txt || true
rm model.txt || true

# Calibrate seems to require we are at the phase center
phase_center=$(phasecenter.py ${obsid}.metafits)
chgcentre ${obsid}.ms $phase_center

# Apply previous calibration solution, if one is present
# This is necessary when recovering from a failed job
if [[ ! -z $prior ]]; then
  applysolutions ${obsid}.ms solutions-${prior}.bin
fi

# Flag any tiles that MWA ops have flagged
# This should be handled automatically but is a bug in Cotter at the moment
flagged=$(getflaggedtiles.py ${obsid}.metafits)
if [[ ! -z $flagged ]]; then
  echo $flagged | xargs flagantennae ${obsid}.ms
fi

# Flag tiles if badantennae file is present
if [[ -f badantennae ]]; then
  cat badantennae | xargs flagantennae ${obsid}.ms
fi

# Construct sky model
srclist_by_beam.py -x -m ${obsid}.metafits -s /home/torrance/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_TGSSgalactic.txt -n 200
cat srclist_pumav3_EoR0aegean_EoR1pietro+ForA_TGSSgalactic_${obsid}_peel200.txt | rts_to_skymodel.py > model.txt

# Add ad-hoc sky models
for model in $(ls | grep '^model\-' | grep '\.txt$'); do
  cat $model | grep -v 'skymodel' >> model.txt
done

if [[ -z $prior ]]; then
  # Ignore previous calibration attempts if no prior is provided
  data="-datacolumn DATA"
fi
calibrate $absmem $data -minuv 60 -maxuv 2600 -m model.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions-${label}.bin

applysolutions ${obsid}.ms solutions-${label}.bin

aoflagger ${obsid}.ms

aocal_plot.py solutions-${label}.bin

mv cal_${label}_started cal_${label}_complete
