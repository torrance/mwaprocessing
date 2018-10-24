#! /bin/bash
#SBATCH -M magnus
#SBATCH --account pawsey0272
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM,BASEDIR

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
suffix=$2
prefix=${obsid}-${suffix}
image=${prefix}.fits

mv ionkludge_scheduled ionkludge_started || touch ionkludge_started

# Construct sky model
srclist_by_beam.py -x -m ${obsid}.metafits -s $BASEDIR/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_TGSSgalactic.txt -n 2000
cat srclist_pumav3_EoR0aegean_EoR1pietro+ForA_TGSSgalactic_${obsid}_peel2000.txt | rts_to_skymodel.py > model-2000.txt

# Search image for sources
BANE $image
aegean --autoload --seedclip 10 --floodclip 4 --table ${prefix}.fits,${prefix}.csv,${prefix}.reg $image

# Cross match with my tool
cross-matcher.py crossmatch model-2000.txt ${prefix}_comp.csv > xm.fits
# Apply fitswarp
fits_warp.py --xm xm.fits --infits $image --suffix warp --ra1 source_ra --dec1 source_dec --ra2 model_ra --dec2 model_dec --plot

mv ionkludge_started ionkludge_complete
