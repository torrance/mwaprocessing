#! /bin/bash
#SBATCH -M zeus
#SBATCH -A pawsey0293
#SBATCH --time=00:30:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task 14
#SBATCH --mem 60G
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM,BASEDIR

set -e
set -x

obsid=$1
label=$2

PATH="~/.local/bin:$PATH"


mv fluxscale_scheduled fluxscale_started || touch fluxscale_started

filename=${obsid}-wsclean-${label}-MFS-I-image-pb.fits

BANE --cores 14 $filename
center=$(pointing.py --degrees ${obsid}.metafits)
freq=$(getfreq.py ${obsid}.metafits)
MIMAS -o region.mim +c $center 12
aegean --cores 14 --autoload --region region.mim --seedclip 100 --floodclip 5 --table $filename $filename
fluxscale.py --gleam $BASEDIR/GLEAM_EGC_v2.fits --aegean $(dirname $filename)/$(basename $filename .fits)_comp.fits --freq $freq > scaling_factor

mv fluxscale_started fluxscale_complete
