#! /bin/bash
#SBATCH -M magnus
#SBATCH -A pawsey0293
#SBATCH --time=24:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
##SBATCH --cpus-per-task 14
##SBATCH --mem 60G
#SBATCH --mail-type FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=ABSMEM,BASEDIR

source ~/.profile

set -e
set -x

# Parse parameters
SHORT='s:o:n:'
LONG='size:,options:,name:'
OPTS=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")
eval set -- "$OPTS"

# Default values
size=7500
options=""

while true; do
  case "$1" in
    -s | --size )
      size="$2"
      shift 2
      ;;
    -o | --options )
      options="$2"
      shift 2
      ;;
    -n | --name )
      name="$2"
      shift 2
      ;;
    -p | --position )
      position="$2"
      shift 2
      ;;
    -- )
      shift
      break
      ;;
    * )
      break;
      ;;
  esac
done

obsid=$1
weight=$2

if [[ -z $weight ]]; then
  echo "No CLEAN weighting specified"
  exit 1
fi

if [[ -z $name ]]; then
  name=$(echo $weight | tr -d '[:space:]')
fi

if [[ ! -z $ABSMEM ]]; then
  absmem="-absmem ${ABSMEM}"
else
  absmem=""
fi

mv image_${name}_scheduled  image_${name}_started || touch image_${name}_started

# Clean up any files from previous job
rm ${obsid}-wsclean-${name}*.fits || true
rm ${obsid}-${name}-stokes*.fits || true
rm ${obsid}-beam*.fits || true
rm *.tmp || true

# Change to pointing direction, then to minw
if [[ -z $pointing ]]; then
  pointing=$(pointing.py ${obsid}.metafits)
fi
chgcentre ${obsid}.ms $pointing
chgcentre -minw -shiftback ${obsid}.ms

scale=$(echo "scale=6; 0.6 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean \
  $absmem \
  -mwa-path $BASEDIR \
  -name ${obsid}-wsclean-${name} \
  -apply-primary-beam \
  -mgain 0.8 \
  -pol i,q,u,v \
  -weight $weight \
  -minuv-l 15 \
  -size $size $size \
  -scale $scale \
  -niter 9999999 \
  -auto-threshold 1.5 \
  -auto-mask 3 \
  -channels-out 4 \
  -fit-spectral-pol 4 \
  -join-channels \
  -nmiter 12 \
  $options \
  ${obsid}.ms

mv image_${name}_started image_${name}_complete
