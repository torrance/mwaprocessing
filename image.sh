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

# Parse parameters
SHORT='s:o:n:'
LONG='size:,options:,name:'
OPTS=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")
eval set -- "$OPTS"

# Default values
size=9000
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
pointing=$(pointing.py ${obsid}.metafits)
chgcentre ${obsid}.ms $pointing
chgcentre -minw -shiftback ${obsid}.ms

scale=$(echo "scale=6; 0.5 / $(getchan.py ${obsid}.metafits)" | bc)
wsclean $absmem -name ${obsid}-wsclean-${name} -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight $weight -minuv-l 15 -size $size $size -scale $scale -niter 300000 -auto-threshold 1.5 -auto-mask 3 $options ${obsid}.ms

# Create beam fits
make_beam.py -f ${obsid}-wsclean-${name}-XX-image.fits -m ${obsid}.metafits --model 2016 --jones

# Rename make_beam.py output for pbcorrect
for file in $(ls | grep XX-image_beam.*\.fits); do
  pol=$(echo $file | sed 's/.*image_beam\(.*\)\.fits/\1/' | tr '[:upper:]' '[:lower:]')
  mv $file ${obsid}-beam-${pol}.fits
done

pbcorrect ${obsid}-wsclean-${name} image.fits ${obsid}-beam ${obsid}-${name}-stokes

mv image_${name}_started image_${name}_complete
