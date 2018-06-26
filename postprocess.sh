#! /bin/bash
#SBATCH -M galaxy
#SBATCH --time=12:00:00
#SBATCH --partition workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 20
#SBATCH --mail-type ALL
#SBATCH --mail-user torrance.hodgson@postgrad.curtin.edu.au
#SBATCH --export=NONE

set -e

obsid=$1

# Apply solutions from preprocess, reflag, and recalibrate

applysolutions ${obsid}.ms solutions.bin

aoflagger ${obsid}.ms

calibrate -minuv 60 -maxuv 2600 -m srclist_pumav3_EoR0aegean_EoR1pietro+ForA_${obsid}_aocal1000.txt -applybeam -j 20 -i 500 ${obsid}.ms solutions.bin

applysolutions ${obsid}.ms solutions.bin

# Do a shallow clean, to be used for selfcal

# scale = 0.5 / chan
wsclean -name wsclean1 -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 1000000 -auto-threshold 5 -auto-mask 8 ${obsid}.ms

beam -2016 -proto wsclean1-XX-image.fits -ms ${obsid}.ms -m ${obsid}.metafits

pbcorrect wsclean1 image.fits beam stokes1

# Selfcal

calibrate -minuv 60 -j 20 -i 500 ${obsid}.ms selfcal-solutions.bin

applysolutions ${obsid}.ms selfcal-solutions.bin

# Do a deep clean

wsclean -name wsclean2 -multiscale -mgain 0.85 -pol xx,xy,yx,yy -joinpolarizations -weight briggs 0 -size 8000 8000 -scale 0.0034 -niter 1000000 -auto-threshold 1 -auto-mask 3 ${obsid}.ms

pbcorrect wsclean2 image.fits beam stokes2
