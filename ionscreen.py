#! /usr/bin/env python
from __future__ import print_function, division

import argparse
import csv
import sys

from astropy.io import fits
from astropy.io.fits import writeto
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as units
from astropy.wcs import WCS
from casacore.tables import table, taql
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xm', required=True)
    parser.add_argument('--ms', required=True)
    parser.add_argument('--center', required=True)
    parser.add_argument('--radius', type=float, default=20, help="TEC map radius (degrees)")
    parser.add_argument('--delta', type=float, default=0.25, help="Pixel size (degrees)")
    args = parser.parse_args()

    # Parse image center value
    center = SkyCoord(args.center, unit=(units.hourangle, units.degree))

    # Open measurement set, and find central time and frequency
    ms = table(args.ms)
    times = sorted(set(ms.getcol('TIME_CENTROID')))
    times = [times[len(times) // 2]]  # We just starting with the central time
    freqs = table(args.ms + '::SPECTRAL_WINDOW').getcell('CHAN_FREQ', 0)  # Assume just one SPW entry
    midfreq = (max(freqs) + min(freqs)) / 2

    print("Calculating TEC values for frequency %f and time %f" % (midfreq, times[0]), file=sys.stderr)

    # prototype = fits.open(args.prototype)[0]
    # header = prototype.header
    # data = prototype.data
    # wcs = WCS(prototype)


    # # Calculate image dimension
    # # We want our TEC image to be at least as large as the original image
    # # Calculating this is yucky due to projection warping
    # width, height = data.shape[3], data.shape[2]  # [ stokes, freq, dec (y), ra (x) ]
    # center_x, center_y = width // 2, height // 2
    # center_ra, center_dec, _, _ = wcs.all_pix2world([[center_x, center_y, 0, 0]], 0)[0]

    # # Calculate maximum angular extent of image
    # corners = wcs.all_pix2world([
    #     [0, 0, 0, 0],
    #     [width, 0, 0, 0],
    #     [0, height, 0, 0],
    #     [width, height, 0, 0]
    # ], 0)
    # max_ra = max(abs(corners.T[0] - center_ra))
    # max_dec = max(abs(corners.T[1] - center_dec))

    # width, height = int((2 * max_ra) // args.delta) + 1, int((2 * max_dec) // args.delta) + 1
    # center_x, center_y = int(width // 2), int(height // 2)


    # Provision TEC fits file and set up headers
    width, height = int((2 * args.radius) // args.delta), int((2 * args.radius) // args.delta)
    center_x, center_y = int(width // 2), int(height // 2)

    print("Creating TEC image of dimesions (%d, %d)" % (width, height), file=sys.stdout)

    data = np.zeros((len(times), 1, 128, height, width), dtype=np.float)  # [time, frequency, antennas, dec, ra]
    Atec = fits.PrimaryHDU(data)

    Atec.header['CTYPE1'] = 'RA---SIN'
    Atec.header['CRPIX1'] = center_x
    Atec.header['CRVAL1'] = center.ra.deg
    Atec.header['CDELT1'] = args.delta
    Atec.header['CUNIT1'] = 'deg'

    Atec.header['CTYPE2'] = 'DEC--SIN'
    Atec.header['CRPIX2'] = center_y
    Atec.header['CRVAl2'] = center.dec.deg
    Atec.header['CDELT2'] = args.delta
    Atec.header['CUNIT2'] = 'deg'

    Atec.header['CTYPE3'] = 'ANTENNA'
    Atec.header['CRPIX3'] = 1
    Atec.header['CRVAL3'] = 0

    Atec.header['CTYPE4'] = 'FREQ'
    Atec.header['CRPIX4'] = 1
    Atec.header['CRVAL4'] = midfreq
    Atec.header['CDELT4'] = 1
    Atec.header['CUNIT4'] = 'Hz'

    Atec.header['CTYPE5'] = 'TIME'
    Atec.header['CRPIX5'] = 1
    Atec.header['CRVAL5'] = times[0]  # FIXME
    Atec.header['CDELT5'] = 1

    wcs = WCS(Atec.header)

    # Process crossmatched sources
    xm = Table.read(args.xm)
    model_positions = np.radians(np.array([xm['model_ra'], xm['model_dec']]))
    source_positions = np.radians(np.array([xm['source_ra'], xm['source_dec']]))
    offsets = np.sin(model_positions - source_positions)

    model_positions_lm = radec_to_lm(model_positions[0], model_positions[1], center.ra.rad, center.dec.rad)

    # Get l,m values for TEC file
    xx, yy = np.meshgrid(range(0, width), range(0, height))
    pixels = np.array([xx.flatten(), yy.flatten()]).T

    ret = wcs.all_pix2world([[x, y, 0, 0, 0] for x, y in pixels], 0)
    grid_lm = radec_to_lm(np.radians(ret.T[0]), np.radians(ret.T[1]), center.ra.rad, center.dec.rad)

    # Compute interpolated position offsets
    delta_l = griddata(model_positions_lm.T, offsets[0], grid_lm.T, fill_value=0)
    delta_m = griddata(model_positions_lm.T, offsets[1], grid_lm.T, fill_value=0)

    # Plot interpolation
    plt.figure()
    plt.quiver(model_positions_lm[0], model_positions_lm[1], offsets[0], offsets[1], angles='xy', scale=0.01, scale_units='xy')
    plt.quiver(grid_lm[0], grid_lm[1], delta_l, delta_m, angles='xy', scale=0.01, scale_units='xy', color='gray')
    plt.gca().invert_xaxis()
    plt.show()

    delta_l = np.reshape(delta_l, (height, width))  # [ dec, ra ]
    delta_m = np.reshape(delta_m, (height, width))

    # Create debugging fits files
    for delta, name in [(delta_l, 'delta-l'), (delta_m, 'delta-m')]:
        hdu = fits.PrimaryHDU(delta)
        hdu.header = Atec.header.copy()
        hdu.writeto(name + '.fits', overwrite=True)

    for i, time in enumerate(times):
        for ant in range(1, 128):
            sys.stderr.write("\rCalculating antenna %d..." % ant)
            sys.stderr.flush()

            tbl = taql("select UVW from $ms where TIME_CENTROID = $time and ANTENNA1 = 0 and ANTENNA2 = $ant")

            if len(tbl) == 0:
                continue
            elif len(tbl) > 1:
                print("Oopsie doodle!")
                exit(1)

            u, v = tbl.getcol('UVW')[0][[True, True, False]]  / (299792458 / midfreq)
            phase = 2 * np.pi * (u * delta_l + v * delta_m)
            data[i, 0, ant, :, :] = phase / -8.44797245E9 * midfreq

    print(" Done.", file=sys.stderr)

    Atec.data = data
    Atec.writeto('tec.fits', overwrite=True)


def radec_to_lm(ra, dec, ra0, dec0):
    # Copied from mwa-reduce/units/imagecoordinates.h
    # Todo: understand this transformation
    deltaAlpha = ra - ra0
    sinDeltaAlpha = np.sin(deltaAlpha)
    cosDeltaAlpha = np.cos(deltaAlpha)
    sinDec = np.sin(dec)
    cosDec = np.cos(dec)
    sinDec0 = np.sin(dec0)
    cosDec0 = np.cos(dec0)

    l = cosDec * sinDeltaAlpha
    m = sinDec * cosDec0 - cosDec * sinDec0 * cosDeltaAlpha
    return np.array([l, m])


if __name__ == '__main__':
    main()
