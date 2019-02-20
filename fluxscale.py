#! /usr/bin/env python
from __future__ import print_function, division

import argparse
import sys

from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    gleam =  Table.read(args.gleam)
    gleam_pos = SkyCoord(gleam['RAJ2000'], gleam['DEJ2000'], unit=(units.degree, units.degree))

    aegean = Table.read(args.aegean)
    aegean_pos = SkyCoord(aegean['ra'], aegean['dec'], unit=(units.degree, units.degree))

    print("Crossmatching %d aegean sources with GLEAM catalog (%d sources)" % (len(aegean_pos), len(gleam_pos)), file=sys.stderr)

    idx_gleam, idx_aegean, d2d, _ = aegean_pos.search_around_sky(gleam_pos, 1*units.arcminute)

    print("Initial cross matches: %d" % len(d2d), file=sys.stderr)

    # Ensure unique matches
    values, counts = np.unique(idx_gleam, return_counts=True)
    duplicates = np.isin(idx_gleam, values[counts > 1])
    idx_gleam = idx_gleam[~duplicates]
    idx_aegean = idx_aegean[~duplicates]
    d2d = d2d[~duplicates]

    values, counts = np.unique(idx_aegean, return_counts=True)
    duplicates = np.isin(idx_aegean, values[counts > 1])
    idx_gleam = idx_gleam[~duplicates]
    idx_aegean = idx_aegean[~duplicates]
    d2d = d2d[~duplicates]

    # Restrict distance to 30''
    idx_gleam = idx_gleam[d2d < 30*units.arcsecond]
    idx_aegean = idx_aegean[d2d < 30*units.arcsecond]
    d2d = d2d[d2d < 30*units.arcsecond]

    print("Final cross matches: %d" % len(d2d), file=sys.stderr)

    gleam = gleam[idx_gleam]
    aegean = aegean[idx_aegean]

    aegean_fluxes, gleam_fluxes = [], []

    freq = args.freq * 1E6
    gleam_freqs = np.array([76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197, 204, 212, 220, 227])
    for gleam_src, aegean_src in zip(gleam, aegean):
        int_fluxes = []
        for gleam_freq in gleam_freqs:
            key = 'int_flux_%03d' % gleam_freq
            int_fluxes.append(gleam_src[key])

        log_int_fluxes = np.log(int_fluxes)
        log_freqs = np.log(gleam_freqs * 1E6)

        fit = np.polyfit(log_freqs, log_int_fluxes, 2)
        interpolated_flux = np.exp(
            fit[0] * np.log(freq)**2 + fit[1] * np.log(freq) + fit[2]
        )

        aegean_fluxes.append(aegean_src['int_flux'])
        gleam_fluxes.append(interpolated_flux)

    ratio = np.array(aegean_fluxes) / np.array(gleam_fluxes)
    print("Median scaling factor: ", np.median(ratio), file=sys.stderr)
    print(np.median(ratio))

    plt.figure()
    plt.boxplot(ratio)
    plt.title(np.percentile(ratio, [5, 25, 50, 75, 95]))
    plt.savefig('fluxratio.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aegean', required=True)
    parser.add_argument('--gleam', required=True)
    parser.add_argument('--freq', type=float, required=True)
    args = parser.parse_args()
    main(args)
