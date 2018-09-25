#! /usr/bin/env python
from __future__ import print_function, division

import argparse

from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader
import astropy.units as u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta')
    args = parser.parse_args()

    header = getheader(args.meta)
    pointing = SkyCoord(header['RA'], header['DEC'], unit=(u.degree, u.degree))
    print(pointing.to_string('hmsdms'))


if __name__ == '__main__':
    main()
