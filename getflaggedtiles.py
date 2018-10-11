#! /usr/bin/env python
from __future__ import print_function, division

import argparse

from astropy.io import fits
from astropy.table import Table
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metafits')
    args = parser.parse_args()

    tbl = Table.read(args.metafits)
    flags = np.array(tbl['Flag'], dtype=np.bool)
    tiles = set(tbl['Antenna'][flags])

    for tile in tiles:
        print(tile)


if __name__ == '__main__':
    main()
