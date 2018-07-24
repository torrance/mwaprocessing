#! /usr/bin/env python

from __future__ import print_function
import argparse

from astropy.io.fits import getheader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metafits')
    args = parser.parse_args()

    metafits = getheader(args.metafits)
    print(metafits['CENTCHAN'])

if __name__ == '__main__':
    main()
