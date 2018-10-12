#! /usr/bin/env python
from __future__ import print_function

import argparse
import csv
import datetime
import os
import sys

from astropy.io.fits import getheader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metafits')
    parser.add_argument('--timedelta', type=int, default=12, help="Maximum allowed time between observation and calibrator (hours)")
    args = parser.parse_args()

    # Get observation date and channels
    header = getheader(args.metafits)
    date = datetime.datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S")
    channels = {int(ch) for ch in header['CHANNELS'].split(',')}  # Set comprehension

    # Load calibrators database
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/calibrators.csv') as f:
        reader = csv.reader(f)
        next(reader)  # Skip headers
        rows = [line for line in reader]

    # Sort calibrators closest in time to observation
    rows = sorted(rows, key=lambda x: abs(datetime.datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S") - date))

    # Find first calibrator observation that matches our channels
    for cal in rows:
        cal_date = datetime.datetime.strptime(cal[1], "%Y-%m-%d %H:%M:%S")
        if abs(date - cal_date) > datetime.timedelta(hours=args.timedelta):
            print("No calibrator observation within %d hours" % args.timedelta, file=sys.stderr)
            exit(1)

        if channels == {int(ch) for ch in cal[18:]}:
            print(cal[0], cal[13])
            exit(0)


if __name__ == '__main__':
      main()
