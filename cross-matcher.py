#! /usr/bin/env python

from __future__ import print_function, division

import argparse
import csv
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io.fits import getheader
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', type=float, default=1, help="Search radius for matching source, in units of beam width")
    parser.add_argument('-e', '--exclusion', type=float, default=2, help="Exclusion zone for other additional source, in units of beam width")
    parser.add_argument('mode', type=str, choices=['crossmatch', 'shift', 'annotate'])
    parser.add_argument('model', type=str, help="Input model in skymodel format")
    parser.add_argument('sources', type=str, help="Aegean sources in csv format")
    args = parser.parse_args()

    # Calculate search areas
    header = getheader(args.sources.replace('_comp.csv', '.fits'))
    radius = args.radius * max(header['BMAJ'], header['BMIN'])
    region = args.exclusion * max(header['BMAJ'], header['BMIN'])
    print('Cross-matching in radius of %f.2", with exclusion zone of %f.2"...' % (radius * 3600, region * 3600), file=sys.stderr)

    # Get model coordinates
    model_ra, model_dec = [], []
    with open(args.model) as f:
        for line in f:
            parts = line.split()
            if parts[0] == 'position':
                ra, dec = parts[1], parts[2]
                model_ra.append(ra)
                model_dec.append(dec)

    model = SkyCoord(ra=model_ra, dec=model_dec, unit=(u.degree, u.degree))

    # Get Aegean sources
    sources_ra, sources_dec = [], []
    with open(args.sources) as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            ra, dec = row[6], row[8]
            sources_ra.append(float(ra))
            sources_dec.append(float(dec))

    sources = SkyCoord(ra=sources_ra, dec=sources_dec, unit=(u.degree, u.degree))

    print("Searching for nearby sources...", file=sys.stderr)
    idx_sources, idx_model, d2d, _ = model.search_around_sky(sources, region*u.degree)

    # Remove sources with multiple matches
    print("Removing ambiguous matches...", file=sys.stderr)
    idx_duplicates = np.zeros_like(idx_sources, dtype=np.bool)
    for i in idx_model:
        idxs = (idx_model == i)
        if sum(idxs) > 1:
            idx_duplicates = np.logical_or(idx_duplicates, idxs)

    idx_sources = idx_sources[~idx_duplicates]
    idx_model = idx_model[~idx_duplicates]
    d2d = d2d[~idx_duplicates]
    print("Done", file=sys.stderr)

    # Remove sources with matches outside the inner radii value
    print("Filtering matches to radius only...", file=sys.stderr)
    idx_sources = idx_sources[d2d < radius*u.degree]
    idx_model = idx_model[d2d < radius*u.degree]
    d2d = d2d[d2d < radius*u.degree]
    print("Done", file=sys.stderr)

    print("Model: %d Sources: %d, cross-matched: %d" % (len(model), len(sources), len(idx_model)), file=sys.stderr)

    if args.mode == 'crossmatch':
        # Create columns
        model_ra = [c.ra.degree for c in model[idx_model]]
        model_dec = [c.dec.degree for c in model[idx_model]]
        source_ra = [c.ra.degree for c in sources[idx_sources]]
        source_dec = [c.dec.degree for c in sources[idx_sources]]

        table = Table(
            data=[model_ra, model_dec, source_ra, source_dec],
            names=('model_ra', 'model_dec', 'source_ra', 'source_dec'),
        )
        table.write(sys.stdout, format='fits')

    elif args.mode == 'shift':
        # Create modified model
        i = 0
        with open(args.model) as m:
            for line in m:
                line = line.rstrip()
                parts = line.split()
                if parts[0] == 'position' and i in idx_model:
                    source = sources[idx_sources[idx_model == i][0]]
                    print('    position %s' % source.to_string("hmsdms"), file=sys.stdout)
                else:
                    print(line, file=sys.stdout)

                if parts[0] == 'position':
                    i += 1

    elif args.mode == 'annotate':
        # Create ds9 region file of matches
        print("global dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1", file=sys.stdout)
        print("icrs", file=sys.stdout)

        for i, j in zip(idx_sources, idx_model):
            print("point %fd %fd # point=circle color=green" % (model[j].ra.deg, model[j].dec.deg), file=sys.stdout)
            print("point %fd %fd # point=circle color=green" % (sources[i].ra.deg, sources[i].dec.deg), file=sys.stdout)

        for i, c in enumerate(sources):
            if i in idx_sources:
                continue
            print("point %fd %fd # point=circle color=gray" % (c.ra.deg, c.dec.deg), file=sys.stdout)

        for i, c in enumerate(model):
            if i in idx_model:
                continue
            print("point %fd %fd # point=circle color=red" % (c.ra.deg, c.dec.deg), file=sys.stdout)

if __name__ == '__main__':
    main()
