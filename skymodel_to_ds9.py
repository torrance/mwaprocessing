#! /usr/bin/env python
from __future__ import print_function

import argparse
import sys

from astropy.coordinates import SkyCoord

model_ra, model_dec = [], []
for line in sys.stdin:
    parts = line.split()
    if parts[0] == 'position':
	_, ra, dec = parts
	model_ra.append(ra)
	model_dec.append(dec)

model = SkyCoord(ra=model_ra, dec=model_dec)

print("global color=red dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1", file=sys.stdout)
print("icrs", file=sys.stdout)
for c in model:
    print("point %fd %fd # point=circle" % (c.ra.deg, c.dec.deg), file=sys.stdout)
