#! /usr/bin/env python
from __future__ import division, print_function

import argparse
import sys

from casacore.tables import table, taql
import matplotlib.pyplot as plt
from numba import njit, prange, float32, float64, complex64
import numpy as np


@njit(parallel=True)
def differ(data, model):
    diff = np.empty((3, data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
    for i in prange(len(data)):
        diff[0, i] = data[i] - model[i]
        diff[1, i] = np.absolute(data[i]) - np.absolute(model[i])
        diff[2, i] = np.angle(data[i]) - np.angle(model[i])

    return diff


@njit([float64(complex64[:], float32[:])], parallel=True)
def complexchisquared(arr, weights):
    count = 0
    for i in prange(len(arr)):
       count += (arr[i].real**2 + arr[i].imag**2) * weights[i]

    return count / (2 * len(arr))


@njit([float64(float32[:], float32[:])], parallel=True)
def chisquared(arr, weights):
    count = 0
    for i in prange(len(arr)):
       count += arr[i]**2 * weights[i]

    return count / len(arr)


parser = argparse.ArgumentParser()
parser.add_argument('mset')
parser.add_argument('--antenna-sigma', type=float, default=3.5)
parser.add_argument('--baseline-sigma', type=float, default=5)
parser.add_argument('--data-column', default='CORRECTED_DATA')
parser.add_argument('--model-column', default='MODEL_DATA')
args = parser.parse_args()

# Extract data from measurement set
print("Reading data into memory...", end="", file=sys.stderr); sys.stderr.flush()
mset = table(args.mset, ack=False)
tbl = taql("select * from $mset where ANTENNA1 <> ANTENNA2 and not FLAG_ROW")

data = tbl.getcol(args.data_column)
model = tbl.getcol(args.model_column)
flags = tbl.getcol("FLAG")
weights = tbl.getcol("WEIGHT_SPECTRUM")
ant1 = tbl.getcol("ANTENNA1")
ant2 = tbl.getcol("ANTENNA2")
print(" Done", file=sys.stderr)

# Calculate difference between data and model
print("Diffing data...", end="", file=sys.stderr); sys.stderr.flush()
data[flags] = np.nan
fulldiff, ampdiff, phasediff = differ(data, model)
ampdiff, phasediff = ampdiff.real, phasediff.real
print(" Done", file=sys.stderr)

baselines = np.zeros((128, 128))
fulls = np.zeros(128)
amps = np.zeros(128)
phases = np.zeros(128)
baselines[:], fulls[:], amps[:], phases[:] = np.nan, np.nan, np.nan, np.nan

# Calculate baseline summaries
print("Calculating chi squared statistics...", end="", file=sys.stderr); sys.stderr.flush()
for a1 in range(128):
    for a2 in range(a1 + 1, 128):
        idx = np.any([
            np.all([ant1 == a1, ant2 == a2], axis=0),
            np.all([ant1 == a2, ant2 == a1], axis=0),
        ], axis=0)

        d = fulldiff[idx][np.isfinite(fulldiff[idx])]
        w = weights[idx][np.isfinite(fulldiff[idx])]

        if len(d) == 0:
            continue

        baselines[a1, a2] = complexchisquared(d, w)

# Calculate antenna summaries
for a in range(128):
    idx = np.any([ant1 == a, ant2 == a], axis=0)

    d = fulldiff[idx][np.isfinite(fulldiff[idx])]
    w = weights[idx][np.isfinite(fulldiff[idx])]
    if len(d) == 0:
        continue

    fulls[a] = complexchisquared(d, w)

    d = ampdiff[idx][np.isfinite(ampdiff[idx])]
    w = weights[idx][np.isfinite(ampdiff[idx])]
    amps[a] = chisquared(d, w)

    d = phasediff[idx][np.isfinite(phasediff[idx])]
    w = weights[idx][np.isfinite(phasediff[idx])]
    phases[a] = chisquared(d, w)

print(" Done", file=sys.stderr)

# Baselines
threshold = np.nanmean(baselines) + args.baseline_sigma * np.nanstd(baselines)
flagged = np.where(baselines > threshold)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title("Baseline Chi Squared")
plt.imshow(baselines)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Baseline Chi Squared (flagged baselines masked)")
baselines[flagged] = np.nan
plt.imshow(baselines)
plt.colorbar()
plt.savefig('baselines.png')

with open('badbaselines.txt', 'w') as f:
    for a1, a2 in np.transpose(flagged):
        print(a1, a2, file=f)

plt.figure(figsize=(12, 12))

# Full
threshold = np.nanmean(fulls) + args.antenna_sigma * np.nanstd(fulls)
fullflagged = np.where(fulls > threshold)

ax = plt.subplot(3, 1, 1)
plt.title("Full ChiSq")
plt.scatter(range(0, 128), fulls)
plt.scatter(fullflagged, fulls[fullflagged])
ax.axhline(threshold, linestyle='--')

# Amps
threshold = np.nanmean(amps) + args.antenna_sigma * np.nanstd(amps)
ampflagged = np.where(amps > threshold)

ax = plt.subplot(3, 1, 2)
plt.title("Amp ChiSq")
plt.scatter(range(0, 128), amps)
plt.scatter(ampflagged, amps[ampflagged])
ax.axhline(threshold, linestyle='--')

# Phases
threshold = np.nanmean(phases) + args.antenna_sigma * np.nanstd(phases)
phaseflagged = np.where(phases > threshold)

ax = plt.subplot(3, 1, 3)
plt.title("Phase ChiSq")
plt.scatter(range(0, 128), phases)
plt.scatter(phaseflagged, phases[phaseflagged])
ax.axhline(threshold, linestyle='--')

flagged = set(list(fullflagged[0]) + list(ampflagged[0]) + list(phaseflagged[0]))
with open('badtiles.txt', 'w') as f:
    for tile in flagged:
        print(tile, file=f)

plt.savefig('tiles.png')
