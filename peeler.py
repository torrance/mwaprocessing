#! /usr/bin/env python

from __future__ import print_function, division

import argparse
from itertools import imap
import sys
from multiprocessing.dummy import Pool

from astropy.io.fits import getheader
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
import astropy.table
import astropy.units as u
from casacore.fitting import fitserver
import casacore.functionals as functionals
from casacore.tables import table, taql
from casacore.measures import measures
from casacore.quanta import quantity
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import mwa_pb.config
import mwa_pb.primary_beam as pb
from scipy.optimize import least_squares
import warnings


speed_of_light = 299792458


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', help="The measurement set from which to peel")
    parser.add_argument('--meta', help="The observation metafits file")
    parser.add_argument('--model', help="The skymodel to peel, in aoskymodel 1.1 format")
    parser.add_argument('--aegean', help="Aegean CSV")
    parser.add_argument('--datacolumn', default='CORRECTED_DATA')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--minuv', type=float, default=0, help="Fit models only on baselines above this minimum uv distance (metres)")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()

    metafits = getheader(args.meta)
    date = Time(metafits['DATE-OBS'], location=mwa_pb.config.MWAPOS)
    delays = [int(d) for d in metafits['DELAYS'].split(',')]
    delays = [delays, delays] # Shuts up mwa_pb
    location = mwa_pb.config.MWAPOS

    print("Observation date: ", date)
    print("Delays: ", delays)

    # Retrieve telescope position
    dm = measures()
    obs = table(args.ms + '/OBSERVATION', ack=False)
    if 'TELESCOPE_NAME' in obs.colnames():
        names = obs.getcol('TELESCOPE_NAME')
        if len(names) == 1:
            obspos = dm.observatory(names[0])
            dm.do_frame(obspos)

        else:
            print("Failed to work out the telescope name of this observation")
            exit(1)
    else:
        print("Measurement set did not provide the telescope name")
        exit(1)
    print("Observation taken using telescope %s" % names[0])

    ms = table(args.ms, readonly=False, ack=False)
    freqs = table(args.ms + '/SPECTRAL_WINDOW', ack=False).getcell('CHAN_FREQ', 0)
    midfreq = (max(freqs) + min(freqs)) / 2
    lambdas = speed_of_light / freqs
    ra0, dec0 = table(args.ms + '/FIELD', ack=False).getcell('PHASE_DIR', 0)[0]  # Phase centre in radians
    chans, pols = ms.getcell(args.datacolumn, 0).shape
    print("There are %d channels" % chans)

    # Check whether width evenly divides the total channels
    if chans % args.width != 0:
        print("Width (%d) does not evenly divide channel number (%d)" % (args.width, chans))
        exit(1)

    widefreqs = np.array([
        np.mean(freqs[i:i+args.width]) for i in range(0, len(freqs), args.width)
    ])
    widelambdas = speed_of_light / widefreqs

    antennas = table(args.ms + '/ANTENNA', ack=False).getcol('POSITION')
    antennas = dm.position(
        'itrf',
        quantity(antennas.T[0], 'm'),
        quantity(antennas.T[1], 'm'),
        quantity(antennas.T[2], 'm'),
    )

    # Create PEELED_DATA column if it does not exist
    if ('CORRECTED_DATA' not in ms.colnames()):
        print("Creating CORRECTED_DATA column...", end=" ")
        sys.stdout.flush()
        coldesc = ms.getcoldesc('DATA')
        coldesc['name'] = 'CORRECTED_DATA'
        ms.addcols(coldesc)

        peeled = ms.getcol(args.datacolumn)
        ms.putcol('CORRECTED_DATA', peeled)
        print("Done")

    # Load models
    with open(args.model) as f:
        models = model_parser(f)
    print("Initialised %d model sources" % len(models))

    # Initialise points based on beam values at the widefreqs
    comps = [c for m in models for c in m.components]

    # Calculate Jones matrices for each model component
    # JBeams[widefreq, component, row, col]
    JBeams = []
    for widefreq in widefreqs:
        altaz = [radec_to_altaz(comp.ra, comp.dec, date, location) for comp in comps]
        altaz = np.array(zip(*altaz))
        JBeam = pb.MWA_Tile_full_EE(np.pi/2 - altaz[0], altaz[1], widefreq, delays=delays, jones=True)
        JBeams.append(JBeam)

    sources = []
    for i, comp in enumerate(comps):
        xx_fluxes = []
        yy_fluxes = []
        for j, widefreq in enumerate(widefreqs):
            stokes = comp.flux(widefreq) # Model flux per Stokes paramter

            # XX = I + V; XY = U + iV; YY = Q - iU; YY = I -Q
            linear = np.array([
                [stokes[0] + stokes[3], stokes[2] + 1j * stokes[3]],
                [stokes[1] - 1j * stokes[2], stokes[0] - stokes[1]],
            ])

            # apparent = JBeam x linear x (JBeam)^H ..... where H is the Hermitian transpose
            JBeam = JBeams[j][i]
            apparent = np.matmul(np.matmul(JBeam, linear), np.conj(JBeam.T))

            # For now (FIX?) we just take the real part
            xx_fluxes.append(np.real(apparent[0, 0]))
            yy_fluxes.append(np.real(apparent[1, 1]))

        xx_fluxes, yy_fluxes = np.array(xx_fluxes), np.array(yy_fluxes)

        # Estimate initial parameters
        log_widefreqs = np.log(widefreqs)
        log_xx_fluxes = np.log(xx_fluxes)
        log_yy_fluxes = np.log(yy_fluxes)
        xx3 = np.polyfit(log_widefreqs, log_xx_fluxes, 0)
        yy3 = np.polyfit(log_widefreqs, log_yy_fluxes, 0)

        sources.append(
            #Point(0, 0, xx3, 0, 0, yy3, comp.ra, comp.dec),
            Gaussian(0, 0, xx3, 0, 0, yy3, 0.1, 0.1, 0, comp.ra, comp.dec),
        )

    # Group sources by a spatial threshold
    threshold = (3 / 60) * np.pi / 180
    partitions = spatial_partition(sources, threshold)

    # Read data minus flagged rows
    tbl = taql("select * from $ms where not FLAG_ROW")
    uvw = tbl.getcol('UVW')
    uvw.flags.writeable = False
    times = tbl.getcol('TIME_CENTROID')
    times.flags.writeable = False
    ant1 = tbl.getcol('ANTENNA1')
    ant1.flags.writeable = False
    ant2 = tbl.getcol('ANTENNA2')
    ant2.flags.writeable = False
    data = tbl.getcol(args.datacolumn)[:, :, [True, False, False, True]].copy() # Just XX, YY

    # Handle flags by setting entries that are flagged as NaN
    flags = tbl.getcol('FLAG')[:, :, [True, False, False, True]]
    data[flags] = np.nan

    # For some reason, it is necessary to rotate onto
    # the current phase direction. Somehow this makes future offsets
    # internally self-consistent.
    dm = measures()
    phasecentre = dm.direction(
        'j2000',
        quantity(ra0, 'rad'),
        quantity(dec0, 'rad'),
    )

    uvw, data = phase_rotate(
        uvw,
        times,
        data,
        ant1,
        ant2,
        obspos,
        phasecentre,
        antennas,
        speed_of_light / freqs,
    )
    # tbl.putcol('UVW', uvw)
    # d = tbl.getcol('DATA')
    # d[:, :, [True, False, False, True]] = data
    # tbl.putcol('DATA', d)

    if args.workers > 0:
        pool = Pool(args.workers)

    for passno in range(args.passes):
        i = 0

        # Order sources by apparent flux at midfreq
        partitions = sorted(
            partitions, reverse=True, key=lambda xs: sum([x.flux(midfreq) for x in xs])
        )

        print("Beginning pass %d...   0%%" % (passno + 1), end="")
        sys.stdout.flush()
        while i < len(partitions):
            # TODO: use prior partitions to estimate updated values for l,m of this partition

            # Find the next batch of partitions that are within some threshold
            # of the currently brightest partition due to process.
            fluxlimit = args.threshold * sum([
                x.flux(midfreq) for x in partitions[i]
            ])
            for n, partition in enumerate(partitions[i:] + [None]):
                if partition is None:
                    break
                elif sum([x.flux(midfreq) for x in partition]) < fluxlimit:
                    break

            batch = partitions[i:i+n]

            data.flags.writeable = False
            if args.workers:
                diffs = pool.imap_unordered(
                    peel_star,
                    zip(
                        [uvw] * len(batch),
                        [times] * len(batch),
                        [data] * len(batch),
                        [ant1] * len(batch),
                        [ant2] * len(batch),
                        batch,
                        [antennas] * len(batch),
                        [freqs] * len(batch),
                        [obspos] * len(batch),
                        [ra0] * len(batch),
                        [dec0] * len(batch),
                        [args] * len(batch),
                        [passno] * len(batch),
                    )
                )
            else:
                diffs = imap(
                    peel,
                    [uvw] * len(batch),
                    [times] * len(batch),
                    [data] * len(batch),
                    [ant1] * len(batch),
                    [ant2] * len(batch),
                    batch,
                    [antennas] * len(batch),
                    [freqs] * len(batch),
                    [obspos] * len(batch),
                    [ra0] * len(batch),
                    [dec0] * len(batch),
                    [args] * len(batch),
                    [passno] * len(batch),
                )
            data.flags.writeable = True

            data = data.copy()  # Avoid changing data for running threads
            for j, diff in enumerate(diffs):
                data += diff
                print("\b\b\b\b% 3d%%" % ((i + j + 1) / len(partitions) * 100), end="")
                sys.stdout.flush()

            i += n

        print("")

    print("\n Finishing peeling. Writing data back to disk...", end=" ")
    sys.stdout.flush()
    peeled = tbl.getcol(args.datacolumn)
    peeled[:, :, 0] = data[:, :, 0]
    peeled[:, :, 3] = data[:, :, 1]
    tbl.putcol('CORRECTED_DATA', peeled)
    ms.close()
    with open('peeled.reg', 'w') as f:
        to_ds9_regions(f, partitions)
    print("Done")


def peel_star(args):
    return peel(*args)


def peel(uvw, times, original, ant1, ant2, partition, antennas, freqs, obspos, ra0, dec0, args, passno):
    # TODO: thread xx and yy streams
    data = original.copy()

    # Add the current model of the source back into the data for subsequent passes
    if passno > 0:
        for source in partition:
            data += source.visibility(uvw, freqs, ra0, dec0)

    # Phase rotate visibilities onto source to peel
    # TODO: rotate onto centroid of partition
    _ra0, _dec0 = partition[0].ra, partition[0].dec
    dm = measures()
    phasecentre = dm.direction(
        'j2000',
        quantity(_ra0, 'rad'),
        quantity(_dec0, 'rad'),
    )

    uvw_rotated, rotated = phase_rotate(
        uvw,
        times,
        data,
        ant1,
        ant2,
        obspos,
        phasecentre,
        antennas,
        speed_of_light / freqs,
    )

    # Filter out baselines beneath minuv length
    uvw_length = (uvw_rotated.T[0]**2 + uvw_rotated.T[1]**2) > args.minuv
    uvw_rotated = uvw_rotated[uvw_length, :]
    rotated = rotated[uvw_length, :]

    # Average bands into chunks of width
    # On first pass, average entire band
    if passno == 0:
        width = len(freqs)
    else:
        width = args.width

    chans = len(freqs)
    averaged = np.zeros((rotated.shape[0], chans // width, 2), dtype=np.complex)
    for j in range(chans // width):
        with warnings.catch_warnings():
            # Ignore "Mean of empty slice" warnings (due to NaN values)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            np.nanmean(rotated[:, j:j+width, :], 1, out=averaged[:, j, :])

    widefreqs = np.array([
        np.mean(freqs[i:i+width]) for i in range(0, chans, width)
    ])

    # Fit model to data
    # Fit to xx (0) and yy (3)
    if passno == 0:
        x0 = [p for source in partition for p in source.minimal_params]
        ret = least_squares(
            minimal_residual,
            x0,
            #method='lm', # Doesn't work in multithreaded environment
            args=(
                uvw_rotated,
                widefreqs,
                partition,
                averaged,
                _ra0,
                _dec0
            )
        )
        if not ret.success:
            print("Failed to converge")

        # Save final values, accounting for potentially
        # different number of params per source
        i = 0
        for source in partition:
            print(np.exp(source.XX2), np.exp(source.YY2))
            n = len(source.minimal_params)
            source.minimal_params = ret.x[i:i+n]
            i += n

        x0 = [p for source in partition for p in source.reduced_params]
        ret = least_squares(
            reduced_residual,
            x0,
            #method='lm', # Doesn't work in multithreaded environment
            args=(
                uvw_rotated,
                widefreqs,
                partition,
                averaged,
                _ra0,
                _dec0
            )
        )
        if not ret.success:
            print("Failed to converge")

        # Save final values, accounting for potentially
        # different number of params per source
        i = 0
        for source in partition:
            print(np.exp(source.XX2), np.exp(source.YY2), source.major, source.minor)
            n = len(source.reduced_params)
            source.reduced_params = ret.x[i:i+n]
            i += n
    else:
        x0 = [p for source in partition for p in source.params]

        ret = least_squares(
            residual,
            x0,
            #method='lm', # Doesn't work in multithreaded environment
            args=(
                uvw_rotated,
                widefreqs,
                partition,
                averaged,
                _ra0,
                _dec0
            ),
        )

        # Save final values, accounting for potentially
        # different number of params per source
        i = 0
        for source in partition:
            n = len(source.params)
            source.params = ret.x[i:i+n]
            i += n

    # Subtract updated source model from data
    for source in partition:
        data -= source.visibility(uvw, freqs, ra0, dec0)

    return np.subtract(data, original, out=data)


def phase_rotate(uvw, times, data, ant1, ant2, obspos, phasecentre, antennas, lambdas):
    data = data.copy()
    dm = measures()
    dm.do_frame(obspos)
    dm.do_frame(phasecentre)

    # Recalculate uvw for new phase position
    new_uvw = np.zeros_like(uvw)

    # Process visibilities by time so that we calculate antenna baselines
    # just once
    for time in set(times):
        epoch = dm.epoch('UTC', quantity(time, 's'))
        dm.do_frame(epoch)
        baselines = dm.as_baseline(antennas)
        antenna_uvw = np.reshape(dm.to_uvw(baselines)['xyz'].get_value(), (-1, 3))

        # Select only those rows for the current time
        # and update uvw values
        idx = times == time
        new_uvw[idx] = antenna_uvw[ant1[idx]] - antenna_uvw[ant2[idx]]

    # Calculate phase offset
    woffset = -2j * np.pi * (new_uvw.T[2] - uvw.T[2])
    data *= np.exp(woffset[:, np.newaxis] / lambdas)[:, :, np.newaxis]

    return new_uvw, data


def minimal_residual(x0, uvw, freqs, sources, V, ra0, dec0):
    model = np.zeros_like(V)

    i = 0
    for source in sources:
        n = len(source.minimal_params)
        source.minimal_params = x0[i:i+n]
        i += n

        model += source.visibility(uvw, freqs, ra0, dec0)

    residuals = abs((V - model).flatten())
    return residuals[np.isfinite(residuals)]


def reduced_residual(x0, uvw, freqs, sources, V, ra0, dec0):
    model = np.zeros_like(V)

    i = 0
    for source in sources:
        n = len(source.reduced_params)
        source.reduced_params = x0[i:i+n]
        i += n

        model += source.visibility(uvw, freqs, ra0, dec0)

    residuals = abs((V - model).flatten())
    return residuals[np.isfinite(residuals)]


def residual(x0, uvw, freqs, sources, V, ra0, dec0):
    model = np.zeros_like(V)

    i = 0
    for source in sources:
        n = len(source.params)
        source.params = x0[i:i+n]
        i += n

        model += source.visibility(uvw, freqs, ra0, dec0)

    residuals = abs((V - model).flatten())
    return residuals[np.isfinite(residuals)]


class Point(object):
    def __init__(self, XX0, XX1, XX2, YY0, YY1, YY2, ra, dec):
        self.XX0 = XX0
        self.XX1 = XX1
        self.XX2 = XX2
        self.YY0 = YY0
        self.YY1 = YY1
        self.YY2 = YY2
        self.ra = ra
        self.dec = dec

        self.converged = True
        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None

    @property
    def reduced_params(self):
        return np.array([
            self.XX2,
            self.YY2,
            self.ra,
            self.dec,
        ])

    @reduced_params.setter
    def reduced_params(self, p):
        self.XX2 = p[0]
        self.YY2 = p[1]
        self.ra = p[2]
        self.dec = p[3]

    @property
    def params(self):
        return np.array([
            self.XX0,
            self.XX1,
            self.XX2,
            self.YY0,
            self.YY1,
            self.YY2,
            # self.ra,
            # self.dec,
        ])

    @params.setter
    def params(self, p):
        self.XX0 = p[0]
        self.XX1 = p[1]
        self.XX2 = p[2]
        self.YY0 = p[3]
        self.YY1 = p[4]
        self.YY2 = p[5]
        # self.ra = p[6]
        # self.dec = p[7]

    def get_lm(self, ra0, dec0):
        return radec_to_lm(self.ra, self.dec, ra0, dec0)

    def set_lm(self, l, m, ra0, dec0):
        self.ra, self.dec = lm_to_radec(l, m, ra0, dec0)

    def flush(self):
        del self.uvw_lambda
        del self.arr1
        del self.arr2
        del self.arr3
        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None

    def visibility(self, uvw, freqs, ra0, dec0):
        lambdas = speed_of_light / freqs
        uvw_lambda = uvw.T[:, :, np.newaxis] / lambdas  # [uvw, rows, chans]

        l, m = self.get_lm(ra0, dec0)

        phase = np.exp(2j * np.pi * (uvw_lambda[0] * l + uvw_lambda[1] * m + uvw_lambda[2] * (np.sqrt(1 - l**2 - m**2) - 1)))  # [rows, chans]
        A_XX = self.fluxXX(freqs)
        A_YY = self.fluxYY(freqs)
        amp = np.array([A_XX, A_YY]).T / np.sqrt(1 - l**2 - m**2)

        return amp * phase[:, :, np.newaxis]

    def fluxXX(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.XX0 * lnfreq**2 + self.XX1 * lnfreq + self.XX2
        )

    def fluxYY(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.YY0 * lnfreq**2 + self.YY1 * lnfreq + self.YY2
        )

    def flux(self, freq):
        return self.fluxXX(freq) + self.fluxYY(freq)


class Gaussian(object):
    def __init__(self, XX0, XX1, XX2, YY0, YY1, YY2, major, minor, pa, ra, dec):
        self.XX0 = XX0
        self.XX1 = XX1
        self.XX2 = XX2
        self.YY0 = YY0
        self.YY1 = YY1
        self.YY2 = YY2
        self.major = major # FWHM, in units of arcminutes
        self.minor = minor # FWHM
        self.pa = pa
        self.ra = ra
        self.dec = dec

        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None

    @property
    def minimal_params(self):
        return np.array([
            self.XX2,
            self.YY2,
            self.ra,
            self.dec,
        ])

    @minimal_params.setter
    def minimal_params(self, p):
        self.XX2 = p[0]
        self.YY2 = p[1]
        self.ra = p[2]
        self.dec = p[3]

    @property
    def reduced_params(self):
        return np.array([
            self.XX2,
            self.YY2,
            self.major,
            self.minor,
            self.pa,
            self.ra,
            self.dec,
        ])

    @reduced_params.setter
    def reduced_params(self, p):
        self.XX2 = p[0]
        self.YY2 = p[1]
        self.major = p[2]
        self.minor = p[3]
        self.pa = p[4]
        self.ra = p[5]
        self.dec = p[6]

    @property
    def params(self):
        return np.array([
            self.XX0,
            self.XX1,
            self.XX2,
            self.YY0,
            self.YY1,
            self.YY2,
            # self.major,
            # self.minor,
            # self.pa,
            # self.ra,
            # self.dec,
        ])

    @params.setter
    def params(self, p):
        self.XX0 = p[0]
        self.XX1 = p[1]
        self.XX2 = p[2]
        self.YY0 = p[3]
        self.YY1 = p[4]
        self.YY2 = p[5]
        # self.major = p[6]
        # self.minor = p[7]
        # self.pa = p[8]
        # self.ra = p[9]
        # self.dec = p[10]

    def get_lm(self, ra0, dec0):
        return radec_to_lm(self.ra, self.dec, ra0, dec0)

    def set_lm(self, l, m, ra0, dec0):
        self.ra, self.dec = lm_to_radec(l, m, ra0, dec0)

    def flush(self):
        del self.uvw_lambda
        del self.arr1
        del self.arr2
        del self.arr3
        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None


    def visibility(self, uvw, freqs, ra0, dec0):
        lambdas = speed_of_light / freqs
        uvw_lambda = uvw.T[:, :, np.newaxis] / lambdas  # [uvw, rows, chans]

        l, m = self.get_lm(ra0, dec0)
        major = np.sin((self.major / 60) * (np.pi / 180))
        minor = np.sin((self.minor / 60) * (np.pi / 180))

        phase = np.exp(2j * np.pi * (uvw_lambda[0] * l + uvw_lambda[1] * m + uvw_lambda[2] * (np.sqrt(1 - l**2 - m**2) - 1)))  # [rows, chans]
        phase *= np.exp(
            -(np.pi**2 / (4 * np.log(2))) * (
                major**2 * (uvw_lambda[0] * np.cos(self.pa + np.pi / 2) - uvw_lambda[1] * np.sin(self.pa + np.pi / 2))**2 +
                minor**2 * (uvw_lambda[0] * np.sin(self.pa + np.pi / 2) + uvw_lambda[1] * np.cos(self.pa + np.pi / 2))**2
            )
        )

        A_XX = self.fluxXX(freqs)
        A_YY = self.fluxYY(freqs)
        amp = np.array([A_XX, A_YY]).T / np.sqrt(1 - l**2 - m**2)

        return amp * phase[:, :, np.newaxis]

    def fluxXX(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.XX0 * lnfreq**2 + self.XX1 * lnfreq + self.XX2
        )

    def fluxYY(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.YY0 * lnfreq**2 + self.YY1 * lnfreq + self.YY2
        )

    def flux(self, freq):
        return self.fluxXX(freq) + self.fluxYY(freq)


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
    return l, m


def lm_to_radec(l, m, ra0, dec0):
    # Copied from mwa-reduce/units/imagecoordinates.h
    # Todo: understand this transformation
    n = np.sqrt(1 - l**2 - m**2)
    cosDec0 = np.cos(dec0)
    sinDec0 = np.sin(dec0)
    deltaAlpha = np.arctan2(l, n * cosDec0 - m * sinDec0)

    ra = deltaAlpha + ra0
    dec = np.arcsin(m * cosDec0 + n * sinDec0)
    return ra, dec


def radec_to_altaz(ra, dec, time, pos):
    coord = SkyCoord(ra, dec, unit=(u.radian, u.radian))
    coord.time = time + pos.lon.hourangle
    coord = coord.transform_to(AltAz(obstime=time, location=pos))
    return coord.alt.rad, coord.az.rad


def to_ds9_regions(f, sources):
    print("global dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1", file=f)
    print("icrs", file=f)
    for i, partition in enumerate(sources):
        for source in partition:
            print("point %.32fd %.32fd # point=circle color=green text={%d}" % (
                source.ra * 180 / np.pi,
                source.dec * 180 / np.pi,
                i,
            ), file=f)


def spatial_partition(sources, threshold):
    # First link sources that directly within threshold
    partitions = []
    for source1 in sources:
        for partition in partitions:
            for source2 in partition:
                if angular_separation(source1.ra, source1.dec, source2.ra, source2.dec) < threshold:
                    break
            else:
                # We didn't match this partition
                continue

            # We did match this partition
            partition.append(source1)
            break
        else:
            # Create a new partition
            partitions.append([source1])

    return partitions


def spatial_offsets(sources, ra0, dec0):
    offsets = []
    for source in sources:
        l0, m0 = radec_to_lm(source._ra, source._dec, ra0, dec0)
        l, m = radec_to_lm(source.ra, source.dec, ra0, dec0)
        offsets.append([l0, r0, l-l0, m-m0])

    return offsets


def spatial_offset_mean(l, m, offsets, sigma):
    Z = 0
    total_l, total_m = 0, 0
    for _l, _m, deltal, deltam in offsets:
        weight = np.exp(-(l - _l)**2 / (2 * sigma**2) + -(m - _m)**2 / (2 * sigma**2))
        Z += weight
        total_l += weight * deltal
        total_m += weight * deltam

    return total_l / Z, total_m / Z


def angular_separation(ra1, dec1, ra2, dec2):
    """
    RA, Dec given in radians
    """
    return np.arccos(
        np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )


def aegean_parser(f):
    tbl = astropy.table.Table.read(f)

    models = []
    for row in tbl:
        # Todo: add test for point source
        position = SkyCoord(row['ra'], row['dec'], unit=(u.degree, u.degree))
        flux = np.array([[row['int_flux'], 0, 0, 0]])
        component = Component(position, flux)
        model = Model(row['uuid'], [component])
        models.append(model)

    return models


def model_parser(f):
    models = []

    next(f)  # Skip version info
    for line in f:
        parts = line.split()
        if parts[0] == 'source':
            models.append(source_parser(f))
        elif parts[0] == '#':
            pass
        else:
            print("Skymodel parsing error: %s" % line)
            exit(1)

    return models


def source_parser(f):
    name, components = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'name':
            name = ' '.join(parts[1:]).strip('"')
        elif parts[0] == 'component':
            components.append(component_parser(f))
        elif parts[0] == '}':
            if name and components:
                return Model(name, components)
            else:
                print("Unexpected }")
                exit(1)
        else:
            print("Skymodel parsing error: %s" % line)
            exit(1)

    print("Unexpected EOF")
    exit(1)


def component_parser(f):
    position, measurements = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'position':
            position = SkyCoord(parts[1], parts[2], unit=(u.hourangle, u.degree))
        elif parts[0] == 'type':
            pass
        elif parts[0] == 'shape':
            pass
        elif parts[0] == 'measurement':
            measurements.append(measurement_parser(f))
        elif parts[0] == '}':
            if position and measurements:
                measurements = np.sort(np.array(measurements), axis=0)
                return Component(position, measurements)
            else:
                print("Unexpected }")
                exit(1)
        else:
            print("Skymodel parsing error: %s" % line)
            exit(1)

    print("Unexpected EOF")
    exit(1)


def measurement_parser(f):
    frequency, fluxdensity = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'frequency':
            # Assume MHz for now
            frequency = float(parts[1]) * 1E6
        elif parts[0] == 'fluxdensity':
            fluxdensity = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
        elif parts[0] == '}':
            if frequency and fluxdensity:
                return [frequency] + fluxdensity
            else:
                print("Unexpected {")
                exit(1)
        else:
            print("SkyModel parsing error: %s" % line)
            exit(1)

    print("Unexpected EOF")
    exit(1)


class Model(object):
    def __init__(self, name, components):
        self.name = name
        self.components = components


class Component(object):
    def __init__(self, position, measurements):
        self.position = position
        self.measurements = measurements

        logfreq = np.log(self.measurements.T[0])
        logflux = np.log(self.measurements.T[1])
        self.coeffs = np.polyfit(logfreq, logflux, 3)

    @property
    def ra(self):
        return self.position.ra.rad

    @property
    def dec(self):
        return self.position.dec.rad

    def flux(self, frequency):
        c = self.coeffs
        logfreq = np.log(frequency)
        return [np.exp(
            c[0] * logfreq**3 +
            c[1] * logfreq**2 +
            c[2] * logfreq +
            c[3]
        ), 0, 0, 0]


if __name__ == '__main__':
    main()
