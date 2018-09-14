#! /usr/bin/env python

from __future__ import print_function, division

import argparse
import sys
from multiprocessing import Pool

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
    if ('PEELED_DATA' not in ms.colnames()):
        print("Creating PEELED_DATA column...", end=" ")
        sys.stdout.flush()
        coldesc = ms.getcoldesc('DATA')
        coldesc['name'] = 'PEELED_DATA'
        ms.addcols(coldesc)

        peeled = ms.getcol(args.datacolumn)
        ms.putcol('PEELED_DATA', peeled)
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
        xx1, xx2, xx3 = np.polyfit(log_widefreqs, log_xx_fluxes, 2)
        yy1, yy2, yy3 = np.polyfit(log_widefreqs, log_yy_fluxes, 2)

        sources.append([
            Point(xx1, xx2, xx3, comp.ra, comp.dec),
            Point(yy1, yy2, yy3, comp.ra, comp.dec),
            #Gaussian(xx1, xx2, xx3, 0, 0, 0, comp.ra, comp.dec),
            #Gaussian(yy1, yy2, yy3, 0, 0, 0, comp.ra, comp.dec),
        ])
    sources = np.array(sources)

    # Group sources by a spatial threshold
    threshold = (3 / 60) * np.pi / 180
    partitions = spatial_partition(sources, threshold)

    # Order sources by apparent flux at midfreq
    partitions = np.array(
        sorted(partitions, reverse=True, key=lambda xs: sum([x[0].flux(midfreq) + x[1].flux(midfreq) for x in xs]))
    )
    partitions = partitions[0:5]

    # Read data minus flagged rows
    tbl = taql("select * from $ms where not FLAG_ROW")
    uvw = tbl.getcol('UVW')
    data = tbl.getcol(args.datacolumn)

    # Handle flags by setting entries that are flagged as NaN
    flags = tbl.getcol('FLAG')
    data[flags] = np.nan

    # Subtract all sources from visibilities
    print("Subtracting current best source models from visibilities...   0%", end="")
    sys.stdout.flush()

    # for i, (source_xx, source_yy) in enumerate(sources):
    #     data[:, :, 0] -= source_xx.visibility(uvw, freqs, ra0, dec0)
    #     source_xx.flush()
    #     data[:, :, 3] -= source_yy.visibility(uvw, freqs, ra0, dec0)
    #     source_yy.flush()

    #     print("\b\b\b\b% 3d%%" % (i / len(sources) * 100), end="")
    #     sys.stdout.flush()

    tbl.putcol('PEELED_DATA', data)
    del uvw
    del data

    print("\b\b\b\bDone")

    for passno in range(args.passes):
        for i, partition in enumerate(partitions):
            print("Peeling partition %d / %d (sources %d; pass %d/%d)" % (i+1, len(partitions), len(partition), passno+1, args.passes))

            # Add source back into data
            if passno > 0:
                print("  > Adding subtracted sources back into data...", end=" ")
                sys.stdout.flush()
                uvw = tbl.getcol('UVW')
                data = tbl.getcol('PEELED_DATA')
                for source_xx, source_yy in partition:
                    data[:, :, 0] += source_xx.visibility(uvw, freqs, ra0, dec0)
                    data[:, :, 3] += source_yy.visibility(uvw, freqs, ra0, dec0)
                tbl.putcol('PEELED_DATA', data)
                del uvw
                del data
                print("Done")

            # Phase rotate visibilities onto source to peel
            print("  > Phase rotating data onto source...", end=" ")
            sys.stdout.flush()
            # TODO: rotate onto centroid of partition
            _ra0, _dec0 = partition[0][0].ra, partition[0][0].dec
            dm = measures()
            phasecentre = dm.direction(
                'j2000',
                quantity(_ra0, 'rad'),
                quantity(_dec0, 'rad'),
            )

            uvw, rotated = phase_rotate(tbl, obspos, phasecentre, antennas, lambdas)
            print("Done")

            # Filter out baselines beneath minuv length
            uvw_length = (uvw.T[0]**2 + uvw.T[1]**2) > args.minuv
            uvw = uvw[uvw_length, :]
            rotated = rotated[uvw_length, :]

            # Average bands into chunks of width
            print("  > Averaging in frequncy space in chunks of %d channels..." % (chans // args.width), end=" ")
            sys.stdout.flush()

            averaged = np.zeros((rotated.shape[0], chans // args.width, pols), dtype=np.complex)
            for j in range(chans // args.width):
                np.nanmean(rotated[:, j:j+args.width, :], 1, out=averaged[:, j, :])

            print("Done")

            # Fit model to data
            print("  > Fitting model to data...", end=" ")
            sys.stdout.flush()

            # Fit to xx (0) and yy (3)
            for k, pol in enumerate([0, 3]):
                x0 = [p for source in partition for p in source[k].params]
                ret = least_squares(
                    residual,
                    x0,
                    method='lm',
                    args=(
                        uvw,
                        widefreqs,
                        [source[k] for source in partition],
                        averaged[:, :, pol],
                        _ra0,
                        _dec0
                    )
                )
                print(np.exp(
                    ret.x[0] * np.log(180E6)**2 + ret.x[1] * np.log(180E6) + ret.x[2]
                ))

                # Save final values, accounting for potentially
                # different number of params per source
                i = 0
                for source in partition:
                    n = len(source[k].params)
                    source[k].params = ret.x[i:i+n]
                    i += n

            print("Done")

            # Subtract updated source model from data
            print("  > Subtracting updated source model from data...", end=" ")
            sys.stdout.flush()
            uvw = tbl.getcol('UVW')
            data = tbl.getcol('PEELED_DATA')
            for source_xx, source_yy in partition:
                data[:, :, 0] -= source_xx.visibility(uvw, freqs, ra0, dec0)
                source_xx.flush()
                data[:, :, 3] -= source_yy.visibility(uvw, freqs, ra0, dec0)
                source_yy.flush()
            tbl.putcol('PEELED_DATA', data)
            del data
            print("Done")

    print("\n Finishing peeling. Writing data back to disk...", end=" ")
    ms.close()
    with open('peeled.reg', 'w') as f:
        to_ds9_regions(f, partitions)
    print("Done")


def phase_rotate(ms, obspos, phasecentre, antennas, lambdas):
    dm = measures()
    dm.do_frame(obspos)
    dm.do_frame(phasecentre)

    times = ms.getcol('TIME_CENTROID')
    uvw = ms.getcol('UVW')
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    data = ms.getcol('PEELED_DATA')
    chans, _ = data[0].shape

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
    # TODO: with clever broadcasting, can do this in one operation
    woffset = -2j * np.pi * (new_uvw.T[2] - uvw.T[2])
    for chan in range(chans):
        data[:, chan, 0] *= np.exp(woffset / lambdas[chan])
        data[:, chan, 1] *= np.exp(woffset / lambdas[chan])
        data[:, chan, 2] *= np.exp(woffset / lambdas[chan])
        data[:, chan, 3] *= np.exp(woffset / lambdas[chan])

    return new_uvw, data


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
    def __init__(self, A0, A1, A2, ra, dec):
        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.ra = ra
        self.dec = dec

        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None

    @property
    def params(self):
        return np.array([
            self.A0,
            self.A1,
            self.A2,
            self.ra,
            self.dec
        ])

    @params.setter
    def params(self, p):
        self.A0 = p[0]
        self.A1 = p[1]
        self.A2 = p[2]
        self.ra = p[3]
        self.dec = p[4]

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
        # uvw_lambdas has dimensions [ms row, freq, uvw]
        lambdas = speed_of_light / freqs
        self.uvw_lambda = uvw[:, :, np.newaxis] / lambdas
        self.uvw_lambda = np.transpose(self.uvw_lambda, (1, 0, 2))

        # Preallocate (or reuse) working arrays
        if self.arr1 is None or self.arr1.shape != (len(uvw), len(freqs)):
            self.arr1 = np.empty((len(uvw), len(freqs)), np.complex)
            self.arr2 = np.empty((len(uvw), len(freqs)), np.complex)
            self.arr3 = np.empty((len(uvw), len(freqs)), np.complex)

        A = self.flux(freqs)

        l, m = self.get_lm(ra0, dec0)

        np.multiply(self.uvw_lambda[0], l, out=self.arr1)
        np.multiply(self.uvw_lambda[1], m, out=self.arr2)
        np.multiply(self.uvw_lambda[2], np.sqrt(1 - l**2 - m**2) - 1, out=self.arr3)
        np.add(self.arr1, self.arr2, out=self.arr1)
        np.add(self.arr1, self.arr3, out=self.arr1)
        np.multiply(self.arr1, 2j * np.pi, out=self.arr1)
        np.exp(self.arr1, out=self.arr1)
        np.multiply(self.arr1, A / np.sqrt(1 - l**2 - m**2), out=self.arr1)
        return self.arr1

    def flux(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.A0 * lnfreq**2 + self.A1 * lnfreq + self.A2
        )


class Gaussian(object):
    def __init__(self, A0, A1, A2, major, minor, pa, ra, dec):
        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.major = major # FWHM, in units of what?
        self.minor = minor # FWHM
        self.pa = pa
        self.ra = ra
        self.dec = dec

        self.uvw_lambda = None
        self.arr1 = None
        self.arr2 = None
        self.arr3 = None

    @property
    def params(self):
        return np.array([
            self.A0,
            self.A1,
            self.A2,
            self.major,
            self.minor,
            self.pa,
            self.ra,
            self.dec
        ])

    @params.setter
    def params(self, p):
        self.A0 = p[0]
        self.A1 = p[1]
        self.A2 = p[2]
        self.major = p[3]
        self.minor = p[4]
        self.pa = p[5]
        self.ra = p[6]
        self.dec = p[7]

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
        # uvw_lambdas has dimensions [ms row, freq, uvw]
        lambdas = speed_of_light / freqs
        self.uvw_lambda = uvw[:, :, np.newaxis] / lambdas
        self.uvw_lambda = np.transpose(self.uvw_lambda, (1, 0, 2))

        # Preallocate (or reuse) working arrays
        if self.arr1 is None or self.arr1.shape != (len(uvw), len(freqs)):
            self.arr1 = np.empty((len(uvw), len(freqs)), np.complex)
            self.arr2 = np.empty((len(uvw), len(freqs)), np.complex)
            self.arr3 = np.empty((len(uvw), len(freqs)), np.complex)

        A = self.flux(freq)

        l, m = self.get_lm(ra0, dec0)

        np.multiply(self.uvw_lambda[0], l, out=self.arr1)
        np.multiply(self.uvw_lambda[1], m, out=self.arr2)
        np.multiply(self.uvw_lambda[2], np.sqrt(1 - l**2 - m**2) - 1, out=self.arr3)
        np.add(self.arr1, self.arr2, out=self.arr1)
        np.add(self.arr1, self.arr3, out=self.arr1)
        np.multiply(self.arr1, 2j * np.pi, out=self.arr1)
        np.exp(self.arr1, out=self.arr1)
        np.multiply(self.arr1, A / np.sqrt(1 - l**2 - m**2), out=self.arr1)

        self.arr1 *= np.exp(
            -np.pi**2 / (4 * np.log(2)) * (
                self.major**2 * (self.uvw_lambda[0] * np.cos(self.pa + np.pi / 2) - self.uvw_lambda[1] * np.sin(self.pa + np.pi / 2))**2 +
                self.minor**2 * (self.uvw_lambda[0] * np.sin(self.pa + np.pi / 2) + self.uvw_lambda[1] * np.cos(self.pa + np.pi / 2))**2
            )
        )

        return self.arr1

    def flux(self, freq):
        lnfreq = np.log(freq)
        return np.exp(
            self.A0 * lnfreq**2 + self.A1 * lnfreq + self.A2
        )


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
        for source_xx, source_yy in partition:
            print("point %.32fd %.32fd # point=circle color=green, text={%d}" % (source_xx.ra * 180 / np.pi, source_xx.dec * 180 / np.pi, i), file=f)
            print("point %.32fd %.32fd # point=circle color=red" % (source_yy.ra * 180 / np.pi, source_yy.dec * 180 / np.pi), file=f)


def spatial_partition(sources, threshold):
    # First link sources that directly within threshold
    partitions = []
    for source in sources:
        source_xx1, _ = source

        for partition in partitions:
            for source_xx2, _ in partition:
                if angular_separation(source_xx1.ra, source_xx1.dec, source_xx2.ra, source_xx2.dec) < threshold:
                    break
            else:
                # We didn't match this partition
                continue

            # We did match this partition
            partition.append(source)
            break
        else:
            # Create a new partition
            partitions.append([source])

    return partitions


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
