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
    obs = table(args.ms + '/OBSERVATION')
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

    ms = table(args.ms, readonly=False)
    freqs = table(args.ms + '/SPECTRAL_WINDOW').getcell('CHAN_FREQ', 0)
    lambdas = speed_of_light / freqs
    ra0, dec0 = table(args.ms + '/FIELD').getcell('PHASE_DIR', 0)[0]  # Phase centre in radians
    chans, pols = ms.getcell(args.datacolumn, 0).shape
    print("There are %d channels" % chans)

    # Check whether width evenly divides the total channels
    if chans % args.width != 0:
        print("Width (%d) does not evenly divide channel number (%d)" % (args.width, chans))

    widefreqs = np.array([
        np.mean(freqs[i:i+args.width]) for i in range(0, len(freqs), args.width)
    ])
    print(widefreqs)
    widelambdas = speed_of_light / widefreqs


    antennas = table(args.ms + '/ANTENNA').getcol('POSITION')
    antennas = dm.position(
        'itrf',
        quantity(antennas.T[0], 'm'),
        quantity(antennas.T[1], 'm'),
        quantity(antennas.T[2], 'm'),
    )


    # Create PEELED_DATA column if it does not exist
    # if ('PEELED_DATA' not in ms.colnames()):
    #     coldesc = ms.getcoldesc('DATA')
    #     coldesc['name'] = 'PEELED_DATA'
    #     ms.addcols(coldesc)

    #     peeled = ms.col('PEELED_DATA')
    #     peeled[:] = np.zeros((chans, pols), dtype=np.complex)

    # Load models
    with open(args.model) as f:
        models = model_parser(f)
    models = models[0:6]
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

    params = []
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

        params.append([
            [xx1, xx2, xx3, comp.ra, comp.dec],
            [yy1, yy2, yy3, comp.ra, comp.dec],
        ])
    params = np.array(params)

    # Subtract all sources from visibilities

    for i, (xx_params, yy_params) in enumerate(params):
        print("Peeling source %d / %d" % (i+1, len(comps)))

        # Phase rotate visibilities onto source to peel
        print("  > Phase rotating data onto source...", end=" ")
        sys.stdout.flush()
        ra0, dec0 = xx_params[3], xx_params[4]
        dm = measures()
        phasecentre = dm.direction(
            'j2000',
            quantity(ra0, 'rad'),
            quantity(dec0, 'rad'),
        )

        uvw, data = phase_rotate(ms, obspos, phasecentre, antennas, lambdas)
        print("Done")

        # Handle flags
        # Remove rows that have been flagged, and set entries that
        # are flagged as NaN
        flags = ms.getcol('FLAG')
        data[flags] = np.nan
        flagged_rows = ms.getcol('FLAG_ROW')
        uvw = uvw[~flagged_rows]
        data = data[~flagged_rows]

        # Subtract current model from visibilities
        print("  > Subtracting current best model from visibilities...", end=" ")
        sys.stdout.flush()

        uvw_lambda = uvw[:, :, np.newaxis] / lambdas
        uvw_lambda = np.transpose(uvw_lambda, (1, 0, 2))

        # Preallocate arrays to be used in subtraction calculations
        arr1 = np.zeros_like(data[:, :, 0])
        arr2 = np.zeros_like(data[:, :, 0])
        arr3 = np.zeros_like(data[:, :, 0])

        for j, (_xx_params, _yy_params) in enumerate(params):
            if i != j:
                A0, A1, A2, ra, dec = _xx_params
                l, m = radec_to_lm(ra, dec, ra0, dec0)
                data[:, :, 0] -= point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3)
                A0, A1, A2, ra, dec = _yy_params
                l, m = radec_to_lm(ra, dec, ra0, dec0)
                data[:, :, 3] -= point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3)

            print(".", end="")
            sys.stdout.flush()
        print("Done")

        # Average bands into chunks of width
        print("  > Averaging in frequncy space in %d width chunks..." % args.width, end=" ")
        sys.stdout.flush()

        averaged = np.zeros((data.shape[0], chans // args.width, pols), dtype=np.complex)
        for j in range(chans // args.width):
            np.nanmean(data[:, j:j+args.width, :], 1, out=averaged[:, j, :])

        print("Done")

        # Fit model to data
        print("  > Fitting model to data...", end=" ")

        # Preallocate working arrays
        arr1 = np.zeros_like(averaged[:, :, 0])
        arr2 = np.zeros_like(averaged[:, :, 0])
        arr3 = np.zeros_like(averaged[:, :, 0])

        uvw_lambda = uvw[:, :, np.newaxis] / widelambdas
        uvw_lambda = np.transpose(uvw_lambda, (1, 0, 2))

        # Fit to xx
        A0, A1, A2, ra, dec = xx_params
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        ret = least_squares(
            residual,
            [A0, A1, A2, l, m],
            method='lm',
            args=(
                uvw_lambda,
                widefreqs,
                averaged[:, :, 0],
                arr1,
                arr2,
                arr3,
            )
        )
        print(np.exp(
            ret.x[0] * np.log(180E6)**2 + ret.x[1] * np.log(180E6) + ret.x[2]
        ))
        A0, A1, A2, l, m = ret.x
        ra, dec = lm_to_radec(l, m, ra0, dec0)
        params[i][0][:] = [A0, A1, A2, ra, dec]


        # Fit to yy
        A0, A1, A2, ra, dec = yy_params
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        ret = least_squares(
            residual,
            [A0, A1, A2, l, m],
            method='lm',
            args=(
                uvw_lambda,
                widefreqs,
                averaged[:, :, 3],
                arr1,
                arr2,
                arr3,
            )
        )
        print(np.exp(
            ret.x[0] * np.log(180E6)**2 + ret.x[1] * np.log(180E6) + ret.x[2]
        ))
        A0, A1, A2, l, m = ret.x
        ra, dec = lm_to_radec(l, m, ra0, dec0)
        params[i][1][:] = [A0, A1, A2, ra, dec]

        print("Done")

    print("Constructing final peeled source...")
    ra0, dec0 = table(args.ms + '/FIELD').getcell('PHASE_DIR', 0)[0]  # Phase centre in radians
    uvw = ms.getcol('UVW')
    data = ms.getcol(args.datacolumn)
    uvw_lambda = uvw[:, :, np.newaxis] / lambdas
    uvw_lambda = np.transpose(uvw_lambda, (1, 0, 2))
    # Preallocate arrays to be used in subtraction calculations
    arr1 = np.zeros_like(data[:, :, 0])
    arr2 = np.zeros_like(data[:, :, 0])
    arr3 = np.zeros_like(data[:, :, 0])

    for xx_params, yy_params in params:
        A0, A1, A2, ra, dec = xx_params
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        data[:, :, 0] -= point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3)
        A0, A1, A2, ra, dec = yy_params
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        data[:, :, 3] -= point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3)

    ms.putcol('PEELED_DATA', data)
    ms.close()


def phase_rotate(ms, obspos, phasecentre, antennas, lambdas):
    dm = measures()
    dm.do_frame(obspos)
    dm.do_frame(phasecentre)

    times = ms.getcol('TIME_CENTROID')
    uvw = ms.getcol('UVW')
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    data = ms.getcol('DATA')
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
    woffset = -2j * np.pi * (new_uvw.T[2] - uvw.T[2])
    for chan in range(chans):
        data[:, chan, 0] = data[:, chan, 0] * np.exp(woffset / lambdas[chan])
        data[:, chan, 1] = data[:, chan, 1] * np.exp(woffset / lambdas[chan])
        data[:, chan, 2] = data[:, chan, 2] * np.exp(woffset / lambdas[chan])
        data[:, chan, 3] = data[:, chan, 3] * np.exp(woffset / lambdas[chan])

    return new_uvw, data


def residual(params, uvw_lambda, freqs, V, arr1, arr2, arr3):
    A0, A1, A2, l, m = params
    model = point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3)
    residuals = abs((V - model).flatten())
    return residuals[np.isfinite(residuals)]


def point(A0, A1, A2, l, m, uvw_lambda, freqs, arr1, arr2, arr3):
    logfreqs = np.log(freqs)
    A = np.exp(
        A0 * logfreqs**2 + A1 * logfreqs + A2
    )

    np.multiply(uvw_lambda[0], l, out=arr1)
    np.multiply(uvw_lambda[1], m, out=arr2)
    np.multiply(uvw_lambda[2], np.sqrt(1 - l**2 - m**2) - 1, out=arr3)
    np.add(arr1, arr2, out=arr1)
    np.add(arr1, arr3, out=arr1)
    np.multiply(arr1, 2j * np.pi, out=arr1)
    np.exp(arr1, out=arr1)
    np.multiply(arr1, A / np.sqrt(1 - l**2 - m**2), out=arr1)
    return arr1

    return np.exp(2j * np.pi *
        (uvw_lambda[0] * l + uvw_lambda[1] * m + uvw_lambda[2] * (np.sqrt(1 - l**2 - m**2) - 1))
        ) * (A.T / np.sqrt(1 - l**2 - m**2))


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


class Model:
    def __init__(self, name, components):
        self.name = name
        self.components = components


class Component:
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
