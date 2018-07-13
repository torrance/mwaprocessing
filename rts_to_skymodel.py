#! /usr/bin/env python

from __future__ import print_function
import argparse
import sys

from astropy.coordinates import SkyCoord
import astropy.units as u


def main():
    parser = argparse.ArgumentParser(
        description="Convert RST source lis to SkyModel 1.1 (aocal). Handles RTS point sources and elliptical gaussians. Ignores shapelets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        default="-",
        help='Input RTS source list, or "-" for stdin',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default="-",
        help='Output SkyModel source list, or "-" for stdout',
    )
    args = parser.parse_args()

    sources = source_parser(args.input)
    args.input.close()

    print("skymodel fileformat 1.1", file=args.output)
    for source in sources:
        source_encoder(args.output, source)
    args.output.close()


def source_parser(f):
    sources = []
    source = None
    component = None
    for i, line in enumerate(f):
        line = line.strip()
        if not line or line[0] == "#":
            continue

        parts = line.split()
        if parts[0] == "SOURCE":
            if source:
                print("Parsing failed: expected ENDSOURCE line %d" % i, file=sys.stderr)
                exit(1)
            if len(parts) != 4:
                print(
                    "parsing failed: expected 4 commands line %d" % i, file=sys.stderr
                )
                exit(1)

            source = Source()
            source.name = parts[1]
            component = Component()
            component.coordinate = SkyCoord(
                ra=parts[2], dec=parts[3], unit=(u.hourangle, u.deg)
            )
        elif not source:
            print("Parsing failed: expected SOURCE line %d" % i, file=sys.stderr)
            exit(1)
        elif parts[0] == "COMPONENT":
            if len(parts) != 3:
                print(
                    "Parsing failed: expected 3 commands line %d" % i, file=sys.stderr
                )

            source.components.append(component)
            component = Component()
            component.coordinate = SkyCoord(
                ra=parts[1], dec=parts[2], unit=(u.hourangle, u.deg)
            )
        elif parts[0] == "ENDCOMPONENT":
            pass
        elif parts[0] == "ENDSOURCE":
            if not source.discard:
                source.components.append(component)
                sources.append(source)
            source = None
        elif parts[0] == "GAUSSIAN":
            if len(parts) != 4:
                print(
                    "Parsing failed: expected 4 commands line %d" % i, file.sys.stderr
                )
                exit(1)

            component.type = "gaussian"
            component.pa = float(parts[1])
            # Convert from arcmins to arcsecond, and from sigma to FWHM
            component.major = float(parts[2]) * 60 / 2.6682
            component.minor = float(parts[3]) * 60 / 2.6682
        elif parts[0] == "FREQ":
            if len(parts) != 6:
                print(
                    "Parsing failed: expected 6 commands line %d" % i, file=sys.stderr
                )
                exit(1)

            freq = float(parts[1]) // 1E+6
            component.measurements[freq] = parts[2:]
        else:
            # We don't understand this source
            print(line, file=sys.stderr)
            print(
                "Discarding source %s because we don't understand it" % source.name,
                file=sys.stderr,
            )
            source.discard = True

    return sources


def source_encoder(f, source):
    print("source {", file=f)
    print('  name "%s"' % source.name, file=f)
    for c in source.components:
        print("  component {", file=f)
        print("    type %s" % c.type, file=f)
        print("    position %s" % c.coordinate.to_string("hmsdms"), file=f)
        if c.type == "gaussian":
            print("    shape %f %f %f" % (c.major, c.minor, c.pa), file=f)
        for freq in sorted(c.measurements.iterkeys()):
            m = c.measurements[freq]
            print("    measurement {", file=f)
            print("      frequency %d MHz" % freq, file=f)
            print("      fluxdensity Jy %s %s %s %s" % (m[0], m[1], m[2], m[3]), file=f)
            print("    }", file=f)
        print("  }", file=f)
    print("}", file=f)


class Source:
    def __init__(self):
        self.discard = False
        self.components = []


class Component:
    def __init__(self):
        self.type = "point"
        self.measurements = {}


if __name__ == "__main__":
    main()
