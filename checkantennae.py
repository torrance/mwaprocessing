#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Use imgcat to inspect solution_[amp|phase].png image and flag tiles')
    parser.add_argument('--force', '-f', action='store_true', help='Reinspect folders that already have a badantennae file')
    args = parser.parse_args()

    dirs = find_dirs('.', args.force)
    for i, d in enumerate(dirs):
        print("%s (%d/%d)" % (d, i+1, len(dirs)))
        inspect(d)

def find_dirs(parent, force):
    dirs = []
    children = [child for child in os.listdir(parent) if os.path.isdir(child)]
    for child in children:
        entries = os.listdir(child)
        if (
            'solutions-pre_amp.png' in entries
            and 'solutions-pre_phase.png' in entries
            and ('badantennae' not in entries or force)
        ):
            dirs.append(child)
    return dirs

def inspect(d):
    bad = []
    try:
        with open(d + '/badantennae') as f:
            for line in f:
                bad.append(int(line.strip()))
    except (IOError, OSError) as e:
        pass

    for plot in ['/solutions-pre_amp.png', '/solutions-pre_phase.png']:
        subprocess.call(['imgcat', d + plot])
        while True:
            txt = raw_input("Enter bad tiles (" + ' '.join([str(x) for x in bad]) + "):")
            if not txt:
                break

            try:
                tile = int(txt)
            except ValueError as e:
                print("That was not an integer")
                continue

            if tile < 0:
                bad.remove(-tile)
            else:
                bad.append(tile)

    with open(d + '/badantennae', 'w') as f:
        for x in bad:
            print(x, file=f)


if __name__ == '__main__':
    main()
