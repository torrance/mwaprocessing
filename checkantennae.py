#! /usr/bin/env python

from __future__ import print_function

from subprocess import Popen, STDOUT
import os
import os.path

FNULL = open(os.devnull, 'w')

for child in os.listdir('.'):
    if os.path.isdir(child):
        entries = os.listdir(child)
        if (
            'solutions_amp.png' in entries
            and 'solutions_phase.png' in entries
            and 'badantennae' not in entries
        ):
            print(child)
            bad = []

            for plot in ['/solutions_amp.png', '/solutions_phase.png']:
                p = Popen(['eog', child + plot], stdout=FNULL, stderr=STDOUT)
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
            
                p.terminate()

            with open(child + '/badantennae', 'w') as f:
                for x in bad:
                    print(x, file=f)

