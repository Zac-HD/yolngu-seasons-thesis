#!/usr/bin/env python3
"""Data analysis and presentation for my thesis."""

import concurrent
import os
import glob
import shutil

import chart
import weather

if __name__ == '__main__':
    if os.path.isdir('../output') and \
            os.path.getmtime('../output') > \
            max(os.path.getmtime(p) for p in glob.glob('*')):
        print('Analysis outputs are (probably) not stale')
    else:
        print('Running analysis...')
        shutil.rmtree('../output/', ignore_errors=True)
        os.mkdir('../output/')
        with concurrent.futures.ProcessPoolExecutor() as ex:
            ex.map(chart.save_out, weather.stations)
        print('Analysis done!')
