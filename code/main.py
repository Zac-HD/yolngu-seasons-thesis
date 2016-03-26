#!/usr/bin/env python3
"""Data analysis and presentation for my thesis."""

import os
import glob
import shutil


mtime = os.path.getmtime
os.chdir(os.path.dirname(__file__))

if os.path.isdir('../output') and \
        mtime('../output') > max(mtime(p) for p in glob.glob('*')):
    print('Analysis outputs are (probably) not stale')
else:
    print('Running analysis...')
    shutil.rmtree('../output/', ignore_errors=True)
    os.mkdir('../output/')

    # pylint:disable=import-error,wrong-import-position
    import chart
    import weather
    import season

    for station_id, name in weather.stations:
        os.mkdir('../output/' + name.lower())
        data = season.add_seasons(weather.data(station_id))
        chart.save_out(data, name.lower())
