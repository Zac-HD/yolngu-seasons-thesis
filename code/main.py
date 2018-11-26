#!/usr/bin/env python3
"""Data analysis and presentation for my thesis."""

from concurrent.futures import ProcessPoolExecutor
import glob
import os
import shutil
import sys

import chart
import weather

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if (
        len(sys.argv) == 1
        and os.path.isdir("../output")
        and (
            os.path.getmtime("../output")
            > max(os.path.getmtime(p) for p in glob.glob("*"))
        )
    ):
        print("Analysis outputs are (probably) not stale")
    else:
        print("Running analysis...")
        shutil.rmtree("../output/", ignore_errors=True)
        os.mkdir("../output/")
        with ProcessPoolExecutor() as ex:
            ex.map(chart.save_out, weather.stations)
        print("Analysis done!")
