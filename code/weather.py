#!/usr/bin/env python3
"""Load up some BOM weather data."""

import pandas as pd

import season
import utils


stations = (
    #("014401", 'warruwi'),
    ("014404", 'milingimbi'),
    ("014405", 'maningrida'),
    ("014508", 'nhulunbuy'),
    ("014517", 'galiwinku'),
    )

_BOM_quality_flags = (
    'Y',  # quality controlled and acceptable
    'N',  # not quality controlled
    'W',  # quality controlled and considered wrong
    'S',  # quality controlled and considered suspect
    'I',  # quality controlled and inconsistent with other known information
    'X')  # no quality information available

_data_cols = ['Month Day Year in YYYY-MM-DD format',
                 *utils.nameof,
                 *utils._quality_cols,
                 *utils._accumulation_pairs]


def raw_station_dataframe(station_number):
    """Return a formatted but unprocessed dataframe for the given station."""
    df = pd.read_csv(
        '../data/DC02D_Data_{}_999999999182877.txt'.format(station_number),
        skipinitialspace=True,
        index_col='Month Day Year in YYYY-MM-DD format',
        parse_dates=True,
        usecols=_data_cols,
        )
    # Discard data from multiple days of accumulation
    for acc, d in utils._accumulation_pairs.items():
        df[d].where(df[acc].fillna(1) == 1, inplace=True)
        del df[acc]
    # Quality fields and wind data are categorical columns
    for c in df.columns:  # pylint:disable=no-member
        if 'quality of ' in c.lower():
            df[c] = df[c].astype("category", categories=_BOM_quality_flags)
        elif ' as 16 compass point text' in c:
            df[c] = df[c].str.strip().astype(
                'category', categories=utils.wind_dirs, ordered=True)
    return df


def data(station_number, *, discard_unchecked=False):
    """Return a dataframe with clean data only for the given station."""
    df = raw_station_dataframe(station_number)
    # Discard low-quality data, then delete quality columns
    low_qual = 'NWSI' if discard_unchecked else 'WSI'
    for d, q in zip(utils.nameof, utils._quality_cols):
        for symbol in low_qual:
            df[d].where(df[q] != symbol, inplace=True)
        del df[q]
    return season.add_seasons(df)
