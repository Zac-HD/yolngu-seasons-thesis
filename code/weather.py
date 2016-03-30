#!/usr/bin/env python3
"""Load up some BOM weather data."""

import pandas as pd

from utils import META, wind_dirs  # pylint:disable=no-name-in-module


stations = (
    #("014401", 'Warruwi'),
    ("014404", 'Milingimbi'),
    #("014405", 'Maningrida'),
    #("014508", 'Nhulunbuy'),
    ("014517", 'Galiwinku'),
    )


def raw_station_dataframe(station_number):
    """Return a formatted but unprocessed dataframe for the given station."""
    df = pd.read_csv(
        '../data/DC02D_Data_{}_999999999112425.txt'.format(station_number),
        skipinitialspace=True,
        # Use date-based index
        parse_dates={'date': ['Month Day Year in YYYY-MM-DD format']},
        index_col='date',
        )
    # Delete spurious columns
    for c in META['discard_cols']:
        del df[c]
    # Discard data from multiple days of accumulation
    for d, acc in META['accumulation_pairs'].items():
        df[d].where(df[acc].fillna(1) == 1, inplace=True)
        del df[acc]
    # Quality fields and wind data are categorical columns
    for c in df.columns:  # pylint:disable=no-member
        if 'quality of ' in c.lower():
            df[c] = df[c].astype("category", categories=META['quality_flags'])
        elif '16 compass point text' in c.lower():
            df[c] = df[c].str.strip().astype(
                'category', categories=wind_dirs, ordered=True)
    return df


def data(station_number, *, discard_metadata=True, discard_unchecked=False):
    """Return a dataframe with clean data only for the given station."""
    df = raw_station_dataframe(station_number)
    # Discard low-quality data, then delete quality columns
    low_qual = 'NWSI' if discard_unchecked else 'WSI'
    for d, q in zip(META['data_cols'], META['quality_cols']):
        for symbol in low_qual:
            df[d].where(df[q] != symbol, inplace=True)
        del df[q]
    # Discard metadata columns
    if discard_metadata:
        for c in META['metadata_cols']:
            del df[c]
    return df
