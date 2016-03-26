#!/usr/bin/env python3
"""Assorted utility functions and metadata."""

import collections

import yaml
import pandas as pd


with open('BOM_format.yml') as f:
    META = yaml.safe_load(f)

wind_dirs = (
    'N', 'NNE', 'NE', 'ENE',
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW',
    'W', 'WNW', 'NW', 'NNW',
    'CALM')

nameof = collections.namedtuple(
    'ColumnNames',
    ['rain', 'maxtemp', 'mintemp', 'dewpoint', 'humid09', 'humid15',
     'windspd09', 'windspd15', 'winddir09', 'winddir15']
    )(*META['data_cols'])

seasons = collections.namedtuple(
    'Seasons',
    ['du', 'ba', 'ma', 'mi', 'da', 'rr']
    )('Dhuludur', 'Barramirri', 'Mayaltha',
      'Midawarr', 'Dharrathamirri', 'Rarrandharr')

_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
           'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
dayofyear_month_labels = [
    '' if (i-14) % 30 else _months.pop(0) for i in range(1, 367)]

chart_panels = ('rain', 'maxtemp', 'mintemp', 'dewpoint',
                'windspd09', 'winddir09', 'windspd15', 'winddir15')


def pivot(df, colname):
    """Return a 2D (year/dayofyear) table for the given column."""
    if hasattr(df, 'columns') and colname not in df.columns:
        colname = nameof._asdict()[colname]
    df['year'] = pd.DatetimeIndex(df.index).year
    df['dayofyear'] = pd.DatetimeIndex(df.index).dayofyear
    ret = df.pivot(index='year', columns='dayofyear', values=colname)
    del df['year']
    del df['dayofyear']
    return ret


def categorical_to_numeric_wind(df, colname):
    """Creates a numerical pivot table of wind for plotting from the df."""
    varname = nameof._asdict()[colname]
    num_wind_dirs = {w: i for i, w in enumerate(wind_dirs)}
    new = pivot(df[varname].dropna().map(num_wind_dirs).to_frame(), varname)
    return new.fillna(17).astype('uint8')


def categorical_to_numeric_season(df, colname):
    """Creates a numerical pivot table of wind for plotting from the df."""
    num_seasons = {s: i for i, s in enumerate(seasons)}
    return pivot(df[colname].dropna().map(num_seasons).to_frame(), colname)
