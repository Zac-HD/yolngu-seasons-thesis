#!/usr/bin/env python3

from collections import namedtuple

import yaml

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.rcParams['figure.figsize'] = (10, 6)

with open('BOM_format.yml') as f:
    META = yaml.safe_load(f)

# provide aliases for the very long column names
nameof = namedtuple('ColumnNames', [
    'rain', 'maxtemp', 'mintemp', 'dewpoint', 'humid09', 'humid15',
    'windspd09', 'windspd15', 'winddir09', 'winddir15']
    )(*META['data_cols'])


def raw_station_dataframe(station_number,
                          *, discard_low_qual=False, discard_accum_above=1):
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
        df[d] = df[d].where(df[acc] == 1)
        del df[acc]
    # Quality fields and wind data are categorical columns
    for c in df.columns:
        if 'quality of ' in c.lower():
            df[c] = df[c].astype("category", categories=META['quality_flags'])
        elif '16 compass point text' in c.lower():
            df[c] = df[c].str.strip().astype(
                'category', categories=META['wind_dirs'], ordered=True)
    return df


def station_dataframe(station_number,
                      *, discard_metadata=True, discard_low_qual=False):
    """Return a dataframe with clean data only for the given station."""
    df = raw_station_dataframe(station_number)
    # Discard low-quality data, then delete quality columns
    for d, q in zip(META['data_cols'], META['quality_cols']):
        if discard_low_qual:
            df[d] = df[d].where(df[q] == 'Y')
        del df[q]
    # Discard metadata columns
    if discard_metadata:
        for c in META['metadata_cols']:
            del df[c]
    # Check for empty columns
    for name, series in df.iteritems():
        if series.empty:
            raise RuntimeWarning('Station {} has not data in column {}'.format(
                station_number, name))
    return df


def pivot(df, colname):
    """Return a 2D (year/dayofyear) table for the given column."""
    if colname not in df.columns:
        colname = nameof._asdict()[colname]
    df['year'] = pd.DatetimeIndex(df.index).year
    df['dayofyear'] = pd.DatetimeIndex(df.index).dayofyear
    ret = df.pivot(index='year', columns='dayofyear', values=colname)
    del df['year']
    del df['dayofyear']
    return ret



s = station_dataframe('014517')  # Galiwinku

vars_ = [s.columns[n] for n in (0, 2, 8)]
rain, temp, wind = [pivot(s.dropna(subset=[var]), var) for var in vars_]

wind = s[nameof.winddir09].dropna().map(
    {w: i for i, w in enumerate(META['wind_dirs'])})
wind = pivot(wind.to_frame(), nameof.winddir09).fillna(18).astype(int)


def heatmap(data, kind=None, **kwargs):
    base_kwargs = {
        'xticklabels': 30,
        'yticklabels': 10,
        'robust': True,
        }
    kind_kwargs = {
        'rain': {
            'cmap': 'Blues',
            'cbar_kws': {'label': 'Daily rainfall (mm)'},
            },
        'temp': {
            'cmap': 'coolwarm',
            'cbar_kws': {'label': 'Daily Max Temperature'},
            },
        'wind': {
            'cmap': mpl.colors.ListedColormap(
                sns.hls_palette(16) + [(1, 1, 1), (0, 0, 0)]),
            'cbar_kws': {'label': 'Wind direction'},
            'robust': False,
            },
        }
    base_kwargs.update(kind_kwargs.get(kind, {}))
    base_kwargs.update(kwargs)
    return sns.heatmap(data, **base_kwargs)


def multipanel(df):
    with sns.axes_style(rc={'axes.facecolor':'black'}):
        fig, axes = plt.subplots(3, sharex=True)
        heatmap(rain, 'rain', ax=axes[0])
        heatmap(temp, 'temp', ax=axes[1])
        heatmap(wind, 'wind', ax=axes[2])
    return fig

multipanel(s).savefig('../output/galiwinku-all.pdf', bbox_inches='tight')

'''

wind_cmap = mpl.colors.ListedColormap(sns.hls_palette(16) + [(1, 1, 1), (0, 0, 0)])


fig, ax = plt.subplots(1, 1)
heatmap(wind, ax=ax, cmap=wind_cmap, cbar_kws={'label': 'Wind direction'})
fig.savefig('../output/galiwinku-wind.pdf', bbox_inches='tight')
'''

# Save summary table for Galiwinku conditions
table = s.describe().to_latex(
    longtable=True,
    float_format=lambda n: '{:.2f}'.format(n),
    na_rep='',
    column_format='l' + 'p{6.5em}' * 10,
    )
with open('../output/galiwinku-summary-table.tex', 'w') as f:
    f.write(table)

