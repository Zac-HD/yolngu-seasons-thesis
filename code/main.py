#!/usr/bin/env python3
"""Data analysis and presentation for my thesis.

Outputs include tables and figures for direct including in the LaTeX source.
"""
# pylint:disable=no-member,unnecessary-lambda

from collections import namedtuple
import os
import shutil

import yaml

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Miscelaneous setup functionality
shutil.rmtree('../output/', ignore_errors=True)
os.mkdir('../output/')

with open('BOM_format.yml') as f:
    META = yaml.safe_load(f)

# provide aliases for the very long column names
nameof = namedtuple(
    'ColumnNames',
    ['rain', 'maxtemp', 'mintemp', 'dewpoint', 'humid09', 'humid15',
     'windspd09', 'windspd15', 'winddir09', 'winddir15']
    )(*META['data_cols'])


def save_figure(fig, name):
    """Save the given figure as a vector pdf file."""
    fig.savefig('../output/{}.pdf'.format(name), bbox_inches='tight')


def latex_table(df, name, **kwargs):
    """Save the given dataframe as a LaTeX longtable"""
    with open('../output/{}.tex'.format(name), 'w') as f:
        f.write(df.to_latex(
            float_format=lambda n: '{:.2f}'.format(n),
            na_rep='', longtable=True, **kwargs))


# Then define functions for the heavy lifting of loading and cleaning data

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
    if hasattr(df, 'columns') and colname not in df.columns:
        colname = nameof._asdict()[colname]
    df['year'] = pd.DatetimeIndex(df.index).year
    df['dayofyear'] = pd.DatetimeIndex(df.index).dayofyear
    ret = df.pivot(index='year', columns='dayofyear', values=colname)
    del df['year']
    del df['dayofyear']
    return ret


# Then a couple of functions to produce beautiful figures

def heatmap(data, kind, **kwargs):
    """Draw a beautiful heatmap in the given axes with variable styles."""
    # Drop years of data with very few observations
    data = data.dropna(thresh=20)
    # construct a list of labels, empty for most days but with month names
    labellist = [''] * 12
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
                  'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        labellist.append(month)
        labellist.extend([''] * 30)
    labellist = labellist[:366]
    # mappings of keyword arguments, to customise style, for each variable
    base_kwargs = {
        'xticklabels': labellist,
        'yticklabels': ['' if d % (len(data.index) // 5) else d
                        for d in data.index],
        'robust': True,
        # TODO:  check DPI for rasterised panel display
        'rasterized': True,
        }
    temp = {'cmap': 'coolwarm'}
    wspd = {'vmin': 10, 'vmax': 40,
            'cmap': sns.light_palette("navy", as_cmap=True)}
    wdir = {'cmap': mpl.colors.ListedColormap(
        sns.hls_palette(16, h=0.25) + [(1, 1, 1), (0, 0, 0)]),
            'robust': False,
            }
    kind_kwargs = {
        'rain': {'cmap': 'Blues', 'vmin': 0, 'vmax': 30,
                 'cbar_kws': {'label': 'Daily rain (mm)\n(5 day mean)'}},
        'maxtemp': {'cbar_kws': {'label': 'Daily max.\ntemperature (C)'},
                    'vmin': 28, 'vmax': 35, **temp},
        'mintemp': {'cbar_kws': {'label': 'Daily min.\ntemperature (C)'},
                    'vmin': 17, 'vmax': 28, **temp},
        'dewpoint': {'cmap': 'YlGnBu', 'vmin': 16, 'vmax': 26,
                     'cbar_kws': {'label': 'Dewpoint\ntemperature (C)'}},
        'humid09': {'cbar_kws': {'label': '9am humidity (%)'}},
        'humid15': {'cbar_kws': {'label': '3pm humidity (%)'}},
        'windspd09': {'cbar_kws': {'label': '9am wind\nspeed (km/h)'}, **wspd},
        'windspd15': {'cbar_kws': {'label': '3pm wind\nspeed (km/h)'}, **wspd},
        'winddir09': {'cbar_kws': {'label': '9am wind\ndirection'}, **wdir},
        'winddir15': {'cbar_kws': {'label': '3pm wind\ndirection'}, **wdir},
        }
    if kind not in kind_kwargs:
        kind = {v: k for k, v in zip(nameof._fields, nameof)}.get(kind)
    return sns.heatmap(np.asarray(data),
                       **{**base_kwargs, **kind_kwargs.get(kind, {}), **kwargs})


def categorical_to_numeric_wind(df, colname):
    """Creates a numerical pivot table of wind for plotting from the df."""
    varname = nameof._asdict()[colname]
    return pivot(
        df[varname].dropna().map(
            {w: i for i, w in enumerate(META['wind_dirs'])}).to_frame(),
        nameof._asdict()[colname]
        ).fillna(17).astype('uint8')


def multipanel(df, *cols, **kwargs):
    """Draw a multi-panel figure with heatmaps for given columns in the df."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not cols:
        cols = df.columns

    context = {
        'axes.facecolor': 'black',
        # TODO:  set correct figsize for whole-page display, inc. fontsize
        'figure.figsize': (10, 1.5 * len(cols)),
        # TODO:  investigate .eps etc to use LaTeX for text, fonts
        }

    func = {  # any needed data transformations
        'rain': lambda df, name: pivot(df[nameof.rain].rolling(
            center=True, window=5, min_periods=1).mean().to_frame(), name),
        'winddir09': lambda df, name: categorical_to_numeric_wind(df, name),
        'winddir15': lambda df, name: categorical_to_numeric_wind(df, name),
        }

    with mpl.rc_context(rc=context):
        fig, axes = plt.subplots(len(cols), sharex=True)
        for name, ax in zip(cols, axes if len(cols) > 1 else [axes]):
            data = func.get(name, lambda df, name: pivot(df, name))(df, name)
            hax = heatmap(data, name, ax=ax, **kwargs)
            hax.set_yticklabels(hax.yaxis.get_majorticklabels(), rotation=0)
    return fig


# And finally do the actual work!

chart_panels = ['rain', 'maxtemp', 'mintemp', 'dewpoint',
                'windspd09', 'winddir09', 'windspd15', 'winddir15']
stations = {
    "014401": 'Warruwi',
    "014404": 'Milingimbi',
    "014405": 'Maningrida',
    "014508": 'Nhulunbuy',
    "014517": 'Galiwinku',
    }

s = station_dataframe('014517')  # Galiwinku
latex_table(s.groupby(s.index.month).mean(),
            'galiwinku-summary-table',
            column_format='l' + 'p{6em}' * 8)

for s, name in stations.items():
    save_figure(multipanel(station_dataframe(s), *chart_panels),
                name.lower() + '-all')
