#!/usr/bin/env python3
"""Turn invisible data into beautiful tables and figures."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# pylint:disable=no-name-in-module
from utils import (
    categorical_to_numeric_wind,
    chart_panels,
    dayofyear_month_labels,
    nameof,
    pivot,
    )


def save_figure(fig, name):
    """Save the given figure as a vector pdf file."""
    fig.savefig('../output/{}.pdf'.format(name), bbox_inches='tight')


def latex_table(df, name, **kwargs):
    """Save the given dataframe as a LaTeX longtable"""
    # pylint:disable=unnecessary-lambda
    with open('../output/{}.tex'.format(name), 'w') as f:
        f.write(df.to_latex(
            float_format=lambda n: '{:.1f}'.format(n),
            na_rep='', longtable=True, **kwargs))


def save_out(data, name):
    """Save a multipanel summary figure and monthly text table."""
    monthly = data[[nameof._asdict()[n] for n in chart_panels]]\
        .groupby(data.index.month)
    modes = monthly[[nameof.winddir09, nameof.winddir15]]\
        .agg(lambda x: x.value_counts().index[0])
    table = pd.merge(monthly.mean(), modes, left_index=True, right_index=True)
    latex_table(table, name + '-monthly-summary',
                column_format='l' + 'p{6em}' * 10)
    save_figure(multipanel(data, *chart_panels), name + '-all')


def heatmap(data, kind, **kwargs):
    """Draw a beautiful heatmap in the given axes with variable styles."""
    # Drop years of data with very few observations
    data = data.dropna(thresh=30)
    # mappings of keyword arguments, to customise style, for each variable
    base_kwargs = {
        'xticklabels': dayofyear_month_labels,
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
    return sns.heatmap(
        np.asarray(data),
        **{**base_kwargs, **kind_kwargs.get(kind, {}), **kwargs})


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
        'winddir09': categorical_to_numeric_wind,
        'winddir15': categorical_to_numeric_wind,
        }

    with mpl.rc_context(rc=context):
        fig, axes = plt.subplots(len(cols), sharex=True)
        for name, ax in zip(cols, axes if len(cols) > 1 else [axes]):
            f = func.get(name) or pivot
            hax = heatmap(f(df, name), name, ax=ax, **kwargs)
            hax.set_yticklabels(hax.yaxis.get_majorticklabels(), rotation=0)
    return fig
