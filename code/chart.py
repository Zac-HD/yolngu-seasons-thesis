#!/usr/bin/env python3
"""Turn invisible data into beautiful tables and figures."""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils
import weather

nameof = utils.nameof


def save_figure(fig, station_name, name):
    """Save the given figure as a vector pdf file."""
    fig.savefig('../output/{}/{}.pdf'.format(station_name, name),
                bbox_inches='tight')


def save_table(df, cols, station_name, name):
    """Save a dataframe as .csv, for manual analysis or typesetting."""
    with open('../output/{}/{}.csv'.format(station_name, name), 'w') as f:
        f.write(df[cols].to_csv(float_format='%.1f'))


def grouped_summary(df, gby=None):
    """Return a table of monthly means, or modes for categorical data."""
    gby = gby if gby is not None else df.index.month
    monthly = df.groupby(gby)
    cat_cols = [c for c in df.columns if df[c].dtype.name == 'category']
    modes = monthly[cat_cols].agg(lambda x: x.value_counts().index[0])
    merged = pd.merge(monthly.mean(), modes, left_index=True, right_index=True)
    return merged[df.columns]


def season_prob(data):
    """Return daily probability of observing each season."""
    is_seasons = ['is_' + n for n in utils.seasons]
    for n in utils.seasons:
        data['is_' + n] = data.raw_season == n
    grp = pd.groupby(data[is_seasons], data.index.dayofyear).sum()
    grp = grp.divide(grp.sum(axis=1), axis='rows')
    return grp.rolling(window=7, center=True).mean()


def save_out(station):
    """Save a multipanel summary figure and monthly text table."""
    station_id, station_name = station
    data = weather.data(station_id)
    os.mkdir('../output/' + station_name)

    monthly = grouped_summary(data)
    save_table(monthly, [nameof._asdict()[n] for n in utils.chart_panels],
               station_name, 'monthly-summary')
    save_table(monthly, ['raw_season'] + list(utils.seasons),
               station_name, 'monthly-seasons')

    save_figure(climograph(data), station_name, 'climograph')
    save_figure(multipanel(data, *utils.chart_panels),
                station_name, 'observations')
    save_figure(multipanel(
        data, *[w for w in utils.nameof._fields if 'windspd' in w]),
                station_name, 'seabreeze-speed')
    save_figure(multipanel(
        data, *[w for w in utils.nameof._fields if 'winddir' in w]),
                station_name, 'seabreeze-direction')
    save_figure(multipanel(data, 'raw_season', *utils.seasons),
                station_name, 'seasons')
    save_figure(lines(
        grouped_summary(data, data.index.dayofyear)[list(utils.seasons)]),
                station_name, 'seasons-daily-index')
    save_figure(lines(
        season_prob(data), filled=True,
        ylabel='Observed season occurence\n(probability, weekly mean)'),
                station_name, 'seasons-daily-prob')
    save_figure(season_pie(data), station_name, 'season-pie')


def lines(daily, ylabel='', filled=False):
    """Draw a line plot of the given columns daily."""
    with sns.axes_style('dark'):
        fig, ax = plt.subplots(figsize=(8, 4))
    if not filled:
        plt.plot(daily)
        ax.set_xlim([-1, 367])
    else:
        daily.plot.area(ax=ax, legend=False)
        ax.set_ylim([0, 1])
    plt.xticks(np.arange(366), utils.dayofyear_month_labels)
    ax.set_xlabel('Month')
    ax.set_ylabel(ylabel or 'Season intensity index (normalised)')
    return fig


def season_pie(data):
    """Draw a pie chart of observed season frequency."""
    fig, ax = plt.subplots()
    vcs = data.raw_season.value_counts(dropna=True, sort=False)
    vcs.name = ''
    vcs.index = ['Dhuludur', 'Barramirri', 'Mayaltha', 'Midawarr',
                 'Dharrath-\namirri', 'Rarrandharr']
    vcs.plot.pie(ax=ax, legend=False, figsize=(3, 3),
                 startangle=90, counterclock=False)
    return fig


def climograph(df):
    """Make a climograph from the given dataframe."""
    temp = [(utils.nameof.maxtemp, 'solid'),
            (utils.nameof.mintemp, 'dashed'),
            (utils.nameof.dewpoint, 'dotted')]
    m = pd.groupby(df, df.index.month)
    with sns.axes_style('dark'):
        fig, ax_rain = plt.subplots(figsize=(8, 4))
        ax = ax_rain.twinx()
        # Plot total rainfall
        ax_rain.bar(m.mean().index, m[utils.nameof.rain].mean(),
                    width=0.6, align='center')
        ax_rain.set_ylabel('Bars: mean rainfall per day (mm)')
        # Plot mean temperatures
        for series, style in temp:
            m[series].mean().plot(
                ax=ax, legend=False, linestyle=style,
                linewidth=4, color='darkred')
    ax.set_ylabel('Lines: daily temperature (deg C)\n'
                  '(Dewpoint < Minimum < Maximum)')
    ax.set_xlabel('Month')
    ax.set_xticks(m.mean().index)
    ax.set_xticklabels(utils._months)
    ax.set_ylim(bottom=0)
    ax_rain.set_xlim([0.5, 12.5])
    ax.set_xlim([0.5, 12.5])
    return fig


def heatmap(data, kind, **kwargs):
    """Draw a beautiful heatmap in the given axes with variable styles."""
    # Drop years of data with very few observations
    data = data.dropna(thresh=30)
    # mappings of keyword arguments, to customise style, for each variable
    base_kwargs = {
        'xticklabels': utils.dayofyear_month_labels,
        'yticklabels': ['' if d % (len(data.index) // 5) else d
                        for d in data.index],
        'robust': True,
        # TODO:  check DPI for rasterised panel display
        'rasterized': True,
        'cbar_kws': {'label': kind + '\n(index)'},
        'cmap': 'Blues',
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
        **{'windspd' + hr: {
            'cbar_kws': {'label': text + ' wind\nspeed (km/h)'}, **wspd}
           for hr, text in zip(utils._wind_hours, utils._hour_names)},
        **{'winddir' + hr: {
            'cbar_kws': {'label': text + ' wind\ndirection'}, **wdir}
           for hr, text in zip(utils._wind_hours, utils._hour_names)},
        'raw_season': {'robust': False, 'cmap': mpl.colors.ListedColormap(
            sns.color_palette(palette='muted', n_colors=6))},
        }
    for s in utils.seasons:
        kind_kwargs[s] = {'vmin': 0, 'vmax': 2}
    if kind not in kind_kwargs:
        kind = {v: k for k, v in zip(nameof._fields, nameof)}.get(kind)
    return sns.heatmap(
        np.asarray(data),
        **{**base_kwargs, **kind_kwargs.get(kind, {}), **kwargs})


def multipanel(df, *cols, **kwargs):
    """Draw a multi-panel figure with heatmaps for given columns in the df."""
    if not cols:
        cols = df.columns
    if isinstance(df, pd.Series):
        df = df.to_frame()
        cols = ['Unknown from Series']

    context = {
        'axes.facecolor': 'black',
        'figure.figsize': (10, 1.5 * len(cols)),
        }

    func = {  # any needed data transformations
        'rain': lambda df, name: utils.pivot(df[nameof.rain].rolling(
            center=True, window=5, min_periods=1).mean().to_frame(), name),
        **{'winddir' + h: utils.categorical_to_numeric_wind
           for h in utils._wind_hours},
        'raw_season': utils.categorical_to_numeric_season,
        }

    with mpl.rc_context(rc=context):
        fig, axes = plt.subplots(len(cols), sharex=True)
        for name, ax in zip(cols, axes if len(cols) > 1 else [axes]):
            f = func.get(name) or utils.pivot
            hax = heatmap(f(df, name), name, ax=ax, **kwargs)
            hax.set_yticklabels(hax.yaxis.get_majorticklabels(), rotation=0)
    return fig
