#!/usr/bin/env python3
"""Detect seasons from weather observations."""

import pandas as pd

from utils import nameof, seasons


def below_mean(df, name):
    # TODO: generalise to 'is within given deciles'
    return (df[name] < df[name].mean()).where(df[name].notnull())


def wind_from(df, *directions, am=True, pm=True):
    """Test if wind is from one of the given directions."""
    assert am or pm, 'Must use at least one of am or pm wind'
    am_name, pm_name = nameof.winddir09, nameof.winddir15
    am_wind = sum([df[am_name] == d for d in directions]).where(
        df[am_name].notnull())
    pm_wind = sum([df[pm_name] == d for d in directions]).where(
        df[pm_name].notnull())
    return (am_wind if am else False) + (pm_wind if pm else False)


def season_indicies(data):
    """Add a column for each season to the dataframe."""
    weekly_rain_days = (data[nameof.rain] > 1).rolling(
        window=7, center=True, min_periods=5).sum()

    season = pd.DataFrame()
    season["Dhuludur"] = sum([
        wind_from(data, "NE", "NNE", "N", "NNW", "NW", "WNW",),
        below_mean(data, nameof.mintemp),
        (weekly_rain_days >= 2),
        ])
    season["Barramirri"] = sum([
        wind_from(data, "N", "NNW", "NW", "WNW", "W"),
        data[nameof.rain] > 10,
        ])
    season["Mayaltha"] = sum([
        wind_from(data, "NNW", "NW", "WNW"),
        (weekly_rain_days <= 3) * 0.5,
        ])
    season["Midawarr"] = sum([
        wind_from(data, "NE", "ENE", "E", "ESE"),
        (weekly_rain_days != 0) * 0.5,
        ])
    season["Dharrathamirri"] = sum([
        weekly_rain_days == 0,
        wind_from(data, "ESE", "SE", "SSE"),
        ])
    season["Rarrandharr"] = sum([
        weekly_rain_days == 0,
        below_mean(data, nameof.dewpoint) * 0.5,
        below_mean(data, nameof.maxtemp) == False,
        ])
    return season


def add_seasons(df):
    """Add a column for each season to the dataframe."""
    seasons_df = season_indicies(df)
    for s in seasons:
        seas = seasons_df[s].rolling(4, center=True, min_periods=2).mean()
        # Normalise to z-score of indices
        df[s] = (seas - seas.mean()) / seas.std()

    # Note that first occurance of max wins, introducing STRONG bias
    df['raw_season'] = df[list(seasons)].idxmax(axis=1).astype(
        'category', categories=seasons, ordered=True)
    return df
