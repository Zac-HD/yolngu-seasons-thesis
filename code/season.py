#!/usr/bin/env python3
"""Detect seasons from weather observations."""

import pandas as pd

from utils import nameof, seasons, wind_dirs


def quantiles(series, low, high):
    """Return a boolean series; true for inputs within the given quantile."""
    return ((series >= series.quantile(low)) &
        (series <= series.quantile(high))).where(series.notnull())


def wind_between(df, start, end, *, am=True, pm=True):
    """Test if wind is from a direction between start and end inclusive, 
    rotating clockwise."""
    assert am or pm, 'Must use at least one of am or pm wind'
    am_name, pm_name = nameof.winddir09, nameof.winddir15
    wd = (wind_dirs[:16] + wind_dirs[:16])
    directions = wd[wd.index(start):]
    directions = directions[:directions.index(end)+1]
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
        wind_between(data, "WNW", "NE"),
        quantiles(data[nameof.mintemp], 0.2, 0.6),
        quantiles(data[nameof.dewpoint], 0.3, 0.7),
        (weekly_rain_days >= 2),
        ])
    season["Barramirri"] = sum([
        wind_between(data, "W", "N"),
        data[nameof.rain].rolling(5, center=True).mean() > 15,
        ])
    season["Mayaltha"] = sum([
        wind_between(data, "WNW", "NNW"),
        quantiles(data[nameof.dewpoint], 0.6, 0.8),
        (weekly_rain_days <= 3) * 0.5,
        ])
    season["Midawarr"] = sum([
        wind_between(data, "NE", "ESE"),
        (weekly_rain_days != 0) * 0.5,
        ])
    season["Dharrathamirri"] = sum([
        weekly_rain_days == 0,
        wind_between(data, "ESE", "SSE"),
        ])
    season["Rarrandharr"] = sum([
        (weekly_rain_days == 0) * 2,
        quantiles(data[nameof.dewpoint], 0, 0.4) * 0.5,
        quantiles(data[nameof.maxtemp], 0.5, 1),
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
    df['raw_season'] = df[list(seasons)]\
        .dropna()\
        .idxmax(axis=1)\
        .astype('category', categories=seasons, ordered=True)
    return df

if __name__ == '__main__':
    import chart
    chart.save_out(("014517", 'galiwinku'))
