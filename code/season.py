#!/usr/bin/env python3
"""Detect seasons from weather observations."""

from utils import nameof, seasons


def below_mean(df, name):
    return df[name] < df[name].mean()


def add_seasons(df):
    """Add a column for each season to the dataframe."""
    # Note that this function is a DEMONSTRATION ONLY
    # and should not be interpreted as more than proof-of-concept work.
    weekly_rain_days = (df[nameof.rain] > 1).rolling(
        window=7, center=True, min_periods=5).sum()

    # TODO - go back to custom aggregation function to carry NaNs through
    df[seasons.du] = sum([
        below_mean(df, nameof.mintemp),
        (weekly_rain_days >= 2) * 0.6,
        ])
    df[seasons.ba] = sum([
        # Wind from NW quadrant
        df[nameof.winddir09] == 'NNW',
        df[nameof.winddir09] == 'NW',
        df[nameof.winddir09] == 'WNW',
        # Rainfall that day over 10 mm
        df[nameof.rain] > 10,
        ])
    df[seasons.ma] = sum([
        # Wind from NW quadrant
        df[nameof.winddir09] == 'NNW',
        df[nameof.winddir09] == 'NW',
        df[nameof.winddir09] == 'WNW',
        # 1, 2, or 3 days of rain per week
        (weekly_rain_days <= 3) * 0.5,
        ])
    df[seasons.mi] = sum([
        # Wind from NE-E quadrant
        df[nameof.winddir09] == 'NNE',
        df[nameof.winddir09] == 'NE',
        df[nameof.winddir09] == 'ENE',
        df[nameof.winddir09] == 'E',
        df[nameof.winddir09] == 'ESE',
        ])
    df[seasons.da] = sum([
        weekly_rain_days == 0,
        # Wind from NE-E quadrant
        df[nameof.winddir09] == 'ESE',
        df[nameof.winddir09] == 'SE',
        df[nameof.winddir09] == 'SSE',
        ])
    df[seasons.rr] = sum([
        weekly_rain_days == 0,
        below_mean(df, nameof.dewpoint),
        ])

    notnull = df[list(nameof)].notnull().all(axis=1)
    for s in seasons:
        # Carry NaNs over from data
        df[s] = df[s].where(notnull)
        # Smooth slightly with rolling 3-day mean
        df[s] = df[s].rolling(3, center=True, min_periods=2).mean()
        # Scale all to [0, 1] range
        df[s] /= df[s].max()

    # Note that first occurance of max wins, introducing STRONG bias
    df['raw_season'] = df[list(seasons)].idxmax(axis=1).astype(
        'category', categories=seasons, ordered=True)
    return df
