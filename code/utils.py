#!/usr/bin/env python3
"""Assorted utility functions and metadata."""

import collections


wind_dirs = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
    "CALM",
)

_wind_hours = ("00", "03", "06", "09", "12", "15", "18", "21")
_hour_names = ("Midnight", "3am", "6am", "9am", "Noon", "3pm", "6pm", "9pm")

nameof = collections.namedtuple(
    # Holds the full names of each data column with compact alias
    "ColumnNames",
    [
        "rain",
        "maxtemp",
        "mintemp",
        "dewpoint",
        *("windspd" + h for h in _wind_hours),
        *("winddir" + h for h in _wind_hours),
    ],
)(
    *[
        "Precipitation in the 24 hours before 9am (local time) in mm",
        "Maximum temperature in 24 hours after 9am (local time) in Degrees C",
        "Minimum temperature in 24 hours before 9am (local time) in Degrees C",
        "Average daily dew point temperature in Degrees C",
        *(
            "Wind speed at {} hours Local Time measured in km/h".format(h)
            for h in _wind_hours
        ),
        *(
            "Wind direction at {} hours Local Time as 16 compass point text".format(h)
            for h in _wind_hours
        ),
    ]
)

_quality_cols = tuple(
    [
        "Quality of precipitation value",
        "Quality of maximum temperature in 24 hours after 9am (local time)",
        "Quality of minimum temperature in 24 hours before 9am (local time)",
        "Quality of overall dew point temperature observations used",
        *(
            "Quality of wind speed at {} hours Local Time".format(h)
            for h in _wind_hours
        ),
        *(
            "Quality of wind direction at {} hours Local Time".format(h)
            for h in _wind_hours
        ),
    ]
)

_accumulation_pairs = {
    "Accumulated number of days over which the precipitation was measured": "Precipitation in the 24 hours before 9am (local time) in mm",
    "Days of accumulation of maximum temperature": "Maximum temperature in 24 hours after 9am (local time) in Degrees C",
    "Days of accumulation of minimum temperature": "Minimum temperature in 24 hours before 9am (local time) in Degrees C",
}

seasons = collections.namedtuple("Seasons", ["du", "ba", "ma", "mi", "da", "rr"])(
    "Dhuludur", "Barramirri", "Mayaltha", "Midawarr", "Dharrathamirri", "Rarrandharr"
)

_months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "July",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
dayofyear_month_labels = [
    "" if (i - 14) % 30 else _months.pop(0) for i in range(1, 367)
]
_months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "July",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


chart_panels = (
    "rain",
    "maxtemp",
    "mintemp",
    "dewpoint",
    "winddir09",
    "winddir15",
    "windspd09",
    "windspd15",
)


def pivot(df, colname):
    """Return a 2D (year/dayofyear) table for the given column."""
    if hasattr(df, "columns") and colname not in df.columns:
        colname = nameof._asdict()[colname]
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    ret = df.pivot(index="year", columns="dayofyear", values=colname)
    del df["year"]
    del df["dayofyear"]
    return ret


def categorical_to_numeric_wind(df, colname):
    """Creates a numerical pivot table of wind for plotting from the df."""
    varname = nameof._asdict()[colname]
    num_wind_dirs = {w: i for i, w in enumerate(wind_dirs)}
    new = pivot(df[varname].dropna().map(num_wind_dirs).to_frame(), varname)
    return new.fillna(17).astype("uint8")


def categorical_to_numeric_season(df, colname):
    """Creates a numerical pivot table of wind for plotting from the df."""
    num_seasons = {s: i for i, s in enumerate(seasons)}
    return pivot(df[colname].dropna().map(num_seasons).to_frame(), colname)
