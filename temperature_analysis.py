"""
Use pandas data analysis framework to read and plot data from NOAA (https://www.ncdc.noaa.gov/cdo-web/)
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# CSV data from 10 weather stations across the Virginia Beach/Norfolk area spanning 1909 - 2021

DATAFILE = "vb_temperatures.csv"
"""
In [207]: temps.describe()
Out[207]:
              TAVG           TMAX           TMIN          TOBS
count  5859.000000  128634.000000  128499.000000  46264.000000
mean     62.027479      69.104786      51.948747     61.447519
std      15.356426      16.114390      15.679403     16.450539
min      15.000000      11.000000     -71.000000      0.000000
25%      50.000000      57.000000      39.000000     48.000000
50%      64.000000      71.000000      53.000000     63.000000
75%      76.000000      82.000000      66.000000     75.000000
max      91.000000     106.000000      92.000000    612.000000
"""

TITLE = "Virginia Beach {} Average Max/Min Temp (1909 - 2021)"
FIGSIZE = (18, 10)


def iterate_by_offset(temps: pd.DataFrame, offset: pd.DateOffset) -> pd.DataFrame:
    """
    temps: DataFrame with daily temperature data with columns DATE, TMAX, TMIN
    offset: DateOffset with desired granularity for rolling average 
    
    Return a DataFrame with rolling average with `offset` granularity over TMAX, TMIN
    """
    start = temps["DATE"].min()
    stop = temps["DATE"].max()
    results = []
    while start <= stop:
        next_period = start + offset
        print(f"Taking average max temp between {start} and {next_period}")
        period_temp_data = temps[(temps["DATE"] >= start) & (temps["DATE"] < next_period)]
        tmax_avg = period_temp_data["TMAX"].aggregate("mean")
        tmin_avg = period_temp_data["TMIN"].aggregate("mean")
        results.append((start, tmax_avg, tmin_avg))
        start += offset
    avg_temps = pd.DataFrame.from_records(results, columns=["date", "tmax", "tmin"])
    return avg_temps


def plot_by_month(temps: pd.DataFrame):
    avg_temps = iterate_by_offset(temps, pd.DateOffset(months=1))
    avg_temps.plot(x="date", y=["tmax", "tmin"], title=TITLE.format("Monthly"), figsize=FIGSIZE)


def plot_by_year(temps: pd.DataFrame):
    avg_temps = iterate_by_offset(temps, pd.DateOffset(years=1))
    avg_temps.plot(x="date", y=["tmax", "tmin"], title=TITLE.format("Yearly"), figsize=FIGSIZE)


if __name__ == "__main__":
    temps = pd.read_csv(DATAFILE, parse_dates=[2])
    plot_by_month(temps)
    plot_by_year(temps)
    plt.show()
