"""
Use pandas data analysis framework to read and plot data from NOAA (https://www.ncdc.noaa.gov/cdo-web/)
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np

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

TITLE = "Virginia Beach Yearly {} Average Temp (1909 - 2021)"
FIGSIZE = (18, 10)

def linear_regression(avg_temps, ax, col_idx):
    avg_temps["time"] = np.arange(len(avg_temps.index))
    X = avg_temps.iloc[:, 3].values.reshape(-1, 1) # time
    Y = avg_temps.iloc[:, col_idx].values.reshape(-1, 1) # tmin/tmax
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    r_2 = round(linear_regressor.score(X, Y), 4)
    return (X, Y, Y_pred, r_2)


def plot_by_year(temps: pd.DataFrame):
    fig, axes = plt.subplots(2)
    fig.set_size_inches(FIGSIZE)

    offset = pd.DateOffset(years=1)
    start = temps["DATE"].min()
    stop = temps["DATE"].max()
    results = []
    month_dict = defaultdict(list)
    while start <= stop:
        next_period = start + offset
        print(f"Taking average temp between {start} and {next_period}")
        period_temp_data = temps[
            (temps["DATE"] >= start) & (temps["DATE"] < next_period)
        ]
        tmax_avg = period_temp_data["TMAX"].aggregate("mean")
        tmin_avg = period_temp_data["TMIN"].aggregate("mean")
        results.append((start, tmax_avg, tmin_avg))
        start += offset
    avg_temps = pd.DataFrame.from_records(results, columns=["date", "tmax", "tmin"])

    for label, ax in zip(["tmin", "tmax"], axes):
        col_idx = 2 if label == "tmin" else 1
        X, Y, Y_pred, r_2 = linear_regression(avg_temps, ax, col_idx)
        ax.scatter(X, Y, label=label)
        ax.plot(X, Y_pred, color="red", label=f"Line of Best Fit: R-squared = {r_2}")
        ax.set_title(TITLE.format(label.lstrip("t").title()))
        ax.set_ylabel("Degrees (F)")
        ax.set_xlabel("Year")
        ax.legend()
    fig.show()


if __name__ == "__main__":
    temps = pd.read_csv(DATAFILE, parse_dates=[2])
    plot_by_year(temps)
    plt.show()
