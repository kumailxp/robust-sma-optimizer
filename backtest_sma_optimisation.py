#!/usr/bin/env python3
import math
from typing import Any
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor, as_completed

from random import randint, randrange
from datetime import timedelta, datetime


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def create_random_date_ranges(minimum_start_date, ranges_to_create):
    results = []
    d1 = datetime.strptime(minimum_start_date, "%Y-%m-%d")
    d2 = datetime.strptime("2023-12-31", "%Y-%m-%d")

    while len(results) < ranges_to_create:
        sd = random_date(d1, d2)
        start_date_as_str = sd.strftime("%Y-%m-%d")
        days_remaining = datetime.utcnow() - sd
        years_remaining = math.floor(days_remaining.days / 365)
        if years_remaining <= 2:
            continue
        rand_end_year = randint(int(years_remaining / 2), years_remaining)
        end_date = datetime(
            year=sd.year + rand_end_year, month=randint(1, 10), day=randint(1, 28)
        )
        end_date_as_str = end_date.strftime("%Y-%m-%d")
        print(f"start date: {start_date_as_str}, end_date: {end_date_as_str}")
        results.append((start_date_as_str, end_date_as_str))
    return results


def group_nearby_values(data_list, max_gap):
    """
    Finds groups of nearby values in a list.

    Args:
        data_list (list): The list of numerical values.
        max_gap (float): The maximum difference between two adjacent
                         values to be considered in the same group.

    Returns:
        list: A list of lists, where each inner list is a group.
    """
    if not data_list:
        return []

    # 1. Sort the list (crucial step for grouping nearby values)
    sorted_list = sorted(data_list)

    # Initialize the list of groups and the first group
    groups = []
    current_group = [sorted_list[0]]

    # 2. Iterate and group
    for i in range(1, len(sorted_list)):
        # Check if the gap to the previous value is within the threshold
        if sorted_list[i] - sorted_list[i - 1] <= max_gap:
            current_group.append(sorted_list[i])
        else:
            # The gap is too large; start a new group
            groups.append(current_group)
            current_group = [sorted_list[i]]

    # 3. Add the last group
    groups.append(current_group)

    return groups


def plot_ranges(data_range, max_sma):

    color_names = []
    for name, _ in mcolors.TABLEAU_COLORS.items():
        color_names.append(name)

    max_colors_len = len(color_names)

    plt.figure(figsize=(12, 6))

    list_of_all_smas = []
    final_smas_data: pd.Series[Any] = []
    for id, (sma_graph, final_bnh, start_date, end_date) in enumerate(data_range):
        plot_data(sma_graph, final_bnh, color_names[id % max_colors_len])
        final_sma = get_best_moving_avgs(sma_graph, final_bnh, start_date, end_date)
        final_smas_data.append(final_sma)

    for id, final_sma in enumerate(final_smas_data):
        plot_best_sma(final_sma, color_names[id % max_colors_len])

    for s in final_smas_data:
        list_of_all_smas.extend([i for i in s.index])

    list_of_all_smas = sorted(list_of_all_smas)
    clustered_results = group_nearby_values(list_of_all_smas, 10)
    print("closly clustered smas:")
    for g in clustered_results:
        print(f"len: {len(g)}, data: {g}")

    plt.ylabel("Final Portfolio Value (USD)")
    plt.xlabel("SMA Period (Days)")
    plt.xticks(np.arange(0, max_sma + 1, 10))
    # plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.show()


def get_best_moving_avgs(results, final_bnh, start_date, end_date):
    results_series = pd.Series(results)
    best_smas = results_series.nlargest(20)
    sma_indicies = [int(i) for i in best_smas.index.values if i > 10]
    sma_indicies = sorted(sma_indicies)
    clustered_results = group_nearby_values(sma_indicies, 10)

    final_smas = pd.Series()
    for v in clustered_results:
        r = list(best_smas[v].values)
        for sma, roi in best_smas.items():
            if roi == max(r) and roi / final_bnh > 1.1:
                final_smas[sma] = roi
                break

    print("-" * 50)
    print(f"HODL: ${final_bnh:,.2f} from {start_date} to {end_date}")
    for i in range(len(final_smas)):
        print(
            f"Good SMA: {final_smas.index[i]}, ROI: ${final_smas.iloc[i]:,.2f}, Ratio (roi/hodl): {final_smas.iloc[i]/final_bnh:.2f}"
        )
    print("-" * 50)
    return final_smas


def plot_best_sma(final_smas, color_name):
    # plt.annotate(f"${final_bnh:,.0f}", # this is the text
    #              (200,final_bnh), # these are the coordinates to position the label
    #              ha='left', color=color_name) # horizontal alignment can be left, right or center

    # transparent_color_2 = list(mcolors.to_rgba(color_name))
    # transparent_color_2[3] = 0.7

    for i in range(len(final_smas)):
        plt.plot(
            final_smas.index[i],
            final_smas.iloc[i],
            "o",
            color=color_name,
            markersize=8,
        )

def plot_data(results, final_bnh, color_name):

    results_series = pd.Series(results)

    transparent_color = list(mcolors.to_rgba(color_name))
    transparent_color[3] = 0.1

    plt.plot(
        results_series.index,
        results_series.values,
        label="SMA Strategy Final Value",
        color=tuple(transparent_color),
        linewidth=1.5,
    )

    plt.axhline(
        final_bnh,
        color=tuple(transparent_color),
        linestyle="--",
        label=f"Buy & Hold Benchmark (${final_bnh:,.0f})",
    )


def backtest_sma_optimization(data, start_date, end_date, min_sma, max_sma):
    # 1. Configuration
    initial_capital = 1000

    print(f"Fetching data for {ticker}...")

    if len(data) == 0:
        print("Error: No data found.")
        return

    # Prepare results storage
    results = {}

    data_filtered = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    # Calculate Buy & Hold returns once on the full dataset
    data_filtered["BTC_Daily_Pct"] = data_filtered["Close"].pct_change()
    data_filtered["Buy_Hold_Value"] = (
        initial_capital * (1 + data_filtered["BTC_Daily_Pct"]).cumprod()
    )

    final_bnh = data_filtered["Buy_Hold_Value"].iloc[-1]

    print(f"Buy & Hold Final Value: ${final_bnh:,.2f}")
    print("-" * 50)
    print(f"Starting simulation for SMA periods {min_sma} to {max_sma}...")

    # 2. Loop Through All SMA Periods
    for sma_period in range(min_sma, max_sma + 1):

        # Calculate SMA on the full 'data' DataFrame (for the full historical period)
        temp_sma = data["Close"].rolling(window=sma_period).mean()

        # Create the working DataFrame for this iteration.
        df = data_filtered.copy()

        # Assign the calculated SMA to the working DataFrame 'df'.
        # Pandas aligns this based on the index (Date).
        df["SMA"] = temp_sma

        # Define the Strategy Logic
        df["Signal"] = np.where(df["Close"] > df["SMA"], 1, 0)

        # Execute Trades
        df["Position"] = df["Signal"].shift(1)

        # Strategy return
        df["Strategy_Daily_Pct"] = df["Position"] * df["BTC_Daily_Pct"]

        # Calculate Cumulative Portfolio Value
        df["Strategy_Value"] = (
            initial_capital * (1 + df["Strategy_Daily_Pct"]).cumprod()
        )

        # Store the final value
        if not df["Strategy_Value"].empty:
            final_strat_value = df["Strategy_Value"].iloc[-1]
            results[sma_period] = final_strat_value

        if sma_period % 100 == 0:
            print(f"Completed SMA {sma_period}...")

    return (results, final_bnh, start_date, end_date)


if __name__ == "__main__":
    min_sma = 1
    max_sma = 200
    data_to_plot = []

    # 1. Configuration
    ticker = "BTC-USD"
    print(f"Fetching data for {ticker}...")

    minimum_start_date = "2016-01-01"

    # Fetch data, starting earlier to ensure rolling mean calculation is stable
    data = yf.download(
        ticker,
        start=minimum_start_date,
        progress=False,
        multi_level_index=False,
        interval="1d",
    )

    date_array = create_random_date_ranges(minimum_start_date, 100)

    futures = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        for s_date, e_date in date_array:
            future = executor.submit(
                backtest_sma_optimization, data, s_date, e_date, min_sma, max_sma
            )
            futures.append(future)
        for future in as_completed(futures):
            try:
                data_to_plot.append(future.result())
            except:
                pass
    plot_ranges(data_to_plot, max_sma)
