#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_ranges(data_range, max_sma):

    color_names = []
    for name, _ in mcolors.TABLEAU_COLORS.items():
        color_names.append(name)

    max_colors_len = len(color_names)

    plt.figure(figsize=(12, 6))

    for id, (result, final_bnh) in enumerate(data_range):
        plot_data(result, final_bnh, color_names[id % max_colors_len])

    plt.ylabel("Final Portfolio Value (USD)")
    plt.xlabel("SMA Period (Days)")
    plt.xticks(np.arange(0, max_sma + 1, 10))
    plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.show()


def plot_data(results, final_bnh, color_name):

    results_series = pd.Series(results)
    best_smas = results_series.nlargest(3)
    print("-" * 50)
    print(f"Buy and Hold: ${final_bnh}")
    print("Best SMAs")
    for i in range(len(best_smas)):
        print(f"SMA: {best_smas.index[i]}, ROI: {best_smas.iloc[i]}")
    print("-" * 50)

    plt.plot(
        results_series.index,
        results_series.values,
        label="SMA Strategy Final Value",
        color=color_name,
        linewidth=1.5,
    )

    transparent_color = list(mcolors.to_rgba(color_name))
    transparent_color[3] = 0.4

    plt.axhline(
        final_bnh,
        color=tuple(transparent_color),
        linestyle="--",
        label=f"Buy & Hold Benchmark (${final_bnh:,.0f})",
    )

    plt.annotate(f"${final_bnh:,.0f}", # this is the text
                 (200,final_bnh), # these are the coordinates to position the label
                 ha='left', color=color_name) # horizontal alignment can be left, right or center

    transparent_color_2 = list(mcolors.to_rgba(color_name))
    transparent_color_2[3] = 0.7

    for i in range(len(best_smas)):
        plt.plot(
            best_smas.index[i],
            best_smas.iloc[i],
            "o",
            color=tuple(transparent_color_2),
            markersize=8,
        )


def backtest_sma_optimization(start_date, end_date, min_sma, max_sma):
    # 1. Configuration
    ticker = "BTC-USD"
    initial_capital = 1000

    print(f"Fetching data for {ticker}...")

    # Fetch data, starting earlier to ensure rolling mean calculation is stable
    data = yf.download(
        ticker,
        start=start_date,
        progress=False,
        multi_level_index=False,
        interval="1d",
    )

    if len(data) == 0:
        print("Error: No data found.")
        return

    # Prepare results storage
    results = {}

    # Calculate Buy & Hold returns once on the full dataset
    data["BTC_Daily_Pct"] = data["Close"].pct_change()
    data["Buy_Hold_Value"] = initial_capital * (1 + data["BTC_Daily_Pct"]).cumprod()

    # Filter the main DataFrame for the simulation period
    # This serves as the clean template for the loop
    data_filtered = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    final_bnh = data_filtered["Buy_Hold_Value"].iloc[-1]

    print(f"Buy & Hold Final Value: ${final_bnh:,.2f}")
    print("-" * 50)
    print(f"Starting simulation for SMA periods {min_sma} to {max_sma}...")

    # 2. Loop Through All SMA Periods
    for sma_period in range(min_sma, max_sma + 1):

        # Calculate SMA on the full 'data' DataFrame (for the full historical period)
        temp_sma = data["Close"].rolling(window=sma_period).mean()

        # Create the working DataFrame for this iteration.
        # 🎯 THE FIX: Copy the *entire* filtered template.
        # We only need 'Close' and 'BTC_Daily_Pct' going forward.
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

        if sma_period % 20 == 0:
            print(f"Completed SMA {sma_period}...")

    return (results, final_bnh)


if __name__ == "__main__":
    min_sma = 1
    max_sma = 200
    data_to_plot = []

    date_array = [("2016-01-01", "2021-12-01"), ("2015-06-01", "2019-12-01"), 
                  ("2017-06-01", "2021-12-01"), ("2019-01-01", "2023-12-01"),
                  ("2015-06-01", "2020-06-01"), ("2020-01-01", "2024-06-01"),
                  ("2021-01-01", "2025-10-01"), ("2018-01-01", "2022-12-01"),]

    for s_date, e_date in date_array:
        data_to_plot.append(backtest_sma_optimization(s_date, e_date, min_sma, max_sma))

    plot_ranges(data_to_plot, max_sma)
