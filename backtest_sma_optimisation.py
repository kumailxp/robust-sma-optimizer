#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_ranges(data_range, max_sma):
    # Plotting the Optimization Results
    plt.figure(figsize=(12, 6))

    for result, final_bnh in data_range:
        plot_data(result, final_bnh)
    # plt.title(f'SMA Period Optimization: Final Portfolio Value ({start_date} - {end_date})')
    plt.ylabel("Final Portfolio Value (USD)")
    plt.xlabel("SMA Period (Days)")
    plt.xticks(np.arange(0, max_sma + 1, 25))
    plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.show()


def plot_data(results, final_bnh):

    results_series = pd.Series(results)
    best_sma_period = results_series.idxmax()
    best_sma_value = results_series.max()
    second_largest_element = results_series.nlargest(2)
    print("-" * 50)
    print(f"🚀 Best SMA Period: {best_sma_period} days")
    print(f"💰 Best SMA Final Value: ${best_sma_value:,.2f}")
    print(f"Comparison to Buy & Hold: {(best_sma_value / final_bnh):.2f}x")
    print(f"2nd Best SMA {second_largest_element.index[-1]}")
    print(f"2nd Result ${second_largest_element.iloc[-1]}")
    print("-" * 50)

    two_best_sma_element = results_series.nlargest(3)

    plt.plot(
        results_series.index,
        results_series.values,
        label="SMA Strategy Final Value",
        color="blue",
        linewidth=1.5,
    )

    plt.axhline(
        final_bnh,
        color="red",
        linestyle="--",
        label=f"Buy & Hold Benchmark (${final_bnh:,.0f})",
    )

    print(two_best_sma_element)
    for i in range(len(two_best_sma_element)):
        plt.plot(
            two_best_sma_element.index[i],
            two_best_sma_element.iloc[i],
            "o",
            color="green",
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
        start="2015-01-01",
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
    # start_date = "2016-01-01"
    # end_date = "2021-12-01"
    data_to_plot = []

    date_array = [("2016-01-01", "2021-12-01"), ("2017-06-01", "2023-12-01")]

    for s_date, e_date in date_array:
        data_to_plot.append(backtest_sma_optimization(s_date, e_date, min_sma, max_sma))

    plot_ranges(data_to_plot, max_sma)
