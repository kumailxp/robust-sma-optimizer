#!/usr/bin/env python
import math
import os
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import yfinance as yf

# Set the backend to 'Agg' (A Graphics Group)
mpl.use("Agg")
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from random import randint, randrange

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import picologging as logging
from rich.progress import Progress
from rich_argparse import RichHelpFormatter

from backtest_plot_sma_strategy_result import plot_all_strategies


def initialise_logger():
    # 1. Define the Logger
    # Get a logger instance, usually named after the module or 'root'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum level to capture (DEBUG and above)

    # 2. Define the Formatter (how the log line looks)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 3. Define the FileHandler
    LOG_FILE_NAME = "app.log"
    file_handler = logging.FileHandler(LOG_FILE_NAME, mode="w")
    file_handler.setLevel(logging.DEBUG)  # File handler logs DEBUG and above
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.CRITICAL)  # Console only logs INFO and above
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = initialise_logger()


def read_file_to_list_efficiently(filename):
    """Reads a file line by line and saves the stripped content to a list."""
    lines_list = []
    try:
        with open(filename) as file:
            for line in file:
                lines_list.append(line.strip())

        return lines_list

    except FileNotFoundError:
        logger.info(f"Error: The file '{filename}' was not found.")
        return []


def random_date(start, end):
    """This function will return a random datetime between two datetime
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
            year=sd.year + rand_end_year, month=randint(1, 10), day=randint(1, 28),
        )
        end_date_as_str = end_date.strftime("%Y-%m-%d")
        logger.info(f"start date: {start_date_as_str}, end_date: {end_date_as_str}")
        results.append((start_date_as_str, end_date_as_str))
    return results


def group_nearby_values(data_list, max_gap):
    """Finds groups of nearby values in a list.

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


def most_frequent(List):
    return max(set(List), key=List.count)


def plot_ranges(data_range, max_sma, ticker, folder_name):
    color_names = []
    for name in mcolors.TABLEAU_COLORS:
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
        list_of_all_smas.extend(list(s.index))

    list_of_all_smas = sorted(list_of_all_smas)
    clustered_results = group_nearby_values(list_of_all_smas, 10)
    clustered_results.sort(key=lambda f: len(f))
    best_smas = []
    for idx, g in enumerate(reversed(clustered_results)):
        if idx == 2:
            break
        best_smas.append(most_frequent(g))

    plt.title(f"Good SMAs for {ticker}")
    plt.ylabel("Final Portfolio Value (USD)")
    plt.xlabel("SMA Period (Days)")
    plt.xticks(np.arange(0, max_sma + 1, 10))
    # plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.3)
    filename = f"best_smas_{ticker}.png"
    save_path = os.path.join(
        folder_name, filename,
    )  # Joins folder_name and filename reliably
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return best_smas


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

    logger.info("-" * 50)
    logger.info(f"HODL: ${final_bnh:,.2f} from {start_date} to {end_date}")
    for i in range(len(final_smas)):
        logger.info(
            f"Good SMA: {final_smas.index[i]}, ROI: ${final_smas.iloc[i]:,.2f}, Ratio (roi/hodl): {final_smas.iloc[i] / final_bnh:.2f}",
        )
    logger.info("-" * 50)
    return final_smas


def plot_best_sma(final_smas, color_name) -> None:
    for i in range(len(final_smas)):
        plt.plot(
            final_smas.index[i],
            final_smas.iloc[i],
            "o",
            color=color_name,
            markersize=8,
        )


def plot_data(results, final_bnh, color_name) -> None:
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

    logger.info(f"Fetching data for {ticker}...")

    if len(data) == 0:
        logger.info("Error: No data found.")
        return None

    # Prepare results storage
    results = {}

    data_filtered = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    # Calculate Buy & Hold returns once on the full dataset
    data_filtered["BTC_Daily_Pct"] = data_filtered["Close"].pct_change()
    data_filtered["Buy_Hold_Value"] = (
        initial_capital * (1 + data_filtered["BTC_Daily_Pct"]).cumprod()
    )

    final_bnh = data_filtered["Buy_Hold_Value"].iloc[-1]

    logger.info(f"Buy & Hold Final Value: ${final_bnh:,.2f}")
    logger.info("-" * 50)
    logger.info(f"Starting simulation for SMA periods {min_sma} to {max_sma}...")

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
            logger.info(f"Completed SMA {sma_period}...")

    return (results, final_bnh, start_date, end_date)


def plot_all(ticker, folder_name, ranges_to_create, start_date, end_date) -> None:
    min_sma = 1
    max_sma = 200
    data_to_plot = []

    # 1. Configuration
    logger.info(f"Fetching data for {ticker}...")

    minimum_start_date = "2014-01-01"
    # Fetch data, starting earlier to ensure rolling mean calculation is stable
    downloaded_data = yf.download(
        ticker,
        start=minimum_start_date,
        progress=False,
        multi_level_index=False,
        interval="1d",
        auto_adjust=True,
    )
    date_array = create_random_date_ranges(minimum_start_date, ranges_to_create)
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor, Progress() as progress:
        task1 = progress.add_task(f"Analysing {ticker}", total=ranges_to_create)
        for s_date, e_date in date_array:
            future = executor.submit(
                backtest_sma_optimization,
                downloaded_data,
                s_date,
                e_date,
                min_sma,
                max_sma,
            )
            futures.append(future)
        for future in as_completed(futures):
            try:
                data_to_plot.append(future.result())
                progress.update(task1, advance=1)
            except Exception as _:
                pass
    b_smas = plot_ranges(data_to_plot, max_sma, ticker, f"{folder_name}/best_smas")
    logger.info(f"Best of all SMAs: {b_smas}")
    if len(b_smas) != 0:
        plot_all_strategies(
            ticker,
            start_date,
            end_date,
            downloaded_data,
            1000,
            b_smas,
            f"{folder_name}/results",
        )


if __name__ == "__main__":
    description_str = """
    \033[38;5;208mDescription:\033[0m
    Simulate sma strategy and backtest the best smas.
    Training dataset is from 2014-01-01 to 2021-01-01.
    Default testing dataset is from 2021-01-01 to 2025-06-01,
    but the user can change this using the --start-date and --end-date flags.
    """
    parser = argparse.ArgumentParser(
        description=description_str, formatter_class=RichHelpFormatter,
    )
    parser.add_argument("-t", "--ticker", nargs="+", help="ticker symbol")
    parser.add_argument(
        "-s",
        "--number-of-simulations",
        help="number of simulations to run (default: 50)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--start-date",
        help="start date for testing dataset",
        type=str,
        default="2021-01-01",
    )
    parser.add_argument(
        "--end-date",
        help="end date for testing dataset",
        type=str,
        default="2025-06-01",
    )
    folder_name = "my_plots"
    os.makedirs(f"{folder_name}/best_smas", exist_ok=True)
    os.makedirs(f"{folder_name}/results", exist_ok=True)

    args = parser.parse_args()
    tickers = (
        args.ticker if args.ticker else read_file_to_list_efficiently("ticker_list.txt")
    )
    for ticker in tickers:
        plot_all(
            ticker,
            folder_name,
            args.number_of_simulations,
            args.start_date,
            args.end_date,
        )
