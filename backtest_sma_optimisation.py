#!/usr/bin/env python
"""Provide functionality for backtesting and optimizing SMA strategies.

This module includes tools for simulating and analyzing SMA-based trading strategies.
"""

# 1. Standard Library Imports
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import randint, randrange
from typing import Any

# 2. Third Party Imports
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import picologging as logging
import yfinance as yf
from rich.progress import Progress
from rich_argparse import RichHelpFormatter

# 3. Local Imports
from backtest_plot_sma_strategy_result import plot_all_strategies

# Set the backend to 'Agg' (A Graphics Group)
mpl.use("Agg")

color_names = list(mcolors.TABLEAU_COLORS)

def initialise_logger() -> logging.Logger:
    """Initialise and configure the logger for the application."""
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file_name = "app.log"
    file_handler = logging.FileHandler(log_file_name, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    log.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.CRITICAL)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    return log

# Global logger instance
LOGGER = initialise_logger()

def read_file_to_list_efficiently(filename: str) -> list[str]:
    """Read a file line by line and save the stripped content to a list."""
    lines_list: list[str] = []
    try:
        with Path(filename).open(encoding="utf-8") as file:
            for line in file:
                lines_list.append(line.strip())
        return lines_list
    except FileNotFoundError:
        LOGGER.info("Error: The file '%s' was not found.", filename)
        return []

def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between two datetime objects."""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

def create_random_date_ranges(minimum_start_date: str,
                            ranges_to_create: int) -> list[tuple[str, str]]:
    """Generate random date ranges based on a minimum start date."""
    results = []
    d1 = datetime.strptime(minimum_start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d2 = datetime.strptime("2023-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc)

    while len(results) < ranges_to_create:
        sd = random_date(d1, d2)
        start_date_as_str = sd.strftime("%Y-%m-%d")
        days_remaining = datetime.now(tz=timezone.utc) - sd
        years_remaining = math.floor(days_remaining.days / 365)
        if years_remaining <= 2:
            continue
        rand_end_year = randint(int(years_remaining / 2), years_remaining)
        end_date = datetime(
            year=sd.year + rand_end_year, month=randint(1, 10), day=randint(1, 28),
            tzinfo=timezone.utc,
        )
        end_date_as_str = end_date.strftime("%Y-%m-%d")
        LOGGER.info("start date: %s, end_date: %s", start_date_as_str, end_date_as_str)
        results.append((start_date_as_str, end_date_as_str))
    return results

def group_nearby_values(data_list: list[float], max_gap: float) -> list[list[float]]:
    """Find groups of nearby values in a list."""
    if not data_list:
        return []
    sorted_list = sorted(data_list)
    groups = []
    current_group = [sorted_list[0]]
    for i in range(1, len(sorted_list)):
        if sorted_list[i] - sorted_list[i - 1] <= max_gap:
            current_group.append(sorted_list[i])
        else:
            groups.append(current_group)
            current_group = [sorted_list[i]]
    groups.append(current_group)
    return groups

def most_frequent(lst: list) -> Any:
    """Find the most frequent element in a list."""
    return max(set(lst), key=lst.count)

def get_best_moving_avgs(results: dict, final_bnh: float,
                         start_date: str, end_date: str) -> pd.Series:
    """Identify the best moving averages based on strategy results."""
    results_series = pd.Series(results)
    best_smas = results_series.nlargest(20)
    sma_indices = sorted([int(i) for i in best_smas.index.values if i > 10])
    clustered_results = group_nearby_values(sma_indices, 10)

    final_smas = pd.Series(dtype=float)
    for group in clustered_results:
        r_values = list(best_smas[group].values)
        for sma, roi in best_smas.items():
            if roi == max(r_values) and roi / final_bnh > 1.1:
                final_smas[sma] = roi
                break

    LOGGER.info("-" * 50)
    LOGGER.info("HODL: $%s from %s to %s", f"{final_bnh:,.2f}", start_date, end_date)
    return final_smas

def plot_best_sma(final_smas: pd.Series, color_name: str) -> None:
    """Plot the best SMA values on a graph."""
    for i in range(len(final_smas)):
        plt.plot(final_smas.index[i], final_smas.iloc[i], "o", color=color_name, markersize=8)

def plot_data(results: dict, final_bnh: float, color_name: str) -> None:
    """Plot the SMA strategy results and the Buy & Hold benchmark."""
    results_series = pd.Series(results)
    rgba = list(mcolors.to_rgba(color_name))
    rgba[3] = 0.1
    plt.plot(results_series.index, results_series.values, color=tuple(rgba), linewidth=1.5)
    plt.axhline(final_bnh, color=tuple(rgba), linestyle="--")

def plot_ranges(data_range: list, max_sma: int,
                ticker_name: str, out_folder: str) -> list[int]:
    """Plot SMA ranges and determine the best SMAs."""
    plt.figure(figsize=(12, 6))
    list_of_all_smas = []
    final_smas_data = []

    for idx, (sma_graph, final_bnh, start, end) in enumerate(data_range):
        c = color_names[idx % len(color_names)]
        plot_data(sma_graph, final_bnh, c)
        final_smas_data.append(get_best_moving_avgs(sma_graph, final_bnh, start, end))

    for idx, final_sma in enumerate(final_smas_data):
        plot_best_sma(final_sma, color_names[idx % len(color_names)])
        list_of_all_smas.extend(list(final_sma.index))

    clustered = group_nearby_values(sorted(list_of_all_smas), 10)
    clustered.sort(key=len)
    best_smas = [most_frequent(g) for g in reversed(clustered[-2:])]

    plt.title(f"Good SMAs for {ticker_name}")
    plt.ylabel("Final Portfolio Value (USD)")
    plt.xlabel("SMA Period (Days)")
    plt.xticks(np.arange(0, max_sma + 1, 10))
    plt.grid(visible=True, which="major", ls="-", alpha=0.3)
    plt.savefig(Path(out_folder) / f"best_smas_{ticker_name}.png", dpi=300)
    plt.close()
    return best_smas

def backtest_sma_optimization(data: pd.DataFrame, config: dict,
                              initial_capital: int = 1000) -> tuple:
    """Backtest and optimize SMA strategies over a given date range."""
    # Config contains: ticker, start_date, end_date, min_sma, max_sma
    results = {}
    mask = (data.index >= config['start_date']) & (data.index <= config['end_date'])
    df_filtered = data[mask].copy()
    df_filtered["Pct_Chg"] = df_filtered["Close"].pct_change()
    final_bnh = (initial_capital * (1 + df_filtered["Pct_Chg"]).cumprod()).iloc[-1]

    for sma_period in range(config['min_sma'], config['max_sma'] + 1):
        temp_sma = data["Close"].rolling(window=sma_period).mean()
        df = df_filtered.copy()
        df["SMA"] = temp_sma
        df["Pos"] = np.where(df["Close"] > df["SMA"], 1, 0)
        df["Strat_Pct"] = df["Pos"].shift(1) * df["Pct_Chg"]
        strat_val = initial_capital * (1 + df["Strat_Pct"]).cumprod()
        if not strat_val.empty:
            results[sma_period] = strat_val.iloc[-1]
    return (results, final_bnh, config['start_date'], config['end_date'])

def plot_all_results(ticker_name: str, out_folder: str, sims: int,
                    start: str, end: str) -> None:
    """Fetch data, run simulations in parallel, and plot results."""
    min_start = "2014-01-01"
    downloaded = yf.download(ticker_name, start=min_start, progress=False,
                             multi_level_index=False, auto_adjust=True)
    date_array = create_random_date_ranges(min_start, sims)
    data_to_plot = []

    with ThreadPoolExecutor(max_workers=4) as executor, Progress() as progress:
        task = progress.add_task(f"Analysing {ticker_name}", total=sims)
        future_to_date = {
            executor.submit(
                backtest_sma_optimization, downloaded,
                {'ticker': ticker_name, 'start_date': s, 'end_date': e,
                 'min_sma': 1, 'max_sma': 200}
            ): (s, e) for s, e in date_array
        }
        for future in as_completed(future_to_date):
            res = future.result()
            if res:
                data_to_plot.append(res)
            progress.update(task, advance=1)

    best = plot_ranges(data_to_plot, 200, ticker_name, f"{out_folder}/best_smas")
    if best:
        plot_all_strategies(ticker_name, start, end, downloaded,
                          1000, best, f"{out_folder}/results")

if __name__ == "__main__":
    DESC = "\033[38;5;208mDescription:\033[0m Simulate SMA strategy optimization."
    arg_parser = argparse.ArgumentParser(description=DESC, formatter_class=RichHelpFormatter)
    arg_parser.add_argument("-t", "--ticker", nargs="+", help="ticker symbol")
    arg_parser.add_argument("-s", "--simulations", type=int, default=50)
    arg_parser.add_argument("--start-date", type=str, default="2021-01-01")
    arg_parser.add_argument("--end-date", type=str, default="2025-06-01")

    BASE_FOLDER = "my_plots"
    for sub in ["best_smas", "results"]:
        os.makedirs(f"{BASE_FOLDER}/{sub}", exist_ok=True)

    ARGS = arg_parser.parse_args()
    T_LIST = ARGS.ticker if ARGS.ticker else read_file_to_list_efficiently("ticker_list.txt")

    for t_sym in T_LIST:
        plot_all_results(t_sym, BASE_FOLDER, ARGS.simulations, ARGS.start_date, ARGS.end_date)
