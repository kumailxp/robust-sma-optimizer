#!/usr/bin/env python3
"""
Module to backtest and plot Simple Moving Average (SMA) strategies.
Compares Buy & Hold performance against various SMA windows.
"""
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def save_plot(ticker_name, start, end, target_folder):
    """
    Finalizes the plot styling, sets logarithmic scale, and saves to disk.
    """
    plt.yscale('log')
    plt.title(f'{ticker_name}: Buy & Hold vs SMA Strategies ({start} - {end})')
    plt.ylabel('Portfolio Value (USD) - Log Scale')
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Format Y-axis to show actual dollar amounts
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    filename = f"sma_result_for_{ticker_name}.png"
    save_path = os.path.join(target_folder, filename)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategy_graph(params):
    """
    Plots a specific SMA strategy equity curve.
    Uses a dictionary to bypass too-many-positional-arguments error.
    """
    data = perform_sma_strategy(
        params['data'], params['start'], params['end'],
        params['sma'], params['capital']
    )
    final_strat = data['Strategy_Value'].iloc[-1]
    if final_strat > params['capital']:
        plt.plot(
            data.index, data['Strategy_Value'],
            label=f"SMA {params['sma']} Strategy",
            color=params['color'], linewidth=1.5
        )
    return final_strat

def plot_buy_and_hold(downloaded_data, start, end, capital):
    """
    Calculates and plots the Buy & Hold benchmark.
    """
    data = downloaded_data[(downloaded_data.index >= start) &
                           (downloaded_data.index <= end)].copy()
    data['Stock_Daily_Pct'] = data['Close'].pct_change()
    data['Buy_Hold_Value'] = capital * (1 + data['Stock_Daily_Pct']).cumprod()
    plt.plot(data.index, data['Buy_Hold_Value'], label='Buy & Hold', color='gray', alpha=0.6)
    return data['Buy_Hold_Value'].iloc[-1]

def fetch_data_from_yahoo(ticker_symbol, start):
    """
    Fetches historical data from Yahoo Finance and handles multi-index columns.
    """
    data = yf.download(ticker_symbol, start=start, progress=False, auto_adjust=True)

    if data.empty:
        print("Error: No data found. Please check your internet connection or ticker.")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker_symbol, axis=1, level=1)

    return data

def perform_sma_strategy(downloaded_data, start, end, sma_period, capital):
    """
    Calculates the SMA strategy logic and cumulative returns.
    """
    data = downloaded_data.copy()
    data['SMA_X'] = data['Close'].rolling(window=sma_period).mean()

    # Filter dates
    data = data[(data.index >= start) & (data.index <= end)].copy()

    # Strategy Logic: Long when Close > SMA
    data['Signal'] = np.where(data['Close'] > data['SMA_X'], 1, 0)
    data['Position'] = data['Signal'].shift(1)
    data['Stock_Daily_Pct'] = data['Close'].pct_change()
    data['Strategy_Daily_Pct'] = data['Position'] * data['Stock_Daily_Pct']
    data['Strategy_Value'] = capital * (1 + data['Strategy_Daily_Pct']).cumprod()
    data['Strategy_Value'] = data['Strategy_Value'].fillna(capital)

    return data

def plot_all_strategies(ticker_name, start, end, data, capital, periods, out_folder):
    """
    Main orchestrator for plotting Buy & Hold and multiple SMA strategies.
    """
    print("-" * 40)
    print(f"RESULTS for {ticker_name} from {start} to {end}")
    print("-" * 40)
    print(f"Initial Investment: ${capital:,.2f}")

    plt.figure(figsize=(12, 6))
    final_bnh = plot_buy_and_hold(data, start, end, capital)
    print(f"Buy & Hold Final:   ${final_bnh:,.2f}")

    colors = ["red", "green", "blue", "orange"]
    for i, sma in enumerate(periods):
        if not sma:
            continue
        # Using a dictionary to pass parameters cleanly
        strat_params = {
            'data': data, 'start': start, 'end': end,
            'sma': sma, 'color': colors[i % len(colors)], 'capital': capital
        }
        final_strat = plot_strategy_graph(strat_params)
        print(f"SMA({sma}) Final:       ${final_strat:,.2f}")
        print(f"Performance Ratio:  {(final_strat / final_bnh):.2f}x vs BNH")

    save_plot(ticker_name, start, end, out_folder)
    print("-" * 40)

if __name__ == "__main__":
    FOLDER_NAME = 'my_plots/results'
    os.makedirs(FOLDER_NAME, exist_ok=True)
    START_DATE = "2021-01-01"
    END_DATE = "2025-06-01"
    INITIAL_CAPITAL = 1000
    SMA_PERIODS = [32, 122]
    TICKER = "CRM"

    DOWNLOADED_DATA = fetch_data_from_yahoo(TICKER, START_DATE)
    if DOWNLOADED_DATA is not None:
        plot_all_strategies(
            TICKER, START_DATE, END_DATE,
            DOWNLOADED_DATA, INITIAL_CAPITAL, SMA_PERIODS, FOLDER_NAME
        )
