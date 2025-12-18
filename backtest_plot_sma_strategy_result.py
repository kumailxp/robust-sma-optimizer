#!/usr/bin/env python3
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_plot(ticker, start_date, end_date, folder_name):

    plt.yscale('log')
    plt.title(f'{ticker}: Buy & Hold vs SMA Strategies ({start_date} - {end_date})')
    plt.ylabel('Portfolio Value (USD) - Log Scale')
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Format Y-axis to show actual dollar amounts instead of scientific notation
    from matplotlib.ticker import ScalarFormatter
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    filename = f"sma_result_for_{ticker}.png"
    save_path = os.path.join(folder_name, filename) # Joins folder_name and filename reliably
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategy_graph(downloaded_data, start_date, end_date, sma_period, color, initial_capital):
    data =  perform_sma_strategy(downloaded_data, start_date, end_date, sma_period, initial_capital)
    final_strat = data['Strategy_Value'].iloc[-1]
    if final_strat > initial_capital:
        plt.plot(data.index, data['Strategy_Value'], label=f'SMA {sma_period} Strategy', color=color, linewidth=1.5)
    return final_strat

def plot_buy_and_hold(downloaded_data, start_date, end_date, initial_capital):
    data = downloaded_data
    data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    data['Stock_Daily_Pct'] = data['Close'].pct_change()
    data['Buy_Hold_Value'] = initial_capital * (1 + data['Stock_Daily_Pct']).cumprod()
    plt.plot(data.index, data['Buy_Hold_Value'], label='Buy & Hold', color='gray', alpha=0.6)
    final_bnh = data['Buy_Hold_Value'].iloc[-1]
    return final_bnh


def fetch_data_from_yahoo(ticker, start_date):
    # We fetch data starting a few months early to allow the SMA calculation to stabilize before 2016
    data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

    # Check if data was retrieved successfully
    if len(data) == 0:
        print("Error: No data found. Please check your internet connection or ticker.")
        return

    # Handle potential MultiIndex columns in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)
    
    return data

def perform_sma_strategy(downloaded_data, start_date, end_date, sma_period, initial_capital):

    # 2. Calculate Indicators
    data = downloaded_data

    data['SMA_X'] = data['Close'].rolling(window=sma_period).mean()

    # 3. Filter data betwen start and end
    data = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    # 4. Define the Strategy Logic
    # Rule: Buy when Close > SMA, Sell when Close < SMA
    # Signal = 1 (Long) if Close > SMA, else 0 (Cash)
    data['Signal'] = np.where(data['Close'] > data['SMA_X'], 1, 0)

    # 5. Execute Trades (Shift logic)
    # Important: If the Signal calculates at the 'Close' of today, 
    # we enter the position for *tomorrow's* price action.
    # We shift the signal by 1 day to avoid look-ahead bias.
    data['Position'] = data['Signal'].shift(1)

    # 6. Calculate Returns
    # Daily percentage change of Bitcoin price
    data['Stock_Daily_Pct'] = data['Close'].pct_change()
    
    # Strategy return: 
    # If Position is 1, we get the stock return. 
    # If Position is 0, we get 0% return (Cash).
    data['Strategy_Daily_Pct'] = data['Position'] * data['Stock_Daily_Pct']

    # 7. Calculate Cumulative Portfolio Value
    # Calculate cumulative product to see value growth over time
    data['Strategy_Value'] = initial_capital * (1 + data['Strategy_Daily_Pct']).cumprod()

    # Handle NaNs created by shifting/rolling
    data['Strategy_Value'] = data['Strategy_Value'].fillna(initial_capital)

    return data

def plot_all_strategies(ticker, start_date, end_date, downloaded_data, initial_capital, sma_periods, folder_name):
    print("-" * 40)
    print(f"RESULTS for {ticker} from {start_date} to {end_date}")
    print("-" * 40)
    print(f"Initial Investment: ${initial_capital:,.2f}")

    sma_periods_with_colors = zip(sma_periods, ["red", "green"])

    plt.figure(figsize=(12, 6))

    final_bnh = plot_buy_and_hold(downloaded_data, start_date, end_date, initial_capital)
    print(f"Buy & Hold Final:   ${final_bnh:,.2f}")

    for sma_period, color in sma_periods_with_colors:
        if not sma_period:
            continue
        final_strat = plot_strategy_graph(downloaded_data, start_date, end_date, sma_period, color, initial_capital)
        print(f"SMA({sma_period}) Final:      ${final_strat:,.2f}")
        print(f"Difference:         {(final_strat / final_bnh):.2f}x better performance")
    
    save_plot(ticker, start_date, end_date, folder_name)
    
    print("-" * 40)


if __name__ == "__main__":
    folder_name = 'my_plots/results'
    os.makedirs(folder_name, exist_ok=True)
    start_date =  "2021-01-01"
    end_date = "2025-06-01"
    initial_capital = 1000
    sma_periods = [32, 122]
    ticker = "CRM"

    downloaded_data = fetch_data_from_yahoo(ticker, start_date)
    plot_all_strategies(ticker, start_date, end_date, downloaded_data, initial_capital, sma_periods, folder_name)
