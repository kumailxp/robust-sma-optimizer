#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_stock_sma_strategy(ticker, start_date, end_date, sma_period):
    # 1. Configuration
    initial_capital = 1000  # Starting with $1,000

    print(f"Fetching data for {ticker}...")
    
    # We fetch data starting a few months early to allow the SMA calculation to stabilize before 2016
    data = yf.download(ticker, start=start_date, progress=False)

    # Check if data was retrieved successfully
    if len(data) == 0:
        print("Error: No data found. Please check your internet connection or ticker.")
        return

    # Handle potential MultiIndex columns in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)

    # 2. Calculate Indicators
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
    data['Buy_Hold_Value'] = initial_capital * (1 + data['Stock_Daily_Pct']).cumprod()
    data['Strategy_Value'] = initial_capital * (1 + data['Strategy_Daily_Pct']).cumprod()

    # Handle NaNs created by shifting/rolling
    data['Buy_Hold_Value'] = data['Buy_Hold_Value'].fillna(initial_capital)
    data['Strategy_Value'] = data['Strategy_Value'].fillna(initial_capital)

    # 8. Print Results
    final_bnh = data['Buy_Hold_Value'].iloc[-1]
    final_strat = data['Strategy_Value'].iloc[-1]
    
    print("-" * 40)
    print(f"RESULTS ({start_date} to Today)")
    print("-" * 40)
    print(f"Initial Investment: ${initial_capital:,.2f}")
    print(f"Buy & Hold Final:   ${final_bnh:,.2f}")
    print(f"SMA({sma_period}) Final:      ${final_strat:,.2f}")
    print(f"Difference:         {(final_strat / final_bnh):.2f}x better performance")
    print("-" * 40)

    # 9. Plotting
    plt.figure(figsize=(12, 6))
    
    # Using Log Scale because growth might be exponential
    # Without Log scale, the 2016-2020 price action would look like a flat line
    plt.plot(data.index, data['Buy_Hold_Value'], label='Buy & Hold', color='gray', alpha=0.6)
    plt.plot(data.index, data['Strategy_Value'], label=f'SMA {sma_period} Strategy', color='green', linewidth=1.5)

    plt.yscale('log')
    plt.title(f'{ticker}: Buy & Hold vs SMA({sma_period}) Strategy ({start_date} - {end_date})')
    plt.ylabel('Portfolio Value (USD) - Log Scale')
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Format Y-axis to show actual dollar amounts instead of scientific notation
    from matplotlib.ticker import ScalarFormatter
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    plt.show()

if __name__ == "__main__":
    backtest_stock_sma_strategy("SAP", "2021-01-01", "2025-06-01", 105)