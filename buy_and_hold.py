import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Example Data (Replace with your actual data) ---
# Create synthetic data for demonstration (e.g., 252 days of a trending asset)
np.random.seed(42) 
initial_price = 100
daily_returns = np.random.normal(0.001, 0.02, 252) # Mean 0.1%, StDev 2%
close_prices = initial_price * np.exp(np.cumsum(daily_returns))
# Create a date range to make the plot look professional
dates = pd.date_range(start='2020-01-01', periods=len(close_prices), freq='B')
close = pd.Series(close_prices, index=dates)

# --- 2. Buy and Hold Backtest Function ---
def backtest_buy_and_hold(close_prices_series):
    """Calculates cumulative log returns for a Buy and Hold strategy."""
    
    # Use pandas series for easier log return calculation
    close_prices = close_prices_series.values
    
    # Buy and Hold Signal (1 for all periods, 0 for the first day's calculation)
    signal = np.ones(len(close_prices))
    signal[0] = 0 # Ensures the strategy starts from Day 1 to Day 2 return

    # Log returns: calculates the percentage change between each day
    log_returns = np.log(close_prices[1:]) - np.log(close_prices[:-1])

    # Strategy returns: Multiply tomorrow's log returns by today's position (signal)
    # We use signal[:-1] to align the position with the return it generates
    strategy_returns = log_returns * signal[:-1] 

    # Calculate cumulative returns (sum of log returns)
    cumulative_log_returns = np.cumsum(strategy_returns)
    
    # Calculate the actual (geometric) cumulative return from log returns
    cumulative_geometric_returns = np.exp(cumulative_log_returns) - 1

    return cumulative_geometric_returns, close_prices_series.index[1:]


# --- 3. Run Backtest and Plot ---

# Run the backtest
cum_returns, dates_for_plot = backtest_buy_and_hold(close)

# Convert the results to a pandas Series for plotting with the correct dates
cumulative_returns_series = pd.Series(cum_returns, index=dates_for_plot)

# Plotting the results
plt.figure(figsize=(12, 6))
cumulative_returns_series.plot(
    title='Buy and Hold Cumulative Returns',
    grid=True,
    legend=False,
    color='darkblue',
    linewidth=2
)
plt.axhline(0, color='grey', linestyle='--', linewidth=1) # Add a line at 0% return
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')

# Format y-axis to show percentage
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.show()