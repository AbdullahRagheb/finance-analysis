import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetching stock data
tickers = ["MCD", "COST", "V", "ADBE", "CVX"]
data = yf.download(tickers, start="2009-12-31", end="2023-12-31")['Adj Close']

# Adding daily risk-free rate (RF)
data['RF'] = 0.0005

# Plotting stock prices
data.plot(figsize=(10, 8), title="Daily Stock Prices")
plt.show()

# Calculating daily returns
returns = data.pct_change().dropna()

# Descriptive statistics for returns
desc_stats = returns.describe().T[['mean', '50%', 'std', 'max', 'min']]
desc_stats.columns = ['Mean', 'Median', 'Standard Deviation', 'Max', 'Min']
print(desc_stats)

# Creating portfolios
returns['Portfolio_1'] = returns['MCD'] * 0.35 + returns['CVX'] * 0.65
returns['Portfolio_2'] = returns['ADBE'] * 0.4 + returns['V'] * 0.6

# Comparing McDonald's return and risk with Portfolio 1
portfolio1_stats = returns[['MCD', 'Portfolio_1']].describe().T[['mean', 'std']]
print(portfolio1_stats)

# Histograms for McDonald's and Portfolio 1
returns[['MCD', 'Portfolio_1']].plot.hist(bins=100, alpha=0.5, figsize=(10, 8))
plt.title("Return Distribution for MCD and Portfolio 1")
plt.show()

# Comparing Portfolio 2 return and risk with Visa's return
portfolio2_stats = returns[['V', 'Portfolio_2']].describe().T[['mean', 'std']]
print(portfolio2_stats)

# Histograms for Adobe and Portfolio 2
returns[['ADBE', 'Portfolio_2']].plot.hist(bins=100, alpha=0.5, figsize=(10, 8))
plt.title("Return Distribution for Adobe and Portfolio 2")
plt.show()

# Expanding descriptive statistics to include portfolios
expanded_stats = returns.describe().T[['mean', 'std']]
expanded_stats['Sharpe Ratio'] = expanded_stats['mean'] / expanded_stats['std']
print(expanded_stats.sort_values('Sharpe Ratio', ascending=False))
