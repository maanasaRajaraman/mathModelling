# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 21:31:18 2025

@author: maana
""" 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["HDFC.csv", "ICICI.csv", "INDUSIND.csv"]
dataframes = []
for file in files:
    df = pd.read_csv(file)
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df = df.sort_values(by='Month')
    df['ror'] = (df['Close'] - df['Open']) / df['Close']
    dataframes.append(df)
 
ror = pd.concat(
    [df['ror'].reset_index(drop=True) for df in dataframes], axis=1
)
ror.columns = [f.split('.')[0] for f in files]

# 2. Compute statistics
mean_returns = ror.mean().values
cov_matrix = ror.cov().values
num_assets = len(files)
num_portfolios = 50000

# 3. Generate random normalized weights
weights = np.random.rand(num_portfolios, num_assets)
weights /= weights.sum(axis=1, keepdims=True)

portfolio_returns = []
portfolio_risks = []

for w in weights:
    ret = np.dot(w, mean_returns)
    var = np.dot(w.T, np.dot(cov_matrix, w))
    portfolio_returns.append(ret)
    portfolio_risks.append(np.sqrt(var))   # risk = std dev

# 5. Plot Efficient Frontier
plt.figure(figsize=(10,6))
plt.scatter(portfolio_risks, portfolio_returns, s=5, color='steelblue')
plt.xlabel('Portfolio Risk (Std. Deviation)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier for Multiple Securities')
plt.grid(True)
plt.show()

# 6. Minimum variance portfolio
min_idx = np.argmin(portfolio_risks)
print(f"Minimum Variance Portfolio Risk: {portfolio_risks[min_idx]*100:.3f}%")
print(f"Return: {portfolio_returns[min_idx]*100:.3f}%")
print(f"Weights: {np.round(weights[min_idx], 3)}")
print(f"Securities: {ror.columns.tolist()}")
