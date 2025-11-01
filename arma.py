# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:00:47 2025

@author: maana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("HDFC.csv")
returns = (df['Close']-df['Open'])/(df['Open']) * 100

# ARMA 
phi = 0.5
theta = 0.3
c = 1

fitted_arma = [returns[0]]
error_arma = [0]

for t in range(1, len(returns)):
    armaPred = c + phi*returns[t-1] + theta*error_arma[-1]
    error = returns[t] - armaPred
    fitted_arma.append(armaPred)
    error_arma.append(error)
    


# ARIMA MODEL
diff_returns = np.diff(returns)
fitted_arima =[diff_returns[0]]
error_arima = [0]

for t in range(0, len(returns)):
    arimaPred = c + phi*diff_returns[t-1] + theta*error_arima[-1]
    error = returns[t] - arimaPred
    fitted_arima.append(arimaPred)
    error_arima.append(error)
    
fitted_arima = np.cumsum(fitted_arima) + returns[0]

plt.figure(figsize=(10,12))
plt.plot(returns, label='Actual Returns', color='black')
plt.plot(fitted_arma, label='ARMA', color = 'green')
plt.plot(fitted_arima, label='ARIMA', color = 'blue')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Returns (%)")
plt.title("Arima, Arma Plot")
plt.show()