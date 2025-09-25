# Exponential Smoothing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# Dataset
periods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
yt = np.array([315, 195, 310, 316, 325, 335, 318, 355, 420, 410, 485, 420, 460, 395, 390, 450, 458, 570, 520, 400, 420, 580, 475, 560])


def exponential_smoothing(yt, lambda_, y1):
    smoothed = np.zeros(len(yt))
    smoothed[0] = y1
    for t in range(1, len(yt)):
        smoothed[t] = lambda_ * yt[t] + (1 - lambda_) * smoothed[t - 1]
    return smoothed

def second_order_smoothing(yt, lambda_):
    # Second-order smoothing (Brown's method)
    n = len(yt)
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    yhat = np.zeros(n)
    
    y1[0] = yt[0]
    y2[0] = yt[0]
    yhat[0] = yt[0]
    
    for t in range(1, n):
        y1[t] = lambda_ * yt[t] + (1 - lambda_) * y1[t - 1]
        y2[t] = lambda_ * y1[t] + (1 - lambda_) * y2[t - 1]
        yhat[t] = 2 * y1[t] - y2[t]
    
    return yhat
    
lambda_02 = 0.2
smoothed_02 = exponential_smoothing(yt, lambda_02, yt[0])

lambda_04 = 0.4
smoothed_04 = exponential_smoothing(yt, lambda_04, yt[0])

secondOrder = second_order_smoothing(yt, lambda_02)
data = {
    'Period': periods,
    'Original': yt,
    'Smoothed (λ = 0.2)': smoothed_02,
    'Smoothed (λ = 0.4)': smoothed_04,
    'secondOrder' : secondOrder
}
df = pd.DataFrame(data)
print(df)

#original vs lambda_ = 0.2 and 0.4
plt.plot(periods, yt, label='Original Data', marker='o', color='black')
plt.plot(periods, smoothed_02, label='Smoothed (λ = 0.2)', marker='o', color='blue')
plt.plot(periods, smoothed_04, label='Smoothed (λ = 0.4)', marker='o', color='red')
plt.plot(periods, secondOrder, label='SecondOrder', marker='o', color='yellow')
plt.xlabel('Period')
plt.ylabel('yt')
plt.title('Original vs Smoothed Data')
plt.legend()
plt.grid(True)
plt.xticks(periods)
plt.tight_layout()
plt.show()

# t test 
def t_test_paired(X, Y):
    if len(X) != len(Y):
        raise ValueError("Both samples must have same length")

    n = len(X) 
    diffs = [X[i] - Y[i] for i in range(n)] 
    mean_diff = sum(diffs) / n 
    sq_diffs = [(d - mean_diff)**2 for d in diffs]
    sd = math.sqrt(sum(sq_diffs) / (n - 1)) 
    t_value = mean_diff / (sd / math.sqrt(n))

    return t_value

# Example
X = smoothed_02
Y = yt

t_val = t_test_paired(X, Y)
print("t statistic =", t_val)
#------------------------------
# ACF - PACF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ACF Calculation (Manual)
def calculate_acf(series, max_lags):
    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2)

    acf_values = []
    for k in range(max_lags + 1):
        if k == 0:
            ck = c0
        else:
            ck = np.sum((series[k:] - mean) * (series[:-k] - mean))
        acf_values.append(ck / c0)
    return acf_values

# PACF Calculation (Manual via Yule-Walker)
def calculate_pacf(series, max_lags):
    pacf_values = [1.0]  # PACF at lag 0 is always 1
    for k in range(1, max_lags + 1):
        X = np.array([series[i:len(series) - k + i] for i in range(k)]).T
        y = series[k:]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pacf_values.append(beta[-1])  # Last coefficient = PACF at lag k
    return pacf_values

# Load data and compute returns
df = pd.read_csv("BARODABANK.csv")
returns = ((df['Close'] - df['Open']) / df['Open']) * 100
returns = returns.dropna().values  # Ensure no NaNs

print("Daily Returns (%):")
print(returns)

# Compute ACF & PACF
lags = len(returns) - 1
max_lags_to_plot = min(50, lags)

acf_vals = calculate_acf(returns, max_lags_to_plot)
pacf_vals = calculate_pacf(returns, max_lags_to_plot)

acf_df = pd.DataFrame({'Lag': range(max_lags_to_plot + 1), 'ACF': acf_vals})
pacf_df = pd.DataFrame({'Lag': range(max_lags_to_plot + 1), 'PACF': pacf_vals})

#
conf_interval = 1.96 / np.sqrt(len(returns))

# --- ACF ---
plt.figure(figsize=(12, 5))
plt.stem(acf_df['Lag'], acf_df['ACF'], use_line_collection=True)
plt.axhline(y=conf_interval, color='r', linestyle='--', label='95% Confidence Interval')
plt.axhline(y=-conf_interval, color='r', linestyle='--')
plt.title('ACF of Daily Returns - BARODABANK')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.ylim(-1, 1)
plt.legend()
plt.show()

# --- PACF ---
plt.figure(figsize=(12, 5))
plt.stem(pacf_df['Lag'], pacf_df['PACF'], use_line_collection=True)
plt.axhline(y=conf_interval, color='r', linestyle='--', label='95% Confidence Interval')
plt.axhline(y=-conf_interval, color='r', linestyle='--')
plt.title('PACF of Daily Returns - BARODABANK')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.ylim(-1, 1)
plt.legend()
plt.show()

# --- Durbin-Watson Test (Manual) ---
def durbin_watson(residuals):
    diff_sq = np.sum(np.diff(residuals) ** 2)
    denom = np.sum(residuals ** 2)
    return diff_sq / denom

# Calculate DW statistic
dw_stat = durbin_watson(returns)
print(f"Durbin-Watson Statistic: {dw_stat:.4f}")

# -------------------------
# Moving Average

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

year = [1991, 1992, 1993, 1994, 1995]
spring = [102, 110, 111, 115, 122]
summer = [120, 126, 128, 135, 144]
fall = [90, 95, 97, 103, 110]
winter = [78, 83, 86, 91, 98]

sn_df = pd.DataFrame({
        'year':year,
        'spring':spring,
        'summer':summer,
        'fall':fall,
        'winter':winter
    })

sn = sn_df.melt(id_vars = 'year', var_name = 'season', value_name = 'value')
sn_order = {'spring':1, 'summer':2, 'fall':3, 'winter':4}
sn['quarter'] = sn['season'].map(sn_order)
sn = sn.sort_values(by = ['year', 'quarter']).reset_index(drop=True)

sn['4_point_moving_avg'] = sn['value'].rolling(window=4, center=True).mean().rolling(window=2, center=True).mean()
sn['percent_of_moving_avg'] = (sn['value']/sn['4_point_moving_avg'])*100

mod_ind = sn.groupby('season')['percent_of_moving_avg'].median()
nf = 400/mod_ind.sum()
sn_indices = mod_ind * nf

sn['sn_index'] = sn['season'].map(sn_indices)
sn['dsn_value'] = (sn['value'] / sn['sn_index']) * 100

dsn_clean = sn.dropna(subset=['dsn_value'])

X_trend = dsn_clean.index.values
y_dsn = dsn_clean['dsn_value'].values

n = len(X_trend)
x_mean = X_trend.mean()
y_mean = y_dsn.mean()

# slope (b1) and intercept (b0)
b1 = ((X_trend - x_mean) * (y_dsn - y_mean)).sum() / ((X_trend - x_mean)**2).sum()
b0 = y_mean - b1 * x_mean

# predict trend for all time points
sn['trend'] = b0 + b1 * sn.index.values

# Add time index column for plotting
sn['time_index'] = sn.index

# Print regression equation
print(f"Linear Trend Equation: y = {b0:.4f} + {b1:.4f} * t")

# Plot with equation on chart
sns.lineplot(data=sn, x='time_index', y='value', label='Original Data')
sns.lineplot(data=sn, x='time_index', y='4_point_moving_avg', label='4 point moving average')
plt.plot(sn['time_index'], sn['trend'], label='Trend Line', color='red')

# Add equation text on plot
eq_text = f"y = {b0:.2f} + {b1:.2f}*t"
plt.text(sn['time_index'].max()*0.5, sn['value'].max()*0.9, eq_text,
         fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.6))

plt.title('Original data vs 4 quarter moving average with Linear Trend')
plt.xlabel('Time index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
