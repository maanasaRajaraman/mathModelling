# q1 - Dynamic system an+1 = r^n * a0

import numpy as np
import matplotlib.pyplot as plt

num_terms = 100
a0_values = np.random.uniform(0.5, 1, 100)
result = []

def compute_a(a0, r, num_terms):
    sequence = []
    for i in range(len(a0)):
        seq = [a0[i]]
        for n in range(1, num_terms):
            next_term = (r[i]**(n-1)) * a0[i]
            seq.append(next_term)
        sequence.append(seq)
    return np.array(sequence)


# i) r = 0
r_values = np.zeros(100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('r=0')
plt.grid(True)
sequences = compute_a(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# ii) 0 < r < 1
r_values = np.random.uniform(0.001, 1, 100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title(' 0 < r < 1')
plt.grid(True)
sequences = compute_a(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# iii) -1 < r < 0
r_values = np.random.uniform(-1, -0.001, 100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('-1 < r < 0')
plt.grid(True)
sequences = compute_a(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# iv) |r| > 1
r_values = np.random.choice(np.concatenate((np.random.uniform(1.001, 2, 100), np.random.uniform(-2, -1.001, 100))), size=100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('|r| > 1')
plt.grid(True)
sequences = compute_a(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()

# q2 - decay of digoxin

import numpy as np
import matplotlib.pyplot as plt

def digoxin_decay_system(c0, dosage, num_periods):
    con = [c0]
    for input in range(num_periods):
        decayed_concentration = con[-1] * 0.5
        new_concentration = decayed_concentration + dosage
        con.append(new_concentration)
    return con
    
initial_concentration = 0
num_periods = 20


dosages = [0.1, 0.2, 0.3]
plt.figure(figsize=(10, 6))
for dosage in dosages:
    con = digoxin_decay_system(initial_concentration, dosage, num_periods)
    plt.plot(range(num_periods + 1), con, marker='x', linestyle='-', label=f'Dosage: {dosage} mg')

plt.xlabel('Dosage Period')
plt.ylabel('Digoxin Concentration (mg)')
plt.title('Concentration Change over time')
plt.grid(True)
plt.legend()
plt.show()

# q3 - Distribution Plots

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_distribution(data, title):
    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')

    # Frequency Polygon
    hist, bins = np.histogram(data, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, hist, 'r-', marker='o', linestyle='dashed', linewidth=1, markersize=4, label='Frequency Polygon')

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency / Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

sample_sizes = [500, 1000, 10000, 100000]

# i) Uniform distribution
for size in sample_sizes:
    uniform_data = np.random.uniform(0, 1, size)
    plot_distribution(uniform_data, f'Uniform Distribution (n={size})')

# ii) Exponential distribution
for size in sample_sizes:
    exponential_data = np.random.exponential(scale=1, size=size)
    plot_distribution(exponential_data, f'Exponential Distribution (n={size})')

# iii) Weibull distribution (using scipy for shape parameter k=2)
for size in sample_sizes:
    weibull_data = stats.weibull_min.rvs(c=2, size=size)
    plot_distribution(weibull_data, f'Weibull Distribution (k=2, n={size})')

# iv) Triangular distribution (using scipy, with mode at 0.5)
for size in sample_sizes:
    triangular_data = stats.triang.rvs(c=0.5, size=size)
    plot_distribution(triangular_data, f'Triangular Distribution (mode=0.5, n={size})')
