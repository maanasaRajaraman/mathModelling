import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(num_points):
    """Estimates the value of pi using Monte Carlo simulation."""
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points)

    distance = np.sqrt(x**2 + y**2)
    inside_circle = distance <= 1
    points_inside_circle = np.sum(inside_circle)

    # Estimate pi (Area of circle / Area of square = pi*r^2 / (2r)^2 = pi/4)
    pi_estimate = 4 * (points_inside_circle / num_points)

    return pi_estimate, x, y, inside_circle

num_points = 100000

# Estimate pi
pi_estimate, x, y, inside_circle = estimate_pi(num_points)

print(f"Estimated value of pi using {num_points} points: {pi_estimate}")

# q5 - News stand purchase
buy_cost = 0.30
sell_price = 0.45
scrap_value = 0.05

news_probabilities = {'good': 0.35, 'fair': 0.45, 'poor': 0.20}


demand_parameters = {
    'good': {'distribution': 'exponential', 'mean': 50},
    'fair': {'distribution': 'normal', 'mean': 50, 'std_dev': 10},
    'poor': {'distribution': 'poisson', 'mean': 50}
}

def simulate_news_type():
    """Simulates the news type for a given day."""
    random_number = np.random.uniform(0, 1)

    if random_number <= news_probabilities['good']:
        return 'good'
    elif random_number <= news_probabilities['good'] + news_probabilities['fair']:
        return 'fair'
    else:
        return 'poor'

def simulate_demand(news_type):
    """Simulates the demand for newspapers based on news type."""
    params = demand_parameters[news_type]
    distribution = params['distribution']

    if distribution == 'exponential':
        demand = np.random.exponential(scale=params['mean'])
    elif distribution == 'normal':
        demand = np.random.normal(loc=params['mean'], scale=params['std_dev'])
    elif distribution == 'poisson':
        demand = np.random.poisson(lam=params['mean'])
    else:
        # Handle unexpected distribution types
        demand = 0
        print(f"Warning: Unknown distribution type '{distribution}' for news type '{news_type}'.")

    # Ensure demand is an integer and within the 0-100 range
    demand = int(round(demand))
    demand = max(0, min(100, demand))

    return demand

def calculate_daily_financials(demand, papers_purchased, buy_cost, sell_price, scrap_value):
    """Calculates the daily financial outcomes of the newsstand."""

    papers_sold = min(demand, papers_purchased)
    daily_revenue = papers_sold * sell_price
    unsold_papers = max(0, papers_purchased - papers_sold)
    salvage_from_scrap = unsold_papers * scrap_value
    cost_of_papers = papers_purchased * buy_cost
    daily_profit = daily_revenue + salvage_from_scrap - cost_of_papers
    loss_of_profit = max(0, demand - papers_sold) * (sell_price - buy_cost)

    return daily_revenue, loss_of_profit, salvage_from_scrap, daily_profit
simulation_days = [200, 500, 1000, 10000]
simulated_results = {}
papers_purchased_daily = 50 # Fixed number of papers purchased daily

for num_days in simulation_days:
    daily_revenues = []
    daily_loss_of_profit = []
    daily_salvage_from_scrap = []
    daily_profits = []

    for day in range(num_days):
        news_type = simulate_news_type()
        demand = simulate_demand(news_type)
        revenue, loss_profit, salvage, profit = calculate_daily_financials(
            demand, papers_purchased_daily, buy_cost, sell_price, scrap_value
        )
        daily_revenues.append(revenue)
        daily_loss_of_profit.append(loss_profit)
        daily_salvage_from_scrap.append(salvage)
        daily_profits.append(profit)

    simulated_results[num_days] = {
        'daily_revenue': daily_revenues,
        'daily_loss_of_profit': daily_loss_of_profit,
        'daily_salvage_from_scrap': daily_salvage_from_scrap,
        'daily_profit': daily_profits
    }

aggregated_results = {}

for num_days, daily_results in simulated_results.items():
    avg_revenue = np.mean(daily_results['daily_revenue'])
    avg_loss_of_profit = np.mean(daily_results['daily_loss_of_profit'])
    avg_salvage_from_scrap = np.mean(daily_results['daily_salvage_from_scrap'])
    avg_profit = np.mean(daily_results['daily_profit'])

    aggregated_results[num_days] = {
        'average_daily_revenue': avg_revenue,
        'average_daily_loss_of_profit': avg_loss_of_profit,
        'average_daily_salvage_from_scrap': avg_salvage_from_scrap,
        'average_daily_profit': avg_profit
    }

print("Aggregated Simulation Results:")
for num_days, averages in aggregated_results.items():
    print(f"\nSimulation over {num_days} days:")
    print(f"  Average Daily Revenue: ${averages['average_daily_revenue']:.2f}")
    print(f"  Average Daily Loss of Profit: ${averages['average_daily_loss_of_profit']:.2f}")
    print(f"  Average Daily Salvage from Scrap: ${averages['average_daily_salvage_from_scrap']:.2f}")
    print(f"  Average Daily Profit: ${averages['average_daily_profit']:.2f}")

# q6 - Booking station

mean_inter_arrival_time = 10
min_service_time = 8
max_service_time = 12
simulation_time = 1000

inter_arrival_times = []
cumulative_arrival_time = 0
while cumulative_arrival_time <= simulation_time:
    arrival_time = np.random.exponential(scale=mean_inter_arrival_time)
    inter_arrival_times.append(arrival_time)
    cumulative_arrival_time += arrival_time

# actual arrival times
arrival_times = np.cumsum(inter_arrival_times)
num_customers = np.sum(arrival_times <= simulation_time)
arrival_times = arrival_times[:num_customers]
service_times = np.random.randint(min_service_time, max_service_time + 1, size=num_customers)

print(f"Generated {num_customers} customers arriving within the simulation time.")
print("First 5 arrival times:", arrival_times[:5])
print("First 5 service times:", service_times[:5])
# 1. Initialize simulation variables
current_time = 0
queue = []
customer_data = [] # (arrival_time, start_service_time, end_service_time, waiting_time)
next_arrival_index = 0
booking_station_free_time = 0

while current_time < simulation_time or queue:
    next_arrival_time = arrival_times[next_arrival_index] if next_arrival_index < len(arrival_times) else float('inf')

    if booking_station_free_time <= current_time and queue:
        next_event_time = current_time # Service starts immediately
    else:
        next_event_time = min(next_arrival_time, booking_station_free_time)

    if next_event_time > simulation_time and not queue:
        break

    current_time = next_event_time
    if current_time == next_arrival_time and next_arrival_index < len(arrival_times):
        queue.append(next_arrival_index)
        next_arrival_index += 1


    if booking_station_free_time <= current_time and queue:
        customer_index = queue.pop(0) # Dequeue the first customer

        service_start_time = current_time

        service_duration = service_times[customer_index]
        service_end_time = service_start_time + service_duration
        waiting_time = service_start_time - arrival_times[customer_index]

        customer_data.append({
            'customer_index': customer_index,
            'arrival_time': arrival_times[customer_index],
            'start_service_time': service_start_time,
            'end_service_time': service_end_time,
            'waiting_time': waiting_time
        })

        booking_station_free_time = service_end_time

print(f"Simulation finished at time {current_time:.2f}")
print(f"Number of customers served: {len(customer_data)}")
if customer_data:
    print(f"First 5 served customers data: {customer_data[:5]}")
