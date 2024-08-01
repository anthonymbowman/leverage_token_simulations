import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Simulation Parameters
YEARS = 2
DAYS_PER_YEAR = 365
MINUTES_PER_DAY = 60 * 24
TOTAL_MINUTES = YEARS * DAYS_PER_YEAR * MINUTES_PER_DAY

# Token Parameters
TARGET_LEVERAGE_RATIO = 3
RECENTERING_SPEED = 0.1
TRADING_FEE = 0.001  # 10 basis points

# Initial Values
INITIAL_NAV = 1
INITIAL_COLLATERAL_PRICE = 1
INITIAL_DEBT_PRICE = 1

# Market Parameters
COLLATERAL_VOLATILITY = 0.52  # 52% annual volatility
DEBT_VOLATILITY = 0  # Assuming stable debt asset

# Simulation Control
SIMULATIONS_PER_P = 100  # Reduced number of simulations
P_RANGE = np.arange(0.05, 0.21, 0.01)

# Analysis Windows
WINDOW_LENGTH_DAYS = 30
WINDOW_STEP_DAYS = 7
WINDOW_LENGTH_MINUTES = WINDOW_LENGTH_DAYS * MINUTES_PER_DAY
WINDOW_STEP_MINUTES = WINDOW_STEP_DAYS * MINUTES_PER_DAY

def generate_price_series(initial_price, volatility, minutes):
    annual_drift = 0  # Assuming no drift for simplicity
    dt = 1 / (MINUTES_PER_DAY * DAYS_PER_YEAR)  # Annualized time step
    returns = np.random.normal(
        (annual_drift - 0.5 * volatility**2) * dt,
        volatility * np.sqrt(dt),
        minutes
    )
    price_series = initial_price * np.exp(np.cumsum(returns))
    return price_series

def simulate_leverage_token(p):
    t = TARGET_LEVERAGE_RATIO
    min_lr = (t + p*t) / (p*t + 1)
    max_lr = (t / (p+1)) / (1 / (t*p + 1))

    # Generate price series
    collateral_prices = generate_price_series(INITIAL_COLLATERAL_PRICE, COLLATERAL_VOLATILITY, TOTAL_MINUTES)
    debt_prices = np.full(TOTAL_MINUTES, INITIAL_DEBT_PRICE)  # Constant debt price

    # Initialize variables
    nav = INITIAL_NAV
    collateral_units = INITIAL_NAV * TARGET_LEVERAGE_RATIO / INITIAL_COLLATERAL_PRICE
    debt_units = INITIAL_NAV * (TARGET_LEVERAGE_RATIO - 1) / INITIAL_DEBT_PRICE

    nav_series = np.zeros(TOTAL_MINUTES)
    rebalance_counts = np.zeros(TOTAL_MINUTES)
    notional_traded_series = np.zeros(TOTAL_MINUTES)

    for i in range(TOTAL_MINUTES):
        collateral_price = collateral_prices[i]
        debt_price = debt_prices[i]

        collateral_value = collateral_units * collateral_price
        debt_value = debt_units * debt_price
        nav = collateral_value - debt_value
        nav_series[i] = nav
        leverage_ratio = collateral_value / nav

        # Check if rebalancing is needed
        if leverage_ratio < min_lr or leverage_ratio > max_lr:
            rebalance_counts[i] = 1
            new_leverage_ratio = (
                leverage_ratio * (1 - RECENTERING_SPEED) +
                TARGET_LEVERAGE_RATIO * RECENTERING_SPEED
            )
            new_leverage_ratio = max(min_lr, min(max_lr, new_leverage_ratio))

            # Calculate rebalance volume
            rebalance_volume = collateral_value - new_leverage_ratio * nav
            
            # Adjust collateral and debt units
            if rebalance_volume > 0:  # Need to decrease leverage
                collateral_units -= rebalance_volume / collateral_price
                debt_units -= rebalance_volume / debt_price
            else:  # Need to increase leverage
                collateral_units += abs(rebalance_volume) / collateral_price
                debt_units += abs(rebalance_volume) / debt_price

            # Apply trading fee and update notional traded
            nav -= abs(rebalance_volume) * TRADING_FEE
            notional_traded_series[i] = abs(rebalance_volume)

    # Analyze multiple windows, starting after the initial window length
    windows = []
    for start in range(WINDOW_LENGTH_MINUTES, TOTAL_MINUTES - WINDOW_LENGTH_MINUTES, WINDOW_STEP_MINUTES):
        end = start + WINDOW_LENGTH_MINUTES
        collateral_perf = collateral_prices[end] / collateral_prices[start] - 1
        nav_perf = nav_series[end] / nav_series[start] - 1
        overall_te = abs(TARGET_LEVERAGE_RATIO * collateral_perf - nav_perf)
        no_downside_te = abs(min(0, TARGET_LEVERAGE_RATIO * collateral_perf - nav_perf))
        
        windows.append({
            'p': p,
            'minLR': min_lr,
            'maxLR': max_lr,
            'rebalance_count': np.sum(rebalance_counts[start:end]),
            'notional_traded': np.sum(notional_traded_series[start:end]),
            'start_collateral_price': collateral_prices[start],
            'end_collateral_price': collateral_prices[end],
            'start_nav': nav_series[start],
            'end_nav': nav_series[end],
            'collateral_performance': collateral_perf,
            'nav_performance': nav_perf,
            'overall_tracking_error': overall_te,
            'no_downside_tracking_error': no_downside_te
        })

    return windows

# Run simulations
results = []
for p in P_RANGE:
    for _ in range(SIMULATIONS_PER_P):
        results.extend(simulate_leverage_token(p))

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
csv_filename = 'leverage_token_simulations_windowed.csv'
df.to_csv(csv_filename, index=False)
print(f"Simulations completed. Results saved to '{csv_filename}'.")