import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')

    # Ensure we have exactly 61 rows
    if len(df) != 61:
        raise ValueError(f"Expected 61 rows, but got {len(df)} rows in {file_path}")

    airdrop_date = df.iloc[30]['Date']  # 31st row (0-indexed) is the airdrop date

    # For robustness check, use only 15 days before the airdrop
    df = df.iloc[15:30]
    df = df.reset_index(drop=True)
    df['T'] = range(-15, 0)
    df['t'] = range(1, len(df) + 1)

    start_date = df['Date'].min() - timedelta(days=1)
    end_date = df['Date'].max() + timedelta(days=1)
    extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
    date_range = pd.date_range(start=start_date, end=end_date)
    extended_mc_data = extended_mc_data.set_index('Date').reindex(date_range).reset_index()
    extended_mc_data = extended_mc_data.rename(columns={'index': 'Date'})
    extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()
    df = pd.merge(df, extended_mc_data, on='Date', how='left')

    return df, airdrop_date

def fit_model(df, metric_name):
    X = sm.add_constant(df[['T', 't', 'MCt']])
    y = df[metric_name]
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data):
    all_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df, airdrop_date = prepare_data(file_path, metric_name, mc_data)
            results = fit_model(df, metric_name)

            result_dict = {
                'protocol': file.split('.')[0],
                'α0 (Intercept)': results.params['const'],
                'α1 (T)': results.params['T'],
                'α2 (t)': results.params['t'],
                'δ (MCt)': results.params['MCt'],
                'p_value (T)': results.pvalues['T'],
                'p_value (t)': results.pvalues['t'],
                'p_value (MCt)': results.pvalues['MCt'],
                'airdrop_date': airdrop_date
            }

            all_results.append(result_dict)

    return pd.DataFrame(all_results)

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')

# Analyze each protocol type
protocol_types = [('TVL', 'TVL', 'DeFi'), ('volume', 'Volume', 'DEX'), ('DAU', 'DAU', 'SocialFi')]

# Perform robustness checks
print("\nPerforming robustness checks:")

# Create 'robustness' folder if it doesn't exist
if not os.path.exists('robustness'):
    os.makedirs('robustness')

for folder, metric, protocol_type in protocol_types:
    print(f"\nRobustness check for {protocol_type}:")
    robustness_results = analyze_protocol_type(folder, metric, protocol_type, mc_data)

    if robustness_results.empty:
        print(f"No robustness check results for {protocol_type}. Skipping...")
        continue

    print(robustness_results)
    print(f"\n{protocol_type} Robustness Check Average Effects:")
    print(f"Average trend effect: {robustness_results['α1 (T)'].mean()}")
    print(f"Average market cap effect: {robustness_results['δ (MCt)'].mean()}")
    print(f"Protocols with significant trend effect: {(robustness_results['p_value (T)'] < 0.05).sum()}/{len(robustness_results)}")
    print(f"Protocols with significant market cap effect: {(robustness_results['p_value (MCt)'] < 0.05).sum()}/{len(robustness_results)}")

    # Save robustness check results in the 'robustness' folder
    robustness_results.to_csv(os.path.join('robustness', f'{protocol_type.lower()}_robustness_check_results.csv'), index=False)

print("\nRobustness check results have been saved in the 'robustness' folder.")