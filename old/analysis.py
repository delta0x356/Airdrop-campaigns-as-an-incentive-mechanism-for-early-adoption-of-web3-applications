import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  # Parse dd/mm/yyyy format
    df = df.sort_values('Date')
    
    # Get the airdrop date (31st row)
    airdrop_date = df.iloc[30]['Date']
    
    # Create time variable T centered at airdrop date (ranges from -30 to 30)
    df['T'] = range(-30, 31)
    
    # Create intervention variable X (0 for pre-airdrop, 1 for post-airdrop including airdrop day)
    df['X'] = np.where(df.index < 30, 0, 1)
    
    # Create interaction term X*T
    df['X_T'] = df['X'] * df['T']
    
    # Create time trend t (1 to 61)
    df['t'] = range(1, 62)
    
    # Create interaction term t*X
    df['t_X'] = df['t'] * df['X']
    
    # Extend market cap data range
    start_date = df['Date'].min() - timedelta(days=1)
    end_date = df['Date'].max() + timedelta(days=1)
    extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
    
    # Reindex extended_mc_data to include all dates
    date_range = pd.date_range(start=start_date, end=end_date)
    extended_mc_data = extended_mc_data.set_index('Date').reindex(date_range).reset_index()
    extended_mc_data = extended_mc_data.rename(columns={'index': 'Date'})
    
    # Forward fill and then backward fill to ensure all dates have a value
    extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()
    
    # Merge with protocol data
    df = pd.merge(df, extended_mc_data, on='Date', how='left')
    
    # Print diagnostic information
    print(f"Protocol: {file_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Airdrop date: {airdrop_date}")
    print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
    print("First few rows of merged data:")
    print(df.head())
    print("\n")
    
    return df

def fit_model(df, metric_name):
    X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt']])
    y = df[metric_name]
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data):
    all_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = prepare_data(file_path, metric_name, mc_data)
            results = fit_model(df, metric_name)
            
            all_results.append({
                'protocol': file.split('.')[0],
                'α0 or β0 or γ0 (Intercept)': results.params['const'],
                'α1 or β1 or γ1 (T)': results.params['T'],
                'α2 or β2 or γ2 (X)': results.params['X'],
                'α3 or β3 or γ3 (X*T)': results.params['X_T'],
                'α4 or β4 or γ4 (t)': results.params['t'],
                'α5 or β5 or γ5 (t*X)': results.params['t_X'],
                'δ (MCt)': results.params['MCt'],
                'p_value (X)': results.pvalues['X'],
                'p_value (X*T)': results.pvalues['X_T'],
                'p_value (MCt)': results.pvalues['MCt']
            })
    
    return pd.DataFrame(all_results)

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')
print("Market cap data loaded. Date range:", mc_data['Date'].min(), "to", mc_data['Date'].max())

# Analyze each protocol type
defi_results = analyze_protocol_type('TVL', 'TVL', 'DeFi', mc_data)
dex_results = analyze_protocol_type('volume', 'Volume', 'DEX', mc_data)
socialfi_results = analyze_protocol_type('DAU', 'DAU', 'SocialFi', mc_data)

# Print results
for df, name in zip([defi_results, dex_results, socialfi_results], ['DeFi', 'DEX', 'SocialFi']):
    print(f"\n{name} Results:")
    print(df)
    print(f"\n{name} Average Effects:")
    print(f"Average immediate effect (α2 or β2 or γ2): {df['α2 or β2 or γ2 (X)'].mean()}")
    print(f"Average slope change (α3 or β3 or γ3): {df['α3 or β3 or γ3 (X*T)'].mean()}")
    print(f"Average market cap effect (δ): {df['δ (MCt)'].mean()}")
    print(f"Protocols with significant immediate effect: {(df['p_value (X)'] < 0.05).sum()}/{len(df)}")
    print(f"Protocols with significant slope change: {(df['p_value (X*T)'] < 0.05).sum()}/{len(df)}")
    print(f"Protocols with significant market cap effect: {(df['p_value (MCt)'] < 0.05).sum()}/{len(df)}")

# Create 'result' folder if it doesn't exist
result_folder = 'result'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Saving results to CSV in the 'result' folder
defi_results.to_csv(os.path.join(result_folder, 'defi_analysis_results.csv'), index=False)
dex_results.to_csv(os.path.join(result_folder, 'dex_analysis_results.csv'), index=False)
socialfi_results.to_csv(os.path.join(result_folder, 'socialfi_analysis_results.csv'), index=False)

print(f"\nResults have been saved in the '{result_folder}' folder.")