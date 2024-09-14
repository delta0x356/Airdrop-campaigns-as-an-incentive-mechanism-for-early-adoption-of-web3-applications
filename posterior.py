import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data, fgi_data, sp500_data, airdrop_date):
    try:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')

        start_date = airdrop_date - timedelta(days=30)
        end_date = airdrop_date + timedelta(days=30)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        if len(df) != 61:
            print(f"Warning: {file_path} does not have exactly 61 days of data centered on the airdrop date.")
            return None

        df['T'] = range(-30, 31)
        df['X'] = np.where(df['Date'] < airdrop_date, 0, 1)
        df['X_T'] = df['X'] * df['T']
        df['t'] = range(1, 62)
        df['t_X'] = df['t'] * df['X']

        # Extend market cap data
        extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
        extended_mc_data = extended_mc_data.set_index('Date').reindex(df['Date']).reset_index()
        extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()

        # Extend Fear and Greed Index data
        extended_fgi_data = fgi_data[(fgi_data['Date'] >= start_date) & (fgi_data['Date'] <= end_date)]
        extended_fgi_data = extended_fgi_data.set_index('Date').reindex(df['Date']).reset_index()
        extended_fgi_data['Fear_Greed_Index'] = extended_fgi_data['Fear_Greed_Index'].ffill().bfill()

        # Extend S&P 500 data
        extended_sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]
        extended_sp500_data = extended_sp500_data.set_index('Date').reindex(df['Date']).reset_index()
        extended_sp500_data['Close'] = extended_sp500_data['Close'].ffill().bfill()

        # Merge all data
        df = pd.merge(df, extended_mc_data[['Date', 'MCt']], on='Date', how='left')
        df = pd.merge(df, extended_fgi_data[['Date', 'Fear_Greed_Index']], on='Date', how='left')
        df = pd.merge(df, extended_sp500_data[['Date', 'Close']], on='Date', how='left')

        print(f"Final columns in df: {df.columns}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Airdrop date: {airdrop_date}")
        print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
        print(f"Fear and Greed Index data available: {df['Fear_Greed_Index'].notna().sum()} / {len(df)} days")
        print(f"S&P 500 data available: {df['Close'].notna().sum()} / {len(df)} days")
        print("First few rows of merged data:")
        print(df.head())
        print("\n")

        return df
    except Exception as e:
        print(f"Error in prepare_data for {file_path}: {str(e)}")
        raise

def fit_model(df, metric_name):
    try:
        X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt', 'Fear_Greed_Index', 'Close']])
        y = df[metric_name]
        model = sm.OLS(y, X).fit()
        return model
    except Exception as e:
        print(f"Error in fit_model for {metric_name}: {str(e)}")
        raise

def check_autocorrelation(model):
    return durbin_watson(model.resid)

def analyze_protocol_type(folder_path, metric_name, protocol_type, mc_data, fgi_data, sp500_data, airdrop_dates):
    autocorrelation_results = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            try:
                protocol_name = file.split('.')[0]
                if protocol_name not in airdrop_dates['Name'].values:
                    print(f"Warning: Protocol '{protocol_name}' not found in airdrop_date.csv. Skipping...")
                    continue
                
                airdrop_date = airdrop_dates[airdrop_dates['Name'] == protocol_name]['Date'].values[0]
                airdrop_date = pd.to_datetime(airdrop_date)
                
                file_path = os.path.join(folder_path, file)
                df = prepare_data(file_path, metric_name, mc_data, fgi_data, sp500_data, airdrop_date)
                
                if df is not None:
                    model = fit_model(df, metric_name)
                    dw_statistic = check_autocorrelation(model)

                    # Calculate Durbin-Watson statistics for Fear and Greed Index and Market Cap
                    fgi_model = sm.OLS(df['Fear_Greed_Index'], sm.add_constant(df[['T']])).fit()
                    mc_model = sm.OLS(df['MCt'], sm.add_constant(df[['T']])).fit()
                    fgi_dw = durbin_watson(fgi_model.resid)
                    mc_dw = durbin_watson(mc_model.resid)

                    autocorrelation_results.append({
                        'protocol': protocol_name,
                        'Durbin-Watson statistic': dw_statistic,
                        'Autocorrelation': 'Positive' if dw_statistic < 1.5 else ('Negative' if dw_statistic > 2.5 else 'No evidence'),
                        'S&P 500 coefficient': model.params['Close'],
                        'S&P 500 p-value': model.pvalues['Close'],
                        'Fear and Greed Index DW': fgi_dw,
                        'Market Cap DW': mc_dw
                    })
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    return pd.DataFrame(autocorrelation_results)

# Load airdrop dates
airdrop_dates = pd.read_csv('airdrop_date.csv')
airdrop_dates['Date'] = pd.to_datetime(airdrop_dates['Date'], format='%d/%m/%Y')

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')
print("Market cap data loaded. Date range:", mc_data['Date'].min(), "to", mc_data['Date'].max())

# Load Fear and Greed Index data
fgi_data = pd.read_csv('fear_greed_index.csv')
fgi_data['Date'] = pd.to_datetime(fgi_data['Date'], format='%d/%m/%Y')
print("Fear and Greed Index data loaded. Date range:", fgi_data['Date'].min(), "to", fgi_data['Date'].max())

# Load S&P 500 data
sp500_data = pd.read_csv('sp500.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], format='%d/%m/%Y')
print("S&P 500 data loaded. Date range:", sp500_data['Date'].min(), "to", sp500_data['Date'].max())

# Analyze each protocol type
protocol_types = [('TVL', 'TVL', 'DeFi'), ('volume', 'Volume', 'DEX'), ('DAU', 'DAU', 'SocialFi')]

# Create posterior_analysis folder if it doesn't exist
posterior_folder = 'posterior_analysis'
if not os.path.exists(posterior_folder):
    os.makedirs(posterior_folder)

for folder, metric, protocol_type in protocol_types:
    print(f"\nAutocorrelation analysis for {protocol_type}:")
    results = analyze_protocol_type(folder, metric, protocol_type, mc_data, fgi_data, sp500_data, airdrop_dates)

    print(results)

    # Save results
    results.to_csv(os.path.join(posterior_folder, f'{protocol_type.lower()}_autocorrelation_results.csv'), index=False)

print(f"\nAutocorrelation analysis results have been saved in the '{posterior_folder}' folder.")