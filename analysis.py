import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import GLS
import os
from datetime import datetime, timedelta

def prepare_data(file_path, metric_name, mc_data, sp500_data, fgi_data, airdrop_date):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')

    start_date = airdrop_date - timedelta(days=30)
    end_date = airdrop_date + timedelta(days=30)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    if len(df) != 61:
        print(f"Warning: {file_path} does not have exactly 61 days of data centered on the airdrop date.")
        return None, None

    df['T'] = range(-30, 31)
    df['X'] = np.where(df['Date'] < airdrop_date, 0, 1)
    df['X_T'] = df['X'] * df['T']
    df['t'] = range(1, 62)
    df['t_X'] = df['t'] * df['X']
    
    # Extend market cap data
    extended_mc_data = mc_data[(mc_data['Date'] >= start_date) & (mc_data['Date'] <= end_date)]
    extended_mc_data = extended_mc_data.set_index('Date').reindex(df['Date']).reset_index()
    extended_mc_data['MCt'] = extended_mc_data['MCt'].ffill().bfill()
    
    # Extend S&P 500 data
    extended_sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]
    extended_sp500_data = extended_sp500_data.set_index('Date').reindex(df['Date']).reset_index()
    extended_sp500_data['Close'] = extended_sp500_data['Close'].ffill().bfill()
    
    # Extend Fear and Greed Index data
    extended_fgi_data = fgi_data[(fgi_data['Date'] >= start_date) & (fgi_data['Date'] <= end_date)]
    extended_fgi_data = extended_fgi_data.set_index('Date').reindex(df['Date']).reset_index()
    extended_fgi_data['Fear_Greed_Index'] = extended_fgi_data['Fear_Greed_Index'].ffill().bfill()
    
    # Merge all data
    df = pd.merge(df, extended_mc_data[['Date', 'MCt']], on='Date', how='left')
    df = pd.merge(df, extended_sp500_data[['Date', 'Close']], on='Date', how='left')
    df = pd.merge(df, extended_fgi_data[['Date', 'Fear_Greed_Index']], on='Date', how='left')

    print(f"Protocol: {file_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Airdrop date: {airdrop_date}")
    print(f"Market cap data available: {df['MCt'].notna().sum()} / {len(df)} days")
    print(f"S&P 500 data available: {df['Close'].notna().sum()} / {len(df)} days")
    print(f"Fear and Greed Index data available: {df['Fear_Greed_Index'].notna().sum()} / {len(df)} days")
    print("First few rows of merged data:")
    print(df.head())
    print("\n")

    return df, airdrop_date

def fit_model(df, metric_name, protocol_type):
    if protocol_type == 'DeFi':
        prefix = 'α'
    elif protocol_type == 'DEX':
        prefix = 'β'
    else:  # SocialFi
        prefix = 'γ'

    X = sm.add_constant(df[['T', 'X', 'X_T', 't', 't_X', 'MCt', 'Fear_Greed_Index', 'Close']])
    y = df[metric_name]

    # First, fit OLS model
    ols_model = sm.OLS(y, X).fit()

    # Fit ARIMA model to the residuals
    arima_model = ARIMA(ols_model.resid, order=(1,0,1)).fit()

    # Adjust the dependent variable
    y_adjusted = y - arima_model.fittedvalues

    # Fit the main model on adjusted data
    adjusted_model = sm.OLS(y_adjusted, X).fit()

    # Test for heteroskedasticity
    _, p_value, _, _ = het_breuschpagan(adjusted_model.resid, adjusted_model.model.exog)
    
    print(f"Heteroskedasticity test p-value: {p_value}")

    # If heteroskedasticity is detected, use GLS
    if p_value < 0.05:
        print("Heteroskedasticity detected. Using GLS.")
        weights = 1 / (adjusted_model.resid ** 2)
        final_model = GLS(y_adjusted, X, weights=weights).fit()
    else:
        print("No heteroskedasticity detected. Using OLS.")
        final_model = adjusted_model

    # Calculate predicted values
    predicted_values = final_model.predict(X) + arima_model.fittedvalues

    results = {
        'protocol': df['protocol_name'].iloc[0],
        f'{prefix}0 (Intercept)': final_model.params['const'],
        f'{prefix}1 (T)': final_model.params['T'],
        f'{prefix}2 (X)': final_model.params['X'],
        f'{prefix}3 (X_T)': final_model.params['X_T'],
        f'{prefix}4 (t)': final_model.params['t'],
        f'{prefix}5 (t_X)': final_model.params['t_X'],
        'δ1 (MCt)': final_model.params['MCt'],
        'δ2 (Fear_Greed_Index)': final_model.params['Fear_Greed_Index'],
        'δ3 (S&P 500)': final_model.params['Close'],
        'p_value (X)': final_model.pvalues['X'],
        'p_value (X_T)': final_model.pvalues['X_T'],
        'p_value (MCt)': final_model.pvalues['MCt'],
        'p_value (Fear_Greed_Index)': final_model.pvalues['Fear_Greed_Index'],
        'p_value (S&P 500)': final_model.pvalues['Close'],
        'ARIMA_AR': arima_model.arparams[0],
        'ARIMA_MA': arima_model.maparams[0],
        'airdrop_date': df['airdrop_date'].iloc[0]
    }

    # Add predicted values and model components to the dataframe
    df['predicted_values'] = predicted_values
    df['residuals'] = y - predicted_values
    for col in X.columns:
        df[f'component_{col}'] = final_model.params[col] * X[col]

    return results, df

# Load airdrop dates
airdrop_dates = pd.read_csv('airdrop_date.csv')
airdrop_dates['Date'] = pd.to_datetime(airdrop_dates['Date'], format='%d/%m/%Y')

# Load market capitalization data
mc_data = pd.read_csv('market_cap.csv')
mc_data['Date'] = pd.to_datetime(mc_data['Date'], format='%d/%m/%Y')
print("Market cap data loaded. Date range:", mc_data['Date'].min(), "to", mc_data['Date'].max())

# Load S&P 500 data
sp500_data = pd.read_csv('sp500.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], format='%d/%m/%Y')
print("S&P 500 data loaded. Date range:", sp500_data['Date'].min(), "to", sp500_data['Date'].max())

# Load Fear and Greed Index data
fgi_data = pd.read_csv('fear_greed_index.csv')
fgi_data['Date'] = pd.to_datetime(fgi_data['Date'], format='%d/%m/%Y')
print("Fear and Greed Index data loaded. Date range:", fgi_data['Date'].min(), "to", fgi_data['Date'].max())

# Analyze each protocol type
protocol_types = [('TVL', 'TVL', 'DeFi'), ('Volume', 'Volume', 'DEX'), ('DAU', 'DAU', 'SocialFi')]

failed_protocols = []  # List to store protocols where analysis failed

for folder, metric, protocol_type in protocol_types:
    print(f"\nAnalysis for {protocol_type}:")
    all_results = []
    
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            protocol_name = file.split('.')[0]
            
            # Check if the protocol name exists in airdrop_dates
            if protocol_name not in airdrop_dates['Name'].values:
                print(f"Warning: Protocol '{protocol_name}' not found in airdrop_date.csv. Skipping...")
                failed_protocols.append((protocol_name, protocol_type, "Not found in airdrop_date.csv"))
                continue
            
            airdrop_date = airdrop_dates[airdrop_dates['Name'] == protocol_name]['Date'].values[0]
            airdrop_date = pd.to_datetime(airdrop_date)
            
            file_path = os.path.join(folder, file)
            
            try:
                df, actual_airdrop_date = prepare_data(file_path, metric, mc_data, sp500_data, fgi_data, airdrop_date)
                
                if df is not None:
                    df['protocol_name'] = protocol_name
                    df['airdrop_date'] = actual_airdrop_date
                    results, df_with_predictions = fit_model(df, metric, protocol_type)
                    all_results.append(results)
                    
                    # Save detailed predictions
                    predictions_folder = 'predicted values'
                    if not os.path.exists(predictions_folder):
                        os.makedirs(predictions_folder)
                    df_with_predictions.to_csv(os.path.join(predictions_folder, f'{protocol_name}_detailed_predictions.csv'), index=False)
                else:
                    failed_protocols.append((protocol_name, protocol_type, "Insufficient data"))
            except Exception as e:
                print(f"Error processing {protocol_name}: {str(e)}")
                failed_protocols.append((protocol_name, protocol_type, str(e)))
                continue

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        print(results_df)
        print(f"\n{protocol_type} Average Effects:")
        print(f"Average immediate effect: {results_df[f'{"α" if protocol_type == "DeFi" else "β" if protocol_type == "DEX" else "γ"}2 (X)'].mean()}")
        print(f"Average slope change: {results_df[f'{"α" if protocol_type == "DeFi" else "β" if protocol_type == "DEX" else "γ"}3 (X_T)'].mean()}")
        print(f"Average market cap effect: {results_df['δ1 (MCt)'].mean()}")
        print(f"Average Fear and Greed Index effect: {results_df['δ2 (Fear_Greed_Index)'].mean()}")
        print(f"Average S&P 500 effect: {results_df['δ3 (S&P 500)'].mean()}")
        print(f"Average ARIMA AR coefficient: {results_df['ARIMA_AR'].mean()}")
        print(f"Average ARIMA MA coefficient: {results_df['ARIMA_MA'].mean()}")
        print(f"Protocols with significant immediate effect: {(results_df['p_value (X)'] < 0.05).sum()}/{len(results_df)}")
        print(f"Protocols with significant slope change: {(results_df['p_value (X_T)'] < 0.05).sum()}/{len(results_df)}")
        print(f"Protocols with significant market cap effect: {(results_df['p_value (MCt)'] < 0.05).sum()}/{len(results_df)}")
        print(f"Protocols with significant Fear and Greed Index effect: {(results_df['p_value (Fear_Greed_Index)'] < 0.05).sum()}/{len(results_df)}")
        print(f"Protocols with significant S&P 500 effect: {(results_df['p_value (S&P 500)'] < 0.05).sum()}/{len(results_df)}")

        # Save results to both 'predictions' and 'result' folders
        results_df.to_csv(os.path.join(predictions_folder, f'{protocol_type.lower()}_analysis_results.csv'), index=False)
        
        result_folder = 'result'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        results_df.to_csv(os.path.join(result_folder, f'{protocol_type.lower()}_analysis_results.csv'), index=False)
    else:
        print(f"No results for {protocol_type}. Skipping...")

print(f"\nResults have been saved in both '{predictions_folder}' and '{result_folder}' folders.")

# Print and save the list of protocols where analysis failed
print("\nProtocols where analysis failed:")
for protocol, protocol_type, reason in failed_protocols:
    print(f"Protocol: {protocol}, Type: {protocol_type}, Reason: {reason}")

failed_df = pd.DataFrame(failed_protocols, columns=['Protocol', 'Type', 'Reason'])
failed_df.to_csv(os.path.join(predictions_folder, 'failed_protocols.csv'), index=False)
failed_df.to_csv(os.path.join(result_folder, 'failed_protocols.csv'), index=False)
print(f"\nList of failed protocols has been saved to both '{predictions_folder}' and '{result_folder}' folders.")