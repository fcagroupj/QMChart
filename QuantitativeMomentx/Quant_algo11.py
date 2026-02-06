#       Quant_algo11.py
# This script calculates the High Quality Momentum (HQM) scores for S&P 500 stocks
#   and saves the results to an Excel file.
#   It uses the stock_data_api module to fetch stock prices.
#   The script computes price returns over multiple time frames,
#   calculates momentum percentiles, and determines the number of shares to buy  
#   based on a fixed portfolio size.
#   Finally, it selects the top 50 stocks with the highest HQM scores.
#   It handles data retrieval, calculations, and output formatting.


import pandas as pd 
import math 
from scipy.stats import percentileofscore as score 
import datetime
from statistics import mean
import os
import glob

def Quant_algo01_Momentum(hqm_dataframe):
    #print(hqm_dataframe)
    ## Calculating Momentum Percentiles
    time_periods = [
                    'One-Year',
                    'Six-Month',
                    'Three-Month',
                    'One-Month'
                    ]

    for row in hqm_dataframe.index:
        for time_period in time_periods:
        
            change_col = f'{time_period} Price Return'
            percentile_col = f'{time_period} Return Percentile'
            if pd.isna(hqm_dataframe.loc[row, change_col]):
                hqm_dataframe.loc[row, change_col] = 0.0

    for row in hqm_dataframe.index:
        for time_period in time_periods:
        
            change_col = f'{time_period} Price Return'
            percentile_col = f'{time_period} Return Percentile'

            hqm_dataframe.loc[row, percentile_col] = score(hqm_dataframe[change_col], hqm_dataframe.loc[row, change_col])/100
    ## Calculating the Number of Shares to Buy
    if(len(hqm_dataframe.index) > 0):
        portfolio_size = 10000000
        position_size = float(portfolio_size) / len(hqm_dataframe.index)
        for i in range(0, len(hqm_dataframe['Ticker'])):
            hqm_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])
            #print(i, hqm_dataframe.loc[i, 'Number of Shares to Buy'], hqm_dataframe['Price'][i], position_size)

    ## Calculating the HQM Score

    for row in hqm_dataframe.index:
        momentum_percentiles = []
        for time_period in time_periods:
            momentum_percentiles.append(hqm_dataframe.loc[row, f'{time_period} Return Percentile'])
        hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)
    return hqm_dataframe
def Quant_save_file(hqm_dataframe2, qm_day2):
    ## Selecting the 50 Best Momentum Stocks
    hqm_dataframe2.sort_values(by = 'HQM Score', ascending = False,inplace= True)
    # hqm_dataframe = hqm_dataframe[:101]     # or save ALL
    hqm_dataframe2.reset_index(drop = True, inplace = True)

    #Print the entire DataFrame    
    # Save hqm_dataframe to an Excel file
    output_filename = f"./reports/hqm_dataframe_{qm_day2}h.xlsx"
    try:
        writer = pd.ExcelWriter(output_filename, engine="xlsxwriter")
        hqm_dataframe2.to_excel(writer, sheet_name="Data", index=False)
        writer.close()
        print(f"Saved Excel file: {output_filename}")
    except Exception as e:
        print(f"Error saving {output_filename}: {e}")

# Helper: read latest HQM score for a ticker from ./reports or ./report
def latest_hqm_score_for(ticker: str):
            try:
                pats = ['./reports/hqm_dataframe_*.xlsx', './report/hqm_dataframe_*.xlsx']
                files = []
                for p in pats:
                    files.extend(glob.glob(p))
                if not files:
                    return None
                latest = sorted(files)[-1]
                df = pd.read_excel(latest)
                # Match columns case-insensitively
                def _match_col(df, target):
                    for c in df.columns:
                        if str(c).strip().lower() == target.lower():
                            return c
                    return None
                col_t = _match_col(df, 'Ticker') or _match_col(df, 'ts_code') or _match_col(df, 'symbol')
                col_s = _match_col(df, 'HQM Score') or _match_col(df, 'hqm score') or _match_col(df, 'hqm')
                if col_t is None or col_s is None:
                    return None
                t = str(ticker or '').strip().upper()
                series_t = df[col_t].astype(str).str.upper().str.strip()
                row = df[series_t == t]
                if row.empty:
                    return None
                v = row.iloc[0][col_s]
                try:
                    return float(v)
                except Exception:
                    return None
            except Exception:
                return None