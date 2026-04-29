import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
from datetime import date

def get_stock_data(stock_id: str, time_period: int):
    end_time = datetime.strptime('2025-12-31', '%Y-%m-%d')
    time_delta_obj = timedelta(days=365 * time_period)
    start_time = (end_time - time_delta_obj).strftime('%Y-%m-%d')
    data = yf.download(stock_id, start=start_time, end=end_time.strftime('%Y-%m-%d'), interval="1d").dropna()
    return data

def get_one_year(stock_id: str):
    # Get the stock data from Yahoo Finance from one year ago to today  
    today = date.today()
    time_delta = timedelta(days=365)
    one_year_ago = today - time_delta
    data = yf.download(f"{stock_id}.TW", start=one_year_ago, end=today, interval="1d")
    return data


def get_three_year(stock_id: str):
    # Get the stock data from Yahoo Finance from three years ago to today
    today = date.today()
    time_delta = timedelta(days=365 * 3)
    three_years_ago = today - time_delta
    data = yf.download(f"{stock_id}.TW", start=three_years_ago, end=today, interval="1d").dropna()
    return data

def get_five_year(stock_id: str):
    # Get the stock data from Yahoo Finance from five years ago to today
    today = date.today()
    time_delta = timedelta(days=365 * 5)
    five_years_ago = today - time_delta
    data = yf.download(f"{stock_id}.TW", start=five_years_ago, end=today, interval="1d").dropna()

    return data

def get_ten_year(stock_id: str):
    # Get the stock data from Yahoo Finance from ten years ago to today
    now = datetime.now()
    ten_years_ago = now - timedelta(days=3652)
    data = yf.download(f"{stock_id}.TW", start=ten_years_ago, end=now, interval="1d").dropna()

    return data

def store_to_csv(stock_id: str, time_period: str, data: dict):
    # Store the data to a CSV file
    os.makedirs('stock_data', exist_ok=True)

    if isinstance(data.columns, pd.MultiIndex):
        # Extract the ticker from MultiIndex and flatten
        data.columns = [col[0] for col in data.columns]
    
    # Reset index to make Date a regular column
    data = data.reset_index()
    
    # Normalize column names to consistent casing but keep original names for output
    cols_lower = {c.lower(): c for c in data.columns}

    # Ensure 'Date' column exists (try to find a date-like column)
    if 'date' not in cols_lower:
        # look for any column name containing 'date'
        date_col = None
        for c in data.columns:
            if 'date' in str(c).lower():
                date_col = c
                break
        if date_col is not None:
            data = data.rename(columns={date_col: 'Date'})
        else:
            # if no date-like column, keep existing columns and add empty Date
            data['Date'] = pd.NaT

    # Ensure 'Adj Close' exists, prefer existing 'Adj Close' then 'Close'
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            data['Adj Close'] = np.nan

    # Desired output columns (exact names). We'll populate by name, not by positional ordering.
    out_columns = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']

    out_df = pd.DataFrame()
    for col in out_columns:
        if col in data.columns:
            out_df[col] = data[col]
        else:
            # Try to find columns case-insensitively
            match = None
            for c in data.columns:
                if c.lower() == col.lower():
                    match = c
                    break
            if match is not None:
                out_df[col] = data[match]
            else:
                # As a last resort, fill with NaN (or appropriate default)
                if col == 'Date':
                    out_df[col] = pd.NaT
                else:
                    out_df[col] = np.nan

    # Save to CSV using exact column names above
    filename = f'stock_data/{stock_id}_{time_period}.csv'
    out_df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python stock_data.py <time_period>")
        print("Time periods: 1year, 3year, 5year, 10year")
        sys.exit(1)

    stock_id = "^GSPC"
    output_stock_id = "S&P500"
    time_period = sys.argv[1].lower()
    period_year_mapping = {
        "1year": 1,
        "3year": 3,
        "5year": 5,
        "10year": 10
    }

    if time_period not in period_year_mapping:
        print(f"Invalid time period. Choose from: {', '.join(period_year_mapping.keys())}")
        sys.exit(1)
    try:
        data = get_stock_data(stock_id, period_year_mapping[time_period])
        store_to_csv(output_stock_id, time_period, data)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)