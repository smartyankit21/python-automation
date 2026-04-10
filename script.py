import os
import warnings
import time
import pandas as pd
import numpy as np
from jugaad_data.nse import stock_df
from datetime import date, timedelta, datetime
import concurrent.futures

warnings.filterwarnings('ignore')


# ---------------- USER INPUT DATE ----------------
def get_target_date():
    return date.today()


# ---------------- STOCK LIST ----------------
def get_tickers_from_file(filepath='companies.csv'):
    if not os.path.exists(filepath):
        print(f'File not found: {filepath}')
        return []
    try:
        df = pd.read_csv(filepath)
        if 'NSE Code' not in df.columns:
            print('NSE Code column missing')
            return []
        tickers = df['NSE Code'].dropna().astype(str).str.strip().tolist()
        return [t for t in tickers if t]
    except Exception as e:
        print(f'CSV Error: {e}')
        return []


# ---------------- FETCH ANALYTICS ----------------
def fetch_stock_analytics(symbol, enddate, retries=3):
    for attempt in range(retries):
        try:
            startdate = enddate - timedelta(days=120)
            df = stock_df(symbol=symbol, from_date=startdate, to_date=enddate, series='EQ')
            if df is None or df.empty or len(df) < 65:
                return None

            required = ['DATE', 'CLOSE', 'PREV. CLOSE', 'VOLUME', 'NO OF TRADES']
            if not all(col in df.columns for col in required):
                return None

            df = df.sort_values('DATE').reset_index(drop=True)
            df['SYMBOL'] = symbol

            df['NO OF TRADES'].replace(0, np.nan, inplace=True)
            df['Price Change'] = (df['CLOSE'] - df['PREV. CLOSE']) / df['PREV. CLOSE'] * 100
            df['per trade volume'] = df['VOLUME'] / df['NO OF TRADES']
            df['PTV SMA20'] = df['per trade volume'].rolling(20).mean()
            df['PTV SMA20'] = df['per trade volume'] / df['PTV SMA20']
            df['PTV SMA20 spike'] = (df['PTV SMA20'] > 1.15).astype(int)
            df['Count PTVSMA 1.15 10d'] = df['PTV SMA20 spike'].rolling(10).sum()
            df['vol ratio currprev'] = df['per trade volume'] / df['per trade volume'].shift(1)
            df['Total Volume Ratio'] = df['VOLUME'] / df['VOLUME'].shift(1)
            df['Vol SMA20'] = df['VOLUME'].rolling(20).mean()
            df['Vol SMA20'] = df['VOLUME'] / df['Vol SMA20']

            volumes = df['VOLUME'].values
            dayssince = []
            for i in range(len(volumes)):
                curr = volumes[i]
                days = np.nan
                for j in range(i - 1, -1, -1):
                    if volumes[j] > curr:
                        days = i - j
                        break
                dayssince.append(days)
            df['Days Since High Vol'] = dayssince

            return df
        except Exception:
            time.sleep(2)
    return None


# ---------------- WORKER FUNCTION ----------------
def process_single_ticker(symbol, targetdate):
    data = fetch_stock_analytics(symbol, targetdate)
    time.sleep(np.random.uniform(0.8, 1.5))
    if data is not None and not data.empty:
        last60days = data.tail(60)
        maxptv60d = last60days['per trade volume'].max()
        latestrow = last60days.iloc[-1]
        latestptv = latestrow['per trade volume']
        latestptvsmaratio = latestrow['PTV SMA20']

        if latestptv >= maxptv60d and latestptvsmaratio > 2:
            return data.tail(1), f'Processed {symbol} | Added (60d High + Ratio > 2)'
        else:
            return None, f'Skipped {symbol} | Failed 60d High or Ratio > 2'
    else:
        return None, f'Skipped {symbol} | Failed to fetch or missing data'


# ---------------- MAIN ----------------
if __name__ == '__main__':
    tickers = get_tickers_from_file('companies.csv')
    if not tickers:
        print('No tickers found')
        exit()

    targetdate = get_target_date()
    print(f'{len(tickers)} stocks using multithreading...')

    master = []
    completedcount = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futuretosymbol = {executor.submit(process_single_ticker, symbol, targetdate): symbol for symbol in tickers}
        for future in concurrent.futures.as_completed(futuretosymbol):
            completedcount += 1
            symbol = futuretosymbol[future]
            try:
                datarow, message = future.result()
                print(f'{completedcount}/{len(tickers)} {message}')
                if datarow is not None:
                    master.append(datarow)
            except Exception as e:
                print(f'{completedcount}/{len(tickers)} Error processing {symbol}: {e}')

    if not master:
        print('No data processed or no stocks met the 60-day highest PTV and Ratio > 2 criteria.')
        exit()

    fulldf = pd.concat(master, ignore_index=True)
    actualdate = fulldf['DATE'].max()
    actualdatestr = str(actualdate)[:10]

    finaldf = fulldf[fulldf['DATE'] == actualdate].copy()
    columns = ['SYMBOL', 'DATE', 'Price Change', 'VOLUME', 'NO OF TRADES', 'per trade volume',
               'PTV SMA20', 'Count PTVSMA 1.15 10d', 'vol ratio currprev',
               'Total Volume Ratio', 'Vol SMA20', 'Days Since High Vol']
    finaldf = finaldf[[c for c in columns if c in finaldf.columns]]

    if 'PTV SMA20' in finaldf.columns:
        finaldf = finaldf.sort_values(by='PTV SMA20', ascending=False)

    numericcols = finaldf.select_dtypes(include=np.number).columns
    finaldf[numericcols] = finaldf[numericcols].round(2)

    filename = f'BreakoutAnalysis60D_{actualdatestr}.csv'
    finaldf.to_csv(filename, index=False)

    print('=' * 60)
    print(f'DONE for date: {actualdatestr}')
    print(f'Saved: {filename}')
    print(f'Criteria: PTV is 60-day high AND PTV/SMA20 > 2')
    print('=' * 60)
