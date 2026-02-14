#       Quant_db_helper15.py
# This script provides helper functions for managing stock price data in USA stock market
#   in an SQLite database. It includes functions to retrieve stock lists,
#   get the last trade date for a stock, and update stock price data
#   using the yfinance API. The script handles database connections,
#   data fetching, and insertion of new price records.
# 
  
import sqlite3
import pandas as pd
import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH15 = ".\\reports\\quant_stock15.db"

#########################################################
def set_stocks_set2():
    # Build US stock list (S&P 500) and save to CSV.
    # Columns: Ticker, Name, Sector, Industry, Country
    # Source: Wikipedia S&P 500 components; yfinance used later for price updates.
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    out_path = './QuantitativeMomentx/us_a_share_list500.csv'
    try:
        # Read the constituents table from Wikipedia
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        html = requests.get(url, headers=headers, timeout=20).text
        tables = pd.read_html(html, match='Symbol')
        if not tables:
            raise RuntimeError('No S&P 500 table found')
        df = tables[0]
        # Normalize column names to expected schema
        col_map = {
            'Symbol': 'Ticker',
            'Security': 'Name',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry',
        }
        df = df.rename(columns=col_map)
        needed = ['Ticker', 'Name', 'Sector', 'Industry']
        for c in needed:
            if c not in df.columns:
                raise RuntimeError(f'Missing column {c} in Wikipedia table')
        out_df = df[needed].copy()
        # Clean tickers and add Country
        out_df['Ticker'] = out_df['Ticker'].astype(str).str.strip().str.upper()
        out_df['Name'] = out_df['Name'].astype(str).str.strip()
        out_df['Sector'] = out_df['Sector'].astype(str).str.strip()
        out_df['Industry'] = out_df['Industry'].astype(str).str.strip()
        out_df['Country'] = 'United States'

        # Save to CSV
        out_df.to_csv(out_path, index=False)
        print(f"[set_stocks_set2] Saved S&P 500 list: {out_path} (rows={len(out_df)})")
        return out_path
    except Exception as e:
        print('[set_stocks_set2] Failed to build S&P 500 list:', e)
        return None
def get_stocks_set2():
    return pd.read_csv('./QuantitativeMomentx/us_a_share_list500.csv', dtype=str)

def get_last_trade_date(conn, ts_code):
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(trade_date)
        FROM daily_price
        WHERE ts_code = ?
    """, (ts_code,))
    row = cur.fetchone()
    return row[0]  # may be None

#########################################################
import yfinance as yf
from datetime import date, timedelta
import time
import numpy as np

D_MAX_SPEED = float(_cfg.get('D_MAX_SPEED', "1250")) / 1000

def _ensure_daily_price_table(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_price (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
        """
    )
    conn.commit()

def updater_daily_thd(ts_code, stop_event, on_update=None):
    conn = sqlite3.connect(DB_PATH15, check_same_thread=False)
    _ensure_daily_price_table(conn)
    
    #Import list of stocks
    stocks_sp = get_stocks_set2()['Ticker']
    stock_i = int(_cfg.get('D_STOCK_UPDATING', "0"))   
    print(f"[updater_daily_thd] updating LOOP 15 {{{stock_i}}}------------------")
    while not stop_event.is_set():
        last_date = get_last_trade_date(conn, ts_code)

        if last_date:
            start = (date.fromisoformat(last_date[:4]+"-"+last_date[4:6]+"-"+last_date[6:]) 
                     + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start = (date.today() - timedelta(days=23*365)).strftime("%Y%m%d")
        # get all data in 23 years
        # start = (date.today() - timedelta(days=23*365)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        # print(f"Updating {ts_code} from {start} to {end}")

        if start <= end:
            try:
                # Convert to YYYY-MM-DD for yfinance
                start_dt = date.fromisoformat(start[:4]+"-"+start[4:6]+"-"+start[6:])
                end_dt = date.fromisoformat(end[:4]+"-"+end[4:6]+"-"+end[6:]) + timedelta(days=1)
                try:
                    df = yf.download(
                        ts_code,
                        start=start_dt.strftime('%Y-%m-%d'),
                        end=end_dt.strftime('%Y-%m-%d'),
                        interval='1d',
                        progress=False,
                    )
                except Exception:
                    df = None
                if df is not None and not df.empty:
                    # Normalize possible MultiIndex columns
                    import pandas as pd
                    def _col(name):
                        x = df[name]
                        return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
                    out = pd.DataFrame({
                        'Open': _col('Open'),
                        'High': _col('High'),
                        'Low': _col('Low'),
                        'Close': _col('Close'),
                        'Volume': _col('Volume'),
                    })
                    out['pre_close'] = out['Close'].shift(1)
                    out['change'] = out['Close'] - out['pre_close']
                    with np.errstate(divide='ignore', invalid='ignore'):
                        out['pct_chg'] = (out['change'] / out['pre_close']) * 100.0
                    out['amount'] = out['Close'] * out['Volume']

                    cur = conn.cursor()
                    count = 0
                    for idx, r in out.iterrows():
                        trade_date = idx.strftime('%Y%m%d') if hasattr(idx, 'strftime') else str(idx)[:10].replace('-', '')
                        def _val(x):
                            try:
                                return float(x)
                            except Exception:
                                return None
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO daily_price
                            (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ts_code,
                                trade_date,
                                _val(r['Open']),
                                _val(r['High']),
                                _val(r['Low']),
                                _val(r['Close']),
                                _val(r['pre_close']),
                                _val(r['change']),
                                _val(r['pct_chg']),
                                _val(r['Volume']),
                                _val(r['amount'])
                            )
                        )
                        count += 1
                    conn.commit()
                    if count > 0:
                        try:
                            if callable(on_update):
                                on_update()
                        except Exception:
                            pass
                        # cfg_util.o_status_line(f"Pulled {stock_i} / {len(stocks_sp)} stocks ... {ts_code} {start}->{end} rows:{count}")
            except Exception as e:
                pass
                # print(f"IgnoredIt {stock_i} / {len(stocks_sp)} stocks ...", ts_code, start, end)
            
        else:
            pass
            # print(f"AlreadyIs {stock_i} / {len(stocks_sp)} stocks ...", ts_code, start, end)
        time.sleep(D_MAX_SPEED)  # update every max time
        if(stock_i < len(stocks_sp)):
            ts_code = stocks_sp[stock_i]
            stock_i += 1
        else:
            stock_i = 0
            ts_code = stocks_sp[stock_i]
            print(f"[updater_daily_thd] updating LOOP 15 {{{stock_i}}}------------------")
        if(stock_i % 10 == 0): cfg_util.write_config({'D_STOCK_UPDATING': str(stock_i)})

    conn.close()

def update_ticker_once(ts_code: str):
    """Update a single ticker using yfinance; returns number of rows inserted."""
    conn = sqlite3.connect(DB_PATH15, check_same_thread=False)
    _ensure_daily_price_table(conn)
    last_date = get_last_trade_date(conn, ts_code)
    if last_date:
        start_dt = date.fromisoformat(f"{last_date[:4]}-{last_date[4:6]}-{last_date[6:]}") + timedelta(days=1)
    else:
        start_dt = date.today() - timedelta(days=365)
    end_dt = date.today() + timedelta(days=1)
    try:
        df = yf.download(
            ts_code,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False,
        )
    except Exception:
        df = None

    inserted = 0
    if df is not None and not df.empty:
        # Normalize possible MultiIndex columns
        import pandas as pd
        def _col(name):
            x = df[name]
            return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
        out = pd.DataFrame({
            'Open': _col('Open'),
            'High': _col('High'),
            'Low': _col('Low'),
            'Close': _col('Close'),
            'Volume': _col('Volume'),
        })
        out['pre_close'] = out['Close'].shift(1)
        out['change'] = out['Close'] - out['pre_close']
        with np.errstate(divide='ignore', invalid='ignore'):
            out['pct_chg'] = (out['change'] / out['pre_close']) * 100.0
        out['amount'] = out['Close'] * out['Volume']

        cur = conn.cursor()
        for idx, r in out.iterrows():
            trade_date = idx.strftime('%Y%m%d') if hasattr(idx, 'strftime') else str(idx)[:10].replace('-', '')
            def _val(x):
                try:
                    return float(x)
                except Exception:
                    return None
            cur.execute(
                """
                INSERT OR REPLACE INTO daily_price
                (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    _val(r['Open']),
                    _val(r['High']),
                    _val(r['Low']),
                    _val(r['Close']),
                    _val(r['pre_close']),
                    _val(r['change']),
                    _val(r['pct_chg']),
                    _val(r['Volume']),
                    _val(r['amount'])
                )
            )
            inserted += 1
        conn.commit()
    conn.close()
    return inserted
# set_stocks_set2()