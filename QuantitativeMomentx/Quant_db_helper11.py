#       Quant_db_helper11.py
# This script provides helper functions for managing stock price data in China stock markets
#   in an SQLite database. It includes functions to retrieve stock lists,
#   get the last trade date for a stock, and update stock price data
#   using the Tushare API. The script handles database connections,
#   data fetching, and insertion of new price records.
# 
  
import sqlite3
import pandas as pd
import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH11 = ".\\reports\\quant_stock06.db"


#########################################################
def get_stocks_set2():
    return pd.read_csv('./QuantitativeMomentx/china_a_share_list6k.csv', dtype=str)

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
import tushare as ts
from datetime import date, timedelta
import time

# tushare.pro API token and speed limit
D_TUSHARE_TOKEN = _cfg.get('D_TUSHARE_TOKEN', "1234567890abc")
D_MAX_SPEED = float(_cfg.get('D_MAX_SPEED', "1250")) / 1000

def updater_daily_thd(ts_code, stop_event, on_update=None):
    conn = sqlite3.connect(DB_PATH11, check_same_thread=False)
    ts.set_token(D_TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    #Import list of stocks
    stocks_sp = get_stocks_set2()['Ticker']
    stock_i = int(_cfg.get('D_STOCK_UPDATING', "0"))   
    print(f'-------------Updating LOOP 11 {{{stock_i}}}------------------')
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

        if start <= end:
            try:
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    fields="trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
                )
                cur = conn.cursor()
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT OR REPLACE INTO daily_price
                        (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ts_code, row["trade_date"], float(row["open"]), float(row["high"]), float(row["low"]), 
                          float(row["close"]), float(row["pre_close"]), float(row["change"]), float(row["pct_chg"]), float(row["vol"]), float(row["amount"])))
                conn.commit()
                row_count = len(df)
                if row_count > 0:
                    try:
                        if callable(on_update):
                            on_update()
                    except Exception:
                        pass
                    cfg_util.o_status_line(f"Pulled {stock_i} / {len(stocks_sp)} stocks ... {ts_code} {start}->{end} rows:{row_count}")
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
            print(f'-------------Updating LOOP 11 {{{stock_i}}}------------------')
        if(stock_i % 10 == 0): cfg_util.write_config({'D_STOCK_UPDATING': str(stock_i)})

    conn.close()

def update_ticker_once(ts_code: str) -> int:
    """Update a single CN ticker using Tushare; returns number of rows inserted."""
    conn = sqlite3.connect(DB_PATH11, check_same_thread=False)
    ts.set_token(D_TUSHARE_TOKEN)
    pro = ts.pro_api()
    last_date = get_last_trade_date(conn, ts_code)
    if last_date:
        start = (date.fromisoformat(last_date[:4]+"-"+last_date[4:6]+"-"+last_date[6:]) + timedelta(days=1)).strftime("%Y%m%d")
    else:
        start = (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    inserted = 0
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        )
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT OR REPLACE INTO daily_price
                (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    row["trade_date"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["pre_close"]),
                    float(row["change"]),
                    float(row["pct_chg"]),
                    float(row["vol"]),
                    float(row["amount"]),
                ),
            )
            inserted += 1
        conn.commit()
    except Exception:
        pass
    conn.close()
    return inserted
