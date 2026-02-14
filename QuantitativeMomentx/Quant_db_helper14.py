#       Quant_db_helper14.py
# This script provides helper functions for managing US stock minute price data
# in an SQLite database using yfinance.

import sqlite3
import pandas as pd
import yfinance as yf
import config_util as cfg_util
import datetime as _dt
import time as _t
import os
import contextlib

_cfg = cfg_util.read_config()
DB_PATH15 = ".\\reports\\quant_stock15.db"


def _ensure_minute_price_table(conn, table_name: str):
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            trade_time TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            vol REAL,
            amount REAL,
            PRIMARY KEY (trade_time)
        )
        """
    )
    # Ensure backward-compatibility: add missing columns for older tables
    conn.commit()


def updater_minute_thd(ts_code, stop_event, ts_day=None, on_update=None):
    conn = sqlite3.connect(DB_PATH15, check_same_thread=False)
    try:
        table_name = cfg_util._sanitize_table_name(ts_code)
        _ensure_minute_price_table(conn, table_name)

        cur = conn.cursor()
        # Determine starting point from last trade_time; default to 7 days ago
        last_dt = None
        try:
            cur.execute(f"SELECT trade_time FROM {table_name} ORDER BY trade_time DESC LIMIT 1")
            row = cur.fetchone()
            if row and row[0]:
                s = str(row[0]).strip()
                if len(s) >= 12:
                    last_dt = _dt.datetime.strptime(s[:12], '%Y%m%d%H%M')
                elif len(s) == 8:
                    last_dt = _dt.datetime.strptime(s, '%Y%m%d')
                print(f"[updater_minute_thd] Last trade_time for {ts_code} is {last_dt}")
        except Exception:
            last_dt = None
        if last_dt is None:
            last_dt = _dt.datetime.now() - _dt.timedelta(days=7)
        start_dt = last_dt + _dt.timedelta(minutes=1)
        end_dt = start_dt + _dt.timedelta(days=1)
        if(end_dt > _dt.datetime.now()): end_dt = _dt.datetime.now()
        # Continuous fetch loop until stop_event is set
        while not (hasattr(stop_event, 'is_set') and stop_event.is_set()):
            try:
                print(f"[updater_minute_thd] Fetching 1m for {ts_code} from {start_dt} to {end_dt}")
                # Suppress yfinance stdout/stderr messages like '1 Failed download...'
                df = yf.download(
                    ts_code,
                    start=start_dt,
                    end=end_dt,
                    interval='1m',
                    progress=False,
                    prepost=False,
                )
            except Exception:
                df = None

            if df is not None and not df.empty:
                # Normalize possible MultiIndex columns produced by yfinance
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
                out['amount'] = out['Close'] * out['Volume']

                inserted = 0
                for idx, r in out.iterrows():
                    # idx is a pandas Timestamp (possibly timezone-aware)
                    try:
                        trade_time = idx.strftime('%Y%m%d%H%M')
                    except Exception:
                        trade_time = str(idx)

                    def _val(x):
                        try:
                            return float(x)
                        except Exception:
                            return None
                    try:
                        cur.execute(
                            f"""
                            INSERT OR REPLACE INTO {table_name}
                            (trade_time, open, high, low, close, vol, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                trade_time,
                                _val(r['Open']),
                                _val(r['High']),
                                _val(r['Low']),
                                _val(r['Close']),
                                _val(r['Volume']),
                                _val(r['amount']),
                            )
                        )
                    except Exception as e:
                        print(f"[updater_minute_thd] Error inserting {trade_time}: {e}")
                        # Drop and re-create table on schema error
                        try:
                            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                            _ensure_minute_price_table(conn, table_name)
                        except Exception:
                            pass
                        break
                    inserted += 1
                conn.commit()
                print(f"[updater_minute_thd] Inserted {inserted} rows into {table_name}")
                # Notify caller (e.g., chart) to refresh view when new data arrives
                try:
                    if inserted > 0 and callable(on_update):
                        on_update()
                except Exception:
                    pass

            # Advance window by 1 day
            start_dt = end_dt
            end_dt = start_dt + _dt.timedelta(days=1)
            if(end_dt > _dt.datetime.now()): end_dt = _dt.datetime.now()
            if(end_dt - start_dt < _dt.timedelta(minutes=1)):
                # No more data to fetch, wait before retrying
                try:
                    _t.sleep(60)
                except Exception:
                    pass
            # brief pause to avoid tight loop
            try:
                _t.sleep(1)
            except Exception:
                pass
    finally:
        conn.close()
