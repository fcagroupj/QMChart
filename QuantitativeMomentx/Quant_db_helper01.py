#       Quant_db_helper01.py
# This script initializes the SQLite database for storing stock price data.
# It creates a table named 'daily_price' with columns for stock code, trade date
# and various price metrics. The table uses a composite primary key on stock code
# and trade date to ensure uniqueness of records.
#
  
import sqlite3
import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH01 = ''
if(_cfg.get('ts_stock_type', 'cn') == 'us'):
    DB_PATH01 = _cfg.get('DB_PATH', ".\\reports\\quant_stock15.db")
else:
    DB_PATH01 = _cfg.get('DB_PATH', ".\\reports\\quant_stock06.db")

def init_db():
    conn = sqlite3.connect(DB_PATH01)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")

    cur.execute("""
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
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_param1 (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            ma5 REAL,
            ma10 REAL,
            ma20 REAL,
            ma30 REAL,
            ma60 REAL,
            ma114 REAL,
            ma_a REAL,
            ma_b REAL,
            ma_c REAL,
            ma_d REAL,
            macd REAL,
            dif REAL,
            dea REAL,
            k REAL,
            d REAL,
            j REAL,
            qm_score REAL,
            qm_2 REAL,
            qm_3 REAL,
            qm_4 REAL,
            qm_5 REAL,
            qm_6 REAL,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)

    conn.commit()
    conn.close()