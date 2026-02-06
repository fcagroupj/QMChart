#       Quant_db_helper16.py
# This script provides a function to save HQM scores into a SQLite database.
#   It reads data from a DataFrame and writes specific fields to the database,
#   while allowing for graceful termination via a stop event.
# 
  
import sqlite3
import pandas as pd
import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH16 = ''
if(_cfg.get('ts_stock_type', 'cn') == 'us'):
    DB_PATH16 = ".\\reports\\quant_stock15.db"
else:
    DB_PATH16 = ".\\reports\\quant_stock06.db"

#########################################################
'''
given hqm_dataframe data read from Quant_algo03_from_db() and save to database table daily_param1
write the records such as ts_code, trade_date, qm_score, Other fields are set to NULL:
trade_date is from in_trade_date
ts_code is from hqm_dataframe['Ticker']
qm_score is from hqm_dataframe['HQM Score']
check stop_event to allow graceful termination during writing database
'''
#########################################################
def Quant_algo05_save_to_db(in_trade_date, hqm_dataframe, stop_event):
    """
    Save HQM scores into the SQLite daily_param1 table.
    - trade_date: from in_trade_date (date or 'YYYYMMDD' string)
    - ts_code: from hqm_dataframe['Ticker']
    - qm_score: from hqm_dataframe['HQM Score']
    Other fields are stored as NULL.
    """
    # Normalize trade_date to 'YYYYMMDD'
    try:
        if hasattr(in_trade_date, "strftime"):
            trade_date = in_trade_date.strftime("%Y%m%d")
        else:
            s = str(in_trade_date or "").strip()
            # Accept 'YYYYMMDD' or 'YYYY-MM-DD'
            trade_date = s.replace("-", "")[:8]
    except Exception:
        trade_date = None

    if not trade_date or len(trade_date) != 8 or not trade_date.isdigit():
        return 0
    if hqm_dataframe is None or len(hqm_dataframe) == 0:
        return 0

    # Robust column matching
    def _match_col(df, target):
        t = str(target).strip().lower()
        for c in df.columns:
            if str(c).strip().lower() == t:
                return c
        return None

    col_t = _match_col(hqm_dataframe, "Ticker")
    col_s = _match_col(hqm_dataframe, "HQM Score")
    if col_t is None or col_s is None:
        return 0

    conn = sqlite3.connect(DB_PATH16, check_same_thread=False)
    cur = conn.cursor()
    
    inserted = 0
    try:
        for _, row in hqm_dataframe.iterrows():
            if stop_event is not None and stop_event.is_set():
                break
            ts_code = str(row[col_t]).strip()
            if not ts_code:
                continue
            val = row[col_s]
            try:
                qm_score = float(val) if pd.notna(val) else None
            except Exception:
                qm_score = None

            cur.execute(
                """
                INSERT OR REPLACE INTO daily_param1
                (ts_code, trade_date, qm_score)
                VALUES (?, ?, ?)
                """,
                (ts_code, trade_date, qm_score)
            )
            inserted += 1
            # Batch commit and optional status
            if inserted % 200 == 0:
                conn.commit()
        conn.commit()
        print(f"  Quant_algo05_save_to_db Saved {inserted} HQM rows for {trade_date}")
    finally:
        conn.close()

    return inserted

# set_stocks_set2()