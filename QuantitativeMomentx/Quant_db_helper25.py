#       Quant_db_helper25.py
# This script calculates the High Quality Momentum (HQM) scores for S&P 500 stocks
#   and saves the results to an Excel file.
#   It uses the stock_data_api module to fetch stock prices.
#   The script computes price returns over multiple time frames,
#   calculates momentum percentiles, and determines the number of shares to buy  
#   based on a fixed portfolio size.
#   Finally, it selects the top 50 stocks with the highest HQM scores.
#   It handles data retrieval, calculations, and output formatting.
  
import sqlite3
import time
import pandas as pd
import numpy as np
import Quant_algo11 as qalg11
import Quant_db_helper16 as qalg16
import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH23 = ''
if(_cfg.get('ts_stock_type', 'cn') == 'us'):
    DB_PATH23 = ".\\reports\\quant_stock15.db"
else:
    DB_PATH23 = ".\\reports\\quant_stock06.db"

#########################################################
from datetime import date, timedelta


def Quant_algo03_read_from_db(qm_day, stop_event):
    # Normalize qm_day to a datetime.date
                # already a date/datetime
    if(qm_day is None): qm_day = date.today()
    # Initialize DataFrame with explicit dtypes to avoid concat/NA dtype warnings
    hqm_dtype_map = {
        'Ticker': 'string',
        'Name': 'string',
        'Price': 'float64',
        'Number of Shares to Buy': 'Int64',  # nullable integer
        'One-Year Price Return': 'float64',
        'One-Year Return Percentile': 'float64',
        'Six-Month Price Return': 'float64',
        'Six-Month Return Percentile': 'float64',
        'Three-Month Price Return': 'float64',
        'Three-Month Return Percentile': 'float64',
        'One-Month Price Return': 'float64',
        'One-Month Return Percentile': 'float64',
        'HQM Score': 'float64',
    }
    hqm_dataframe = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in hqm_dtype_map.items()})
    conn = sqlite3.connect(DB_PATH23, check_same_thread=False)

    #Import list of stocks
    stocks_sp = cfg_util.get_stocks_set3()['Ticker']
    stocks_name = cfg_util.get_stocks_set3()['name']
    stock_i = 0
    print('  Reading LOOP for HQM data from database ...')
    while not stop_event.is_set():
        if(stock_i < len(stocks_sp)):
            ts_code = stocks_sp[stock_i]
            ts_name = stocks_name[stock_i]
            stock_i += 1
        else:
            break

        cur = conn.cursor()
        cur.execute("""
            SELECT trade_date, close
            FROM daily_price
            WHERE ts_code = ?
            ORDER BY trade_date DESC
        """, (ts_code,))

        rows = cur.fetchall()
        if rows:
            def nearest(days):
                target = (qm_day - timedelta(days=days)).strftime("%Y%m%d")
                for d, c in rows:
                    if d <= target:
                        return c
                return None
            price_0d, price_1m, price_3m, price_6m, price_12m = nearest(0), nearest(30), nearest(90), nearest(180), nearest(365)
            if(price_0d is None): continue
            if(price_1m is None): continue
            if(price_3m is None): continue
            if(price_6m is None): continue
            if(price_12m is None): continue
            hqm_dataframe.loc[len(hqm_dataframe)] = [
                ts_code,
                ts_name,
                price_0d,
                pd.NA,
                price_12m,
                np.nan,
                price_6m,
                np.nan,
                price_3m,
                np.nan,
                price_1m,
                np.nan,
                np.nan
            ]
            if(len(hqm_dataframe) % 500 == 0 or len(hqm_dataframe) == 1):
                # o_msg = f"Read {len(hqm_dataframe)} / {len(stocks_sp)} stocks ...", ts_code, price_0d, price_1m, price_3m, price_6m, price_12m
                # cfg_util.o_status_line(o_msg)
                pass
            
        time.sleep(0.01)  # read every 10 ms

    conn.close()
    return hqm_dataframe
# High Quality Momentum (HQM) Strategy Implementation
# this function calculates the HQM scores for the given DataFrame from the database
# starting from 20040101
def reader_db_qm_thd(qm_day2, stop_event1): 
    # update today at first
    day_qm_end = date.today() 
    hqm_dataframe2 = Quant_algo03_read_from_db(day_qm_end, stop_event1)
    if(len(hqm_dataframe2) >= 100 and (not stop_event1.is_set())):
        hqm_dataframe2 = qalg11.Quant_algo01_Momentum(hqm_dataframe2)
        qalg11.Quant_save_file(hqm_dataframe2, day_qm_end) # save QualtM DataFrame to Excel

    # update history data before today
    day_qm_start = _cfg.get('D_QM_UPDATING', "20040101")
    # loop from day_qm_start to day_qm_end
    day_qm = date(int(day_qm_start[0:4]), int(day_qm_start[4:6]), int(day_qm_start[6:8]))
    start_dt = date(2004, 1, 1)
    while((day_qm <= day_qm_end) and (not stop_event1.is_set())):
        qm_day1 = day_qm   # .strftime("%Y%m%d")
        #calculate the progress by qm_day1 - day_qm_start divided by day_qm_end - day_qm_start
        try:
            elapsed_days = (qm_day1 - start_dt).days
            total_days = (day_qm_end - start_dt).days
            if total_days <= 0:
                progress_pct = 0.0
            else:
                progress_pct = max(0.0, min(100.0, (elapsed_days * 100.0) / float(total_days)))
        except Exception:
            progress_pct = 0.0

        print(f'-------------Calculating HQM for date {{{qm_day1}}}------------------  [{progress_pct:.1f}%]')
        hqm_dataframe1 = Quant_algo03_read_from_db(qm_day1, stop_event1)
        if(len(hqm_dataframe1) >= 100 and (not stop_event1.is_set())):
            print(f'  Quant_algo03_read_from_db {{{len(hqm_dataframe1)}}}  data for date {{{qm_day1}}}')
            
            hqm_dataframe1 = qalg11.Quant_algo01_Momentum(hqm_dataframe1)
            inserted = qalg16.Quant_algo05_save_to_db(qm_day1, hqm_dataframe1, stop_event1)
            if(inserted > 100 and (not stop_event1.is_set())):
                cfg_util.write_config({'D_QM_UPDATING': qm_day1.strftime("%Y%m%d")})
                # check qm_day1 is today, if yes, then break
                if(qm_day1 >= date.today()):    
                    qalg11.Quant_save_file(hqm_dataframe1, qm_day1) # save QualtM DataFrame to Excel
            else:
                print(f' {{{inserted}}} data is saved for date {{{qm_day1}}}, skip to the next day')
        else:
            print(f'  Not enough data {{{len(hqm_dataframe1)}}} for date {{{qm_day1}}}, skip to the next day')
        day_qm = day_qm + timedelta(days=1)
    # all are finished
    stop_event1.set()   # 🔔 SIGNAL: task finished


# Load price data from DB and prepare rows/closes for plotting
def load_ts_daily_db_data(ts_code):
    print(f"[load_ts_daily_db_data] Loading data for {ts_code} from database...")
    resolved_code, resolved_name = cfg_util._resolve_ts_code(ts_code)

    conn = sqlite3.connect(DB_PATH23)
    cur = conn.cursor()
    try:
            cur.execute(
                f"""
                SELECT trade_date, high, low, close, vol
                FROM daily_price
                WHERE ts_code = ?
                ORDER BY trade_date ASC
                """,
                (resolved_code,),
            )
            raw_rows = cur.fetchall()
    except Exception:
        raw_rows = []
    conn.close()

    if not raw_rows:
        return None

    closes = []
    highs = []
    lows = []
    rows = []  # keep (date, close) for existing interactions
    volumes = []
    for d, h, l, c, v in raw_rows:
            if c is None:
                continue
            try:
                closes.append(float(c))
                highs.append(float(h) if h is not None else np.nan)
                lows.append(float(l) if l is not None else np.nan)
                rows.append((d, c))
                volumes.append(float(v) if v is not None else np.nan)
            except Exception:
                continue

    if len(closes) < 2:
        return None

    # ---- Precompute indicators: MA, DIF/DEA/MACD, K/D/J ----
    n = len(closes)
    # Moving averages for standard windows including 114
    ma_windows = [5, 10, 20, 30, 60, 114]
    ma_map = {}
    try:
        arr = np.array(closes, dtype=float)
        for w in ma_windows:
            if n >= w:
                kernel = np.ones(w, dtype=float) / w
                valid = np.convolve(arr, kernel, mode='valid')
                # pad front with NaN to align length
                pad = [np.nan] * (w - 1)
                ma_map[w] = pad + valid.tolist()
            else:
                ma_map[w] = [np.nan] * n
    except Exception:
        ma_map = {w: [np.nan] * n for w in ma_windows}

    # MACD components (DIF, DEA) and histogram (MACD)
    try:
        s = pd.Series(closes, dtype=float)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        dif_vals = (ema12 - ema26).tolist()
        dea_vals = pd.Series(dif_vals, dtype=float).ewm(span=9, adjust=False).mean().tolist()
        macd_vals = [((d if d is not None else np.nan) - (e if e is not None else np.nan)) * 2.0
                     if (d is not None and e is not None) else np.nan
                     for d, e in zip(dif_vals, dea_vals)]
    except Exception:
        dif_vals = [np.nan] * n
        dea_vals = [np.nan] * n
        macd_vals = [np.nan] * n

    # KDJ (period=9)
    try:
        highs_arr = np.array(highs, dtype=float)
        lows_arr = np.array(lows, dtype=float)
        closes_arr = np.array(closes, dtype=float)
        period = 9
        rsv = np.full(n, np.nan, dtype=float)
        for i in range(n):
            start_i = max(0, i - period + 1)
            hi = np.nanmax(highs_arr[start_i:i+1])
            lo = np.nanmin(lows_arr[start_i:i+1])
            if np.isnan(hi) or np.isnan(lo) or hi == lo:
                rsv[i] = np.nan
            else:
                rsv[i] = (closes_arr[i] - lo) / (hi - lo) * 100.0
        k_vals = np.full(n, np.nan, dtype=float)
        d_vals = np.full(n, np.nan, dtype=float)
        j_vals = np.full(n, np.nan, dtype=float)
        k_prev = 50.0
        d_prev = 50.0
        for i in range(n):
            if np.isnan(rsv[i]):
                k_vals[i] = k_prev
            else:
                k_vals[i] = (2.0/3.0) * k_prev + (1.0/3.0) * rsv[i]
            d_vals[i] = (2.0/3.0) * d_prev + (1.0/3.0) * k_vals[i]
            j_vals[i] = 3.0 * k_vals[i] - 2.0 * d_vals[i]
            k_prev = k_vals[i]
            d_prev = d_vals[i]
        k_vals = k_vals.tolist()
        d_vals = d_vals.tolist()
        j_vals = j_vals.tolist()
    except Exception:
        k_vals = [np.nan] * n
        d_vals = [np.nan] * n
        j_vals = [np.nan] * n
    # ---- RSI(14) calculation ----
    try:
        s_close = pd.Series(closes, dtype=float)
        delta = s_close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14.0, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi14_vals = (100.0 - (100.0 / (1.0 + rs))).tolist()
    except Exception:
        rsi14_vals = [np.nan] * n
    # ---- OBV calculation moved here ----
    try:
        vols_arr = np.array(volumes, dtype=float)
        if len(vols_arr) != n:
            vols_arr = np.full(n, np.nan, dtype=float)
        closes_arr = np.array(closes, dtype=float)
        obv = np.zeros(n, dtype=float)
        for i in range(1, n):
            v = vols_arr[i]
            if not np.isfinite(v):
                v = 0.0
            if np.isnan(closes_arr[i-1]) or np.isnan(closes_arr[i]):
                obv[i] = obv[i-1]
            elif closes_arr[i] > closes_arr[i-1]:
                obv[i] = obv[i-1] + v
            elif closes_arr[i] < closes_arr[i-1]:
                obv[i] = obv[i-1] - v
            else:
                obv[i] = obv[i-1]
        obv_vals = obv.tolist()
        # 20-day simple moving average of OBV (smoothing)
        w = 20
        try:
            if n >= w:
                kernel = np.ones(w, dtype=float) / float(w)
                valid = np.convolve(obv, kernel, mode='valid')
                pad = [np.nan] * (w - 1)
                obv_ma20_vals = pad + valid.tolist()
            else:
                obv_ma20_vals = [np.nan] * n
        except Exception:
            obv_ma20_vals = [np.nan] * n
    except Exception:
        obv_vals = [np.nan] * n
        obv_ma20_vals = [np.nan] * n
    # look up qm_score from database table daily_param1 with resolved_code and trade_date, putting into a list qm_scores
    qm_scores = []
    try:
        conn = sqlite3.connect(DB_PATH23)
        cur = conn.cursor()
        for d, _ in rows:
            cur.execute(
                """
                SELECT qm_score
                FROM daily_param1
                WHERE ts_code = ? AND trade_date = ?
                """,
                (resolved_code, d),
            )
            result = cur.fetchone()
            if result and result[0] is not None:
                qm_scores.append(float(result[0]))
            else:
                qm_scores.append(np.nan)
        conn.close()
    except Exception:
        qm_scores = [np.nan] * n
    print(f"[load_ts_daily_db_data] Loaded {len(closes)} data points for {resolved_code}", rows[0][0], "to", rows[-1][0])
    return {
        "rows": rows,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "ma": ma_map,
        "dif": dif_vals,
        "dea": dea_vals,
        "macd": macd_vals,
        "k": k_vals,
        "d": d_vals,
        "j": j_vals,
        "rsi14": rsi14_vals,
        "resolved_ticker": resolved_code,
        "resolved_name": resolved_name,
        "qm_scores": qm_scores,
        "obv": obv_vals,
        "obv_ma20": obv_ma20_vals,
    }

# Delete the most recent N daily rows (price + params) for a ticker
def delete_last_n_days(ts_code: str, n: int = 10) -> int:
    """
    Delete the most recent n daily rows for the given ticker from
    tables daily_price and daily_param1. Returns the number of deleted days.
    """
    if n is None:
        n = 10
    try:
        resolved_code, _ = cfg_util._resolve_ts_code(ts_code)
    except Exception:
        resolved_code = ts_code

    try:
        conn = sqlite3.connect(DB_PATH23, check_same_thread=False)
        cur = conn.cursor()
        # Fetch latest n dates for this ticker
        cur.execute(
            """
            SELECT trade_date
            FROM daily_price
            WHERE ts_code = ?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            (resolved_code, int(n)),
        )
        recent = cur.fetchall()
        dates = [r[0] for r in recent]
        if not dates:
            conn.close()
            return 0

        # Delete corresponding rows in both tables
        cur.executemany(
            "DELETE FROM daily_price WHERE ts_code = ? AND trade_date = ?",
            [(resolved_code, d) for d in dates],
        )
        try:
            cur.executemany(
                "DELETE FROM daily_param1 WHERE ts_code = ? AND trade_date = ?",
                [(resolved_code, d) for d in dates],
            )
        except Exception:
            # If params table missing, ignore
            pass

        conn.commit()
        deleted = len(dates)
        print(f"[delete_last_n_days] Deleted {deleted} days for {resolved_code}: {dates[-1]} .. {dates[0]}")
        conn.close()
        return deleted
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print(f"[delete_last_n_days] Error: {e}")
        return 0

# get stock data report from current database
# read table of daily_price in database, return the number of rows of each stock, also including starting date and ending date
# save the report to a excel file
def get_stock_data_report():
    # Connect to SQLite and aggregate daily_price stats
    try:
        conn = sqlite3.connect(DB_PATH23)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts_code,
                   COUNT(*) AS row_count,
                   MIN(trade_date) AS start_date,
                   MAX(trade_date) AS end_date
            FROM daily_price
            GROUP BY ts_code
            ORDER BY ts_code
            """
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print("[get_stock_data_report] DB query failed:", e)
        return None

    if not rows:
        print("[get_stock_data_report] No data found in daily_price.")
        return None

    # Build DataFrame
    report_df = pd.DataFrame(rows, columns=[
        'Ticker', 'Row Count', 'Start Date', 'End Date'
    ])

    # Attach stock names when available
    try:
        stocks_df = cfg_util.get_stocks_set3()
        if {'Ticker', 'name'}.issubset(stocks_df.columns):
            name_map = dict(zip(stocks_df['Ticker'].astype(str), stocks_df['name'].astype(str)))
            report_df['Name'] = report_df['Ticker'].astype(str).map(name_map).fillna('')
            # Reorder columns to place Name next to Ticker
            report_df = report_df[['Ticker', 'Name', 'Row Count', 'Start Date', 'End Date']]
    except Exception as e:
        print("[get_stock_data_report] Could not attach stock names:", e)

    # Save to Excel (fallback to CSV if Excel writer not available)
    out_xlsx = './reports/stock_data_report.xlsx'
    try:
        report_df.to_excel(out_xlsx, index=False)
        print(f"[get_stock_data_report] Saved Excel report: {out_xlsx} (rows={len(report_df)})")
        return out_xlsx
    except Exception as e:
        print("[get_stock_data_report] Excel write failed, falling back to CSV:", e)
        out_csv = './reports/stock_data_report.csv'
        try:
            report_df.to_csv(out_csv, index=False)
            print(f"[get_stock_data_report] Saved CSV report: {out_csv} (rows={len(report_df)})")
            return out_csv
        except Exception as e2:
            print("[get_stock_data_report] CSV write also failed:", e2)
            return None

# get_stock_data_report()