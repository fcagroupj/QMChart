#       Quant_db_helper24.py
# This script 
  
import sqlite3
import time
import pandas as pd
import numpy as np
import datetime as _dt
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import config_util as cfg_util

_cfg = cfg_util.read_config()
DB_PATH23 = ''
if(_cfg.get('ts_stock_type', 'cn') == 'us'):
    DB_PATH23 = ".\\reports\\quant_stock15.db"
else:
    DB_PATH23 = ".\\reports\\quant_stock06.db"

#########################################################
from datetime import date, timedelta

# Load price data from DB and prepare rows/closes for plotting
def load_ts_minute_db_data(ts_code):
    resolved_code, resolved_name = cfg_util._resolve_ts_code(ts_code)
    table_name = cfg_util._sanitize_table_name(ts_code)
    print(f"[load_ts_minute_db_data] Loading data for {ts_code} from database table {table_name}...")
    conn = sqlite3.connect(DB_PATH23)
    cur = conn.cursor()
    try:
            cur.execute(
                f"""
                SELECT trade_time, high, low, close, vol
                FROM {table_name}
                WHERE trade_time > ?
                ORDER BY trade_time ASC
                """,
                (0,),
            )
            raw_rows = cur.fetchall()
    except Exception:
        raw_rows = []
    print(f"[load_ts_minute_db_data] Loaded {len(raw_rows)} data")
    conn.close()

    if not raw_rows:
        return None
    # convert trade_time in raw_rows into eastern/new york time
    try:
        if _cfg.get('ts_stock_type', 'cn') == 'us' and ZoneInfo is not None:
            est = ZoneInfo('America/New_York')
            processed = []
            for d, h, l, c, v in raw_rows:
                try:
                    ds = str(d).strip()
                    if len(ds) >= 12:
                        dt_utc = _dt.datetime.strptime(ds[:12], '%Y%m%d%H%M').replace(tzinfo=_dt.timezone.utc)
                        dt_est = dt_utc.astimezone(est)
                        d_conv = dt_est.strftime('%Y%m%d%H%M')
                    else:
                        d_conv = d
                except Exception:
                    d_conv = d
                processed.append((d_conv, h, l, c, v))
            raw_rows = processed
    except Exception:
        pass

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
            # Convert minute trade_time (YYYYMMDDHHMM) to daily trade_date (YYYYMMDD)
            try:
                d_str = str(d).strip()
                trade_date = d_str[:8] if len(d_str) >= 8 else d_str
            except Exception:
                trade_date = d
            cur.execute(
                """
                SELECT qm_score
                FROM daily_param1
                WHERE ts_code = ? AND trade_date = ?
                """,
                (resolved_code, trade_date),
            )
            result = cur.fetchone()
            if result and result[0] is not None:
                qm_scores.append(float(result[0]))
            else:
                qm_scores.append(np.nan)
        conn.close()
    except Exception:
        qm_scores = [np.nan] * n
    print(f"[load_ts_minute_db_data] Loaded {len(closes)} data points for {resolved_code}", rows[0][0], "to", rows[-1][0])
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
        "resolved_ticker": resolved_code,
        "resolved_name": resolved_name,
        "qm_scores": qm_scores,
        "obv": obv_vals,
        "obv_ma20": obv_ma20_vals,
    }

