#       Quant_db_helper21.py
# This script plot price charts with technical indicators for given stock price data.  
# It uses OpenCV and NumPy for rendering the charts.
# The script includes functions to find suitable Chinese fonts for rendering text.

import pandas as pd
import numpy as np
import os

import cv2
import config_util as cfg_util
from typing import Optional, Tuple

#########################################################

def _find_chinese_font() -> str:
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",   # Microsoft YaHei
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf", # SimHei
        r"C:\\Windows\\Fonts\\simsun.ttc", # SimSun
        r"C:\\Windows\\Fonts\\msjh.ttc",   # Microsoft JhengHei
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return path
        except Exception:
            continue
    return ""


def plot_average_price_line(ma_window, closes, xs, top_margin, plot_h, min_c, max_c, img, color):
    # draw 5-day moving average of stocks with closes
    n = len(closes)
    if n >= ma_window:
        closes_arr = np.array(closes, dtype=float)
        kernel = np.ones(ma_window, dtype=float) / ma_window
        ma5 = np.convolve(closes_arr, kernel, mode='valid')

        xs_ma = xs[ma_window - 1:]
        ys_ma = top_margin + (max_c - ma5) * (plot_h) / (max_c - min_c)
        pts_ma = np.column_stack((xs_ma, ys_ma)).astype(np.int32)

        cv2.polylines(img, [pts_ma], False, color, 1, lineType=cv2.LINE_AA)    

# Render plot image and metadata from loaded price data
_LAST_SAVED_VIEW: Optional[Tuple[int, int]] = None


def render_price_plot(price_data, width=1000, height=400, view=None, ma_flags=None, persist_view: bool = True, tick_target=None):
    if not price_data:
        return None

    rows = list(price_data.get("rows", []))
    closes = list(price_data.get("closes", []))
    highs = list(price_data.get("highs", []))
    lows = list(price_data.get("lows", []))
    volumes = list(price_data.get("volumes", []))
    resolved_code = price_data.get("resolved_ticker")
    ts_name = price_data.get("resolved_name")
    # Optional: precomputed indicators provided by loader
    pre_dif = price_data.get("dif")
    pre_dea = price_data.get("dea")
    pre_macd = price_data.get("macd")
    pre_k = price_data.get("k")
    pre_d = price_data.get("d")
    pre_j = price_data.get("j")
    pre_obv = price_data.get("obv")
    pre_obv_ma20 = price_data.get("obv_ma20")
    ma_map = price_data.get("ma")

    # Apply view range (start,end) if provided
    applied_view = None
    if view is not None and isinstance(view, tuple) and len(view) == 2:
        start, end = int(view[0]), int(view[1])
        start = max(0, start)
        end = min(len(rows) - 1, end)
        if end >= start:
            applied_view = (start, end)
            rows = rows[start:end + 1]
            closes = closes[start:end + 1]
            highs = highs[start:end + 1]
            lows = lows[start:end + 1]
            # Slice precomputed arrays to the selected window
            def _slice(arr):
                try:
                    return arr[start:end+1] if arr is not None else None
                except Exception:
                    return None
            pre_dif = _slice(pre_dif)
            pre_dea = _slice(pre_dea)
            pre_macd = _slice(pre_macd)
            pre_k = _slice(pre_k)
            pre_d = _slice(pre_d)
            pre_j = _slice(pre_j)
            volumes = _slice(volumes)
            pre_obv = _slice(pre_obv)
            pre_obv_ma20 = _slice(pre_obv_ma20)
            if isinstance(ma_map, dict):
                try:
                    ma_map = {w: _slice(v) for w, v in ma_map.items()}
                except Exception:
                    pass
    else:
        # default applied view is whole range (if available)
        if len(rows) > 0:
            applied_view = (0, len(rows) - 1)

    if len(closes) < 2:
        return None

    W, H = int(width), int(height)
    # Move plot to the left and reserve space on the right for Y-axis ticks
    left_margin, right_margin, top_margin, bottom_margin = 2, 80, 2, 50
    # Set plotting UI background to black
    img = np.full((H, W, 3), 0, dtype=np.uint8)

    # Price scale
    min_c = float(np.min(closes))
    max_c = float(np.max(closes))
    pad = (max_c - min_c) * 0.05
    if pad == 0:
        pad = max_c * 0.05 if max_c != 0 else 1.0
    min_c -= pad
    max_c += pad

    n = len(closes)
    plot_w = W - left_margin - right_margin
    plot_h_total = H - top_margin - bottom_margin
    # Decide whether to draw MACD/KDJ/OBV, QM (schema2) and cross markers based on flags
    draw_macd = True
    draw_kdj = True
    draw_cross = True
    draw_qm = False
    draw_obv = False
    try:
        if ma_flags is not None:
            draw_macd = bool(ma_flags.get('macd', True))
            draw_kdj = bool(ma_flags.get('kdj', True))
            draw_cross = bool(ma_flags.get('cross', True))
            draw_qm = bool(ma_flags.get('schema2', False))
            draw_obv = bool(ma_flags.get('obv', False))
    except Exception:
        draw_macd = True
        draw_kdj = True
        draw_cross = True
        draw_qm = False
        draw_obv = False
    # Split into panels: price (top), then dynamic set of subpanels (MACD, KDJ, OBV, QM)
    sep_h = 8
    macd_top = None; macd_bottom = None; macd_h = 0
    kdj_top = None; kdj_bottom = None; kdj_h = 0
    obv_top = None; obv_bottom = None; obv_h = 0
    qm_top = None; qm_bottom = None; qm_h = 0
    # Decide default heights for subpanels
    sub_heights = []
    if draw_macd:
        sub_heights.append(('macd', int(max(80, plot_h_total * 0.28))))
    if draw_kdj:
        sub_heights.append(('kdj', int(max(60, plot_h_total * 0.22))))
    if draw_obv:
        sub_heights.append(('obv', int(max(60, plot_h_total * 0.22))))
    if draw_qm:
        sub_heights.append(('qm', int(max(60, plot_h_total * 0.20))))
    total_sub = sum(h for _, h in sub_heights)
    price_h = max(60, plot_h_total - total_sub - sep_h * (len(sub_heights) if sub_heights else 0))
    price_top = top_margin
    price_bottom = price_top + price_h
    y_cursor = price_bottom
    for name, h in sub_heights:
        top = y_cursor + sep_h
        bottom = top + h
        if name == 'macd':
            macd_top, macd_bottom, macd_h = top, bottom, h
        elif name == 'kdj':
            kdj_top, kdj_bottom, kdj_h = top, bottom, h
        elif name == 'obv':
            obv_top, obv_bottom, obv_h = top, bottom, h
        elif name == 'qm':
            qm_top, qm_bottom, qm_h = top, bottom, h
        y_cursor = bottom

    xs = np.linspace(left_margin, W - right_margin, n)
    ys = price_top + (max_c - np.array(closes)) * (price_h) / (max_c - min_c)
    pts = np.column_stack((xs, ys)).astype(np.int32)

    # moving averages (toggleable)
    default_colors = {
        5:  (240, 240, 240),
        10: (220, 220, 220),
        20: (200, 200, 200),
        30: (180, 180, 180),
        60: (160, 160, 160),
        114: (140, 140, 140),
    }
    if ma_flags is None:
        ma_flags = {w: True for w in default_colors}
    for w, color in default_colors.items():
        if ma_flags.get(w, False):
            # Use precomputed MA if available; fallback to compute
            if isinstance(ma_map, dict) and isinstance(ma_map.get(w), list):
                try:
                    ma_arr = np.array(ma_map[w], dtype=float)
                    mask = np.isfinite(ma_arr)
                    if np.any(mask):
                        xs_ma = xs[mask]
                        ys_ma = price_top + (max_c - ma_arr[mask]) * (price_h) / (max_c - min_c)
                        pts_ma = np.column_stack((xs_ma, ys_ma)).astype(np.int32)
                        cv2.polylines(img, [pts_ma], False, color, 1, lineType=cv2.LINE_AA)
                except Exception:
                    plot_average_price_line(w, closes, xs, price_top, price_h, min_c, max_c, img, color)
            else:
                plot_average_price_line(w, closes, xs, price_top, price_h, min_c, max_c, img, color)

    # plot area and line
    # Price panel box and line
    cv2.rectangle(img, (left_margin, price_top), (W - right_margin, price_bottom), (50, 50, 50), 1)
    # Draw price line segments colored by day-to-day close change
    try:
        _cfg = cfg_util.read_config()
        stock_type = _cfg.get('ts_stock_type', 'cn')
    except Exception:
        stock_type = 'cn'
    color_up = (0, 0, 200)   # red (CN convention)
    color_down = (0, 150, 0) # green
    if stock_type == 'us':
        color_up, color_down = (0, 150, 0), (0, 0, 200)
    neutral = (150, 150, 150)
    for i in range(1, n):
        c_prev = closes[i-1]
        c_curr = closes[i]
        if not (np.isfinite(c_prev) and np.isfinite(c_curr)):
            continue
        x1, y1 = int(xs[i-1]), int(ys[i-1])
        x2, y2 = int(xs[i]), int(ys[i])
        if c_curr > c_prev:
            col = color_up
        elif c_curr < c_prev:
            col = color_down
        else:
            col = neutral
        cv2.line(img, (x1, y1), (x2, y2), col, 2, lineType=cv2.LINE_AA)

    # --- Schema 3: Divergence flags on MA20 using OBV MA20 ---
    try:
        draw_schema3 = bool(ma_flags.get('schema3', False)) if ma_flags is not None else False
    except Exception:
        draw_schema3 = False
    if draw_schema3:
        try:
            # Build MA20 series: prefer precomputed from loader
            if isinstance(ma_map, dict) and isinstance(ma_map.get(20), list):
                ma20 = np.array(ma_map.get(20), dtype=float)
                if len(ma20) != n:
                    ma20 = ma20[:n] if len(ma20) > n else np.pad(ma20, (0, n - len(ma20)), constant_values=np.nan)
            else:
                if n >= 20:
                    arr_c = np.array(closes, dtype=float)
                    kernel20 = np.ones(20, dtype=float) / 20.0
                    valid20 = np.convolve(arr_c, kernel20, mode='valid')
                    ma20 = np.concatenate([np.full(19, np.nan, dtype=float), valid20])
                else:
                    ma20 = np.full(n, np.nan, dtype=float)

            # OBV MA20: prefer precomputed; fallback to compute from OBV/volumes
            if isinstance(pre_obv_ma20, list):
                obv_ma20 = np.array(pre_obv_ma20, dtype=float)
                if len(obv_ma20) != n:
                    obv_ma20 = obv_ma20[:n] if len(obv_ma20) > n else np.pad(obv_ma20, (0, n - len(obv_ma20)), constant_values=np.nan)
            else:
                try:
                    # derive OBV first
                    if pre_obv is not None and isinstance(pre_obv, list):
                        obv_tmp = np.array(pre_obv, dtype=float)
                        if len(obv_tmp) != n:
                            obv_tmp = obv_tmp[:n] if len(obv_tmp) > n else np.pad(obv_tmp, (0, n - len(obv_tmp)), constant_values=np.nan)
                    else:
                        vols_arr = np.array(volumes, dtype=float)
                        if len(vols_arr) != n:
                            vols_arr = np.full(n, np.nan, dtype=float)
                        closes_arr = np.array(closes, dtype=float)
                        obv_tmp = np.zeros(n, dtype=float)
                        for i2 in range(1, n):
                            v2 = vols_arr[i2]
                            if not np.isfinite(v2):
                                v2 = 0.0
                            if np.isnan(closes_arr[i2-1]) or np.isnan(closes_arr[i2]):
                                obv_tmp[i2] = obv_tmp[i2-1]
                            elif closes_arr[i2] > closes_arr[i2-1]:
                                obv_tmp[i2] = obv_tmp[i2-1] + v2
                            elif closes_arr[i2] < closes_arr[i2-1]:
                                obv_tmp[i2] = obv_tmp[i2-1] - v2
                            else:
                                obv_tmp[i2] = obv_tmp[i2-1]
                    # smooth to MA20
                    if n >= 20:
                        k20 = np.ones(20, dtype=float) / 20.0
                        valid_o20 = np.convolve(obv_tmp, k20, mode='valid')
                        obv_ma20 = np.concatenate([np.full(19, np.nan, dtype=float), valid_o20])
                    else:
                        obv_ma20 = np.full(n, np.nan, dtype=float)
                except Exception:
                    obv_ma20 = np.full(n, np.nan, dtype=float)

            # Compute thresholds scaled to ranges to reduce noise
            rng_p = max(1e-6, (max_c - min_c))
            if np.any(np.isfinite(obv_ma20)):
                rng_o = float(np.nanmax(obv_ma20) - np.nanmin(obv_ma20))
            else:
                rng_o = 1.0
            rng_o = max(1e-6, rng_o)
            tau_p = 0.002 * rng_p
            tau_o = 0.01  * rng_o
            win = 10

            def y_price(v: float) -> float:
                return price_top + (max_c - float(v)) * (price_h) / (max_c - min_c)

            last_flag_i = -999
            for i in range(win, n):
                dp = ma20[i] - ma20[i - win]
                dq = obv_ma20[i] - obv_ma20[i - win]
                if not (np.isfinite(dp) and np.isfinite(dq)):
                    continue
                # Divergence: opposite directions across thresholds
                bearish = (dp > tau_p and dq < -tau_o)
                bullish = (dp < -tau_p and dq >  tau_o)
                if not (bearish or bullish):
                    continue
                # avoid clutter: minimum spacing between flags
                if i - last_flag_i < int(win * 0.6):
                    continue
                last_flag_i = i
                x = int(xs[i])
                y = int(y_price(ma20[i]))
                pole_h = max(10, int(H / 80))
                y_top = max(price_top + 2, y - pole_h // 2)
                y_bot = min(price_bottom - 2, y + pole_h // 2)
                cv2.line(img, (x, y_top), (x, y_bot), (150, 150, 150), 1, lineType=cv2.LINE_AA)
                if bearish:
                    pts = np.array([[x, y_top], [x + 12, y_top + 6], [x, y_top + 12]], dtype=np.int32)
                    cv2.fillConvexPoly(img, pts, (0, 0, 200), lineType=cv2.LINE_AA)
                else:
                    ym = int((y_top + y_bot) / 2)
                    pts = np.array([[x, ym], [x + 12, ym - 6], [x, ym - 12]], dtype=np.int32)
                    cv2.fillConvexPoly(img, pts, (0, 150, 0), lineType=cv2.LINE_AA)
        except Exception:
            pass

    # Prepare MA114 series for schema gating (NaN where unavailable)
    ma114_vals = np.full(n, np.nan, dtype=float)
    try:
        if isinstance(ma_map, dict) and ma_map.get(114) is not None:
            arr = np.array(ma_map.get(114), dtype=float)
            if len(arr) == n:
                ma114_vals = arr
            else:
                # fallback pad if lengths mismatch
                ma114_vals[:min(n, len(arr))] = arr[:min(n, len(arr))]
        else:
            if n >= 114:
                closes_arr = np.array(closes, dtype=float)
                kernel = np.ones(114, dtype=float) / 114.0
                conv = np.convolve(closes_arr, kernel, mode='valid')
                ma114_vals[113:] = conv
    except Exception:
        pass

    # --- MACD calculations ---
    dif_vals = None
    dea_vals = None
    macd_vals = None
    if draw_macd:
        try:
            if pre_dif is not None and pre_dea is not None and pre_macd is not None:
                dif_vals = np.array(pre_dif, dtype=float)
                dea_vals = np.array(pre_dea, dtype=float)
                macd_vals = np.array(pre_macd, dtype=float)
            else:
                s = pd.Series(closes, dtype=float)
                ema12 = s.ewm(span=12, adjust=False).mean()
                ema26 = s.ewm(span=26, adjust=False).mean()
                dif_vals = (ema12 - ema26).to_numpy()
                dea_vals = pd.Series(dif_vals).ewm(span=9, adjust=False).mean().to_numpy()
                macd_vals = (dif_vals - dea_vals) * 2.0

            # Scale for MACD panel
            macd_min = float(np.nanmin([np.nanmin(dif_vals), np.nanmin(dea_vals), np.nanmin(macd_vals)]))
            macd_max = float(np.nanmax([np.nanmax(dif_vals), np.nanmax(dea_vals), np.nanmax(macd_vals)]))
            if macd_max == macd_min:
                macd_max += 1e-6
            # zero line y
            def macd_y(vals):
                return macd_top + (macd_max - np.array(vals, dtype=float)) * (macd_h) / (macd_max - macd_min)
            y_dif = macd_y(dif_vals)
            y_dea = macd_y(dea_vals)
            y_zero = macd_y(np.zeros(n))

            # MACD panel box
            cv2.rectangle(img, (left_margin, macd_top), (W - right_margin, macd_bottom), (50, 50, 50), 1)
            # zero line
            cv2.line(img, (left_margin, int(y_zero[0])), (W - right_margin, int(y_zero[0])), (220, 220, 220), 1, lineType=cv2.LINE_AA)

            # DIF and DEA lines
            pts_dif = np.column_stack((xs, y_dif)).astype(np.int32)
            pts_dea = np.column_stack((xs, y_dea)).astype(np.int32)
            cv2.polylines(img, [pts_dif], False, (0, 120, 255), 1, lineType=cv2.LINE_AA)
            cv2.polylines(img, [pts_dea], False, (255, 120, 0), 1, lineType=cv2.LINE_AA)

            # MACD histogram bars
            # approximate bar half-width based on spacing
            if n > 1:
                dx = (xs[-1] - xs[0]) / (n - 1)
            else:
                dx = 2.0
            half_w = max(1, int(dx * 0.35))
            for i in range(n):
                x = int(xs[i])
                y1 = int(macd_y(macd_vals[i]))
                y0 = int(y_zero[i])
                # Mainland convention: red for positive (up), green for negative (down)
                _cfg = cfg_util.read_config()
                color_bar = (0, 0, 200) if macd_vals[i] >= 0 else (0, 150, 0)
                if(_cfg.get('ts_stock_type', 'cn') == 'us'):
                    color_bar = (0, 150, 0) if macd_vals[i] >= 0 else (0, 0, 200)
                cv2.rectangle(img, (x - half_w, min(y0, y1)), (x + half_w, max(y0, y1)), color_bar, -1)

            # Draw MACD golden/death cross markers
            try:
                if not draw_cross:
                    raise Exception("skip cross")
                schema1_on = False
                try:
                    schema1_on = bool(ma_flags.get('schema1', False)) if ma_flags is not None else False
                except Exception:
                    schema1_on = False
                radius = max(3, int(dx * 0.25))
                for i in range(1, n):
                    prev_delta = dif_vals[i-1] - dea_vals[i-1]
                    curr_delta = dif_vals[i] - dea_vals[i]
                    if np.isnan(prev_delta) or np.isnan(curr_delta):
                        continue
                    is_golden = (prev_delta < 0 and curr_delta > 0)
                    is_death = (prev_delta > 0 and curr_delta < 0)
                    if not (is_golden or is_death):
                        continue
                    # schema 1 gating by MA114 vs close
                    if schema1_on and np.isfinite(ma114_vals[i]):
                        close_i = float(closes[i])
                        ma114_i = float(ma114_vals[i])
                        if is_golden and close_i < ma114_i:
                            continue
                        if is_death and close_i > ma114_i:
                            continue
                    d1p, d1c = dif_vals[i-1], dif_vals[i]
                    d2p, d2c = dea_vals[i-1], dea_vals[i]
                    den = (d1c - d1p) - (d2c - d2p)
                    if den == 0 or np.isnan(den):
                        t = 0.5
                    else:
                        t = (d2p - d1p) / den
                        t = float(np.clip(t, 0.0, 1.0))
                    val_cross = d1p + t * (d1c - d1p)
                    x_cross = int(xs[i-1] + t * (xs[i] - xs[i-1]))
                    y_cross = int(macd_y(val_cross))
                    color = (0, 150, 0) if is_golden else (0, 0, 200)
                    cv2.circle(img, (x_cross, y_cross), radius, color, -1, lineType=cv2.LINE_AA)
            except Exception:
                pass
        except Exception:
            # If MACD fails, skip without breaking the price plot
            pass

    # --- KDJ calculations and panel ---
    k_vals = None
    d_vals = None
    j_vals = None
    if draw_kdj and kdj_top is not None:
        try:
            if pre_k is not None and pre_d is not None and pre_j is not None:
                k_vals = np.array(pre_k, dtype=float)
                d_vals = np.array(pre_d, dtype=float)
                j_vals = np.array(pre_j, dtype=float)
            else:
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
        except Exception:
            pass

    if draw_kdj and kdj_top is not None and k_vals is not None and d_vals is not None and j_vals is not None:
        try:
            kdj_min = 0.0
            kdj_max = 100.0
            def kdj_y(vals):
                return kdj_top + (kdj_max - np.array(vals, dtype=float)) * (kdj_h) / (kdj_max - kdj_min)
            y_k = kdj_y(k_vals)
            y_d = kdj_y(d_vals)
            y_j = kdj_y(j_vals)

            # KDJ panel box
            cv2.rectangle(img, (left_margin, kdj_top), (W - right_margin, kdj_bottom), (50, 50, 50), 1)
            # K, D, J lines
            pts_k = np.column_stack((xs, y_k)).astype(np.int32)
            pts_d_ = np.column_stack((xs, y_d)).astype(np.int32)
            pts_j = np.column_stack((xs, y_j)).astype(np.int32)
            cv2.polylines(img, [pts_k], False, (0, 180, 180), 1, lineType=cv2.LINE_AA)    # K: cyan
            cv2.polylines(img, [pts_d_], False, (200, 0, 200), 1, lineType=cv2.LINE_AA)   # D: magenta
            cv2.polylines(img, [pts_j], False, (120, 120, 120), 1, lineType=cv2.LINE_AA)  # J: gray

            # Draw KDJ golden/death cross markers (K vs D)
            try:
                if not draw_cross:
                    raise Exception("skip cross")
                if n > 1:
                    dx = (xs[-1] - xs[0]) / (n - 1)
                else:
                    dx = 2.0
                radius = max(3, int(dx * 0.25))
                for i in range(1, n):
                    prev_delta = k_vals[i-1] - d_vals[i-1]
                    curr_delta = k_vals[i] - d_vals[i]
                    if np.isnan(prev_delta) or np.isnan(curr_delta):
                        continue
                    is_golden = (prev_delta < 0 and curr_delta > 0)
                    is_death = (prev_delta > 0 and curr_delta < 0)
                    if not (is_golden or is_death):
                        continue
                    # schema 1 gating by MA114 vs close
                    schema1_on = False
                    try:
                        schema1_on = bool(ma_flags.get('schema1', False)) if ma_flags is not None else False
                    except Exception:
                        schema1_on = False
                    if schema1_on and np.isfinite(ma114_vals[i]):
                        close_i = float(closes[i])
                        ma114_i = float(ma114_vals[i])
                        if is_golden and close_i < ma114_i:
                            continue
                        if is_death and close_i > ma114_i:
                            continue
                    k_p, k_c = k_vals[i-1], k_vals[i]
                    d_p, d_c = d_vals[i-1], d_vals[i]
                    den = (k_c - k_p) - (d_c - d_p)
                    if den == 0 or np.isnan(den):
                        t = 0.5
                    else:
                        t = (d_p - k_p) / den
                        t = float(np.clip(t, 0.0, 1.0))
                    val_cross = k_p + t * (k_c - k_p)
                    x_cross = int(xs[i-1] + t * (xs[i] - xs[i-1]))
                    y_cross = int(kdj_top + (kdj_max - val_cross) * (kdj_h) / (kdj_max - kdj_min))
                    color = (0, 150, 0) if is_golden else (0, 0, 200)
                    cv2.circle(img, (x_cross, y_cross), radius, color, -1, lineType=cv2.LINE_AA)
            except Exception:
                pass
            # Legend
            try:
                legend_y2 = max(kdj_top + 15, top_margin + 12)
                font_kdj = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_kdj = max(0.4, min(0.7, H / 1000))
                cv2.putText(img, "KDJ", (left_margin + 4, legend_y2), font_kdj, font_scale_kdj, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(img, "K", (left_margin + 60, legend_y2), font_kdj, font_scale_kdj, (0, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(img, "D", (left_margin + 90, legend_y2), font_kdj, font_scale_kdj, (200, 0, 200), 1, cv2.LINE_AA)
                cv2.putText(img, "J", (left_margin + 120, legend_y2), font_kdj, font_scale_kdj, (120, 120, 120), 1, cv2.LINE_AA)
            except Exception:
                pass
        except Exception:
            pass

    # --- Schema 1 composite marker: per KDJ-golden subsequence, draw one upward triangle at the first MACD-golden after the first KDJ-golden (MA114-gated) ---
    try:
        schema1_on = False
        try:
            schema1_on = bool(ma_flags.get('schema1', False)) if ma_flags is not None else False
        except Exception:
            schema1_on = False
        if schema1_on and dif_vals is not None and dea_vals is not None and k_vals is not None and d_vals is not None:
            # derive sizing from horizontal spacing
            if n > 1:
                dx_price = (xs[-1] - xs[0]) / (n - 1)
            else:
                dx_price = 2.0
            tri_h = max(12, int(dx_price * 0.6))
            tri_w = max(12, int(dx_price * 0.6))
            # Track KDJ-golden subsequences (reset on MA114-gated KDJ-death)
            in_subseq = False
            first_kdj_golden_idx = None
            triangle_drawn = False
            for i in range(1, n):
                # KDJ crosses
                k_p = k_vals[i-1]
                d_p = d_vals[i-1]
                k_c = k_vals[i]
                d_c = d_vals[i]
                if np.isnan(k_p) or np.isnan(d_p) or np.isnan(k_c) or np.isnan(d_c):
                    continue
                prev_delta_kdj = k_p - d_p
                curr_delta_kdj = k_c - d_c
                is_kdj_golden = (prev_delta_kdj < 0 and curr_delta_kdj > 0)
                is_kdj_death = (prev_delta_kdj > 0 and curr_delta_kdj < 0)
                # MA114 gating for KDJ crosses
                kdj_cross_allowed = True
                if np.isfinite(ma114_vals[i]):
                    close_i = float(closes[i])
                    ma114_i = float(ma114_vals[i])
                    if is_kdj_golden and close_i < ma114_i:
                        kdj_cross_allowed = False
                    if is_kdj_death and close_i > ma114_i:
                        kdj_cross_allowed = False
                # Reset subsequence on allowed KDJ-death
                if is_kdj_death and kdj_cross_allowed:
                    in_subseq = False
                    first_kdj_golden_idx = None
                    triangle_drawn = False
                    continue
                # Start subsequence on first allowed KDJ-golden
                if is_kdj_golden and kdj_cross_allowed and not in_subseq:
                    in_subseq = True
                    first_kdj_golden_idx = i
                    triangle_drawn = False
                    # Continue to look for MACD golden
                if in_subseq and not triangle_drawn and first_kdj_golden_idx is not None:
                    # MACD golden check at i
                    d1p = dif_vals[i-1]
                    d1c = dif_vals[i]
                    d2p = dea_vals[i-1]
                    d2c = dea_vals[i]
                    if np.isnan(d1p) or np.isnan(d1c) or np.isnan(d2p) or np.isnan(d2c):
                        continue
                    prev_delta_macd = d1p - d2p
                    curr_delta_macd = d1c - d2c
                    is_macd_golden = (prev_delta_macd < 0 and curr_delta_macd > 0)
                    # MA114 gating for MACD golden
                    macd_allowed = True
                    if np.isfinite(ma114_vals[i]):
                        close_i = float(closes[i])
                        ma114_i = float(ma114_vals[i])
                        if is_macd_golden and close_i < ma114_i:
                            macd_allowed = False
                    if is_macd_golden and macd_allowed and i >= first_kdj_golden_idx:
                        x_day = int(xs[i])
                        y_day = int(ys[i])
                        apex = (x_day, max(price_top + 4, y_day - tri_h))
                        left = (x_day - tri_w , y_day + tri_h )
                        right = (x_day + tri_w , y_day + tri_h )
                        pts_tri = np.array([apex, left, right], dtype=np.int32)
                        cv2.fillConvexPoly(img, pts_tri, (0, 150, 0), lineType=cv2.LINE_AA)
                        triangle_drawn = True

            # Per KDJ-death subsequence (separated by KDJ-golden), draw one downward triangle at the first MACD-dead after the first KDJ-dead (MA114-gated)
            in_subseq_dead = False
            first_kdj_death_idx = None
            triangle_drawn_dead = False
            for i in range(1, n):
                # KDJ crosses
                k_p = k_vals[i-1]
                d_p = d_vals[i-1]
                k_c = k_vals[i]
                d_c = d_vals[i]
                if np.isnan(k_p) or np.isnan(d_p) or np.isnan(k_c) or np.isnan(d_c):
                    continue
                prev_delta_kdj = k_p - d_p
                curr_delta_kdj = k_c - d_c
                is_kdj_golden = (prev_delta_kdj < 0 and curr_delta_kdj > 0)
                is_kdj_death = (prev_delta_kdj > 0 and curr_delta_kdj < 0)
                # MA114 gating for KDJ crosses
                kdj_cross_allowed = True
                if np.isfinite(ma114_vals[i]):
                    close_i = float(closes[i])
                    ma114_i = float(ma114_vals[i])
                    if is_kdj_golden and close_i < ma114_i:
                        kdj_cross_allowed = False
                    if is_kdj_death and close_i > ma114_i:
                        kdj_cross_allowed = False
                # Reset death-subsequence on allowed KDJ-golden
                if is_kdj_golden and kdj_cross_allowed:
                    in_subseq_dead = False
                    first_kdj_death_idx = None
                    triangle_drawn_dead = False
                    continue
                # Start death-subsequence on first allowed KDJ-death
                if is_kdj_death and kdj_cross_allowed and not in_subseq_dead:
                    in_subseq_dead = True
                    first_kdj_death_idx = i
                    triangle_drawn_dead = False
                # MACD dead check at i within death subsequence
                if in_subseq_dead and not triangle_drawn_dead and first_kdj_death_idx is not None:
                    d1p = dif_vals[i-1]
                    d1c = dif_vals[i]
                    d2p = dea_vals[i-1]
                    d2c = dea_vals[i]
                    if np.isnan(d1p) or np.isnan(d1c) or np.isnan(d2p) or np.isnan(d2c):
                        continue
                    prev_delta_macd = d1p - d2p
                    curr_delta_macd = d1c - d2c
                    is_macd_death = (prev_delta_macd > 0 and curr_delta_macd < 0)
                    # MA114 gating for MACD death
                    macd_allowed = True
                    if np.isfinite(ma114_vals[i]):
                        close_i = float(closes[i])
                        ma114_i = float(ma114_vals[i])
                        if is_macd_death and close_i > ma114_i:
                            macd_allowed = False
                    if is_macd_death and macd_allowed and i >= first_kdj_death_idx:
                        x_day = int(xs[i])
                        y_day = int(ys[i])
                        apex = (x_day, min(price_bottom - 4, y_day + tri_h))
                        left = (x_day - tri_w , y_day - tri_h )
                        right = (x_day + tri_w , y_day - tri_h )
                        pts_tri = np.array([apex, left, right], dtype=np.int32)
                        cv2.fillConvexPoly(img, pts_tri, (0, 0, 200), lineType=cv2.LINE_AA)
                        triangle_drawn_dead = True
    except Exception:
        pass

    # Title
    title = f"{resolved_code} - {ts_name}" if ts_name else f"{resolved_code}"

    font_path = _find_chinese_font()
    try:
        from PIL import Image, ImageDraw, ImageFont
        import cv2 as _cv
        font_size = max(18, min(36, H // 25))
        # Try loading a CJK-capable font; handle TTC index variants
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                for idx in range(5):
                    try:
                        font = ImageFont.truetype(font_path, font_size, index=idx)
                        break
                    except Exception:
                        continue
        if font is None:
            font = ImageFont.load_default()

        rgb = _cv.cvtColor(img, _cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        # Use textbbox for wider Pillow compatibility
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = int(bbox[2] - bbox[0])
        th = int(bbox[3] - bbox[1])
        # Center horizontally; move to very top vertically
        tx = max(left_margin, int((W - tw) / 2))
        ty = 2
        # White title text for dark background
        draw.text((tx, ty), title, font=font, fill=(255, 255, 255))
        img = _cv.cvtColor(np.array(pil_img), _cv.COLOR_RGB2BGR)
        title_box = (tx, ty - th, tx + tw, ty + th)
    except Exception:
        # Fallback: OpenCV cannot render CJK reliably; keep ASCII-only if PIL unavailable
        font_title = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = max(0.6, min(1.0, H / 700))
        (tw, th), _ = cv2.getTextSize(title, font_title, title_scale, 2)
        tx = max(left_margin, int((W - tw) / 2))
        # Baseline just below the very top
        ty = th + 2
        # White title text for dark background
        cv2.putText(img, title, (tx, ty), font_title, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
        title_box = (tx, ty - th, tx + tw, ty)

    # axes
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.7, H / 1000))
    # Light grey axes and grid for dark background
    color_axis = (50, 50, 50)
    color_grid = (50, 50, 50)

    # Y ticks axis vertical with adaptive integer steps; target tick count configurable
    rng = max_c - min_c
    allowed_steps = [5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]
    try:
        target_ticks = int(tick_target) if tick_target is not None else 10
    except Exception:
        target_ticks = 10
    target_ticks = max(4, min(20, target_ticks))
    best_step = allowed_steps[0]
    best_diff = float('inf')
    for step in allowed_steps:
        start_tick = int(np.ceil(min_c / step) * step)
        end_tick = int(np.floor(max_c / step) * step)
        count = 1 + max(0, (end_tick - start_tick) // step)
        diff = abs(count - target_ticks)
        if diff < best_diff:
            best_diff = diff
            best_step = step
    # Compute ticks using selected step
    start_tick = int(np.ceil(min_c / best_step) * best_step)
    end_tick = int(np.floor(max_c / best_step) * best_step)
    ticks = list(range(start_tick, end_tick + 1, best_step)) if end_tick >= start_tick else []
    # Fallback to min/max if no ticks (e.g., very small range)
    if not ticks:
        ticks = [int(round(min_c)), int(round(max_c))]
    for val_i in ticks:
        y = int(price_top + (max_c - val_i) * price_h / (max_c - min_c))
        # Horizontal grid line across the price panel
        cv2.line(img, (left_margin, y), (W - right_margin, y), color_grid, 1, lineType=cv2.LINE_AA)
        # Y-axis tick mark on the right side (draw inward)
        cv2.line(img, (W - right_margin - 6, y), (W - right_margin, y), color_axis, 1)
        # Right-side label within the right margin area
        label = str(val_i)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        x_text = min(W - tw - 5, W - right_margin + 10)
        cv2.putText(img, label, (x_text, y + th // 2), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    # X ticks axis horizontal
    num_xticks = min(10, n)
    idxs = np.linspace(0, n - 1, num_xticks)
    idxs = np.unique(np.clip(np.round(idxs).astype(int), 0, n - 1))
    for i in idxs:
        x = int(xs[i])
        cv2.line(img, (x, H - bottom_margin), (x, H - bottom_margin + 6), color_axis, 1)
        dstr = str(rows[i][0])
        if len(dstr) == 8:
            label = f"{dstr[2:4]}/{dstr[4:6]}/{dstr[6:8]}"
        else:
            label = dstr
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        x_text = int(x - tw / 2)
        y_text = H - bottom_margin + 25 + th
        x_text = max(5, min(x_text, W - tw - 5))
        cv2.putText(img, label, (x_text, y_text), font, font_scale, (150, 150, 150), 1, cv2.LINE_AA)

    # MACD legend
    if draw_macd and macd_top is not None:
        try:
            legend_y = macd_top + 15
            cv2.putText(img, "MACD", (left_margin + 4, legend_y), font, font_scale, (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(img, "DIF", (left_margin + 60, legend_y), font, font_scale, (0, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "DEA", (left_margin + 110, legend_y), font, font_scale, (255, 120, 0), 1, cv2.LINE_AA)
        except Exception:
            pass

    # --- Schema 2: QM subplot ---
    if draw_qm and qm_top is not None and qm_bottom is not None:
        try:
            qm_vals = price_data.get("qm_scores")
            # Slice QM to the current applied view length
            if applied_view is not None and qm_vals is not None:
                a, b = applied_view
                try:
                    qm_vals = qm_vals[a:b+1]
                except Exception:
                    pass
            if isinstance(qm_vals, list) and len(qm_vals) == n:
                qm_arr = np.array(qm_vals, dtype=float)
                mask = np.isfinite(qm_arr)
                if np.any(mask):
                    min_qm = float(np.nanmin(qm_arr[mask]))
                    max_qm = float(np.nanmax(qm_arr[mask]))
                    if max_qm == min_qm:
                        max_qm = min_qm + 1.0
                    y_qm = qm_top + (max_qm - qm_arr) * (qm_bottom - qm_top) / (max_qm - min_qm)
                    # QM panel box
                    cv2.rectangle(img, (left_margin, qm_top), (W - right_margin, qm_bottom), (50, 50, 50), 1)
                    # Middle reference line at QM = 0.5
                    try:
                        y_mid = qm_top + (max_qm - 0.5) * (qm_bottom - qm_top) / (max_qm - min_qm)
                        y_mid_i = int(max(qm_top, min(qm_bottom, round(y_mid))))
                        cv2.line(img, (left_margin, y_mid_i), (W - right_margin, y_mid_i), (100, 100, 100), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                    # Right-side QM tick marks and labels for 0, 0.5, 1
                    try:
                        for v in (0.0, 0.5):
                            y_v = qm_top + (max_qm - v) * (qm_bottom - qm_top) / (max_qm - min_qm)
                            y_vi = int(max(qm_top, min(qm_bottom, round(y_v))))
                            # Tick mark on right edge of QM panel
                            cv2.line(img, (W - right_margin - 6, y_vi), (W - right_margin, y_vi), (50, 50, 50), 1)
                            # Label just outside to the right, like price Y ticks
                            lbl = "0.5" if abs(v - 0.5) < 1e-9 else ("1" if abs(v - 1.0) < 1e-9 else "0")
                            (tw_v, th_v), _ = cv2.getTextSize(lbl, font, font_scale, 1)
                            x_text_v = min(W - tw_v - 5, W - right_margin + 10)
                            cv2.putText(img, lbl, (x_text_v, y_vi + th_v // 2), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                    # QM line (filter to finite points)
                    mask_line = np.isfinite(y_qm)
                    if np.any(mask_line):
                        xs_qm = xs[mask_line]
                        y_qm_plot = y_qm[mask_line]
                        pts_qm = np.column_stack((xs_qm, y_qm_plot)).astype(np.int32)
                        # Color: red for CN, green otherwise
                        try:
                            _cfg = cfg_util.read_config()
                            stock_type = _cfg.get('ts_stock_type', 'cn')
                        except Exception:
                            stock_type = 'cn'
                        qm_color = (0, 0, 200) if stock_type == 'cn' else (0, 150, 0)
                        cv2.polylines(img, [pts_qm], False, qm_color, 1, lineType=cv2.LINE_AA)
                    # Legend
                    cv2.putText(img, "QM", (left_margin + 4, qm_top + 15), font, font_scale, (80, 80, 80), 1, cv2.LINE_AA)
        except Exception:
            pass
    # Prepare OBV return values
    obv_vals = None
    obv_ma20_vals = None
    # --- OBV subplot ---
    if draw_obv and obv_top is not None and obv_bottom is not None:
        try:
            # Compact number formatter for axis labels
            def _fmt_compact(v: float) -> str:
                try:
                    av = abs(float(v))
                except Exception:
                    return str(v)
                if av >= 1e12:
                    return f"{v/1e12:.1f}T"
                if av >= 1e9:
                    return f"{v/1e9:.1f}B"
                if av >= 1e6:
                    return f"{v/1e6:.1f}M"
                if av >= 1e3:
                    return f"{v/1e3:.1f}K"
                return f"{v:.0f}"
            # Prefer precomputed OBV; fallback to compute if missing
            if pre_obv is not None and isinstance(pre_obv, list):
                obv = np.array(pre_obv, dtype=float)
                # Align length to n
                if len(obv) != n:
                    obv = obv[:n] if len(obv) > n else np.pad(obv, (0, n - len(obv)), constant_values=np.nan)
            else:
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
            # OBV MA20 smoothing: prefer precomputed; fallback to compute from obv
            if pre_obv_ma20 is not None and isinstance(pre_obv_ma20, list):
                obv_ma20 = np.array(pre_obv_ma20, dtype=float)
                if len(obv_ma20) != n:
                    obv_ma20 = obv_ma20[:n] if len(obv_ma20) > n else np.pad(obv_ma20, (0, n - len(obv_ma20)), constant_values=np.nan)
            else:
                try:
                    w = 20
                    if n >= w:
                        kernel = np.ones(w, dtype=float) / float(w)
                        valid = np.convolve(obv, kernel, mode='valid')
                        pad = np.full(w-1, np.nan, dtype=float)
                        obv_ma20 = np.concatenate([pad, valid])
                    else:
                        obv_ma20 = np.full(n, np.nan, dtype=float)
                except Exception:
                    obv_ma20 = np.full(n, np.nan, dtype=float)
            # Scale includes both OBV and its smoothing for consistent axis
            try:
                obv_min = float(np.nanmin([np.nanmin(obv), np.nanmin(obv_ma20)]))
                obv_max = float(np.nanmax([np.nanmax(obv), np.nanmax(obv_ma20)]))
            except Exception:
                obv_min = float(np.nanmin(obv))
                obv_max = float(np.nanmax(obv))
            if obv_max == obv_min:
                obv_max = obv_min + 1e-6
            y_obv = obv_top + (obv_max - obv) * (obv_bottom - obv_top) / (obv_max - obv_min)
            y_obv_ma20 = obv_top + (obv_max - obv_ma20) * (obv_bottom - obv_top) / (obv_max - obv_min)
            # OBV panel box
            cv2.rectangle(img, (left_margin, obv_top), (W - right_margin, obv_bottom), (50, 50, 50), 1)
            # Zero-reference line at first value baseline
            base_val = obv[0]
            y_base = obv_top + (obv_max - base_val) * (obv_bottom - obv_top) / (obv_max - obv_min)
            cv2.line(img, (left_margin, int(y_base)), (W - right_margin, int(y_base)), (80, 80, 80), 1, lineType=cv2.LINE_AA)
            # Right-side OBV ticks: min, base, max (compact labels)
            try:
                tick_vals = [obv_min, base_val, obv_max]
                # De-duplicate while preserving order
                seen = set()
                uniq = []
                for tv in tick_vals:
                    key = float(tv)
                    if key not in seen:
                        seen.add(key)
                        uniq.append(tv)
                for tv in uniq:
                    y_tv = obv_top + (obv_max - tv) * (obv_bottom - obv_top) / (obv_max - obv_min)
                    yi = int(max(obv_top, min(obv_bottom, round(y_tv))))
                    # Tick mark on right edge of OBV panel
                    cv2.line(img, (W - right_margin - 6, yi), (W - right_margin, yi), (50, 50, 50), 1)
                    # Compact label to the right
                    lbl = _fmt_compact(tv)
                    (tw_v, th_v), _ = cv2.getTextSize(lbl, font, font_scale, 1)
                    x_text_v = min(W - tw_v - 5, W - right_margin + 10)
                    cv2.putText(img, lbl, (x_text_v, yi + th_v // 2), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
            except Exception:
                pass
            # OBV line (blue)
            pts_obv = np.column_stack((xs, y_obv)).astype(np.int32)
            cv2.polylines(img, [pts_obv], False, (255, 0, 0), 1, lineType=cv2.LINE_AA)
            # OBV MA20 smoothing line (yellow), draw only finite points
            try:
                mask_ma = np.isfinite(y_obv_ma20)
                if np.any(mask_ma):
                    xs_ma = xs[mask_ma]
                    y_ma_plot = y_obv_ma20[mask_ma]
                    pts_ma = np.column_stack((xs_ma, y_ma_plot)).astype(np.int32)
                    cv2.polylines(img, [pts_ma], False, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            except Exception:
                pass
            # Legend
            cv2.putText(img, "OBV", (left_margin + 4, obv_top + 15), font, font_scale, (80, 80, 80), 1, cv2.LINE_AA)
            # Expose OBV for hover tooltip
            obv_vals = obv.tolist()
            obv_ma20_vals = obv_ma20.tolist()
        except Exception:
            pass

    # Persist the applied view to config.xml (opt-in to avoid excessive writes)
    if persist_view:
        try:
            global _LAST_SAVED_VIEW
            if applied_view is not None and applied_view != _LAST_SAVED_VIEW:
                cfg_util.write_config({'view_start': str(applied_view[0]), 'view_end': str(applied_view[1])})
                _LAST_SAVED_VIEW = applied_view
        except Exception:
            pass

    return {
        "img": img,
        "xs": xs,
        "ys": ys,
        "rows": rows,
        "closes": closes,
        "volumes": volumes,
        "ma": ma_map,
        "dif": dif_vals.tolist() if dif_vals is not None else None,
        "dea": dea_vals.tolist() if dea_vals is not None else None,
        "macd": macd_vals.tolist() if macd_vals is not None else None,
        "k": k_vals.tolist() if k_vals is not None else None,
        "d": d_vals.tolist() if d_vals is not None else None,
        "j": j_vals.tolist() if j_vals is not None else None,
        "obv": obv_vals if obv_vals is not None else None,
        "obv_ma20": obv_ma20_vals if obv_ma20_vals is not None else None,
        "left_margin": left_margin,
        "right_margin": right_margin,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "W": W,
        "H": H,
        "title_box": title_box,
        "title_text": title,
        "resolved_ticker": resolved_code,
        "resolved_name": ts_name,
    }
