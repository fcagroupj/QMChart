#
#       qmchart.py
#  This script displays an interactive price chart for a specified stock using OpenCV.
#  It supports zooming, panning, and toggling moving averages (MA).
#  It also integrates with a background thread to update stock data.
#  Clicking the "QM" label starts a background process to compute High Quality Momentum (HQM) scores.
#  The stock code can be edited via a dialog box triggered by clicking the title.
#       hover tooltip, showing date, price, and indicator values.
#       Plot Options menu  
#       QM label
#       moving averages (MA) lines.
#       X ticks, Y ticks
#       The price subplot
#       MACD subplot, DIF, DEA.
#       KDJ subplot, K, D, J.
#       golden cross (green), dead cross (red)
#       schema1, show green triangle for Stock buy points, based on MACD, DIF crossing DEA, and only when price is above MA114, 
#               Show red triangle for Stock sell points when price is below MA114. Check https://www.youtube.com/watch?v=K2u31j4R7-s&t=979s. 
#       schema2, QM subplot shows QM data  
#       schema3, show buy/sell points based on OBV volume indicator, check https://www.youtube.com/watch?v=4YXQRdLFYNc
#                draw red flags for bearish and green for bullish divergences
#       schema4, check the stock bottom with RSI Oversold Bounce, show blue triangle if it's bottom
#               Draw the stock top with RSI Overbought Reversal, show orange triangle if it's top

import config_util as cfg_util
import cv2
import ctypes
import sys
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import threading
import Quant_db_helper01 as qdb01  
import Quant_db_helper11 as qdb11  
import Quant_db_helper14 as qdb14  
import Quant_db_helper15 as qdb15  
from Quant_db_helper21 import render_price_plot, drawTitleBox
from Quant_db_helper25 import load_ts_daily_db_data, reader_db_qm_thd
from Quant_db_helper25 import delete_last_n_days
from Quant_db_helper24 import load_ts_minute_db_data
import Quant_algo11 as qalg11
import time
from datetime import datetime, timedelta
try:
    # Single-run update helpers (US/CN)
    from Quant_db_helper15 import update_ticker_once as us_update_once
except Exception:
    us_update_once = None
try:
    from Quant_db_helper11 import update_ticker_once as cn_update_once
except Exception:
    cn_update_once = None

def format_compact_number(v: float) -> str:
    try:
        v = float(v)
        av = abs(v)
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

def main():
    cfg = cfg_util.read_config()
    ts_sk_code = cfg.get('ts_stock_code', '301060.SZ')
    ts_stock_type = cfg.get('ts_stock_type', 'cn')
    ts_stock_code, resolved_name = cfg_util._resolve_ts_code(ts_sk_code)
    qdb01.init_db()
    # when data is changed, set value to trigger update in main loop
    st_data_changed = 1
    # When any event is changed, set value to trigger plotting update in main loop
    st_view_changed = 1
    # get the maximum width and height from OS window size
    width, height = 1000, 600
    try:
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
            width = ctypes.windll.user32.GetSystemMetrics(0)
            height = ctypes.windll.user32.GetSystemMetrics(1) - 80  # taskbar height
    except Exception:
        pass
    # Moving-average toggle state (restored from config)
    def _on(val: str | None, default: str = '1') -> bool:
        v = (val if val is not None else default).strip().lower()
        return v not in ('0', 'false', 'no', '')
    ma_flags = {
        5:   _on(cfg.get('ma5')),
        10:  _on(cfg.get('ma10')),
        20:  _on(cfg.get('ma20')),
        30:  _on(cfg.get('ma30')),
        60:  _on(cfg.get('ma60')),
        114: _on(cfg.get('ma114')),
        'macd': _on(cfg.get('macd')),
        'kdj':  _on(cfg.get('kdj')),
        'rsi':  _on(cfg.get('rsi'), default='0'),
        'volume': _on(cfg.get('volume'), default='1'),
        'obv':  _on(cfg.get('obv'), default='0'),
        'cross': _on(cfg.get('cross')),
        'schema1': _on(cfg.get('schema1'), default='0'),
        'schema2': _on(cfg.get('schema2'), default='0'),
        'schema3': _on(cfg.get('schema3'), default='0'),
        'schema4': _on(cfg.get('schema4'), default='0'),
    }
    # Tunables: drag sensitivity and Y-axis tick target
    try:
        drag_sensitivity = float((cfg.get('drag_sensitivity') or '1.0'))
    except Exception:
        drag_sensitivity = 1.0
    drag_sensitivity = max(0.2, min(5.0, drag_sensitivity))
    try:
        axis_yticks_target = int((cfg.get('axis_yticks_target') or '10'))
    except Exception:
        axis_yticks_target = 10
    axis_yticks_target = max(4, min(20, axis_yticks_target))
    # Restore last view window target from config if valid; actual data load happens in update_st_data_db
    init_view = None
    price_data = None
    # Initial rendering will be handled by update_st_view_plotting via st_view_changed
    plot = None
    if(True):
        base = np.full((width, height, 3), 0, dtype=np.uint8)
        xs, ys = None, None
        # Initialize plotting series variables so nonlocal bindings exist
        rows = None
        dif = None
        dea = None
        k = None
        d = None
        j = None
        rsi14 = None
        obv = None
        ma_data = None
        W, H = base.shape[1], base.shape[0]
        left_margin, right_margin, top_margin, bottom_margin = 0, 0, 0, 0
        title_box = (833, -27, 1087, 31)
        current_ticker =  ts_stock_code
        hqm_score_val = qalg11.latest_hqm_score_for(current_ticker)
        # Title is drawn only in update_st_view_plotting()

            # Current ticker and HQM score
        view_start, view_end = 0, 0
        full_n = 0
        full_rows = []
        # Drag/pan state
        dragging = False
        drag_start_x = 0
        drag_start_view_start = view_start
        drag_start_view_end = view_end
        last_shift = 0
        # QuantM clickable label state
        quantm_box = None
        t2_quantm_running = False
        t2_quantm_event = threading.Event()
        t2_quantm = None
        # Background data loader thread (t3_loader) for minute/daily DB reads
        t3_loader = None
        # Remember the last stock input dialog to avoid multiple Tk instances
        dlg_input_stock = None

        # --- Timeframe toggle state (1=1M active, 2=1D active) ---
        try:
            ts_data_typ = int(cfg.get('ts_data_type'))
        except Exception:
            ts_data_typ = 1
        timeframe_boxes = {}

        win_title = f"QMChart V1.0.45"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(win_title, width, height)
        except Exception:
            pass

        # Helper to draw the QM label in the top-right and record its box
        def draw_quantm_label(img):
            nonlocal quantm_box, W, top_margin, t2_quantm_running, t2_quantm_event
            nonlocal hqm_score_val
            label = "QM" if hqm_score_val is None else f"QM {hqm_score_val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, min(1.0, H / 700))
            (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
            pad_x, pad_y = 10, 8
            x2 = W - 10
            y1 = 1  # max(1, top_margin - 16)
            x1 = x2 - (tw + pad_x * 2)
            y2 = y1 + th + int(pad_y * 1.2)
            quantm_box = (x1, y1, x2, y2)
            # Button background and border
            cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)
            # Text color: black (idle), red (running), green (finished)
            if t2_quantm_running and not t2_quantm_event.is_set():
                text_color = (0, 0, 255)
            elif t2_quantm_event.is_set():
                text_color = (0, 180, 0)
            else:
                text_color = (0, 0, 0)
            cv2.putText(img, label, (x1 + pad_x, y1 + th + pad_y - 2), font, scale, text_color, 2, cv2.LINE_AA)

        # Helper to draw "1M" and "1D" labels on top-left and record boxes
        def draw_timeframe_labels(img):
            nonlocal timeframe_boxes, ts_data_typ, left_margin, top_margin, W, H
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, min(1.0, H / 700))
            pad_x, pad_y = 8, 6
            gap = 6
            # Text sizes
            (tw_m, th_m), _ = cv2.getTextSize("1M", font, scale, 2)
            (tw_d, th_d), _ = cv2.getTextSize("1D", font, scale, 2)
            # Positions
            x = max(5, left_margin + 6)
            y = 1
            # 1M box
            x1m = x
            y1m = y
            x2m = x1m + tw_m + pad_x * 2
            y2m = y1m + th_m + int(pad_y * 1.2)
            # 1D box to the right
            x1d = x2m + gap
            y1d = y1m
            x2d = x1d + tw_d + pad_x * 2
            y2d = y1d + th_d + int(pad_y * 1.2)

            # Active: green; Inactive: black
            active_bg = (0, 180, 0)
            inactive_bg = (0, 0, 0)
            border = (120, 120, 120)
            text_color = (255, 255, 255)

            # 1M is active if ts_data_typ == 2
            bg_1m = active_bg if ts_data_typ == 1 else inactive_bg
            bg_1d = active_bg if ts_data_typ == 2 else inactive_bg

            # Draw 1M
            cv2.rectangle(img, (x1m, y1m), (x2m, y2m), bg_1m, -1)
            cv2.rectangle(img, (x1m, y1m), (x2m, y2m), border, 1)
            cv2.putText(img, "1M", (x1m + pad_x, y1m + th_m + pad_y - 2), font, scale, text_color, 2, cv2.LINE_AA)
            # Draw 1D
            cv2.rectangle(img, (x1d, y1d), (x2d, y2d), bg_1d, -1)
            cv2.rectangle(img, (x1d, y1d), (x2d, y2d), border, 1)
            cv2.putText(img, "1D", (x1d + pad_x, y1d + th_d + pad_y - 2), font, scale, text_color, 2, cv2.LINE_AA)

            timeframe_boxes = {'1M': (x1m, y1m, x2m, y2m), '1D': (x1d, y1d, x2d, y2d)}

        # --- Right-click Moving Average menu state ---
        ma_menu_visible = False
        ma_menu_box = None
        ma_item_boxes = []
        ma_colors = {5: (240,240,240), 10:(220,220,220), 20:(200,200,200), 30:(180,180,180), 60:(160,160,160), 114:(140,140,140)}
        ma_menu_pos = None  # (x, y) where menu is drawn

        def draw_ma_menu(img, x=20, y=20):
            nonlocal ma_menu_box, ma_item_boxes
            pad_x, pad_y = 10, 8
            item_h = 24
            items = [5, 10, 20, 30, 60, 114, 'macd', 'kdj', 'rsi', 'volume', 'obv', 'cross', 'schema1', 'schema2', 'schema3', 'schema4']
            width_box = 180
            height_box = pad_y*2 + item_h*len(items) + 22
            x1, y1, x2, y2 = x, y, x + width_box, y + height_box
            ma_menu_box = (x1, y1, x2, y2)
            ma_item_boxes = []
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (180,180,180), 1)
            cv2.putText(img, "Plot Options", (x1+8, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
            for i, w in enumerate(items):
                iy = y1 + 30 + i*item_h
                box = (x1+10, iy-14, x1+30, iy+6)
                ma_item_boxes.append((w, box))
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (120,120,120), 1)
                if ma_flags.get(w, False):
                    cv2.line(img, (box[0]+4, iy-4), (box[2]-4, iy+4), (0,150,0), 2, cv2.LINE_AA)
                    cv2.line(img, (box[0]+4, iy+4), (box[2]-4, iy-4), (0,150,0), 2, cv2.LINE_AA)
                sample_x1, sample_y1, sample_x2, sample_y2 = x1+40, iy-12, x1+60, iy+6
                if isinstance(w, int):
                    cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), ma_colors[w], -1)
                    cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (150,150,150), 1)
                    cv2.putText(img, f"MA{w}", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                else:
                    if w == 'macd':
                        # MACD sample: DIF/DEA lines and a small histogram bar
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        cv2.line(img, (sample_x1+1, sample_y2-2), (sample_x2-1, sample_y1+2), (0,120,255), 1, cv2.LINE_AA)
                        cv2.line(img, (sample_x1+1, sample_y1+2), (sample_x2-1, sample_y2-2), (255,120,0), 1, cv2.LINE_AA)
                        cx = (sample_x1 + sample_x2)//2
                        cy = (sample_y1 + sample_y2)//2
                        # Red (positive) above zero, Green (negative) below zero
                        cv2.rectangle(img, (cx-2, cy-6), (cx+2, cy-1), (0,0,200), -1)   # red
                        cv2.rectangle(img, (cx-2, cy+1), (cx+2, cy+6), (0,150,0), -1)   # green
                        cv2.putText(img, "MACD", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'kdj':
                        # KDJ sample: three small colored lines
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        cv2.line(img, (sample_x1+1, sample_y2-2), (sample_x2-1, sample_y1+1), (0,180,180), 1, cv2.LINE_AA)   # K
                        cv2.line(img, (sample_x1+1, sample_y1+2), (sample_x2-1, sample_y2-2), (200,0,200), 1, cv2.LINE_AA)   # D
                        cv2.line(img, (sample_x1+1, (sample_y1+sample_y2)//2), (sample_x2-1, (sample_y1+sample_y2)//2 + 1), (120,120,120), 1, cv2.LINE_AA)  # J
                        cv2.putText(img, "KDJ", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'rsi':
                        # RSI sample: center line with 30/70 guides and a small zigzag
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        mid = (sample_y1 + sample_y2)//2
                        y30 = sample_y1 + int((sample_y2 - sample_y1) * 0.7)
                        y70 = sample_y1 + int((sample_y2 - sample_y1) * 0.3)
                        cv2.line(img, (sample_x1+1, y30), (sample_x2-1, y30), (80,80,80), 1, cv2.LINE_AA)
                        cv2.line(img, (sample_x1+1, y70), (sample_x2-1, y70), (80,80,80), 1, cv2.LINE_AA)
                        pts = np.array([[sample_x1+2, mid+3], [sample_x1+10, mid-3], [sample_x2-2, mid+2]], dtype=np.int32)
                        cv2.polylines(img, [pts], False, (0,160,255), 1, cv2.LINE_AA)
                        cv2.putText(img, "RSI", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'volume':
                        # Volume sample: tiny bar/cylinder below baseline
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        my = (sample_y1 + sample_y2)//2
                        bx = (sample_x1 + sample_x2)//2
                        cv2.rectangle(img, (bx-6, my+2), (bx+6, sample_y2-2), (0,150,0), -1)
                        try:
                            cv2.ellipse(img, (bx, my+2), (6, 3), 0, 0, 360, (120,240,120), -1, lineType=cv2.LINE_AA)
                        except Exception:
                            pass
                        cv2.putText(img, "Volume", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'obv':
                        # OBV sample: simple small line representation (blue)
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        my = (sample_y1 + sample_y2)//2
                        cv2.line(img, (sample_x1+2, my+3), (sample_x1+10, my-3), (255,0,0), 1, cv2.LINE_AA)
                        cv2.line(img, (sample_x1+10, my-3), (sample_x2-2, my+2), (255,0,0), 1, cv2.LINE_AA)
                        cv2.putText(img, "OBV", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'cross':
                        # Cross sample: green/red circles
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        cx = (sample_x1 + sample_x2)//2
                        cy = (sample_y1 + sample_y2)//2
                        cv2.circle(img, (cx-4, cy), 3, (0,150,0), -1, lineType=cv2.LINE_AA)
                        cv2.circle(img, (cx+4, cy), 3, (0,0,200), -1, lineType=cv2.LINE_AA)
                        cv2.putText(img, "Cross", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'schema1':
                        # Schema 1 sample: MA114 baseline with gated markers (illustrative)
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        my = (sample_y1 + sample_y2)//2
                        cv2.line(img, (sample_x1+1, my), (sample_x2-1, my), (20,20,20), 1, cv2.LINE_AA)
                        cv2.circle(img, (sample_x1+6, my-6), 3, (0,150,0), -1, lineType=cv2.LINE_AA)  # golden above
                        cv2.circle(img, (sample_x2-6, my+6), 3, (0,0,200), -1, lineType=cv2.LINE_AA)  # death below
                        cv2.putText(img, "schema 1", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'schema2':
                        # Schema 2 sample: tiny QM sparkline
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        cv2.line(img, (sample_x1+2, sample_y2-2), (sample_x1+12, sample_y1+4), (120,200,120), 1, cv2.LINE_AA)
                        cv2.line(img, (sample_x1+12, sample_y1+4), (sample_x2-2, sample_y2-6), (200,160,80), 1, cv2.LINE_AA)
                        cv2.putText(img, "schema 2", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'schema3':
                        # Schema 3 sample: MA20 with a small flag
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        my = (sample_y1 + sample_y2)//2
                        cv2.line(img, (sample_x1+2, my), (sample_x2-2, my-2), (200,200,200), 1, cv2.LINE_AA)
                        pole_x = sample_x1 + 10
                        cv2.line(img, (pole_x, my-8), (pole_x, my+8), (120,120,120), 1, cv2.LINE_AA)
                        pts = np.array([[pole_x, my-8], [pole_x+10, my-4], [pole_x, my]], dtype=np.int32)
                        cv2.fillConvexPoly(img, pts, (0,150,0), lineType=cv2.LINE_AA)
                        cv2.putText(img, "schema 3", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
                    elif w == 'schema4':
                        # Schema 4 sample: bottom marker triangle under price
                        cv2.rectangle(img, (sample_x1, sample_y1), (sample_x2, sample_y2), (230,230,230), 1)
                        base_y = sample_y2 - 3
                        cx = (sample_x1 + sample_x2)//2
                        tri = np.array([[cx, base_y-8], [cx-8, base_y], [cx+8, base_y]], dtype=np.int32)
                        cv2.fillConvexPoly(img, tri, (255, 128, 0), lineType=cv2.LINE_AA)
                        cv2.putText(img, "schema 4", (x1+70, iy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1, cv2.LINE_AA)
        # Schedule data load in a background thread and trigger view refresh on completion
        def update_st_data_db():
            nonlocal price_data, full_rows, full_n, view_start, view_end, init_view
            nonlocal st_view_changed, ts_stock_code, ts_data_typ, t3_loader
            # If a previous loader is still running, wait briefly for it to finish to avoid overlap
            try:
                if t3_loader is not None and t3_loader.is_alive():
                    try:
                        t3_loader.join(timeout=0.2)
                    except Exception:
                        pass
                # Define loader target
                def _t3_load_and_refresh():
                    nonlocal price_data, full_rows, full_n, view_start, view_end, init_view
                    nonlocal st_view_changed, ts_stock_code, ts_data_typ
                    try:
                        # Decide timeframe and load from DB
                        if ts_data_typ == 1:
                            new_price_data = load_ts_minute_db_data(ts_stock_code)
                        else:
                            new_price_data = load_ts_daily_db_data(ts_stock_code)

                        if not isinstance(new_price_data, dict):
                            return

                        # Assign new data
                        price_data = new_price_data
                        full_rows = price_data.get('rows', [])[:]
                        full_n = len(full_rows)

                        # Compute initial view from config if valid
                        init_view = None
                        try:
                            vs_raw = (cfg.get('view_start') or '').strip()
                            ve_raw = (cfg.get('view_end') or '').strip()
                            vs = int(vs_raw) if vs_raw != '' else -1
                            ve = int(ve_raw) if ve_raw != '' else -1
                            if 0 <= vs <= ve < full_n:
                                init_view = (vs, ve)
                        except Exception:
                            init_view = None

                        if init_view is not None:
                            view_start, view_end = init_view
                        else:
                            view_start = 0
                            view_end = max(0, full_n - 1)

                        # Trigger plotting update after data refresh
                        st_view_changed += 1
                    except Exception:
                        # Best-effort: ignore loader exceptions to avoid crashing UI loop
                        pass
                # Start new loader thread
                t3_loader = threading.Thread(target=_t3_load_and_refresh, name="t3_loader")
                t3_loader.daemon = True
                t3_loader.start()
            except Exception:
                pass

        # Helper to (re)start the background data updater thread based on timeframe
        def restart_t1_updater():
            nonlocal t1_updater, t1_updater_event, current_ticker, ts_data_typ, ts_stock_type
            nonlocal st_data_changed, st_view_changed
            # Callback to mark data/view changed when minute updater inserts rows
            def _t1_minute_data():
                nonlocal st_data_changed, st_view_changed
                st_data_changed = 1
                # st_view_changed = 1
            try:
                # Stop existing updater if running
                if t1_updater is not None and t1_updater.is_alive():
                    t1_updater_event.set()
                    for _ in range(200):
                        if not t1_updater.is_alive():
                            break
                        time.sleep(0.05)
                # Start a new updater with a fresh event
                t1_updater_event = threading.Event()
                if ('us' == ts_stock_type):
                    if ts_data_typ == 1:
                        t1_updater = threading.Thread(target=qdb14.updater_minute_thd, args=(current_ticker, t1_updater_event, "abc"), kwargs={'on_update': _t1_minute_data})
                        t1_updater.start()
                    else:
                        # Prefer single-run update to avoid full-loop spam
                        def _us_once():
                            try:
                                if callable(us_update_once):
                                    us_update_once(current_ticker)
                                else:
                                    # Fallback to full updater for current ticker only
                                    qdb15.updater_daily_thd(current_ticker, threading.Event(), on_update=_t1_minute_data)
                            except Exception:
                                pass
                            # Trigger reload after update
                            _t1_minute_data()
                        t1_updater = threading.Thread(target=_us_once, name="us_update_once")
                        t1_updater.daemon = True
                        t1_updater.start()
                else:
                    if ts_data_typ == 1:
                        # Minute for CN not defined; no-op
                        pass
                    else:
                        def _cn_once():
                            try:
                                if callable(cn_update_once):
                                    cn_update_once(current_ticker)
                                else:
                                    qdb11.updater_daily_thd(current_ticker, threading.Event(), on_update=_t1_minute_data)
                            except Exception:
                                pass
                            _t1_minute_data()
                        t1_updater = threading.Thread(target=_cn_once, name="cn_update_once")
                        t1_updater.daemon = True
                        t1_updater.start()
            except Exception:
                pass
        # Centralized updater: render once and parse results into UI state, then redraw overlays
        def update_st_view_plotting():
            nonlocal base, xs, ys, rows, dif, dea, k, d, j, rsi14, obv, ma_data
            nonlocal left_margin, right_margin, top_margin, bottom_margin, W, H, title_box
            nonlocal win_title, ts_stock_code, resolved_name
            nonlocal view_start, view_end
            try:
                # 1) Prepare base image and plotting state
                tt = None
                if price_data is None:
                    # No data available: keep current base size and set minimal margins
                    try:
                        H, W = base.shape[0], base.shape[1]
                    except Exception:
                        H, W = 600, 1000
                    left_margin = max(2, left_margin or 2)
                    right_margin = max(0, right_margin or 0)
                    top_margin = max(0, top_margin or 0)
                    bottom_margin = max(0, bottom_margin or 0)
                else:
                    new_plot = render_price_plot(
                        price_data, width, height,
                        view=(view_start, view_end),
                        ma_flags=ma_flags,
                        persist_view=False,
                        tick_target=axis_yticks_target
                    )
                    if new_plot is None:
                        return
                    base = new_plot["img"]
                    xs = new_plot["xs"]
                    ys = new_plot["ys"]
                    rows = new_plot["rows"]
                    dif = new_plot.get("dif")
                    dea = new_plot.get("dea")
                    k = new_plot.get("k")
                    d = new_plot.get("d")
                    j = new_plot.get("j")
                    rsi14 = new_plot.get("rsi14")
                    obv = new_plot.get("obv")
                    ma_data = new_plot.get("ma")
                    left_margin = new_plot["left_margin"]
                    right_margin = new_plot["right_margin"]
                    top_margin = new_plot["top_margin"]
                    bottom_margin = new_plot["bottom_margin"]
                    W = new_plot["W"]
                    H = new_plot["H"]
                    tt = new_plot.get("title_text")

                # 2) Single-pass overlays: title, QM label, timeframe, optional MA menu
                title_text = str(tt or (f"{ts_stock_code} - {resolved_name}" if resolved_name else ts_stock_code or "")).strip()
                if not title_text:
                    title_text = "set a stock"
                base, title_box = drawTitleBox(base, title_text, left_margin)
                draw_quantm_label(base)
                draw_timeframe_labels(base)
                overlay = base.copy()
                if ma_menu_visible and ma_menu_pos:
                    draw_ma_menu(overlay, x=ma_menu_pos[0], y=ma_menu_pos[1])
                cv2.imshow(win_title, overlay)
            except Exception:
                pass

        # Mouse callback to draw crosshair and show date/price
        def on_mouse(event, x, y, flags, userdata):
            nonlocal ts_stock_code, base, xs, ys, rows
            nonlocal left_margin, right_margin, top_margin, bottom_margin, W, H, title_box
            nonlocal full_rows, full_n, view_start, view_end
            nonlocal dragging, drag_start_x, drag_start_view_start, drag_start_view_end, last_shift
            nonlocal quantm_box, t2_quantm_running, t2_quantm, t2_quantm_event
            nonlocal price_data, ma_menu_visible, ma_menu_pos
            nonlocal dlg_input_stock
            nonlocal dif, dea, k, d, j, rsi14, obv, ma_data
            nonlocal current_ticker, hqm_score_val
            nonlocal timeframe_boxes, ts_data_typ
            nonlocal st_view_changed, st_data_changed
            nonlocal t1_updater, t1_updater_event
            

            # Click timeframe labels first
            if event == cv2.EVENT_LBUTTONDOWN and timeframe_boxes:
                def _hit(box):
                    x1, y1, x2, y2 = box
                    return x1 <= x <= x2 and y1 <= y <= y2
                if _hit(timeframe_boxes.get('1M', (0,0,0,0))) and ts_data_typ != 1:
                    ts_data_typ = 1
                    try:
                        cfg_util.write_config({'ts_data_type': '1'})
                    except Exception:
                        pass
                    draw_timeframe_labels(base)
                    # Reload data for new timeframe
                    st_data_changed = 1
                    # Restart background updater to minute
                    restart_t1_updater()
                    cv2.imshow(win_title, base)
                    return
                if _hit(timeframe_boxes.get('1D', (0,0,0,0))) and ts_data_typ != 2:
                    ts_data_typ = 2
                    try:
                        cfg_util.write_config({'ts_data_type': '2'})
                    except Exception:
                        pass
                    draw_timeframe_labels(base)
                    # Reload data for new timeframe
                    st_data_changed = 1
                    # Restart background updater to daily
                    restart_t1_updater()
                    cv2.imshow(win_title, base)
                    return
            # Right-click: toggle MA menu
            # Right-click: toggle MA menu at click position
            if event == cv2.EVENT_RBUTTONDOWN:
                ma_menu_visible = not ma_menu_visible
                if ma_menu_visible:
                    # Clamp the menu position so it stays within window bounds
                    menu_w = 180
                    # reflect new item count (6 MAs + MACD + KDJ + Volume + OBV + Cross + schema1 + schema2 + schema3)
                    menu_h = 10 + 24*14 + 22
                    mx = max(5, min(x, W - menu_w - 5))
                    my = max(5, min(y, H - menu_h - 5))
                    ma_menu_pos = (mx, my)
                    overlay = base.copy()
                    draw_ma_menu(overlay, x=ma_menu_pos[0], y=ma_menu_pos[1])
                    # Ensure QM label stays on top of overlays
                    cv2.imshow(win_title, overlay)
                return

            # If MA menu visible, handle clicks to toggle items
            if ma_menu_visible and event == cv2.EVENT_LBUTTONDOWN:
                if ma_menu_box:
                    x1m, y1m, x2m, y2m = ma_menu_box
                    if x1m <= x <= x2m and y1m <= y <= y2m:
                        for w, box in ma_item_boxes:
                            bx1, by1, bx2, by2 = box
                            if bx1 <= x <= bx2 and by1 <= y <= by2:
                                ma_flags[w] = not ma_flags.get(w, True)
                                # If schema1 is enabled from the menu, auto-enable MACD and KDJ
                                if w == 'schema1' and ma_flags.get('schema1', False):
                                    ma_flags['macd'] = True
                                    ma_flags['kdj'] = True
                                    ma_flags[114] = True
                                # If schema1 is disabled from the menu, auto-disable MACD and KDJ
                                if w == 'schema1' and not ma_flags.get('schema1', False):
                                    ma_flags['macd'] = False
                                    ma_flags['kdj'] = False
                                    ma_flags[114] = False
                                # If schema3 is enabled, auto-enable MA20 and OBV
                                if w == 'schema3' and ma_flags.get('schema3', False):
                                    ma_flags[20] = True
                                    ma_flags['obv'] = True
                                # If schema3 is disabled, auto-disable MA20 and OBV
                                if w == 'schema3' and not ma_flags.get('schema3', False):
                                    ma_flags[20] = False
                                    ma_flags['obv'] = False
                                # If schema4 is toggled, mirror RSI panel visibility
                                if w == 'schema4' and ma_flags.get('schema4', False):
                                    ma_flags['rsi'] = True
                                if w == 'schema4' and not ma_flags.get('schema4', False):
                                    ma_flags['rsi'] = False
                                # Persist Plot Options to config
                                try:
                                    cfg_util.write_config({
                                        'ma5':  '1' if ma_flags.get(5, False) else '0',
                                        'ma10': '1' if ma_flags.get(10, False) else '0',
                                        'ma20': '1' if ma_flags.get(20, False) else '0',
                                        'ma30': '1' if ma_flags.get(30, False) else '0',
                                        'ma60': '1' if ma_flags.get(60, False) else '0',
                                        'ma114': '1' if ma_flags.get(114, False) else '0',
                                        'macd': '1' if ma_flags.get('macd', False) else '0',
                                        'kdj':  '1' if ma_flags.get('kdj', False) else '0',
                                        'rsi':  '1' if ma_flags.get('rsi', False) else '0',
                                        'volume': '1' if ma_flags.get('volume', False) else '0',
                                        'obv':  '1' if ma_flags.get('obv', False) else '0',
                                        'cross': '1' if ma_flags.get('cross', False) else '0',
                                        'schema1': '1' if ma_flags.get('schema1', False) else '0',
                                        'schema2': '1' if ma_flags.get('schema2', False) else '0',
                                        'schema3': '1' if ma_flags.get('schema3', False) else '0',
                                        'schema4': '1' if ma_flags.get('schema4', False) else '0',
                                        'schema5': '1' if ma_flags.get('schema5', False) else '0',
                                    })
                                except Exception:
                                    pass
                                # Trigger centralized update
                                st_view_changed += 1
                                return
                # click outside closes menu
                ma_menu_visible = False
                cv2.imshow(win_title, base)
                return
            # Click on title to edit stock code
            if event == cv2.EVENT_LBUTTONDOWN and title_box is not None:
                # Destroy previous dialog if it exists
                if dlg_input_stock is not None:
                    try:
                        dlg_input_stock.destroy()
                    except Exception:
                        pass
                    dlg_input_stock = None
                x1, y1, x2, y2 = title_box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    try:
                        dlg_input_stock = tk.Tk()
                        dlg_input_stock.withdraw()
                        new_code = simpledialog.askstring(
                            "Edit Stock Code",
                            f"Enter stock ts_code (current: {ts_stock_code}):",
                            parent=dlg_input_stock,
                            initialvalue=ts_stock_code
                        )
                        try:
                            dlg_input_stock.destroy()
                        except Exception:
                            pass
                        dlg_input_stock = None
                    except Exception:
                        new_code = input(f"Enter stock ts_code (current: {ts_stock_code}): ")
                    if new_code is not None:
                        new_code = new_code.strip()
                    if new_code:
                        # Update state and trigger background data reload
                        ts_stock_code, resolved_name = cfg_util._resolve_ts_code(new_code)
                        try:
                            cfg_util.write_config({'ts_stock_code': ts_stock_code})
                        except Exception:
                            pass
                        # Refresh ticker and HQM score
                        current_ticker = ts_stock_code
                        hqm_score_val = qalg11.latest_hqm_score_for(current_ticker)
                        # Request data reload
                        st_data_changed = 1
                        # Restart background updater to minute
                        restart_t1_updater()
                        return
            # Click on QuantM to start background reader thread
            if event == cv2.EVENT_LBUTTONDOWN and quantm_box is not None:
                qx1, qy1, qx2, qy2 = quantm_box
                if qx1 <= x <= qx2 and qy1 <= y <= qy2:
                    if not t2_quantm_running:
                        t2_quantm_running = True
                        # Start reader thread to build HQM dataframe
                        t2_quantm_event.clear()                        
                        t2_quantm = threading.Thread(target=reader_db_qm_thd, args=(None, t2_quantm_event), daemon=True)
                        t2_quantm.start()
                        # Immediate visual feedback (red)
                        draw_quantm_label(base)
                        cv2.imshow(win_title, base)
                        # Watcher to flip QM label to green when finished
                        def _quantm_watch():
                            try:
                                t2_quantm_event.wait()
                                # Refresh HQM score after file is written
                                hqm_score_val = qalg11.latest_hqm_score_for(current_ticker)
                                draw_quantm_label(base)
                                cv2.imshow(win_title, base)
                            except Exception:
                                pass
                        threading.Thread(target=_quantm_watch, daemon=True).start()
                    return
            # Start drag within plot area for panning
            if event == cv2.EVENT_LBUTTONDOWN:
                if left_margin <= x <= W - right_margin and top_margin <= y <= H - bottom_margin and full_n >= 2:
                    dragging = True
                    drag_start_x = x
                    drag_start_view_start = view_start
                    drag_start_view_end = view_end
                    last_shift = 0
                    return
            # End drag
            if event == cv2.EVENT_LBUTTONUP:
                if dragging:
                    dragging = False
                    # Persist final view range once on drag end
                    try:
                        cfg_util.write_config({'view_start': str(view_start), 'view_end': str(view_end)})
                    except Exception:
                        pass
                    return
            # Drag move: pan view
            if event == cv2.EVENT_MOUSEMOVE and dragging:
                plot_w = (W - left_margin - right_margin)
                L = max(1, drag_start_view_end - drag_start_view_start + 1)
                dx = x - drag_start_x
                # Translate pixel drift to index shift (tunable sensitivity)
                shift = int(round(-dx * L / float(plot_w) * drag_sensitivity))
                if shift != last_shift:
                    new_start = drag_start_view_start + shift
                    new_end = drag_start_view_end + shift
                    # Clamp to bounds; preserve window size
                    if new_start < 0:
                        new_end += -new_start
                        new_start = 0
                    if new_end > full_n - 1:
                        shift_back = new_end - (full_n - 1)
                        new_start -= shift_back
                        new_end = full_n - 1
                        if new_start < 0:
                            new_start = 0
                    # Update view state and trigger centralized update
                    view_start, view_end = new_start, new_end
                    last_shift = shift
                    st_view_changed += 1
                return

            # Mouse wheel zoom around current mouse position
            if event == cv2.EVENT_MOUSEWHEEL:
                if left_margin <= x <= W - right_margin and top_margin <= y <= H - bottom_margin and full_n >= 2:
                    # Current view length
                    L = max(1, view_end - view_start + 1)
                    # index within current view
                    try:
                        idx_view = int(np.argmin(np.abs(xs - x)))
                    except Exception:
                        idx_view = L // 2
                    idx_global = view_start + idx_view
                    # Zoom direction: flags > 0 => zoom in, else out
                    zoom_in = flags > 0
                    factor = 0.8 if zoom_in else 1.25
                    min_len = max(20, min(50, full_n // 20))  # adaptive minimal window
                    max_len = full_n
                    new_len = int(round(L * factor))
                    new_len = max(min_len, min(max_len, new_len))
                    if new_len == L:
                        pass # print("Zoom limit reached", factor, min_len, max_len, new_len)
                    else:
                        # Keep relative position in view
                        r = idx_view / float(L)
                        new_start = int(round(idx_global - r * new_len))
                        new_end = new_start + new_len - 1
                        # Clamp to bounds, preserve window size
                        if new_start < 0:
                            new_end += -new_start
                            new_start = 0
                        if new_end > full_n - 1:
                            shift = new_end - (full_n - 1)
                            new_start -= shift
                            new_end = full_n - 1
                            if new_start < 0:
                                new_start = 0
                        # Update view state and trigger centralized update
                        view_start, view_end = new_start, new_end
                        st_view_changed += 1
                        # Persist view range after zoom event
                        try:
                            cfg_util.write_config({'view_start': str(view_start), 'view_end': str(view_end)})
                        except Exception:
                            pass
                return
            if event == cv2.EVENT_MOUSEMOVE and left_margin <= x <= W - right_margin and top_margin <= y <= H - bottom_margin:    
                if(xs is None or ys is None or rows is None):
                    return            
                idx = int(np.argmin(np.abs(xs - x)))
                px = int(xs[idx])
                py = int(ys[idx])
                dstr = str(rows[idx][0])
                price_raw = rows[idx][1]
                try:
                    price_val = float(price_raw)
                except Exception:
                    price_val = price_raw
                date_label = f"{dstr[2:4]}/{dstr[4:6]}/{dstr[6:8]}" if len(dstr) == 8 else dstr
                overlay = base.copy()
                # Crosshair lines
                cv2.line(overlay, (x, top_margin), (x, H - bottom_margin), (180, 180, 180), 1, cv2.LINE_AA)
                cv2.line(overlay, (left_margin, py), (W - right_margin, py), (180, 180, 180), 1, cv2.LINE_AA)
                # Point marker
                cv2.circle(overlay, (px, py), 4, (0, 120, 255), -1, lineType=cv2.LINE_AA)

                # Label box with DIF/DEA and K/D/J values if available, the hover tooltip shows
                if isinstance(price_val, float):
                    base_label = f"{date_label}   {price_val:.2f}"
                else:
                    base_label = f"{date_label}   {price_val}"
                try:
                    parts = [base_label]
                    # MA114 if toggled on
                    if ma_flags.get(114, False) and isinstance(ma_data, dict):
                        ma114 = ma_data.get(114)
                        if isinstance(ma114, list) and 0 <= idx < len(ma114):
                            v = ma114[idx]
                            try:
                                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                    parts.append(f"MA114:{float(v):.2f}")
                            except Exception:
                                pass
                    # DIF/DEA if MACD is toggled on
                    if ma_flags.get('macd', False) and dif is not None and dea is not None and 0 <= idx < len(dif) and 0 <= idx < len(dea):
                        dv = dif[idx]
                        ev = dea[idx]
                        if dv is not None and ev is not None:
                            parts.append(f"DIF:{float(dv):.3f}")
                            parts.append(f"DEA:{float(ev):.3f}")
                    # RSI value if RSI panel is toggled on
                    if ma_flags.get('rsi', False) and rsi14 is not None and 0 <= idx < len(rsi14):
                        rv = rsi14[idx]
                        if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                            parts.append(f"RSI:{float(rv):.2f}")
                    # K/D/J if KDJ is toggled on
                    '''
                    if ma_flags.get('kdj', False) and k is not None and d is not None and j is not None and 0 <= idx < len(k) and 0 <= idx < len(d) and 0 <= idx < len(j):
                        kv = k[idx]
                        dv2 = d[idx]
                        jv = j[idx]
                        if kv is not None and dv2 is not None and jv is not None:
                            parts.append(f"K:{float(kv):.2f}")
                            parts.append(f"D:{float(dv2):.2f}")
                            parts.append(f"J:{float(jv):.2f}")
                    '''
                    # QM value if schema2 (QM) is toggled on
                    if ma_flags.get('schema2', False):
                        try:
                            qm_vals = price_data.get('qm_scores') if isinstance(price_data, dict) else None
                            if isinstance(qm_vals, list):
                                gi = view_start + idx
                                if 0 <= gi < len(qm_vals):
                                    qmv = qm_vals[gi]
                                    if qmv is not None:
                                        qmvf = float(qmv)
                                        if not (isinstance(qmvf, float) and np.isnan(qmvf)):
                                            parts.append(f"QM:{qmvf:.2f}")
                        except Exception:
                            pass
                    # OBV value if OBV is toggled on (compact format)
                    if ma_flags.get('obv', False):
                        try:
                            # OBV is returned in the plot for the current view
                            if 'obv' in locals() and obv is not None and 0 <= idx < len(obv):
                                ov = obv[idx]
                                if ov is not None and not (isinstance(ov, float) and np.isnan(ov)):
                                    parts.append(f"OBV:{format_compact_number(ov)}")
                        except Exception:
                            pass
                    label = "   ".join(parts)
                except Exception:
                    label = base_label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                bx = min(max(x + 10, 5), W - tw - 15)
                by = min(max(y - 10, th + 10), H - th - 10)
                cv2.rectangle(overlay, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), (255, 255, 255), -1)
                cv2.rectangle(overlay, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), (180, 180, 180), 1)
                cv2.putText(overlay, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

                if ma_menu_visible and ma_menu_pos:
                    draw_ma_menu(overlay, x=ma_menu_pos[0], y=ma_menu_pos[1])
                cv2.imshow(win_title, overlay)

        cv2.setMouseCallback(win_title, on_mouse)
        # Trigger the first update via st_view_changed; centralized updater will render
        st_view_changed = 1
        # Start updater thread
        t1_updater_event = threading.Event()
        t1_updater = None
        restart_t1_updater()
        # Keyboard loop: handle '1'/'END' to anchor to end, '7'/'HOME' to anchor to start
        KEY_END = 2686976   # Windows HighGUI END key code (may not be detected)
        KEY_HOME = 2359296  # Windows HighGUI HOME key code (may vary)
        KEY_ESC = 27
        KEY_Q = ord('q')
        KEY_ONE = ord('1')
        KEY_3 = ord('3')
        KEY_PGDN = 2621440   # Windows HighGUI PageDown key code (may vary)
        KEY_SEVEN = ord('7')
        KEY_9 = ord('9')
        KEY_PGUP = 2162688   # Windows HighGUI PageUp key code (may vary)
        KEY_5 = ord('5')
        KEY_0 = ord('0')
        KEY_BACKSPACE = 8  # Common Backspace code in HighGUI

        # Helper: stop QM thread cooperatively
        def _t2_stop_quantm():
            nonlocal t2_quantm_running, t2_quantm_event, t2_quantm
            try:
                if t2_quantm_running and t2_quantm is not None and t2_quantm.is_alive():
                    t2_quantm_event.set()
                    for _ in range(200):  # ~10s max
                        if not t2_quantm.is_alive():
                            break
                        time.sleep(0.05)
                    t2_quantm_running = False
            except Exception:
                pass
        
        # Helper: apply a new view range and update/persist UI
        def apply_view(new_start: int, new_end: int, persist: bool = True) -> bool:
            nonlocal view_start, view_end, st_view_changed
            try:
                view_start, view_end = new_start, new_end
                st_view_changed += 1
                if persist:
                    try:
                        cfg_util.write_config({'view_start': str(view_start), 'view_end': str(view_end)})
                    except Exception:
                        pass
                return True
            except Exception:
                return False

        # ---- View computation helpers ----
        def _date_at(idx: int):
            try:
                dstr = str(full_rows[idx][0])
                if len(dstr) == 8:
                    return datetime.strptime(dstr, "%Y%m%d").date()
                return None
            except Exception:
                return None

        def compute_anchor_end_one_year():
            try:
                new_end = max(0, full_n - 1)
                dt_end = _date_at(new_end)
                if dt_end:
                    dt_start_target = dt_end - timedelta(days=365)
                    i = new_end
                    start_idx = 0
                    while i >= 0:
                        di = _date_at(i)
                        if di is None or di < dt_start_target:
                            start_idx = i + 1
                            break
                        i -= 1
                    if i < 0:
                        start_idx = 0
                    new_start = max(0, min(start_idx, new_end))
                    return new_start, new_end
                approx = 250
                new_end = max(0, full_n - 1)
                new_start = max(0, new_end - approx)
                return new_start, new_end
            except Exception:
                new_end = max(0, full_n - 1)
                new_start = max(0, new_end - 250)
                return new_start, new_end

        def compute_page_forward_one_year():
            try:
                curr_len = max(1, view_end - view_start + 1)
                end_idx = max(0, min(full_n - 1, view_end))
                dt_end = _date_at(end_idx)
                if dt_end:
                    dt_target_end = dt_end + timedelta(days=365)
                    i = end_idx
                    new_end = end_idx
                    while i < full_n:
                        di = _date_at(i)
                        if di is None or di > dt_target_end:
                            new_end = max(end_idx, i - 1)
                            break
                        i += 1
                    if i >= full_n:
                        new_end = full_n - 1
                    new_start = max(0, new_end - curr_len + 1)
                    return new_start, new_end
                approx = 250
                new_end = min(full_n - 1, end_idx + approx)
                new_start = max(0, new_end - curr_len + 1)
                return new_start, new_end
            except Exception:
                curr_len = max(1, view_end - view_start + 1)
                new_end = min(full_n - 1, view_end + 250)
                new_start = max(0, new_end - curr_len + 1)
                return new_start, new_end

        def compute_page_backward_one_year():
            try:
                curr_len = max(1, view_end - view_start + 1)
                start_idx = max(0, min(full_n - 1, view_start))
                dt_start = _date_at(start_idx)
                if dt_start:
                    dt_target_start = dt_start - timedelta(days=365)
                    i = start_idx
                    new_start = start_idx
                    while i >= 0:
                        di = _date_at(i)
                        if di is None or di < dt_target_start:
                            new_start = max(0, i + 1)
                            break
                        i -= 1
                    if i < 0:
                        new_start = 0
                    new_end = min(full_n - 1, new_start + curr_len - 1)
                    return new_start, new_end
                approx = 250
                new_start = max(0, start_idx - approx)
                new_end = min(full_n - 1, new_start + curr_len - 1)
                return new_start, new_end
            except Exception:
                curr_len = max(1, view_end - view_start + 1)
                new_start = max(0, view_start - 250)
                new_end = min(full_n - 1, new_start + curr_len - 1)
                return new_start, new_end

        def compute_anchor_end_same_length():
            # Preserve current view length, anchoring the window to the latest available date
            curr_len = max(1, view_end - view_start + 1)
            new_end = max(0, full_n - 1)
            new_start = max(0, new_end - curr_len + 1)
            return new_start, new_end

        def compute_anchor_start_same_length():
            curr_len = max(1, view_end - view_start + 1)
            new_start = 0
            new_end = min(full_n - 1, new_start + curr_len - 1)
            return new_start, new_end

        def compute_full_view():
            return 0, max(0, full_n - 1)
        # Once ST data is changed, plotting will be updated via update_st_view_plotting()
        while True:
            if(st_data_changed > 0):
                update_st_data_db()
                st_data_changed = 0
            if(st_view_changed > 0):
                update_st_view_plotting()
                st_view_changed = 0
            # Detect window close via title bar 'X' and stop QM thread
            try:
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    _t2_stop_quantm()
                    break
            except Exception:
                _t2_stop_quantm()
                break
            k = cv2.waitKey(20) & 0xFFFFFFFF
            if k == -1: continue
            if(k == 4294967295): continue
            if k in (KEY_ESC, KEY_Q): break
                # Cooperatively stop QM thread before exit
            # else: print('key', k)
            if k == KEY_0:
                # Anchor the current view to show at most 1 year ending at the latest date
                new_start, new_end = compute_anchor_end_one_year()
                apply_view(new_start, new_end, persist=True)
            elif k in (KEY_ONE, KEY_END):
                # Anchor the current view to end at the latest date, preserving window length
                new_start, new_end = compute_anchor_end_same_length()
                apply_view(new_start, new_end, persist=True)
            elif k in (KEY_3, KEY_PGDN):
                # Page forward by up to 1 year, preserving current window length
                new_start, new_end = compute_page_forward_one_year()
                apply_view(new_start, new_end, persist=True)
            elif k in (KEY_9, KEY_PGUP):
                # Page backward by up to 1 year, preserving current window length
                new_start, new_end = compute_page_backward_one_year()
                apply_view(new_start, new_end, persist=True)
            elif k in (KEY_SEVEN, KEY_HOME):
                # Anchor the current view to start at the very beginning, preserving window length
                new_start, new_end = compute_anchor_start_same_length()
                apply_view(new_start, new_end, persist=True)
            elif k == KEY_5:
                # Show the full dataset range
                new_start, new_end = compute_full_view()
                apply_view(new_start, new_end, persist=True)
            elif k == KEY_BACKSPACE:
                # Backspace: when 1D view is active, delete last 10 days from DB for current ticker
                try:
                    if ts_data_typ == 2 and ts_stock_code:
                        removed = delete_last_n_days(ts_stock_code, 10)
                        if removed > 0:
                            # Trigger immediate reload to reflect deletion
                            st_data_changed = 1
                            # Then ask updater to fetch latest rows for current ticker
                            restart_t1_updater()
                            print(f"[qmchart] Removed {removed} daily rows for {ts_stock_code} and requested updater fetch")
                except Exception as e:
                    print(f"[qmchart] Backspace deletion failed: {e}")
        cv2.destroyAllWindows()
        t1_updater_event.set()
        t1_updater.join()
        # Stop QuantM thread if running
        try:
            _t2_stop_quantm()
        except Exception:
            pass
        t3_loader.join()

if __name__ == "__main__":
    main()
