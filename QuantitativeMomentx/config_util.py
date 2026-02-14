#
#       config_util.py
# This script manages reading and writing configuration settings
#   to and from an XML file. It provides default values for various
#   configuration parameters related to stock data processing.
#
import os
import xml.etree.ElementTree as ET
import sys
import pandas as pd
import difflib

CONFIG_PATH = '.\\config.xml'

DEFAULTS = {
    'ts_stock_type': 'cn',                              # 'cn' for China stocks, 'us' for USA stocks
    'ts_stock_code': '301060.SZ',
    'ts_stock_list': 'china_a_share_list6k.csv',        # not used
    'DB_PATH': '.\\reports\\quant_stock06.db',          # not used
    'D_TUSHARE_TOKEN': '6d4a772da291aef3b584bb3aa213d1b1e0ef521bffc34eadad241e5a',
    'D_MAX_SPEED': '1250',
    'D_STOCK_UPDATING': '0',
    'D_MA_DEFAULT': 'simple',                            # Default moving average type
    'D_MA_PERIOD': '20',                                 # Default moving average period
    # Timeframe: 1 = 1D (daily), 2 = 1M (minute)
    'ts_data_type': '2',
    # Persisted chart view (start/end indices within full rows)
    'view_start': '',
    'view_end': '',
    # Persisted MA options (1=on, 0=off)
    'ma5': '0',
    'ma10': '0',
    'ma20': '0',
    'ma30': '0',
    'ma60': '0',
    'ma114': '1',
    'macd': '0',
    'kdj': '0',
    'rsi': '0',
    'volume': '1',
    'obv': '0',
    'cross': '0',
    'schema1': '0',
    'schema2': '0',
    'schema3': '0',
    'schema4': '0',
    'D_QM_UPDATING': '20040101',
}


def read_config(path: str | None = None) -> dict:
    cfg_path = path or CONFIG_PATH
    data = DEFAULTS.copy()
    try:
        if os.path.exists(cfg_path):
            tree = ET.parse(cfg_path)
            root = tree.getroot()
            for key in DEFAULTS.keys():
                node = root.find(key)
                if node is not None and node.text:
                    data[key] = node.text.strip()
    except Exception:
        # Return defaults on parse error
        pass
    return data


def write_config(updates: dict, path: str | None = None) -> None:
    cfg_path = path or CONFIG_PATH
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    current = read_config(cfg_path)
    current.update({k: str(v) for k, v in updates.items() if v is not None})

    root = ET.Element('config')
    for key in DEFAULTS.keys():
        el = ET.SubElement(root, key)
        el.text = current.get(key, DEFAULTS[key])
    tree = ET.ElementTree(root)
    tree.write(cfg_path, encoding='utf-8', xml_declaration=True)

# Write a status line that overwrites the current console line (no newline)
def o_status_line(text: str, width: int = 120):
    try:
        sys.stdout.write('\r' + text.ljust(width))
        sys.stdout.flush()
    except Exception:
        # Fallback to regular print if needed
        print(text)
# save the data to database table tb_m_[ts_code]
def _sanitize_table_name(ts_code: str) -> str:
    # Build tb_m_<ts_code> in lowercase and replace non-alphanumeric characters
    s = str(ts_code).lower()
    safe = ''.join(ch if ch.isalnum() else '_' for ch in s)
    return f"tb_m_{safe}"   

_cfg = read_config()
def get_stocks_set3():
    f_stock_list = ''
    if(_cfg.get('ts_stock_type', 'cn') == 'us'):
        f_stock_list ='us_a_share_list500.csv'
    else:       
        f_stock_list = 'china_a_share_list6k.csv'
    return pd.read_csv('./QuantitativeMomentx/' + f_stock_list, dtype=str)

def get_stock_name(ts_code: str) -> str:
    try:
        df = get_stocks_set3()
        match = df[df['Ticker'] == ts_code]
        if not match.empty and 'name' in match.columns:
            return str(match.iloc[0]['name'])
    except Exception:
        pass
    return ""

# Resolve ts_code to best match from stock list
def _resolve_ts_code(s: str):
        raw = str(s or '').strip()
        q = raw.upper().replace(' ', '').replace('-', '').replace('_', '')
        try:
            df = get_stocks_set3()
        except Exception:
            return raw, get_stock_name(raw)
        tickers = df['Ticker'].astype(str).str.upper().fillna('')
        names = df['name'].astype(str).fillna('')
        digits = ''.join(ch for ch in q if ch.isdigit())
        def sim(a,b):
            try:
                return difflib.SequenceMatcher(None, a, b).ratio()
            except Exception:
                return 0.0
        best = (0.0, q, get_stock_name(q))
        for code, name in zip(tickers, names):
            score = sim(q, code)
            if digits:
                if code.startswith(digits):
                    score += 0.6
                elif digits in code:
                    score += 0.3
            if q.endswith('.SZ') and code.endswith('.SZ'):
                score += 0.2
            if q.endswith('.SH') and code.endswith('.SH'):
                score += 0.2
            if any((ord(ch) > 127) or ch.isalpha() for ch in raw):
                score += 0.5 * sim(raw.upper(), str(name).upper())
            if score > best[0]:
                best = (score, code, name)
        return best[1], best[2]
     