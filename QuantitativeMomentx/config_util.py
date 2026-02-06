#
#       config_util.py
# This script manages reading and writing configuration settings
#   to and from an XML file. It provides default values for various
#   configuration parameters related to stock data processing.
#
import os
import xml.etree.ElementTree as ET
import sys

CONFIG_PATH = '.\\config.xml'

DEFAULTS = {
    'ts_stock_type': 'us',                              # 'cn' for China stocks, 'us' for USA stocks
    'ts_stock_code': 'aapl',
    'ts_stock_list': 'china_a_share_list6k.csv',        # not used
    'DB_PATH': '.\\reports\\quant_stock06.db',          # not used
    'D_TUSHARE_TOKEN': 'ApplyYoursOnTushare.pro',
    'D_MAX_SPEED': '1250',
    'D_STOCK_UPDATING': '0',
    'D_MA_DEFAULT': 'simple',                            # Default moving average type
    'D_MA_PERIOD': '20',                                 # Default moving average period
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
    'obv': '0',
    'cross': '0',
    'schema1': '0',
    'schema2': '0',
    'schema3': '0',
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