from datetime import date
import sys, os
# Ensure this script's folder is on sys.path for direct module import
sys.path.insert(0, os.path.dirname(__file__))
import Quant_db_helper15 as q15

class _Stop:
    def is_set(self):
        return True

if __name__ == '__main__':
    # Use today's date; adjust as needed
    ts_code = 'AAPL'
    ts_day = date.today()
    q15.updater_minute_thd(ts_code, ts_day, _Stop())
    print('Minute update done for', ts_code, ts_day)
