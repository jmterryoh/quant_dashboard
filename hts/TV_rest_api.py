import time
import pandas as pd

# https://pypi.org/project/tradingview-datafeed/
# https://github.com/rongardF/tvdatafeed
from tvDatafeed import TvDatafeed, Interval

def get_tvdata(stock_code, stock_name, data_count, interval=Interval.in_1_minute):
    # 1. Tradingview 에서 과거 분봉데이터 가져오기
    # 0.3초~1.0초 소요
    # 에러가 가끔 발생되므로 에러발생시에는 2초후에 반복
    retry_count = 0
    while retry_count < 10:
        try:
            tv = TvDatafeed()
            tvdata = pd.DataFrame()
            # 1일: 381 [360(6시간) + 20(20분) + 1(30분 종가 1개)]
            tvdata = tv.get_hist(symbol=stock_code, exchange='KRX', interval=interval, n_bars=data_count)
            # when success
            if tvdata is not None:
                if not tvdata.empty:
                    return tvdata
        except Exception as e:
            print(f"Error in tv.get_hist({stock_code}, {stock_name}): ", e)
            retry_count += 1
            time.sleep(2)
    return pd.DataFrame()