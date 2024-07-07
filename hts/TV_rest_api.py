import time
import pytz
from datetime import datetime, timedelta
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

def get_tvdata_from_vdt_days_before(stock_code, stock_name, selected_minutes, vdt, days_before=0):

    vdt_date = pytz.timezone('Asia/Seoul').localize(datetime.strptime(vdt, "%Y%m%d"))
    vdt_delta = vdt_date - timedelta(days=0)
    today = datetime.now(pytz.timezone('Asia/Seoul'))
    days_difference = (today - vdt_delta).days + days_before

    interval = Interval.in_5_minute
    if selected_minutes:
        if selected_minutes == '1분':
            data_count = days_difference * 381  # 1분봉 기준 1시간: 60개, 7.5시간: 60 * 6 + 20 + 1
            interval = Interval.in_1_minute
        elif selected_minutes == '3분':
            data_count = days_difference * 127  # 3분봉 기준 1시간: 20개, 7.5시간: 20 * 6 + 6 + 1
            interval = Interval.in_3_minute
        elif selected_minutes == '5분':
            data_count = days_difference * 77 # 5분봉 기준 1시간: 12개, 7.5시간: 12 * 6 + 4 + 1
            interval = Interval.in_5_minute
        elif selected_minutes == '15분':
            data_count = days_difference * 30  # 15분봉 기준 1시간: 4개, 7.5시간: 30
            interval = Interval.in_15_minute
        elif selected_minutes == '30분':
            data_count = days_difference * 15  # 30분봉 기준 1시간: 2개, 7.5시간: 15
            interval = Interval.in_30_minute
        elif selected_minutes == '1시간':
            data_count = days_difference * 8  # 1시간 기준 1시간: 1개, 7.5시간: 8
            interval = Interval.in_1_hour
        elif selected_minutes == '1일':
            data_count = days_difference   # 1일 기준, 1개
            interval = Interval.in_daily

    tvdata = get_tvdata(stock_code=stock_code, stock_name=stock_name, data_count=data_count, interval=interval)
    if tvdata is not None and not tvdata.empty:

        # datatime 형식을 string 형식으로 변환
        tvdata.drop(columns=['symbol'], inplace=True)
        tvdata.index.name = 'time'

        #valley_string_datetime = vdt + "000000"
        price_df = tvdata.copy()

        return price_df

    return None