import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import json
import time
import pytz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from util import zigzag as zz
from pages import sidebar as sb
from datetime import datetime, timedelta
from chart import TRV_lightchart_min_vwap as chart
from hts import YF_api as yf
from hts import TV_rest_api as tv
from middleware import md_client as dc

from tvDatafeed import TvDatafeed, Interval

global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH
global BB

global stocklist_df, selected_stockname
stocklist_df = {}
selected_stockname = None
selected_minutes = None


multiplier1 = 0.5
multiplier2 = 1.0
multiplier3 = 1.5
multiplier4 = 2.0


# 한국의 주요 공휴일 리스트 (2024년)
korean_holidays = [
    "2024-06-06", # 현충일
    "2024-08-15", # 광복절
    "2024-09-16", "2024-09-17", "2024-09-18", # 추석 연휴
    "2024-10-03", # 개천절
    "2024-10-09", # 한글날
    "2024-12-25" # 성탄절
]

current_page = "pages/page_chart_analysis_vwap.py"

def get_detected_dates(year_from, month_from, day_from):
    global korean_holidays

    # 한국 시간대 설정
    korea_timezone = pytz.timezone('Asia/Seoul')

    # 오늘 날짜 설정 (한국 시간대)
    end_date = datetime.now(korea_timezone)
    current_time = datetime.now(korea_timezone).strftime("%H%M%S")

    # 시작 날짜 설정
    start_date = korea_timezone.localize(datetime(year_from, month_from, day_from))

    # 문자열을 datetime 객체로 변환하고, 타임존 정보 추가
    korean_holidays = [korea_timezone.localize(datetime.strptime(date, "%Y-%m-%d")) for date in korean_holidays]

    # 결과 날짜 리스트 초기화
    business_days = []

    # 시작 날짜부터 종료 날짜까지 반복
    current_date = start_date
    while current_date <= end_date:
        # 주말이 아니고 공휴일이 아닌 경우
        if current_date.weekday() < 5 and current_date not in korean_holidays:
            #print(current_date, end_date, current_time)
            if current_date.date() == end_date.date() and current_time <= "160500":
                pass
            else:
                business_days.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    # 날짜 목록을 내림차순으로 정렬하고 상위 20개 추출
    business_days = sorted(business_days, reverse=True)[:20]        

    return business_days

# 다음 영업일을 계산하는 코드
# date_string: 기준일, holidays_datetime: 공휴일 목록(datetime 형식)
def get_next_business_day(date_string):
    global korean_holidays

    # 주어진 문자열을 datetime 객체로 변환
    date_obj = datetime.strptime(date_string, "%Y%m%d")
    
    # 다음 날 계산
    next_day_obj = date_obj + timedelta(days=1)
        
    # 다음 영업일 찾기 (주말 및 공휴일 건너뛰기)
    while next_day_obj.weekday() >= 5 or next_day_obj in korean_holidays:
        next_day_obj += timedelta(days=1)
    
    # 결과를 %Y%m%d 형식의 문자열로 변환
    next_day_str = next_day_obj.strftime("%Y%m%d")
    
    return next_day_str

def get_detected_stocks(idt):
    # task_name = "get_algo_stocks_increase10_by_date"
    # params = { "idt": f"{idt}" }
    # 20240701 increase10_history ADD 만 보여주던 목록을 snapshot 목록으로 변경함
    if idt >= "20240624":
        task_name = "get_algo_stocks_increase10_snapshot_by_date"
        params = { "ssdt": f"{idt}" }
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    stocklist_df = pd.DataFrame(respose["return"]["data"])
                    if stocklist_df is not None and not stocklist_df.empty:
                        stocknames = stocklist_df['name'].tolist()
                        return True, stocknames, stocklist_df
    else:
        task_name = "get_algo_stocks_increase10_by_date"
        params = { "idt": f"{idt}" }
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    stocklist_df = pd.DataFrame(respose["return"]["data"])
                    if stocklist_df is not None and not stocklist_df.empty:
                        stocknames = stocklist_df['name'].tolist()
                        return True, stocknames, stocklist_df

    return False, None, None

def get_tvdata_from_vdt(stock_code, stock_name, selected_minutes, vdt, days_more):

    vdt_date = pytz.timezone('Asia/Seoul').localize(datetime.strptime(vdt, "%Y%m%d"))
    today = datetime.now(pytz.timezone('Asia/Seoul'))
    vdt_one_week_ago = vdt_date - timedelta(days=0)
    days_difference = (today - vdt_one_week_ago).days + days_more

    interval = Interval.in_5_minute
    if selected_minutes:
        if selected_minutes == '1분':
            data_count = days_difference * 381  # 1분봉 기준 1시간: 60개, 7.5시간: 450
            interval = Interval.in_1_minute
        elif selected_minutes == '3분':
            data_count = days_difference * 127  # 3분봉 기준 1시간: 20개, 7.5시간: 150
            interval = Interval.in_3_minute
        elif selected_minutes == '5분':
            data_count = days_difference * 77 # 5분봉 기준 1시간: 12개, 7.5시간: 90
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

    tvdata = tv.get_tvdata(stock_code=stock_code, stock_name=stock_name, data_count=data_count, interval=interval)
    if tvdata is not None and not tvdata.empty:

        # datatime 형식을 string 형식으로 변환
        #tvdata = tvdata.reset_index()
        #tvdata['time'] = tvdata['datetime'].dt.strftime('%Y%m%d%H%M%S')
        #tvdata.set_index('time', inplace=True)
        #tvdata.sort_index()
        #tvdata['symbol'] = tvdata['symbol'].apply(lambda x: x.split(':')[1])  # symbol 값 KRX:329180 형식
        tvdata.drop(columns=['symbol'], inplace=True)
        tvdata.index.name = 'time'

        valley_string_datetime = vdt + "000000"
        price_df = tvdata.loc[tvdata.index >= valley_string_datetime].copy()

        return price_df

    return None

# Anchored VWAP 
def calculate_anchored_vwap(df, anchor_string_date, anchor_base, price_base, increase10_string_date, multiplier1, multiplier2, multiplier3, multiplier4):

    start_time = anchor_string_date + "000000"
    #end_time = anchor_string_date + "235959"
    end_time = increase10_string_date + "235959"  # anchor_datetime 시점변경: anchor 탐지일 ~ 장대양봉일

    oneday_df = df.loc[(start_time <= df.index) & (df.index <= end_time)].copy()
    #oneday_df.set_index('datetime', inplace=True)

    anchor_datetime = ""
    if anchor_base == "min":
        anchor_datetime = oneday_df[price_base].idxmin()
    elif anchor_base == "max":
        anchor_datetime = oneday_df[price_base].idxmax()

    #print(start_time, end_time, anchor_datetime)

    # Use .loc[] to avoid setting values on a slice of DataFrame
    df1 = df.loc[df.index >= anchor_datetime].copy()
    #df1.set_index('datetime', inplace=True)
 
    # VWAP 계산
    # VWAP Formula : ACC (close * volume) / ACC (volume)
    data = pd.DataFrame({
        f'{price_base}': df1[price_base],  # close value base
        'volume': df1['volume'],
        'bpvol': (df1[price_base] * df1['volume']),  # base_price * volume
        'bpbpvol': (df1[price_base] * df1[price_base] * df1['volume']),  # base_price * base_price * volume
        'vwap': (df1[price_base] * df1['volume']).cumsum() / df1['volume'].cumsum()
    })
    df1['accvol'] = data['volume'].cumsum()
    df1['accbpvol'] = data['bpvol'].cumsum()
    df1['accbpbpvol'] =  data['bpbpvol'].cumsum()
    df1['vwap'] = data['vwap']

    # STD 1, 2 계산
    if multiplier1 is not None or multiplier2 is not None or multiplier3 is not None or multiplier3 is not None:
        # data['TPP'] = (df1['close'] * df1['close'] * df1['volume']).cumsum() / df1['volume'].cumsum()
        # STD Formula : sqrt( (ACC(close X close * volume) / ACC(volume)) - VWAP * VWAP )
        # multiple 1: 1.28, mutiple 2: 2.01, multiple 3: 2.51
        data['tpp'] = (data['bpbpvol']).cumsum() / data['volume'].cumsum() # volume!! 
        data['vw*vw'] = data['vwap'] * data['vwap']
        data['std'] = (data['tpp'] - data['vw*vw']) ** 0.5
        if multiplier1 is not None:
            df1.loc[:, 'std1p'] = data['vwap'] + data['std'] * multiplier1
            df1.loc[:, 'std1m'] = data['vwap'] - data['std'] * multiplier1
        if multiplier2 is not None:
            df1.loc[:, 'std2p'] = data['vwap'] + data['std'] * multiplier2
            df1.loc[:, 'std2m'] = data['vwap'] - data['std'] * multiplier2
        if multiplier3 is not None:
            df1.loc[:, 'std3p'] = data['vwap'] + data['std'] * multiplier3
            df1.loc[:, 'std3m'] = data['vwap'] - data['std'] * multiplier3
        if multiplier4 is not None:
            df1.loc[:, 'std4p'] = data['vwap'] + data['std'] * multiplier4
            df1.loc[:, 'std4m'] = data['vwap'] - data['std'] * multiplier4

    return df1

# Input : df = Dataframe(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
#         anchor_date = anchord datetime
#         multiplier1 = vwap band 1
#         multiplier2 = vwap band 2 
# Output : Dataframe(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'STD+1', 'STD-1', 'STD+2', 'STD-2'])
def calculate_vwap_bands(df, anchor_date, anchor_base, price_base, volume_base, vwap_name, multiplier1, multiplier2):
    # anchor_date = datetime.strptime(anchor_date, "%Y%m%d").strftime("%Y-%m-%d")
    # start_time = anchor_date + " 00:00:00"
    # end_time = anchor_date + " 23:59:59"  # anchor_datetime 시점변경: anchor 탐지일 ~ 장대양봉일

    # oneday_df = df.loc[(start_time <= df.index) & (df.index <= end_time)].copy()

    # anchor_datetime = ""
    # if anchor_base == "min":
    #     anchor_datetime = oneday_df[price_base].idxmin()
    # elif anchor_base == "max":
    #     anchor_datetime = oneday_df[price_base].idxmax()
    # df = df.loc[df.index >= anchor_datetime].copy()

    # VWAP Formula : ACC (Close * Volume) / ACC (Volume)
    data = pd.DataFrame({
        f'{price_base}': df[price_base],  # Close value base
        f'{volume_base}': df[volume_base],
        'TP': (df[price_base] * df[volume_base]),
        vwap_name: (df[price_base] * df[volume_base]).cumsum() / df[volume_base].cumsum()
    })
    df[vwap_name] = data[vwap_name]

    if multiplier1 is not None or multiplier2 is not None:
        # data['TPP'] = (df1['Close'] * df1['Close'] * df1['Volume']).cumsum() / df1['Volume'].cumsum()
        # STD Formula : sqrt( (ACC(Close X Close * Volume) / ACC(Volume)) - VWAP * VWAP )
        # multiple 1: 1.28, mutiple 2: 2.01, multiple 3: 2.51
        data['TPP'] = (data[price_base] * data['TP']).cumsum() / data[volume_base].cumsum() # Volume!! 
        data['VW2'] = data[vwap_name] * data[vwap_name]
        data['STD'] = (data['TPP'] - data['VW2']) ** 0.5
        if multiplier1 is not None:
            df.loc[:, 'STD+1'] = data[vwap_name] + data['STD'] * multiplier1
            df.loc[:, 'STD-1'] = data[vwap_name] - data['STD'] * multiplier1
        if multiplier2 is not None:
            df.loc[:, 'STD+2'] = data[vwap_name] + data['STD'] * multiplier2
            df.loc[:, 'STD-2'] = data[vwap_name] - data['STD'] * multiplier2

    return df

def calculate_vwap_only(df, vwap_name, price_base, volume_base):    
    # Use .loc[] to avoid setting values on a slice of DataFrame
    #df1 = df.loc[df.index >= anchor_date].copy()

    # VWAP Formula : ACC (Close * Volume) / ACC (Volume)
    data = pd.DataFrame({
        f'{price_base}': df[price_base],  # Close value base
        f'{volume_base}': df[volume_base],
        vwap_name: (df[price_base] * df[volume_base]).cumsum() / df[volume_base].cumsum()
    })
    df[vwap_name] = data[vwap_name]

    return df

def anchored_vwap_to_database(price_df, stock_code, stock_name, anchor_string_date, price_base, increase10_string_date, multiplier1, multiplier2, multiplier3, multiplier4):
    try:
        # 1. VWAP 계산(종가기준)
        # 0.003초 소요
        diff = 0.0
        vwap_df = calculate_anchored_vwap(df=price_df, anchor_string_date=anchor_string_date, anchor_base="min", price_base=price_base, increase10_string_date=increase10_string_date,
                                          multiplier1=multiplier1, multiplier2=multiplier2, multiplier3=multiplier3, multiplier4=multiplier4)
        vwap_df.fillna(0, inplace=True) # NaN 을 0으로 변경
        df = vwap_df.tail(1)
        for index, row in df.iterrows():
            v2 = int(row['std2p'])
            v1 = int(row['std1p'])
            diff = (v2-v1)/v1 * 100
        #     if diff >= 3.0:
        #         close = int(row['close'])
        #         vw = int(row['vwap'])
        #         if close > vw:
        #             print(f"{stock_name}: {diff}%")

        return True, vwap_df, round(diff, 1)

    except Exception as e:
        print("Error in anchored_vwap_to_database: ", e)
        return False, None

def calculate_bollinger_bands(price_df, window=20, multiplier2=1.0, multiplier4=2.0):
    try:
        # Calculate rolling mean and standard deviation
        rolling_mean = price_df['close'].rolling(window=window).mean()
        rolling_std = price_df['close'].rolling(window=window).std()

        # Calculate Bollinger Bands
        std2p = rolling_mean + multiplier2 * rolling_std
        std2m = rolling_mean - multiplier2 * rolling_std
        std4p = rolling_mean + multiplier4 * rolling_std
        std4m = rolling_mean - multiplier4 * rolling_std

        # Create a new DataFrame to store Bollinger Bands data
        bollinger_bands = price_df.copy()
        bollinger_bands['BL_middle'] = rolling_mean
        bollinger_bands['BL_std2p'] = std2p
        bollinger_bands['BL_std2m'] = std2m
        bollinger_bands['BL_std4p'] = std4p
        bollinger_bands['BL_std4m'] = std4m

        bollinger_bands.dropna(inplace=True)

        return True, bollinger_bands

    except Exception as e:
        print("Error in calculate_bollinger_bands: ", e)
        return False, None

# 초기화 : session 에 control값을 대신하는 sv로 시작하는 key 값이 있는 경우(클릭 등의 이벤트로 화면이 갱신되는 경우) session 에서 값을 읽어서 global 변수를 채운다. -> 각 control 값은 global 변수로 다시 세팅
# 각 control 의 key값으로 control에 직접 접근하지 않고, sv_key값에 복사하여 session 에 저장하고 page 를 재구성할 때 sv_key값을 control 의 value 로 입력해서 contorl을 관리
# 각 control 에 event(on_click, on_change) 발생시 event 콜백함수안에 sv_key 값에 control 값을 저장하고 page 를 다시 구성할 때 sv_key 값을 읽어와서 control 값으로 세팅
# 각 control값을 변경해야 할 경우, 변경 event 콜백함수안에서 sv_key 값에 변경을 원하는 값으로 세팅하면, page 를 다시 구성하면서 sv_key 값을 읽어와서 control 값이 변경됨
# 중요!! 순서: key를 보유한 control에 event 발생(사용자)->session key값에 event에 해당하는 값을 저장(streamlit)->control 콜백함수 호출(streamlit)->session sv_key값에 session key
def init_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)

    global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH
    global BB

    emaA = False if 'sv_emaA' not in st.session_state else st.session_state['sv_emaA']
    emaB = False if 'sv_emaB' not in st.session_state else st.session_state['sv_emaB']
    emaC = False if 'sv_emaC' not in st.session_state else st.session_state['sv_emaC']
    emaD = False if 'sv_emaD' not in st.session_state else st.session_state['sv_emaD']

    BB = False if 'sv_BB' not in st.session_state else st.session_state['sv_BB']


def clear_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)
    
    st.session_state['sv_emaA'], st.session_state['sv_emaB'], st.session_state['sv_emaC'], st.session_state['sv_emaD'] = False, False, False, False

def on_change_stock():
    clear_session_control_values()

# 선택한 종목의 indicator 를 로드할 경우, sv_key 값을 DB에서 읽어온 값으로 세팅해주면, 콜백함수 리턴 후 page 재구성시에 init_session_control_values 함수안에서 sv_key 값으로 control 값이 세팅됨
def on_click_load_indicators():
    global stocklist_df 

    one_month_ago = datetime.today() - timedelta(days=30)

    if 'stock' not in st.session_state:
        return

    stock = st.session_state['stock']
    stock = stocklist_df.loc[stocklist_df['name'] == stock, ['market', 'code']].values
    uidx, market, code = 1, stock[0][0], stock[0][1]

    task_name = "get_technical_indicators"
    params = {'uidx': 1, 'market': f"{market}", 'code': f"{code}"}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                indicators_df = pd.DataFrame(respose["return"]["data"])
                if not indicators_df.empty:
                    indicators = indicators_df.loc[indicators_df['indicators'].notnull(), 'indicators'].iloc[0]
                    indicators_dict = json.loads(indicators)
                    ema_params = indicators_dict['ema']
                    st.session_state['sv_emaA'] = True if 'EMA_A' in ema_params else False
                    st.session_state['sv_emaB'] = True if 'EMA_B' in ema_params else False
                    st.session_state['sv_emaC'] = True if 'EMA_C' in ema_params else False
                    st.session_state['sv_emaD'] = True if 'EMA_D' in ema_params else False
                    vwap_params = indicators_dict['vwap']

def on_change_emaA():
    st.session_state['sv_emaA'] = st.session_state['emaA']

def on_change_emaB():
    st.session_state['sv_emaB'] = st.session_state['emaB']

def on_change_emaC():
    st.session_state['sv_emaC'] = st.session_state['emaC']

def on_change_emaD():
    st.session_state['sv_emaD'] = st.session_state['emaD']

def on_change_BB():
    st.session_state['sv_BB'] = st.session_state['BB']


# 관심종목 등록
def on_click_save_stock_insterest():
    global selected_stockname

    if selected_stockname:
        stock = stocklist_df.loc[stocklist_df['name'] == selected_stockname, ['market', 'code']].values
        if stock.any():
            market = stock[0][0]
            code = stock[0][1]
            name = selected_stockname    
            ret_Ok, output = dc.insert_stock_interest(uidx=1,
                                                    market=market,
                                                    code=code,
                                                    name=name,
                                                    pattern="차트분석",
                                                    description="")

# 직전 valley 일자를 찾는 함수
def find_recent_valley_before_date(zigzag_df, input_date, days_before=90):
    # input_date 형식 변환
    target_date = datetime.strptime(str(input_date), '%Y%m%d').date()
    
    # 입력된 날짜의 value 찾기
    target_value = None
    zigzag_df = zigzag_df.iloc[::-1]
    for index, row in zigzag_df.iterrows():
        entry_date = row['time'].date()
        if entry_date == target_date and row['pivot'] == 'valley':
            target_value = row['value']
            break
    
    # 해당 valley 값의 -1% 아래값
    if target_value is None:
        return None    
    target_value = int(target_value * 0.99)

    # 오늘 날짜 설정 (한국 시간대)
    three_months_ago = target_date - timedelta(days=days_before)

    # 입력된 날짜 이전의 가장 최근 valley의 날짜 찾기
    recent_valley_date = None
    for index, row in zigzag_df.iterrows():
        entry_date = row['time'].date()
        if three_months_ago <= entry_date < target_date and row['pivot'] == 'valley' and row['value'] < target_value:
            recent_valley_date = entry_date
            break
    
    if recent_valley_date is None:
        return None

    return recent_valley_date.strftime('%Y%m%d')

def string_datetime_to_timestamp(value):
    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    utc_dt = pytz.utc.localize(dt)
    kst_dt = utc_dt.astimezone(pytz.timezone('Asia/Seoul'))
    timestamp = int(time.mktime(kst_dt.timetuple()))
    return timestamp

def main():

    global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH
    global BB
    global stocklist_df, selected_stockname, selected_minutes

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    ss.check_session(current_page)
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    # column subtitle
    col_subtitle_style = '<p style="color:white;font-size: 18px;font-weight:bold;padding-left:6px;margin-bottom:2px;">{title}</p>'

    stock_code = "005930.KS"
    stock_code_only = "005930"
    stock_name = "삼성전자"
    stocknames = None
    stock = None

    ema_A_length, ema_B_length, ema_C_length, ema_D_length = 0, 0, 0, 0
    bollinger_ma = 0

    idt, i10dt, vdt, pattern = None, None, None, None
    data_count = None
    interval = Interval.in_15_minute

    band_gap = 0
    selected_idt = ""

    with st.container():

        init_session_control_values()
    
        reload = False

        # 체크박스와 날짜선택박스를 포함하는 4개의 컬럼 생성
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            detected_dates = get_detected_dates(year_from=2024, month_from=6, day_from=1)
            selected_idt = st.selectbox(label="탐지일 선택", options=detected_dates, key="d_dates")

        with col2:
            if selected_idt:
                success, stocknames, stocklist_df = get_detected_stocks(idt = selected_idt)
                if success:
                    selected_stockname = st.selectbox('주식종목 선택', stocknames, key="stock", on_change=on_change_stock)
                    if selected_stockname:
                        stock = stocklist_df.loc[stocklist_df['name'] == selected_stockname, ['market', 'code']].values
                        stock_code = stock[0][1]+"."+stock[0][0]
                        stock_code_only = stock[0][1]
                        stock_name = selected_stockname
                        stock = stocklist_df.loc[stocklist_df['name'] == selected_stockname, ['market', 'idt', 'i10dt', 'vdt', 'pattern']].values
                        market, idt, i10dt, vdt, pattern = stock[0][0], stock[0][1], stock[0][2], stock[0][3], stock[0][4]

        with col3:
            col31, col32 = st.columns(2)
            if selected_stockname:
                with col31:
                    ema_A_length = 5 if col31.toggle(label="5 MA", key="emaA", value=emaA, on_change=on_change_emaA) else 0
                    ema_B_length = 10 if col31.toggle(label="10 MA", key="emaB", value=emaB, on_change=on_change_emaB) else 0
                    pass
                with col32:
                    ema_C_length = 60 if col32.toggle(label="60 MA", key="emaC", value=emaC, on_change=on_change_emaC) else 0            
                    ema_D_length = 381 if col32.toggle(label="381 MA", key="emaD", value=emaD, on_change=on_change_emaD) else 0
        with col4:
            col41, col42 = st.columns(2)
            with col41:
                #dt = datetime.strptime(vdt, "%Y%m%d")
                #string_date = dt.strftime('%Y-%m-%d')
                if pattern:
                    st.text_input('조검검색', value=pattern, disabled=True, key="valley_date")
            with col42:
                if i10dt:
                    dt = datetime.strptime(i10dt, "%Y%m%d")
                    string_date = dt.strftime('%Y-%m-%d')
                    st.text_input('장대양봉일', value=string_date, disabled=True, key="increase10_date")
        with col5:
            col51, col52 = st.columns(2)
            with col51:
                if idt:
                    dt = datetime.strptime(idt, "%Y%m%d")
                    string_date = dt.strftime('%Y-%m-%d')
                    st.text_input('탐지일', value=string_date, disabled=True, key="detected_date")
        with col6:
            col61, col62 = st.columns(2)
            with col61:
                if vdt:
                    vdt_date = pytz.timezone('Asia/Seoul').localize(datetime.strptime(vdt, "%Y%m%d"))
                    today = datetime.now(pytz.timezone('Asia/Seoul'))
                    vdt_one_week_ago = vdt_date - timedelta(days=0)
                    days_difference = (today - vdt_one_week_ago).days
                    selected_minutes = st.selectbox('분봉 선택', index=0, options=['1분','3분','5분','15분','30분','1시간','1일'], key="minutes")
                    if selected_minutes:
                        if selected_minutes == '1분':
                            data_count = days_difference * 450  # 1분봉 기준 1시간: 60개, 7.5시간: 450
                            interval = Interval.in_1_minute
                        elif selected_minutes == '3분':
                            data_count = days_difference * 150  # 3분봉 기준 1시간: 20개, 7.5시간: 150
                            interval = Interval.in_3_minute
                        elif selected_minutes == '5분':
                            data_count = days_difference * 90 + 2 * 90 # 5분봉 기준 1시간: 12개, 7.5시간: 90
                            interval = Interval.in_5_minute
                        elif selected_minutes == '15분':
                            data_count = days_difference * 30 + 4 * 30 # 15분봉 기준 1시간: 4개, 7.5시간: 30
                            interval = Interval.in_15_minute
                        elif selected_minutes == '30분':
                            data_count = days_difference * 15 + 5 * 15  # 30분봉 기준 1시간: 2개, 7.5시간: 15
                            interval = Interval.in_30_minute
                        elif selected_minutes == '1시간':
                            data_count = days_difference * 8 + 6 * 8 # 1시간 기준 1시간: 1개, 7.5시간: 8
                            interval = Interval.in_1_hour
                        elif selected_minutes == '1일':
                            data_count = days_difference + 50  # 1일 기준, 1개
                            interval = Interval.in_daily
            with col62:
                if selected_stockname:
                    dummy_ma = 0 if col62.toggle(label="00 BB", key="DM", disabled=True, label_visibility="hidden") else 0
                    bollinger_ma = 33 if col62.toggle(label="33 BB", key="BB", value=BB, on_change=on_change_BB) else 0
                    if interval == Interval.in_1_minute:
                        bollinger_ma = int(bollinger_ma * 5 / 1)
                    elif interval == Interval.in_3_minute:
                        bollinger_ma = int(bollinger_ma * 5 / 3)
                    elif interval == Interval.in_5_minute:
                        bollinger_ma = int(bollinger_ma * 5 / 5)
                    elif interval == Interval.in_15_minute:
                        bollinger_ma = int(bollinger_ma * 5 / 15)
                    elif interval == Interval.in_30_minute:
                        if bollinger_ma > 0:
                            bollinger_ma = 20
                    elif interval == Interval.in_1_hour:
                        if bollinger_ma > 0:
                            bollinger_ma = 10
                    elif interval == Interval.in_daily:
                        if bollinger_ma > 0:
                            bollinger_ma = 5


        if data_count is None:
            return
        
        # for index, row in stocklist_df.iterrows():

        #     market, idt, i10dt, vdt, pattern = row['market'], row['idt'], row['i10dt'], row['vdt'], row['pattern']
        #     stock_code_only = row['code']
        #     stock_name = row['name']

        # 체크박스와 날짜선택박스 변수 초기화
        vwap_param = {}

        #df = yf.fetch_stock_data(symbol=stock_code, period="10d", interval="5m")
        #df.index.name = "Date"

        # 1. Tradingview 에서 과거 분봉데이터 가져오기
        vwap_df = pd.DataFrame()
        bollinger_df = pd.DataFrame()
        tvdata = tv.get_tvdata(stock_code=stock_code_only, stock_name=stock_name, data_count=data_count, interval=interval)
        if tvdata is not None and not tvdata.empty:

            # datatime 형식을 string 형식으로 변환
            tvdata = tvdata.reset_index()
            tvdata['time'] = tvdata['datetime'].dt.strftime('%Y%m%d%H%M%S')
            tvdata.set_index('time', inplace=True)
            tvdata.sort_index()
            tvdata['symbol'] = tvdata['symbol'].apply(lambda x: x.split(':')[1])  # symbol 값 KRX:329180 형식

            valley_string_datetime = vdt + "000000"
            price_df = tvdata.loc[tvdata.index >= valley_string_datetime].copy()

            # 2. vwap 계산 후 return
            success, df, band_gap = anchored_vwap_to_database(price_df=price_df, stock_code=stock_code_only, stock_name=stock_name, price_base="low", anchor_string_date=vdt,
                                                              increase10_string_date=i10dt, multiplier1=multiplier1, multiplier2=multiplier2, multiplier3=multiplier3, multiplier4=multiplier4)
            if success:
                df.drop(columns=['symbol'], inplace=True)
                df.reset_index(inplace=True)
                del df['time']
                df.rename(columns={'datetime':'time'}, inplace=True)
                df['time'] = df['time'].dt.strftime('%Y-%m-%d  %H:%M:%S')
                df.set_index('time', inplace=True)
                vwap_df = df

                with col52:
                    st.text_input('밴드갭', value=f"{band_gap}%", disabled=True, key="band_gap")

            # 3. Bollinger band 계산 후 return
            success, bollinger_df = calculate_bollinger_bands(price_df=price_df, window=bollinger_ma, multiplier2=multiplier2, multiplier4=multiplier4)
            if success:
                bollinger_df.drop(columns=['symbol'], inplace=True)
                bollinger_df.reset_index(inplace=True)
                del bollinger_df['time']
                bollinger_df.rename(columns={'datetime':'time'}, inplace=True)
                bollinger_df['time'] = bollinger_df['time'].dt.strftime('%Y-%m-%d  %H:%M:%S')
                bollinger_df.set_index('time', inplace=True)
                #print(bollinger_df)

            tvdata.drop(columns=['symbol'], inplace=True)
            tvdata.reset_index(inplace=True)
            del tvdata['time']
            tvdata.rename(columns={'datetime':'time'}, inplace=True)
            tvdata['time'] = tvdata['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            tvdata.set_index('time', inplace=True)

        len_df = len(tvdata)
        if len_df <= ema_A_length: ema_A_length = 0
        if len_df <= ema_B_length: ema_B_length = 0
        if len_df <= ema_C_length: ema_C_length = 0
        if len_df <= ema_D_length: ema_D_length = 0
        if len_df <= bollinger_ma: bollinger_ma = 0
        
        ema_param = {}
        if ema_A_length > 0:
            ema_param.update({'EMA_A': {'length': ema_A_length, 'color': 'orange', 'linewidth': 2}})
        if ema_B_length > 0:
            #ema_param.update({'EMA_B': {'length': ema_B_length, 'color': 'mediumblue', 'linewidth': 2}})
            ema_param.update({'EMA_B': {'length': ema_B_length, 'color': 'orange', 'linewidth': 2}})
        if ema_C_length > 0:
            ema_param.update({'EMA_C': {'length': ema_C_length, 'color': 'red', 'linewidth': 2}})
        if ema_D_length > 0:
            ema_param.update({'EMA_D': {'length': ema_D_length, 'color': 'crimson', 'linewidth': 2}})

        indicators_params = {'ema': ema_param, 'vwap': vwap_param}

        # 장대양봉일 이후(매수 모니터링 시작일) 분봉데이터 추출, 일봉데이터와 합쳐서 일봉기준 vwap 계산에 사용
        idt_string_datetime = datetime.strptime(i10dt, "%Y%m%d")
        idt_string_datetime = idt_string_datetime.replace(hour=0, minute=0, second=0)
        idt_string_datetime = idt_string_datetime.strftime("%Y-%m-%d %H:%M:%S")
        price_idt_df = tvdata.loc[tvdata.index >= idt_string_datetime].copy()
        #price_idt_df.reset_index(inplace=True)
        #price_idt_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        #price_idt_df.set_index('time', inplace=True)
        #price_idt_df.index.name = 'Date'
        #print(price_idt_df)

        # 직전저점 vwap
        vwap_1day_df = pd.DataFrame()
        previous_vdt, price_1day_df = zz.get_previous_valley_datetime(stock_code=stock_code_only, stock_name=stock_name, vdt=vdt, base_price="close", days_before=5)

        # 직전저점(previous_vdt)과 저점(vdt)사이의 고점
        vwap_high2_dataframe = pd.DataFrame()

        # 장대양봉일부터 최고점
        vwap_highest_dataframe = pd.DataFrame()

        # previous_vdt 와 vdt 가 동일할 경우에는 일봉기준 vwap 을 사용하지 않고 분봉 vwap 을 사용
        if previous_vdt:

            pvdt = datetime.strptime(previous_vdt, "%Y%m%d%H%M%S").strftime("%Y%m%d")
            price_1day_df = get_tvdata_from_vdt(stock_code=stock_code_only, stock_name=stock_name, selected_minutes=selected_minutes, vdt=pvdt, days_more=False)

            price_pvdt_df = price_1day_df.loc[price_1day_df.index >= previous_vdt].copy()
            price_pvdt_df = price_pvdt_df.reset_index()
            price_pvdt_df['time'] = price_pvdt_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            price_pvdt_df.set_index('time', inplace=True)
            #price_pvdt_df = pd.concat([price_pvdt_df, price_idt_df])

            vwap_1day_df = calculate_vwap_bands(df=price_pvdt_df, anchor_base="min", anchor_date=previous_vdt, price_base="low", volume_base='volume', vwap_name="vwap", multiplier1=multiplier2, multiplier2=multiplier4)
            # Data index 를 time 으로 변경, 그래프 생성시 time, vwap 으로 생성, time 컬럼의 형식을 문자열로 변환, 그래프 생성시 time 컬럼의 문자열을 timestamp 로 변경
            vwap_1day_df = vwap_1day_df.reset_index()
            #vwap_1day_df = vwap_1day_df.rename(columns={'Date': 'time'})


            # vwap_high2_dataframe 직전저점(previous_vdt)과 저점(vdt)사이의 고점
            i10dt_string_datetime = datetime.strptime(i10dt, "%Y%m%d")
            i10dt_string_datetime = i10dt_string_datetime.replace(hour=0, minute=0, second=0)
            i10dt_string_datetime = i10dt_string_datetime.strftime("%Y-%m-%d %H:%M:%S")
            pvdt_string_datetime = datetime.strptime(previous_vdt, "%Y%m%d%H%M%S")
            pvdt_string_datetime = pvdt_string_datetime.replace(hour=0, minute=0, second=0)
            pvdt_string_datetime = pvdt_string_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            price_pvdt_df = tvdata.loc[(tvdata.index >= pvdt_string_datetime) & (tvdata.index < i10dt_string_datetime)].copy()
            price_high_index = price_pvdt_df['high'].idxmax() # 고점찾기
            price_pvdt_df = tvdata.loc[tvdata.index >= price_high_index].copy() # 고점이후 데이터 
            vwap_high2_dataframe = calculate_vwap_only(df=price_pvdt_df, vwap_name="vwap", price_base="low", volume_base="volume")
            vwap_high2_dataframe = vwap_high2_dataframe.reset_index()

            #st.text(price_pvdt_df)
            #print(pvdt_string_datetime, i10dt_string_datetime, price_high_index)

            # vwap_highest_dataframe 장대양봉일(i10dt) 이후 최고점
            i10dt_string_datetime = datetime.strptime(i10dt, "%Y%m%d")
            i10dt_string_datetime = i10dt_string_datetime.replace(hour=0, minute=0, second=0)
            i10dt_string_datetime = i10dt_string_datetime.strftime("%Y-%m-%d %H:%M:%S")
            price_pvdt_df = tvdata.loc[tvdata.index >= i10dt_string_datetime].copy()
            price_high_index = price_pvdt_df['high'].idxmax() # 고점찾기
            price_pvdt_df = tvdata.loc[tvdata.index >= price_high_index].copy() # 고점이후 데이터 
            vwap_highest_dataframe = calculate_vwap_only(df=price_pvdt_df, vwap_name="vwap", price_base="low", volume_base="volume")
            vwap_highest_dataframe = vwap_highest_dataframe.reset_index()

            #print(pvdt_string_datetime, i10dt_string_datetime, price_high_index)
            #print(vwap_high2_dataframe)


        # Save, Load button
        with col6:
            col61, col62 = st.columns(2)
 
        show_volume = False
        click_events_dy = chart.get_stock_chart(  symbol=stock_code
                                                , selected_idt = get_next_business_day(selected_idt)
                                                , dataframe=tvdata
                                                , vwap_dataframe=vwap_df
                                                , vwap_band_gap = band_gap
                                                , vwap_high1_dataframe = None
                                                , vwap_high2_dataframe = vwap_high2_dataframe
                                                , vwap_highest_dataframe = vwap_highest_dataframe
                                                , vwap_1day_dataframe = vwap_1day_df
                                                , bollinger_dataframe=bollinger_df
                                                , indicators_params=indicators_params
                                                , pane_name="pane_daily"
                                                , time_minspacing=3
                                                , show_volume=show_volume
                                                , chart_height=650)        
        
if __name__ == '__main__':
    main()