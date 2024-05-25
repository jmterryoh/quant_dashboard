import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import json
import pytz
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from datetime import datetime, timedelta
from chart import TRV_lightchart_min_vwap as chart
from hts import YF_api as yf
from hts import TV_rest_api as tv
from middleware import md_client as dc

from tvDatafeed import TvDatafeed, Interval

global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH

global stocklist_df, selected_stockname
stocklist_df = {}
selected_stockname = None
selected_minutes = None


multiplier1 = 0.5
multiplier2 = 1.0
multiplier3 = 1.5
multiplier4 = 2.0

current_page = "pages/page_chart_analysis_vwap.py"

# Anchored VWAP 
def calculate_anchored_vwap(df, anchor_string_date, anchor_base, price_base, multiplier1, multiplier2, multiplier3, multiplier4):

    start_time = anchor_string_date + "000000"
    end_time = anchor_string_date + "235959"

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

def anchored_vwap_to_database(price_df, stock_code, stock_name, anchor_string_date, increase10_string_date, multiplier1, multiplier2, multiplier3, multiplier4):
    try:
        # 1. VWAP 계산(종가기준)
        # 0.003초 소요
        diff = 0.0
        vwap_df = calculate_anchored_vwap(df=price_df, anchor_string_date=anchor_string_date, anchor_base="min", price_base="close", multiplier1=multiplier1, multiplier2=multiplier2, multiplier3=multiplier3, multiplier4=multiplier4)
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




# 초기화 : session 에 control값을 대신하는 sv로 시작하는 key 값이 있는 경우(클릭 등의 이벤트로 화면이 갱신되는 경우) session 에서 값을 읽어서 global 변수를 채운다. -> 각 control 값은 global 변수로 다시 세팅
# 각 control 의 key값으로 control에 직접 접근하지 않고, sv_key값에 복사하여 session 에 저장하고 page 를 재구성할 때 sv_key값을 control 의 value 로 입력해서 contorl을 관리
# 각 control 에 event(on_click, on_change) 발생시 event 콜백함수안에 sv_key 값에 control 값을 저장하고 page 를 다시 구성할 때 sv_key 값을 읽어와서 control 값으로 세팅
# 각 control값을 변경해야 할 경우, 변경 event 콜백함수안에서 sv_key 값에 변경을 원하는 값으로 세팅하면, page 를 다시 구성하면서 sv_key 값을 읽어와서 control 값이 변경됨
# 중요!! 순서: key를 보유한 control에 event 발생(사용자)->session key값에 event에 해당하는 값을 저장(streamlit)->control 콜백함수 호출(streamlit)->session sv_key값에 session key
def init_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)

    global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH

    emaA = False if 'sv_emaA' not in st.session_state else st.session_state['sv_emaA']
    emaB = False if 'sv_emaB' not in st.session_state else st.session_state['sv_emaB']
    emaC = False if 'sv_emaC' not in st.session_state else st.session_state['sv_emaC']
    emaD = False if 'sv_emaD' not in st.session_state else st.session_state['sv_emaD']
    emaE = False if 'sv_emaE' not in st.session_state else st.session_state['sv_emaE']
    emaF = False if 'sv_emaF' not in st.session_state else st.session_state['sv_emaF']
    emaG = False if 'sv_emaG' not in st.session_state else st.session_state['sv_emaG']
    emaH = False if 'sv_emaH' not in st.session_state else st.session_state['sv_emaH']


def clear_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)
    
    st.session_state['sv_emaA'], st.session_state['sv_emaB'], st.session_state['sv_emaC'], st.session_state['sv_emaD'] = False, False, False, False
    st.session_state['sv_emaE'], st.session_state['sv_emaF'], st.session_state['sv_emaH'], st.session_state['sv_emaG'] = False, False, False, False

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
                    st.session_state['sv_emaE'] = True if 'EMA_E' in ema_params else False
                    st.session_state['sv_emaF'] = True if 'EMA_F' in ema_params else False
                    st.session_state['sv_emaG'] = True if 'EMA_G' in ema_params else False
                    st.session_state['sv_emaH'] = True if 'EMA_H' in ema_params else False
                    vwap_params = indicators_dict['vwap']

def on_change_emaA():
    st.session_state['sv_emaA'] = st.session_state['emaA']

def on_change_emaB():
    st.session_state['sv_emaB'] = st.session_state['emaB']

def on_change_emaC():
    st.session_state['sv_emaC'] = st.session_state['emaC']

def on_change_emaD():
    st.session_state['sv_emaD'] = st.session_state['emaD']

def on_change_emaE():
    st.session_state['sv_emaE'] = st.session_state['emaE']

def on_change_emaF():
    st.session_state['sv_emaF'] = st.session_state['emaF']

def on_change_emaG():
    st.session_state['sv_emaG'] = st.session_state['emaG']

def on_change_emaH():
    st.session_state['sv_emaH'] = st.session_state['emaH']


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

def main():

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
    interval = Interval.in_15_minute

    global emaA, emaB, emaC, emaD, emaE, emaF, emaG, emaH
    global stocklist_df, selected_stockname, selected_minutes

    task_name = "get_algo_stocks_increase10"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                stocklist_df = pd.DataFrame(respose["return"]["data"])
                if stocklist_df is not None:
                    stocknames = stocklist_df['name'].tolist()

    with st.container():

        init_session_control_values()
    
        reload = False

        # 체크박스와 날짜선택박스를 포함하는 4개의 컬럼 생성
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if stocknames:
                selected_stockname = st.selectbox('주식종목 선택', stocknames, key="stock", on_change=on_change_stock)
                if selected_stockname:
                    stock = stocklist_df.loc[stocklist_df['name'] == selected_stockname, ['market', 'code']].values
                    stock_code = stock[0][1]+"."+stock[0][0]
                    stock_code_only = stock[0][1]
                    stock_name = selected_stockname
                    stock = stocklist_df.loc[stocklist_df['name'] == selected_stockname, ['market', 'idt', 'i10dt', 'vdt', 'pattern']].values
                    market, idt, i10dt, vdt, pattern = stock[0][0], stock[0][1], stock[0][2], stock[0][3], stock[0][4]

        with col2:
            col21, col22 = st.columns(2)
            with col21:
                ema_A_length = 10 if col21.toggle(label="10", key="emaA", value=emaA, on_change=on_change_emaA) else 0
                ema_B_length = 15 if col21.toggle(label="15", key="emaB", value=emaB, on_change=on_change_emaB) else 0
            with col22:
                ema_C_length = 21 if col22.toggle(label="21", key="emaC", value=emaC, on_change=on_change_emaC) else 0            
                ema_D_length = 33 if col22.toggle(label="33", key="emaD", value=emaD, on_change=on_change_emaD) else 0

        with col3:
            col31, col32 = st.columns(2)
            with col31:
                ema_E_length = 66 if col31.toggle(label="66", key="emaE", value=emaE, on_change=on_change_emaE) else 0
                ema_F_length = 112 if col31.toggle(label="112", key="emaF", value=emaF, on_change=on_change_emaF) else 0
            with col32:
                ema_G_length = 224 if col32.toggle(label="224", key="emaG", value=emaG, on_change=on_change_emaG) else 0            
                ema_H_length = 448 if col32.toggle(label="448", key="emaH", value=emaH, on_change=on_change_emaH) else 0
        with col4:
            col41, col42 = st.columns(2)
            with col41:
                dt = datetime.strptime(vdt, "%Y%m%d")
                string_date = dt.strftime('%Y-%m-%d')
                st.text_input('직전저점일', value=string_date, disabled=True, key="valley_date")
            with col42:
                dt = datetime.strptime(i10dt, "%Y%m%d")
                string_date = dt.strftime('%Y-%m-%d')
                st.text_input('장대양봉일', value=string_date, disabled=True, key="increase10_date")
        with col5:
            col51, col52 = st.columns(2)
            with col51:
                dt = datetime.strptime(i10dt, "%Y%m%d")
                string_date = dt.strftime('%Y-%m-%d')
                st.text_input('탐지일', value=string_date, disabled=True, key="detected_date")
        with col6:
            col61, col62 = st.columns(2)
            with col61:
                vdt_date = pytz.timezone('Asia/Seoul').localize(datetime.strptime(vdt, "%Y%m%d"))
                today = datetime.now(pytz.timezone('Asia/Seoul'))
                vdt_one_week_ago = vdt_date - timedelta(days=0)
                days_difference = (today - vdt_one_week_ago).days
                selected_minutes = st.selectbox('분봉 선택', index=2, options=['3분','5분','15분','30분'], key="minutes")
                if selected_minutes:
                    if selected_minutes == '3분':
                        data_count = days_difference * 150  # 3분봉 기준 1시간: 20개, 7.5시간: 150
                        interval = Interval.in_3_minute
                    elif selected_minutes == '5분':
                        data_count = days_difference * 90  # 5분봉 기준 1시간: 12개, 7.5시간: 90
                        interval = Interval.in_5_minute
                    elif selected_minutes == '15분':
                        data_count = days_difference * 30  # 15분봉 기준 1시간: 4개, 7.5시간: 30
                        interval = Interval.in_15_minute
                    elif selected_minutes == '30분':
                        data_count = days_difference * 15  # 30분봉 기준 1시간: 2개, 7.5시간: 15
                        interval = Interval.in_30_minute

 
        # for index, row in stocklist_df.iterrows():

        #     market, idt, i10dt, vdt, pattern = row['market'], row['idt'], row['i10dt'], row['vdt'], row['pattern']
        #     stock_code_only = row['code']
        #     stock_name = row['name']

        # 체크박스와 날짜선택박스 변수 초기화
        vwap_param = {}

        #df = yf.fetch_stock_data(symbol=stock_code, period="10d", interval="5m")
        #df.index.name = "Date"

        # 1. Tradingview 에서 과거 분봉데이터 가져오기
        df = pd.DataFrame()
        tvdata = tv.get_tvdata(stock_code=stock_code_only, stock_name=stock_name, data_count=data_count, interval=interval)
        if not tvdata.empty:

            # datatime 형식을 string 형식으로 변환
            tvdata = tvdata.reset_index()
            tvdata['time'] = tvdata['datetime'].dt.strftime('%Y%m%d%H%M%S')
            tvdata.set_index('time', inplace=True)
            tvdata.sort_index()
            tvdata['symbol'] = tvdata['symbol'].apply(lambda x: x.split(':')[1])  # symbol 값 KRX:329180 형식

            valley_string_datetime = vdt + "000000"
            price_df = tvdata.loc[tvdata.index >= valley_string_datetime].copy()

            # 2. vwap 계산 후 DB에 저장
            success, vwap_df, band_gap = anchored_vwap_to_database(price_df=price_df, stock_code=stock_code_only, stock_name=stock_name, anchor_string_date=vdt,
                                                                   increase10_string_date=i10dt, multiplier1=multiplier1, multiplier2=multiplier2, multiplier3=multiplier3, multiplier4=multiplier4)
            if success:
                vwap_df.drop(columns=['symbol'], inplace=True)
                vwap_df.reset_index(inplace=True)
                del vwap_df['time']
                vwap_df.rename(columns={'datetime':'time'}, inplace=True)
                vwap_df['time'] = vwap_df['time'].dt.strftime('%Y-%m-%d  %H:%M:%S')
                vwap_df.set_index('time', inplace=True)
                df = vwap_df

                with col52:
                    st.text_input('밴드갭', value=f"{band_gap}%", disabled=True, key="band_gap")

            tvdata.drop(columns=['symbol'], inplace=True)
            tvdata.reset_index(inplace=True)
            del tvdata['time']
            tvdata.rename(columns={'datetime':'time'}, inplace=True)
            tvdata['time'] = tvdata['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            tvdata.set_index('time', inplace=True)

        len_df = len(df)
        if len_df <= ema_A_length: ema_A_length = 0
        if len_df <= ema_B_length: ema_B_length = 0
        if len_df <= ema_C_length: ema_C_length = 0
        if len_df <= ema_D_length: ema_D_length = 0
        if len_df <= ema_E_length: ema_E_length = 0
        if len_df <= ema_F_length: ema_F_length = 0
        if len_df <= ema_G_length: ema_G_length = 0
        if len_df <= ema_H_length: ema_H_length = 0
        
        ema_param = {}
        if ema_A_length > 0:
            ema_param.update({'EMA_A': {'length': ema_A_length, 'color': 'royalblue', 'linewidth': 1}})
        if ema_B_length > 0:
            ema_param.update({'EMA_B': {'length': ema_B_length, 'color': 'mediumblue', 'linewidth': 1}})
        if ema_C_length > 0:
            ema_param.update({'EMA_C': {'length': ema_C_length, 'color': 'red', 'linewidth': 1}})
        if ema_D_length > 0:
            ema_param.update({'EMA_D': {'length': ema_D_length, 'color': 'crimson', 'linewidth': 1}})
        if ema_E_length > 0:
            ema_param.update({'EMA_E': {'length': ema_E_length, 'color': 'green', 'linewidth': 1}})
        if ema_F_length > 0:
            ema_param.update({'EMA_F': {'length': ema_F_length, 'color': 'orange', 'linewidth': 1}})
        if ema_G_length > 0:
            ema_param.update({'EMA_G': {'length': ema_G_length, 'color': 'black', 'linewidth': 1}})
        if ema_H_length > 0:
            ema_param.update({'EMA_H': {'length': ema_H_length, 'color': 'gray', 'linewidth': 1}})

        indicators_params = {'ema': ema_param, 'vwap': vwap_param}

        # Save, Load button
        with col6:
            col61, col62 = st.columns(2)
 
        show_volume = False
        click_events_dy = chart.get_stock_chart(  symbol=stock_code
                                                , dataframe=tvdata
                                                , vwap_dataframe=df
                                                , indicators_params=indicators_params
                                                , pane_name="pane_daily"
                                                , time_minspacing=3
                                                , show_volume=show_volume
                                                , chart_height=650)        
        
if __name__ == '__main__':
    main()