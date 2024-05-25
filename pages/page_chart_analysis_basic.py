import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from datetime import datetime, timedelta
from chart import TRV_lightchart as chart
from hts import YF_api as yf
from middleware import md_client as dc
import json


global emaA, emaB, emaC, emaD
global vwap1_cb, vwap1_dt, vwap1_mt1, vwap1_mt2
global vwap2_cb, vwap2_dt, vwap2_mt1, vwap2_mt2
global vwap3_cb, vwap3_dt, vwap3_mt1, vwap3_mt2

global stocklist_df, selected_stockname
stocklist_df = {}
selected_stockname = None

current_page = "pages/page_chart_analysis_basic.py"


# 초기화 : session 에 control값을 대신하는 sv로 시작하는 key 값이 있는 경우(클릭 등의 이벤트로 화면이 갱신되는 경우) session 에서 값을 읽어서 global 변수를 채운다. -> 각 control 값은 global 변수로 다시 세팅
# 각 control 의 key값으로 control에 직접 접근하지 않고, sv_key값에 복사하여 session 에 저장하고 page 를 재구성할 때 sv_key값을 control 의 value 로 입력해서 contorl을 관리
# 각 control 에 event(on_click, on_change) 발생시 event 콜백함수안에 sv_key 값에 control 값을 저장하고 page 를 다시 구성할 때 sv_key 값을 읽어와서 control 값으로 세팅
# 각 control값을 변경해야 할 경우, 변경 event 콜백함수안에서 sv_key 값에 변경을 원하는 값으로 세팅하면, page 를 다시 구성하면서 sv_key 값을 읽어와서 control 값이 변경됨
# 중요!! 순서: key를 보유한 control에 event 발생(사용자)->session key값에 event에 해당하는 값을 저장(streamlit)->control 콜백함수 호출(streamlit)->session sv_key값에 session key
def init_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)

    global emaA, emaB, emaC, emaD
    global vwap1_cb, vwap1_dt, vwap1_mt1, vwap1_mt2
    global vwap2_cb, vwap2_dt, vwap2_mt1, vwap2_mt2
    global vwap3_cb, vwap3_dt, vwap3_mt1, vwap3_mt2

    emaA = True if 'sv_emaA' not in st.session_state else st.session_state['sv_emaA']
    emaB = True if 'sv_emaB' not in st.session_state else st.session_state['sv_emaB']
    emaC = True if 'sv_emaC' not in st.session_state else st.session_state['sv_emaC']
    emaD = False if 'sv_emaD' not in st.session_state else st.session_state['sv_emaD']

    vwap1_cb = False if 'sv_vwap1_cb' not in st.session_state else st.session_state['sv_vwap1_cb']
    vwap1_dt = one_month_ago if 'sv_vwap1_dt' not in st.session_state else (one_month_ago if st.session_state['sv_vwap1_dt'] is None else st.session_state['sv_vwap1_dt'])
    vwap1_mt1 = False if 'sv_vwap1_mt1' not in st.session_state else st.session_state['sv_vwap1_mt1']
    vwap1_mt2 = False if 'sv_vwap1_mt2' not in st.session_state else st.session_state['sv_vwap1_mt2']

    vwap2_cb = False if 'sv_vwap2_cb' not in st.session_state else st.session_state['sv_vwap2_cb']
    vwap2_dt = one_month_ago if 'sv_vwap2_dt' not in st.session_state else (one_month_ago if st.session_state['sv_vwap2_dt'] is None else st.session_state['sv_vwap2_dt'])
    vwap2_mt1 = False if 'sv_vwap2_mt1' not in st.session_state else st.session_state['sv_vwap2_mt1']
    vwap2_mt2 = False if 'sv_vwap2_mt2' not in st.session_state else st.session_state['sv_vwap2_mt2']

    vwap3_cb = False if 'sv_vwap3_cb' not in st.session_state else st.session_state['sv_vwap3_cb']
    vwap3_dt = one_month_ago if 'sv_vwap3_dt' not in st.session_state else (one_month_ago if st.session_state['sv_vwap3_dt'] is None else st.session_state['sv_vwap3_dt'])
    vwap3_mt1 = False if 'sv_vwap3_mt1' not in st.session_state else (0 if st.session_state['sv_vwap3_mt1'] is None else st.session_state['sv_vwap3_mt1'])
    vwap3_mt2 = False if 'sv_vwap3_mt2' not in st.session_state else (0 if st.session_state['sv_vwap3_mt2'] is None else st.session_state['sv_vwap3_mt2'])

def clear_session_control_values():
    one_month_ago = datetime.today() - timedelta(days=30)
    
    st.session_state['sv_emaA'], st.session_state['sv_emaB'], st.session_state['sv_emaC'], st.session_state['sv_emaD'] = True, True, True, False
    st.session_state['sv_vwap1_cb'], st.session_state['sv_vwap1_dt'], st.session_state['sv_vwap1_mt1'], st.session_state['sv_vwap1_mt2'] = False, one_month_ago, False, False
    st.session_state['sv_vwap2_cb'], st.session_state['sv_vwap2_dt'], st.session_state['sv_vwap2_mt1'], st.session_state['sv_vwap2_mt2'] = False, one_month_ago, False, False
    st.session_state['sv_vwap3_cb'], st.session_state['sv_vwap3_dt'], st.session_state['sv_vwap3_mt1'], st.session_state['sv_vwap3_mt2'] = False, one_month_ago, False, False

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
                    st.session_state['sv_vwap1_dt'], st.session_state['sv_vwap1_mt1'], st.session_state['sv_vwap1_mt2'] = (datetime.strptime(vwap_params['VWAP_1']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_1']['mt1'], vwap_params['VWAP_1']['mt2']) if 'VWAP_1' in vwap_params else (None, False, False)
                    st.session_state['sv_vwap2_dt'], st.session_state['sv_vwap2_mt1'], st.session_state['sv_vwap2_mt2'] = (datetime.strptime(vwap_params['VWAP_2']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_2']['mt1'], vwap_params['VWAP_2']['mt2']) if 'VWAP_2' in vwap_params else (None, False, False)
                    st.session_state['sv_vwap3_dt'], st.session_state['sv_vwap3_mt1'], st.session_state['sv_vwap3_mt2'] = (datetime.strptime(vwap_params['VWAP_3']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_3']['mt1'], vwap_params['VWAP_3']['mt2']) if 'VWAP_3' in vwap_params else (None, False, False)
                    st.session_state['sv_vwap1_cb'] = True if st.session_state['sv_vwap1_dt'] else False
                    st.session_state['sv_vwap2_cb'] = True if st.session_state['sv_vwap2_dt'] else False
                    st.session_state['sv_vwap3_cb'] = True if st.session_state['sv_vwap3_dt'] else False
                    st.session_state['sv_vwap1_dt'] = st.session_state['sv_vwap1_dt'] if st.session_state['sv_vwap1_dt'] else one_month_ago
                    st.session_state['sv_vwap2_dt'] = st.session_state['sv_vwap2_dt'] if st.session_state['sv_vwap2_dt'] else one_month_ago
                    st.session_state['sv_vwap3_dt'] = st.session_state['sv_vwap3_dt'] if st.session_state['sv_vwap3_dt'] else one_month_ago

def on_change_emaA():
    st.session_state['sv_emaA'] = st.session_state['emaA']

def on_change_emaB():
    st.session_state['sv_emaB'] = st.session_state['emaB']

def on_change_emaC():
    st.session_state['sv_emaC'] = st.session_state['emaC']

def on_change_emaD():
    st.session_state['sv_emaD'] = st.session_state['emaD']

def on_change_vwap1_cb():
    st.session_state['sv_vwap1_cb'] = st.session_state['vwap1_cb']

def on_change_vwap2_cb():
    st.session_state['sv_vwap2_cb'] = st.session_state['vwap2_cb']

def on_change_vwap3_cb():
    st.session_state['sv_vwap3_cb'] = st.session_state['vwap3_cb']

def on_change_vwap1_dt():
    st.session_state['sv_vwap1_dt'] = st.session_state['vwap1_dt']

def on_change_vwap2_dt():
    st.session_state['sv_vwap2_dt'] = st.session_state['vwap2_dt']

def on_change_vwap3_dt():
    st.session_state['sv_vwap3_dt'] = st.session_state['vwap3_dt']

def on_change_vwap1_mt1():
    st.session_state['sv_vwap1_mt1'] = st.session_state['vwap1_mt1']

def on_change_vwap1_mt2():
    st.session_state['sv_vwap1_mt2'] = st.session_state['vwap1_mt2']

def on_change_vwap2_mt1():
    st.session_state['sv_vwap2_mt1'] = st.session_state['vwap2_mt1']

def on_change_vwap2_mt2():
    st.session_state['sv_vwap2_mt2'] = st.session_state['vwap2_mt2']

def on_change_vwap3_mt1():
    st.session_state['sv_vwap3_mt1'] = st.session_state['vwap3_mt1']

def on_change_vwap3_mt2():
    st.session_state['sv_vwap3_mt2'] = st.session_state['vwap3_mt2']

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
    stock_name = "삼성전자"
    stocknames = None
    stock = None

    global emaA, emaB, emaC, emaD
    global vwap1_cb, vwap1_dt, vwap1_mt1, vwap1_mt2
    global vwap2_cb, vwap2_dt, vwap2_mt1, vwap2_mt2
    global vwap3_cb, vwap3_dt, vwap3_mt1, vwap3_mt2
    global stocklist_df, selected_stockname

    task_name = "get_stocklist_from_db"
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
                    stock_name = selected_stockname

        with col2:
            col21, col22 = st.columns(2)
            with col21:
                ema_A_length = 33 if col21.toggle(label="33", key="emaA", value=emaA, on_change=on_change_emaA) else 0
                ema_B_length = 112 if col21.toggle(label="112", key="emaB", value=emaB, on_change=on_change_emaB) else 0
            with col22:
                ema_C_length = 224 if col22.toggle(label="224", key="emaC", value=emaC, on_change=on_change_emaC) else 0            
                ema_D_length = 448 if col22.toggle(label="448", key="emaD", value=emaD, on_change=on_change_emaD) else 0

        # 체크박스와 날짜선택박스 변수 초기화
        vwap_param = {}
        with col3:
            col31, col32 = st.columns(2)
            with col31:
                checkbox1 = col31.checkbox("VW1", key="vwap1_cb", value=vwap1_cb, on_change=on_change_vwap1_cb)
                date_input1 = col31.date_input(label="시작일", key="vwap1_dt", disabled=not checkbox1, label_visibility="collapsed", value=vwap1_dt, on_change=on_change_vwap1_dt)
            with col32:
                if checkbox1:
                    vwap1mt1 = col32.checkbox(label="B1", key="vwap1_mt1", value=vwap1_mt1, on_change=on_change_vwap1_mt1)
                    vwap1mt2 = col32.checkbox(label="B2", key="vwap1_mt2", value=vwap1_mt2, on_change=on_change_vwap1_mt2)
                    vb1_mt1 = 1 if vwap1mt1 else 0
                    vb1_mt2 = 2 if vwap1mt2 else 0 
                    vwap_param.update({'VWAP_1': {'dt': date_input1.strftime('%Y-%m-%d'), 'mt1': vb1_mt1, 'mt2': vb1_mt2, 'color':'blue'}})

        with col4:
            col41, col42 = st.columns(2)
            with col41:
                checkbox2 = col41.checkbox("VW2", key="vwap2_cb", value=vwap2_cb, on_change=on_change_vwap2_cb)
                date_input2 = col41.date_input(label="시작일", key="vwap2_dt", disabled=not checkbox2, label_visibility="collapsed", value=vwap2_dt, on_change=on_change_vwap2_dt)
            with col42:
                if checkbox2:
                    vwap2mt1 = col42.checkbox(label="B1", key="vwap2_mt1", value=vwap2_mt1, on_change=on_change_vwap2_mt1)
                    vwap2mt2 = col42.checkbox(label="B2", key="vwap2_mt2", value=vwap2_mt2, on_change=on_change_vwap2_mt2)
                    vb2_mt1 = 1 if vwap2mt1 else 0
                    vb2_mt2 = 2 if vwap2mt2 else 0 
                    vwap_param.update({'VWAP_2': {'dt': date_input2.strftime('%Y-%m-%d'), 'mt1': vb2_mt1, 'mt2': vb2_mt2, 'color':'purple'}})

        with col5:
            col51, col52 = st.columns(2)
            with col51:
                checkbox3 = col51.checkbox("VW3", key="vwap3_cb", value=vwap3_cb, on_change=on_change_vwap3_cb)
                date_input3 = col51.date_input(label="시작일", key="vwap3_dt", disabled=not checkbox3, label_visibility="collapsed", value=vwap3_dt, on_change=on_change_vwap3_dt)
            with col52:
                if checkbox3:
                    vwap3mt1 = col52.checkbox(label="B1", key="vwap3_mt1", value=vwap3_mt1, on_change=on_change_vwap3_mt1)
                    vwap3mt2 = col52.checkbox(label="B2", key="vwap3_mt2", value=vwap3_mt2, on_change=on_change_vwap3_mt2)
                    vb3_mt1 = 1 if vwap3mt1 else 0
                    vb3_mt2 = 2 if vwap3mt2 else 0 
                    vwap_param.update({'VWAP_3': {'dt': date_input3.strftime('%Y-%m-%d'), 'mt1': vb3_mt1, 'mt2': vb3_mt2, 'color':'green'}})

        df = yf.fetch_stock_data(symbol=stock_code, period="5y", interval="1d")
        len_df = len(df)
        if len_df <= ema_A_length: ema_A_length = 0
        if len_df <= ema_B_length: ema_B_length = 0
        if len_df <= ema_C_length: ema_C_length = 0
        if len_df <= ema_D_length: ema_D_length = 0
        
        ema_param = {}
        if ema_A_length > 0:
            ema_param.update({'EMA_A': {'length': ema_A_length, 'color': 'red', 'linewidth': 1}})
        if ema_B_length > 0:
            ema_param.update({'EMA_B': {'length': ema_B_length, 'color': 'orange', 'linewidth': 1}})
        if ema_C_length > 0:
            ema_param.update({'EMA_C': {'length': ema_C_length, 'color': 'black', 'linewidth': 2}})
        if ema_D_length > 0:
            ema_param.update({'EMA_D': {'length': ema_D_length, 'color': 'gray', 'linewidth': 1}})

        indicators_params = {'ema': ema_param, 'vwap': vwap_param}

        # Save, Load button
        with col6:
            col61, col62 = st.columns(2)
            with col61:
                if st.button(key="save_indicators", label="지표저장"):
                    if stock.any():
                        uidx, market, code = 1, stock[0][0], stock[0][1]
                        json_indicators = json.dumps(indicators_params)

                        task_name = "insert_technical_indicator"
                        params = {'uidx': uidx,
                                'market': f"{market}",
                                'code': f"{code}",
                                'name': f"{selected_stockname}",
                                'indicators': f"{json_indicators}"}
                        respose = dc.fetch_result_from_remote_server(task_name, params)
                        if "return" in respose:
                            if "result" in respose["return"]:
                                if respose["return"]["result"] == "error":
                                    st.error(respose["return"]["data"])
                st.button(key="load_indicators", label="지표로드", on_click=on_click_load_indicators)

            with col62:
                st.button(key="save_stock_interest", label="관심등록", on_click=on_click_save_stock_insterest)

        show_volume = False
        click_events_dy = chart.get_stock_chart(  symbol=stock_code
                                                , dataframe=df
                                                , indicators_params=indicators_params
                                                , pane_name="pane_daily"
                                                , time_minspacing=3
                                                , show_volume=show_volume
                                                , chart_height=650)        
        
if __name__ == '__main__':
    main()