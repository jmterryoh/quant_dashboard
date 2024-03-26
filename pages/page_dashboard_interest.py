import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from chart import TRV_lightchart_rightpane as chart
from hts import YF_api as yf
from db import db_client as dc
from datetime import datetime
import json
import asyncio


#@st.cache_data
def get_stocklist_interest():
    df = {}

    task_name = "get_stocklist_interest"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df = pd.DataFrame(respose["return"]["data"])
            elif respose["return"]["result"] == "error":
                st.error(respose["return"]["data"])
        else:
            st.error("network error")
    else:
        st.error("network error")

    return df    

#@st.cache_data
def get_stock_indicators(uidx, market, code):

    df = {}

    task_name = "get_technical_indicators"
    params = {'uidx': uidx, 'market': f"{market}", 'code': f"{code}"}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                indicators_df = pd.DataFrame(respose["return"]["data"])
                if not indicators_df.empty:
                    indicators = indicators_df.loc[indicators_df['indicators'].notnull(), 'indicators'].iloc[0]
                    indicators_dict = json.loads(indicators)
                    ema_params = indicators_dict['ema']
                    ema_df = {}
                    if 'EMA_A' in ema_params:
                        ema_df.update({'EMA_A': {'length': 33, 'color': 'red', 'linewidth': 1}})
                    if 'EMA_B' in ema_params:
                        ema_df.update({'EMA_B': {'length': 112, 'color': 'orange', 'linewidth': 1}})
                    if 'EMA_C' in ema_params:
                        ema_df.update({'EMA_C': {'length': 224, 'color': 'black', 'linewidth': 2}})
                    if 'EMA_D' in ema_params:
                        ema_df.update({'EMA_D': {'length': 448, 'color': 'gray', 'linewidth': 1}})

                    vwap_params = indicators_dict['vwap']
                    vwap_df = {}
                    vwap_dt, vwap_mt1, vwap_mt2 = (datetime.strptime(vwap_params['VWAP_1']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_1']['mt1'], vwap_params['VWAP_1']['mt2']) if 'VWAP_1' in vwap_params else (None, False, False)
                    if vwap_dt: 
                        vwap_df.update({'VWAP_1': {'dt': vwap_dt, 'mt1': vwap_mt1, 'mt2': vwap_mt2, 'color':'blue'}})
                    vwap_dt, vwap_mt1, vwap_mt2 = (datetime.strptime(vwap_params['VWAP_2']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_2']['mt1'], vwap_params['VWAP_2']['mt2']) if 'VWAP_2' in vwap_params else (None, False, False)
                    if vwap_dt: 
                        vwap_df.update({'VWAP_1': {'dt': vwap_dt, 'mt1': vwap_mt1, 'mt2': vwap_mt2, 'color':'blue'}})
                    vwap_dt, vwap_mt1, vwap_mt2 = (datetime.strptime(vwap_params['VWAP_3']['dt'], "%Y-%m-%d").date(), vwap_params['VWAP_3']['mt1'], vwap_params['VWAP_3']['mt2']) if 'VWAP_3' in vwap_params else (None, False, False)
                    if vwap_dt: 
                        vwap_df.update({'VWAP_1': {'dt': vwap_dt, 'mt1': vwap_mt1, 'mt2': vwap_mt2, 'color':'blue'}})

                    df = {'ema': ema_df, 'vwap': vwap_params}

    return df

def display_stock_charts(market, name, code, indicators_params, cycle, period, interval, time_minspacing, height=350):

    df = {}
    stock_code = code + "." + market

    df = yf.fetch_stock_data(symbol=stock_code, period=period, interval=interval)
    if len(df) > 0:
        click_events = chart.get_stock_chart(symbol=stock_code,
                                             dataframe=df,
                                             indicators_params=indicators_params,
                                             pane_name=f"pane_{code}_{period.lower()}",
                                             time_minspacing=time_minspacing,
                                             show_volume=False,
                                             chart_height=height)
    else:
        st.write("No data available")

def display_page():

    df = {}
    df = get_stocklist_interest()    
    if df is None:
        return
    
    column_list = []  # 열을 저장할 리스트

    for index, row in df.iterrows():
        market = row['market']
        code = row['code']
        name = row['name']
        pattern = row['pattern']
        udt = row['udt']

        # Add subtitle
        col_subtitle = sc.create_column_subtitle(name)
        st.markdown(col_subtitle, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        column_list.append([col1, col2])
        with col1:
            indicators_params_dy = {}
            indicators_params_dy = get_stock_indicators(uidx=1, market=market, code=code)
            display_stock_charts(market=market, name=name, code=code, indicators_params=indicators_params_dy, cycle="일봉", period="5y", interval="1d", time_minspacing=15, height=350)

        with col2:
            if sc.get_screen_width() >= 800:
                #indicators_params_wk = {}
                #indicators_params_dy = get_stock_indicators(uidx=1, market=market, code=code)
                #display_stock_charts(market=market, name=name, code=code, indicators_params=indicators_params_wk, cycle="주봉", period="10y", interval="1wk", height=350)
                indicators_params_dy = {}
                indicators_params_dy = get_stock_indicators(uidx=1, market=market, code=code)
                display_stock_charts(market=market, name=name, code=code, indicators_params=indicators_params_dy, cycle="일봉", period="3y", interval="1d", time_minspacing=3, height=350)


button_refresh = None
async def update_screen():
    while True:
        ss.check_session('pages/page_dashboard_interest_m.py')
        if button_refresh:
            display_page()
        await asyncio.sleep(30)
        st.rerun()


if __name__ == "__main__":

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    ss.check_session('pages/page_dashboard_interest.py')
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    button_refresh = st.toggle(label="정보갱신 30초", key="button_refresh", value=True)

    try:
        # async run the draw function, sending in all the
        # widgets it needs to use/populate
        asyncio.run(update_screen())
    except Exception as e:
        print(f'error...{type(e)}')
        raise
    finally:    
        # some additional code to handle user clicking stop
        print('finally')
