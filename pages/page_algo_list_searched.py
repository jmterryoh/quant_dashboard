import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from chart import TRV_lightchart as chart
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from hts import YF_api as yf
from middleware import md_client as dc
from util import screen as scr
from util import trvta as tt

global grid1


def display_search_results():

    # Your code to display search results in ag-Grid
    # 자동매매대상
    # Add subtitle
    col_subtitle1 = scr.create_column_subtitle('자동매매대상')
    st.markdown(col_subtitle1, unsafe_allow_html=True)

    task_name = "get_algo_stocks_increase10"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df1 = pd.DataFrame(respose["return"]["data"])
    if not df1.empty:
        recommendation_list = []
        codes = df1["code"].tolist()
        recommendation_list = tt.get_tradingview_ta(codes)
        if recommendation_list:
            recommendation_df = pd.DataFrame(recommendation_list)
            df1 = pd.merge(df1, recommendation_df, on='code', how='outer')
            df1 = df1.sort_values(by='pattern')

        # ag-Grid 옵션 설정
        gb1 = GridOptionsBuilder.from_dataframe(df1[["pattern", "idt", "i10dt", "code", "market", "name", "close", "sector", "recommendation", "indicator"]])
        # configure selection
        gb1.configure_selection(selection_mode="single", use_checkbox=True)
        gb1.configure_column("pattern", header_name="검색그룹", width=110)
        gb1.configure_column("idt", header_name="탐지일", width=90)
        gb1.configure_column("i10dt", header_name="급등일", width=90)
        gb1.configure_column("code", header_name="코드", width=70)
        gb1.configure_column("market", header_name="시장", width=70)
        gb1.configure_column("name", header_name="종목명", width=140)
        gb1.configure_column("close", header_name="현재가(원)", width=80)
        gb1.configure_column("sector", header_name="업종", width=100)
        gb1.configure_column("recommendation", header_name="추천", width=80)
        gb1.configure_column("indicator", header_name="지표", width=80)

        #gb.configure_side_bar()
        grid1_options = gb1.build()

        # Streamlit에서 ag-Grid를 표시
        grid1 = AgGrid(df1,
                       height=760, #760
                       width="100%",
                       gridOptions=grid1_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       #key="grid1",
                       #reload_data=True,
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid1

def display_stock_charts(stock_name, stock_code, indicators_params, cycle, period, interval, time_minspacing=3, height=350):

    df = {}
    col1, col2 = st.columns(2)
    with col1:
        # Add subtitle
        chart_subtitle = scr.create_column_subtitle(f"{cycle}-{stock_name}")
        st.markdown(chart_subtitle, unsafe_allow_html=True)
    with col2:
        # Add a checkbox for toggling volume display
        show_volume = st.checkbox("거래량", value=False, key=f"cb_vol_{cycle}")
        #show_volume = False
        if stock_code:
            df = yf.fetch_stock_data(symbol=stock_code, period=period, interval=interval)

    if len(df) > 0:
        click_events = chart.get_stock_chart(symbol=stock_code,
                                             dataframe=df,
                                             indicators_params=indicators_params,
                                             pane_name=f"pane_{period.lower()}",
                                             time_minspacing=time_minspacing,
                                             show_volume=show_volume,
                                             chart_height=height)
    else:
        st.write("No data available")


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    ss.check_session('pages/page_list.searched.py')
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    global grid1

    grid1 = None
    stock_code1, stock_name1 = None, None
    selected_stock_code, selected_stock_name = None, None
    grid1_selected_rows = None

    col1, col2 = st.columns(2)

    with col1:
        grid1 = display_search_results()
        if grid1:
            if 'selected_rows' in grid1:
                grid1_selected_rows = grid1['selected_rows']
                if grid1_selected_rows:
                    stock_code1 = grid1_selected_rows[0]['code'] + "." + grid1_selected_rows[0]['market']
                    stock_name1 = grid1_selected_rows[0]['name']

    with col2:

        vdt = ""
        if grid1_selected_rows:
            selected_stock_name, selected_stock_code = stock_name1, stock_code1

            success, inc_df = dc.get_algo_stock_increase10(code=grid1_selected_rows[0]['code'])
            if success:
                vdt = inc_df.loc[0, 'vdt']
                vdt = datetime.strptime(vdt, "%Y%m%d")
                vdt = vdt.strftime("%Y-%m-%d")

        indicators_params_dy = {
            'ema': {
#                'EMA_A': {'length': 5, 'color': 'gray', 'linewidth': 2},
                'EMA_B': {'length': 10, 'color': 'blue', 'linewidth': 2},
                'EMA_C': {'length': 33, 'color': 'red', 'linewidth': 2},
#                'EMA_D': {'length': 62, 'color': 'green', 'linewidth': 2},
                'EMA_E': {'length': 112, 'color': 'orange', 'linewidth': 2},
                'EMA_F': {'length': 224, 'color': 'black', 'linewidth': 2},
            },
            'vwap': {
                'VWAP_1': {'dt': f'{vdt}', 'mt1': 1, 'mt2': 2, 'color':'green'}
            }
        }
        display_stock_charts(selected_stock_name, selected_stock_code, indicators_params=indicators_params_dy, cycle="일봉", period="2y", interval="1d", time_minspacing=30, height=350)
        indicators_params_wk = {
            'ema': {
                'EMA_A': {'length': 7, 'color': 'red', 'linewidth': 1},
                'EMA_B': {'length': 22, 'color': 'orange', 'linewidth': 1},
                'EMA_C': {'length': 45, 'color': 'black', 'linewidth': 2},
            },
        }        
        display_stock_charts(selected_stock_name, selected_stock_code, indicators_params=indicators_params_wk, cycle="주봉", period="10y", interval="1wk", height=350)

if __name__ == '__main__':
    main()