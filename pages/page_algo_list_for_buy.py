import streamlit as st
import pandas as pd
from pathlib import Path
import sys
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

TRD_BUY_SELL = {
      "sell" : "01"
    , "buy" : "02"
}

uidx = 1
current_page = "pages/page_algo_list_for_buy.py"

grid1, grid2 = None, None
effective_date, selected_algo_buy, selected_algo_sell, selected_aidx = "", None, None, 0
stoploss_price, allocated_amount = 0, 200000

def display_interest_list():

    # Your code to display interest list in ag-Grid
    # 관심종목
    # Add subtitle
    col_subtitle1 = scr.create_column_subtitle('관심종목')
    st.markdown(col_subtitle1, unsafe_allow_html=True)

    df1 = {}

    task_name = "get_stocklist_interest"
    params = {}
    response = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in response:
        if "result" in response["return"]:
            if response["return"]["result"] == "success":
                df1 = pd.DataFrame(response["return"]["data"])

    if df1 is not None:
        recommendation_list = []
        codes = df1["code"].tolist()
        recommendation_list = tt.get_tradingview_ta(codes)
        if recommendation_list:
            recommendation_df = pd.DataFrame(recommendation_list)
            df1 = pd.merge(df1, recommendation_df, on='code', how='outer')
            df1 = df1.sort_values(by='pattern')

        # 포멧 변환
        df1['open'] = df1['open'].apply(lambda x: "{:,.0f}".format(x))
        df1['close'] = df1['close'].apply(lambda x: "{:,.0f}".format(x))

        # ag-Grid 옵션 설정
        gb1 = GridOptionsBuilder.from_dataframe(df1[["pattern", "code", "market", "name", "sector", "recommendation", "indicator", "open", "close", "increase", "time"]])
        # configure selection
        gb1.configure_selection(selection_mode="single", use_checkbox=True)
        gb1.configure_column("pattern", header_name="검색그룹", width=140)
        gb1.configure_column("code", header_name="코드", width=70)
        gb1.configure_column("market", header_name="시장", width=70)
        gb1.configure_column("name", header_name="종목명", width=140)
        gb1.configure_column("sector", header_name="업종", width=100)
        gb1.configure_column("recommendation", header_name="추천", width=80)
        gb1.configure_column("indicator", header_name="지표", width=90)
        gb1.configure_column("open", header_name="시가", width=70)
        gb1.configure_column("close", header_name="현가", width=70)
        gb1.configure_column("increase", header_name="변동", width=70)
        gb1.configure_column("time", header_name="시점", width=90)

        #gb.configure_side_bar()
        grid1_options = gb1.build()

        # Streamlit에서 ag-Grid를 표시
        grid1 = AgGrid(df1,
                       height=405, #340, #760
                       width="100%",
                       gridOptions=grid1_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       #key="grid1",
                       #reload_data=True,
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid1
                
def display_add_button(col):
    global grid1

    grid1_selected_rows = None
    if not grid1:
        return None
    
    if 'selected_rows' in grid1:
        grid1_selected_rows = grid1['selected_rows']

    with col:
        if grid1_selected_rows:
            button_add = st.button(label="▼ 자동매수 추가", type="primary", on_click=on_click_add)
        else:
            button_add = st.button(label="▼ 자동매수 추가", type="secondary", on_click=on_click_add)

    return button_add

def on_click_add():
    global grid1, uidx, effective_date, selected_algo_buy, selected_algo_sell, allocated_amount, stoploss_price, selected_aidx
    selected_effective_date = ""

    if effective_date:
        selected_effective_date = effective_date.strftime("%Y%m%d")
    if grid1:
        if "selected_rows" in grid1:
            selected_rows = grid1["selected_rows"]
            if selected_rows:
                task_name = "insert_algo_stock_for_buy"
                params = {'uidx': uidx,
                          'aidx': selected_aidx,
                          'market': f"{selected_rows[0]['market']}",
                          'code': f"{selected_rows[0]['code']}",
                          'name': f"{selected_rows[0]['name']}",
                          'pattern': f"{selected_rows[0]['pattern']}",
                          'stoploss_price' : f"{stoploss_price}",
                          'effective_date': f"{selected_effective_date}",
                          'allocated_amount': allocated_amount,
                          'algorithm_buy': f"{selected_algo_buy}",
                          'algorithm_sell': f"{selected_algo_sell}",
                          'description': "" }
                response = dc.fetch_result_from_remote_server(task_name, params)
                if "return" in response:
                    if "result" in response["return"]:
                        if response["return"]["result"] == "error":
                            st.error(response["return"]["data"])


def display_remove_button(col):
    global grid2

    grid2_selected_rows = None
    if grid2:
        if 'selected_rows' in grid2:
            grid2_selected_rows = grid2['selected_rows']

    with col:
        if grid2_selected_rows:
            button_del = st.button(label="▲ 자동매수 삭제", type="primary", on_click=on_click_remove)
        else:
            button_del = st.button(label="▲ 자동매수 삭제", type="secondary", on_click=on_click_remove)

    return button_del

def on_click_remove():
    global grid2, uidx

    if grid2:
        if "selected_rows" in grid2:
            selected_rows = grid2["selected_rows"]
            if selected_rows:
                task_name = "delete_algo_stock_for_buy"
                params = {'idx': f"{selected_rows[0]['idx']}"}
                response = dc.fetch_result_from_remote_server(task_name, params)
                if "return" in response:
                    if "result" in response["return"]:
                        if response["return"]["result"] == "error":
                            st.error(response["return"]["data"])

def get_algorithm_list():
    task_name = "get_algo_list_all"
    params = {}
    response = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in response:
        if "result" in response["return"]:
            if response["return"]["result"] == "success":
                df = pd.DataFrame(response["return"]["data"])
                algo_list_buy = df[(df['del'] == 0) & (df['buy_sell_classification'] == TRD_BUY_SELL['buy'])]['name'].tolist()
                algo_list_sell = df[(df['del'] == 0) & (df['buy_sell_classification'] == TRD_BUY_SELL['sell'])]['name'].tolist()
                return algo_list_buy, algo_list_sell
            
    return None, None

def get_users_account():
    global uidx

    task_name = "get_users_account"
    params = {"uidx" : uidx}
    response = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in response:
        if "result" in response["return"]:
            if response["return"]["result"] == "success":
                users_account_df = pd.DataFrame(response["return"]["data"])
                return users_account_df

    return pd.DataFrame()


def display_algo_list_for_buy():

    # 자동매매 종목
    # Add subtitle
    col_subtitle2 = scr.create_column_subtitle('자동매매 종목')
    st.markdown(col_subtitle2, unsafe_allow_html=True)

    df2 = {}

    task_name = "get_algo_stocklist_for_buy"
    params = {}
    response = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in response:
        if "result" in response["return"]:
            if response["return"]["result"] == "success":
                df2 = pd.DataFrame(response["return"]["data"])

    if not df2.empty:
        # tradingview ta 
        recommendation_list = []
        codes = df2["code"].tolist()
        recommendation_list = tt.get_tradingview_ta(codes)
        if recommendation_list:
            recommendation_df = pd.DataFrame(recommendation_list)
            df2 = pd.merge(df2, recommendation_df, on='code', how='outer')
            df2 = df2.sort_values(by='pattern')

        # 'effective_date' 컬럼을 datetime 형식으로 변환하여 'converted' 컬럼에 적용
        df2['converted_date'] = pd.to_datetime(df2['effective_date'], format='%Y%m%d').dt.strftime('%m-%d')
        df2['stoploss_price'] = df2['stoploss_price'].apply(lambda x: "{:,.0f}".format(x))
        df2['allocated_amount'] = df2['allocated_amount'].apply(lambda x: "{:,.0f}".format(x))
        df2['open'] = df2['open'].apply(lambda x: "{:,.0f}".format(x))
        df2['close'] = df2['close'].apply(lambda x: "{:,.0f}".format(x))

        # ag-Grid 옵션 설정
        gb2 = GridOptionsBuilder.from_dataframe(df2[["pattern","idx","code","market","name","open","close","converted_date","stoploss_price","allocated_amount","algorithm_buy","algorithm_sell"]])
        # configure selection
        gb2.configure_selection(selection_mode="single", use_checkbox=True)
        gb2.configure_column("pattern", header_name="검색그룹", width=140)
        gb2.configure_column("idx", header_name="인덱스", width=0, hide=True)
        gb2.configure_column("code", header_name="코드", width=70)
        gb2.configure_column("market", header_name="시장", width=70)
        gb2.configure_column("name", header_name="종목명", width=140)
        gb2.configure_column("open", header_name="시가", width=70)
        gb2.configure_column("close", header_name="현가", width=70)
        gb2.configure_column("converted_date", header_name="시작일", width=80)
        gb2.configure_column("stoploss_price", header_name="손절가", width=80)
        gb2.configure_column("allocated_amount", header_name="할당금액", width=80)
        gb2.configure_column("algorithm_buy", header_name="매수기법", width=170)
        gb2.configure_column("algorithm_sell", header_name="매도기법", width=170)

        #gb.configure_side_bar()
        grid2_options = gb2.build()

        # Streamlit에서 ag-Grid를 표시
        grid2 = AgGrid(df2,
                       height=340, #760
                       width="100%",
                       gridOptions=grid2_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       #key="grid2",
                       #reload_data=True, 
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid2
    else:
        # 공란 표시
        st.info("No data")

def display_stock_charts(stock_name, stock_code, indicators_params, cycle, period, interval, height=350):

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
                                             time_minspacing=3,
                                             show_volume=show_volume,
                                             chart_height=height)
    else:
        st.write("No data available")


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    ss.check_session(current_page)
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    # selected_aidx : users_account idx 
    global grid1, grid2, effective_date, selected_algo_buy, selected_algo_sell, stoploss_price, allocated_amount, selected_aidx

    stock_market1, stock_code1, stock_name1 = "", "", ""
    stock_market2, stock_code2, stock_name2 = "", "", ""
    selected_stock_code, selected_stock_name = None, None
    grid1_selected_rows, grid2_selected_rows = None, None

    col11, col12 = st.columns([2,1])

    # 관심종목
    with col11:
        grid1 = display_interest_list()
        if grid1:
            if 'selected_rows' in grid1:
                grid1_selected_rows = grid1['selected_rows']
                if grid1_selected_rows:
                    stock_market1 = grid1_selected_rows[0]['market']
                    stock_code1 = grid1_selected_rows[0]['code']
                    stock_name1 = grid1_selected_rows[0]['name']
    with col12:
        col_subtitle = scr.create_column_subtitle('설정(관심종목 to 자동매수)')
        st.markdown(col_subtitle, unsafe_allow_html=True)

        col121, col122 = st.columns(2)
        disabled = True
        if grid1_selected_rows:
            disabled = False

        with col121:
            stock_market1 = st.text_input('시장', help='ex) KS:코스피, KQ:코스닥', value=stock_market1, disabled=True, key="stock_market1")
            stock_code1 = st.text_input('종목코드', help='ex) 005930', value=stock_code1, disabled=True, key="stock_code1")
            stock_name1 = st.text_input('종목명', help='ex) 삼성전자', value=stock_name1, disabled=True, key="stock_markstock_name1et1")
            effective_date = st.date_input('거래시작일', help='자동매수 시작일', disabled=disabled, key="stock_date1")
            users_account_df = get_users_account()
            if not users_account_df.empty:
                account_numbers = users_account_df['account_number'].tolist()
                selected_account_number = st.selectbox(label="자동매매 계좌", options=account_numbers, key="selected_account_number", help='자동매매 계좌', placeholder="계좌 선택", disabled=disabled, label_visibility='visible')
                if selected_account_number:
                    selected_idx_list = users_account_df.loc[users_account_df['account_number'] == selected_account_number, ['idx']].values
                    selected_aidx = int(selected_idx_list[0][0]) # int64 to int

        with col122:
            stoploss_price = st.number_input('손절가', min_value=0, step=1, format='%d', help='ex) 10000, 200000', value=stoploss_price, disabled=disabled, key="stock_price1")
            allocated_amount = st.number_input('할당금액', min_value=0, step=10000, format='%d', help='ex) 10000, 10000', value=allocated_amount, disabled=disabled, key="stock_amount1")
            list_buy, list_sell = get_algorithm_list()
            if list_buy:
                selected_algo_buy = st.selectbox(label="매수기법", options=list_buy, key="selected_algo_buy1", help='매수기법', placeholder="매수기법 선택", disabled=disabled, label_visibility='visible')
            if list_sell:
                selected_algo_sell = st.selectbox(label="매도기법", options=list_sell, key="selected_algo_sell1", help='매도기법', placeholder="매도기법 선택", disabled=disabled, label_visibility='visible')

    # blank row
    col00 = st.columns(1)

    # 자동매매 추가
    col21, col22, col23, col24, col25, col26, col27, col28, col29 = st.columns(9)
    with col24:
        display_add_button(col=col24)

    # blank row
    col00 = st.columns(1)

    # 자동매수 종목
    col31, col32 = st.columns([2,1])
    with col31:
        grid2 = display_algo_list_for_buy()
        if grid2:
            if 'selected_rows' in grid2:
                grid2_selected_rows = grid2['selected_rows']
                if grid2_selected_rows:
                    stock_market2 = grid2_selected_rows[0]['market']
                    stock_code2 = grid2_selected_rows[0]['code']
                    stock_name2 = grid2_selected_rows[0]['name']
    with col32:

        indicators_params_dy = {
            'ema': {
                'EMA_A': {'length': 33, 'color': 'red', 'linewidth': 1},
                'EMA_B': {'length': 112, 'color': 'orange', 'linewidth': 1},
                'EMA_C': {'length': 224, 'color': 'black', 'linewidth': 2},
            },
        }
        if grid1_selected_rows and grid2_selected_rows:
            selected_stock_name, selected_stock_code = None, None
            st.write(f"{stock_name1}과 {stock_name2} 중 하나만 선택하세요.")
        elif grid1_selected_rows:
            selected_stock_name, selected_stock_code = stock_name1, (stock_code1  + "." + stock_market1)
            display_stock_charts(selected_stock_name, selected_stock_code, indicators_params=indicators_params_dy, cycle="일봉", period="5y", interval="1d", height=330)
        elif grid2_selected_rows:
            selected_stock_name, selected_stock_code = stock_name2, (stock_code2  + "." + stock_market2)
            display_stock_charts(selected_stock_name, selected_stock_code, indicators_params=indicators_params_dy, cycle="일봉", period="5y", interval="1d", height=330)

    # 자동매수 삭제
    with col26:
        display_remove_button(col=col26)


if __name__ == '__main__':
    main()