import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from chart import TRV_lightchart as chart
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from hts import YF_api as yf
from middleware import md_client as dc
from util import screen as scr
from util import trvta as tt
import time

global grid1, grid2

def alert_warning(message):
    alert = st.warning(message)
    time.sleep(2)
    alert.empty()

def display_interest_list():

    # 관심종목
    # Add subtitle
    col_subtitle1 = scr.create_column_subtitle('관심종목')
    st.markdown(col_subtitle1, unsafe_allow_html=True)

    df1 = {}

    task_name = "get_stocklist_interest"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df1 = pd.DataFrame(respose["return"]["data"])
            elif respose["return"]["result"] == "error":
                st.error(respose["return"]["data"])
        else:
            st.error("network error")
    else:
        st.error("network error")
        
    if not df1.empty:                
        recommendation_list = []
        codes = df1["code"].tolist()
        recommendation_list = tt.get_tradingview_ta(codes)
        if recommendation_list:
            recommendation_df = pd.DataFrame(recommendation_list)
            df1 = pd.merge(df1, recommendation_df, on='code', how='outer')
            df1 = df1.sort_values(by='pattern')

        # ag-Grid 옵션 설정
        gb1 = GridOptionsBuilder.from_dataframe(df1[["pattern", "code", "market", "name", "sector", "recommendation", "indicator"]])
        # configure selection
        gb1.configure_selection(selection_mode="single", use_checkbox=True)
        gb1.configure_column("pattern", header_name="검색그룹", width=140)
        gb1.configure_column("code", header_name="코드", width=70)
        gb1.configure_column("market", header_name="시장", width=70)
        gb1.configure_column("name", header_name="종목명", width=140)
        gb1.configure_column("sector", header_name="업종", width=100)
        gb1.configure_column("recommendation", header_name="추천", width=100)
        gb1.configure_column("indicator", header_name="지표", width=160)

        #gb.configure_side_bar()
        grid1_options = gb1.build()

        # Streamlit에서 ag-grid를 표시
        grid1 = AgGrid(df1,
                       height=330, #760
                       width="100%",
                       gridOptions=grid1_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       #key="grid1",
                       #reload_data=True, 
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid1

def display_owned_list():

    # 관심종목
    # Add subtitle
    col_subtitle2 = scr.create_column_subtitle('보유종목')
    st.markdown(col_subtitle2, unsafe_allow_html=True)

    df2 = {}

    task_name = "get_stocks_owned_all"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df2 = pd.DataFrame(respose["return"]["data"])
            elif respose["return"]["result"] == "error":
                st.error(respose["return"]["data"])
        else:
            st.error("network error")
    else:
        st.error("network error")

    if not df2.empty:                
        
        df2 = df2[df2['quantity'] > 0]
        if df2.empty:
            return

        recommendation_list = []
        codes = df2["code"].tolist()
        recommendation_list = tt.get_tradingview_ta(codes)
        if recommendation_list:
            recommendation_df = pd.DataFrame(recommendation_list)
            df2 = pd.merge(df2, recommendation_df, on='code', how='outer')
            df2 = df2.sort_values(by='name')

        df2['price'] = df2['price'].astype(int)
        df2['amount'] = df2['amount'].astype(int)
        # ag-Grid 옵션 설정
        gb2 = GridOptionsBuilder.from_dataframe(df2[["market", "code", "name", "price", "quantity", "amount", "sector", "recommendation", "indicator"]])
        # configure selection
        gb2.configure_selection(selection_mode="single", use_checkbox=True)
        gb2.configure_column("code", header_name="코드", width=70)
        gb2.configure_column("market", header_name="시장", width=70)
        gb2.configure_column("name", header_name="종목명", width=140)
        gb2.configure_column("price", header_name="보유가", width=100
                            , type=["numericColumn","numberColumnFilter"]
                            , valueGetter="data.price.toLocaleString('ko-KR')", precision=0)
        gb2.configure_column("quantity", header_name="주식수", width=100
                            , type=["numericColumn","numberColumnFilter"]
                            , valueGetter="data.quantity.toLocaleString('ko-KR')", precision=0)
        gb2.configure_column("amount", header_name="금액", width=120
                            , type=["numericColumn","numberColumnFilter"]
                            , valueGetter="data.amount.toLocaleString('ko-KR')", precision=0)
        gb2.configure_column("sector", header_name="업종", width=100)
        gb2.configure_column("recommendation", header_name="추천", width=80)
        gb2.configure_column("indicator", header_name="지표", width=140)

        #gb.configure_side_bar()
        grid2_options = gb2.build()

        # Streamlit에서 ag-grid를 표시
        grid2 = AgGrid(df2,
                       height=330, #760
                       width="100%",
                       gridOptions=grid2_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       #key="grid2",
                       #reload_data=True, 
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid2
                
def display_add_button(col22):
    global grid1

    grid1_selected_rows = None
    if not grid1:
        return None
    
    if 'selected_rows' in grid1:
        grid1_selected_rows = grid1['selected_rows']

    with col22:
        if grid1_selected_rows:
            button_add = st.button(label="▼ 보유추가(매수)", type="primary", on_click=on_click_buy)
        else:
            button_add = st.button(label="▼ 보유추가(매수)", type="secondary", on_click=on_click_buy)

    return button_add

def on_click_buy():
    global grid1

    if grid1:
        if "selected_rows" in grid1:
            selected_rows = grid1["selected_rows"]
            if selected_rows:
                stock_price1 = st.session_state['stock_price1']
                stock_quantity1 = st.session_state['stock_quantity1']
                stock_date1 = st.session_state['stock_date1']
                trdt = stock_date1.strftime('%Y%m%d')
                trtype = "BUY"
                if stock_price1 <= 0:
                    alert_warning("매수가를 입력해 주세요.")
                    return                
                if stock_quantity1 <= 0:
                    alert_warning("매수주식수를 입력해 주세요.")
                    return                
                  
                ret_Ok, output = dc.update_owned_stock_transaction(uidx=1,
                                                                   market=selected_rows[0]['market'],
                                                                   code=selected_rows[0]['code'],
                                                                   name=selected_rows[0]['name'],
                                                                   price=stock_price1,
                                                                   quantity=stock_quantity1,
                                                                   trtype=trtype,
                                                                   trdt=trdt,
                                                                   reason="")
            else:
                alert_warning("보유추가(매수)할 종목을 선택해 주세요.")                

def display_remove_button(col23):
    global grid2

    grid2_selected_rows = None
    if not grid2:
        return None

    if 'selected_rows' in grid2:
        grid2_selected_rows = grid2['selected_rows']

    with col23:
        if grid2_selected_rows:
            button_del = st.button(label="▲ 보유삭제(매도)", type="primary", key="button_del", on_click=on_click_sell)
        else:
            button_del = st.button(label="▲ 보유삭제(매도)", type="secondary", key="button_del", on_click=on_click_sell)

    return button_del

def on_click_sell():
    global grid2

    if grid2:
        if "selected_rows" in grid2:
            selected_rows = grid2["selected_rows"]
            if selected_rows:
                stock_price2 = st.session_state['stock_price2']
                stock_quantity2 = st.session_state['stock_quantity2']
                stock_date2 = st.session_state['stock_date2']
                trdt = stock_date2.strftime('%Y%m%d')
                trtype = "SELL"
                if stock_price2 <= 0:
                    alert_warning("매도가를 입력해 주세요.")
                    return                
                if stock_quantity2 <= 0:
                    alert_warning("매도주식수를 입력해 주세요.")
                    return
                if selected_rows[0]['quantity'] < stock_quantity2:
                    alert_warning("보유주식수보다 매도주식수가 많을 수 없습니다.")
                    return
                  
                ret_Ok, output = dc.update_owned_stock_transaction(uidx=1,
                                                                   market=selected_rows[0]['market'],
                                                                   code=selected_rows[0]['code'],
                                                                   name=selected_rows[0]['name'],
                                                                   price=stock_price2,
                                                                   quantity=stock_quantity2,
                                                                   trtype=trtype,
                                                                   trdt=trdt,
                                                                   reason="")

            else:
                alert_warning("보유추가(매수)할 종목을 선택해 주세요.")    

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    ss.check_session('pages/page_list.owned.py')
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    global grid1, grid2

    stock_market1, stock_code1, stock_name1, stock_price1, stock_quantity1, stock_date1 = "", "", "", 0, 0, None
    stock_market2, stock_code2, stock_name2, stock_price2, stock_quantity2, stock_date2 = "", "", "", 0, 0, None
    selected_stock_code, selected_stock_name = None, None
    grid1_selected_rows, grid2_selected_rows = None, None

    col11, col12 = st.columns(2)

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
        col121, col122 = st.columns(2)
        disabled = True
        if grid1_selected_rows:
            disabled = False

        with col121:
            stock_market1 = st.text_input('시장', help='ex) KS:코스피, KQ:코스닥', value=stock_market1, disabled=True, key="stock_market1")
            stock_code1 = st.text_input('종목코드', help='ex) 005930', value=stock_code1, disabled=True, key="stock_code1")
            stock_name1 = st.text_input('종목명', help='ex) 삼성전자', value=stock_name1, disabled=True, key="stock_markstock_name1et1")
        with col122:
            stock_price1 = st.number_input('매수가', min_value=0, step=1, format='%d', help='ex) 10000, 200000', disabled=disabled, key="stock_price1")
            stock_quantity1 = st.number_input('수량', min_value=0, step=1, format='%d', help='ex) 10, 1000', disabled=disabled, key="stock_quantity1")
            stock_date1 = st.date_input('거래일', help='거래일', disabled=disabled, key="stock_date1")

    # 보유추가
    col21, col22, col23, col24, col25, col26, col27, col28, col29 = st.columns(9)
    with col24:
        display_add_button(col24)

    # 보유종목
    col31, col32 = st.columns(2)
    with col31:
        grid2 = display_owned_list()
        if grid2:
            if 'selected_rows' in grid2:
                grid2_selected_rows = grid2['selected_rows']
                if grid2_selected_rows:
                    stock_market2 = grid2_selected_rows[0]['market']
                    stock_code2 = grid2_selected_rows[0]['code']
                    stock_name2 = grid2_selected_rows[0]['name']
    with col32:
        col321, col322 = st.columns(2)
        disabled = True
        if grid2_selected_rows:
            disabled = False

        with col321:
            stock_market2 = st.text_input('시장', help='ex) KS:코스피, KQ:코스닥', value=stock_market2, disabled=True, key="stock_market2")
            stock_code2 = st.text_input('종목코드', help='ex) 005930', value=stock_code2, disabled=True, key="stock_code2")
            stock_name2 = st.text_input('종목명', help='ex) 삼성전자', value=stock_name2, disabled=True, key="stock_name2")
        with col322:
            stock_price2 = st.number_input('매도가', min_value=0, step=1, format='%d', help='ex) 10000, 200000', disabled=disabled, key="stock_price2")
            stock_quantity2 = st.number_input('수량', min_value=0, step=1, format='%d', help='ex) 10, 1000', disabled=disabled, key="stock_quantity2")
            stock_date2 = st.date_input('거래일', help='거래일', disabled=disabled, key="stock_date2")

    # 보유삭제
    with col26:
        display_remove_button(col26)



if __name__ == '__main__':
    main()