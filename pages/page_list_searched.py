import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import session as ss
from util import screen as sc
from pages import sidebar as sb
from chart import TRV_lightchart_rightpane as chart
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from hts import YF_api as yf
from db import db_client as dc
from util import screen as scr

global grid1, grid2


def display_search_results():

    # Your code to display search results in ag-Grid
    # 조건검색종목
    # Add subtitle
    idt, seq = None, None
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        col_subtitle1 = scr.create_column_subtitle('조건검색종목')
        st.markdown(col_subtitle1, unsafe_allow_html=True)
    with col2:
        task_name = "get_stocklist_searched_dates"
        params = {}
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    searched_dates_df = pd.DataFrame(respose["return"]["data"])
                    #print(searched_dates_df)
                    searched_dates_df['dates'] = searched_dates_df['idt'] + "-" + searched_dates_df['seq'].astype(str)
                    searched_dates = searched_dates_df['dates'].tolist()
                    selected_searched_date = st.selectbox(label="목록", options=searched_dates, key="cb_dates", label_visibility='collapsed')
                    style_narrow = """
                    <style>
                    .st-bk {padding-right:0!important;}
                    .st-bi {padding-bottom:0!important;}
                    .st-bh {padding-top:0!important;}
                    </style>
                    """
                    st.markdown(style_narrow, unsafe_allow_html=True)

                    if selected_searched_date:
                        selected_date = searched_dates_df.loc[searched_dates_df['dates'] == selected_searched_date, ['idt', 'seq']].values
                        idt = selected_date[0][0]
                        seq = selected_date[0][1]

    df1 = None
    if idt and seq:
        task_name = "get_stocklist_searched"
        params = {'idt': f"{idt}", 'seq': seq}
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    df1 = pd.DataFrame(respose["return"]["data"])
    else:
        task_name = "get_stocklist_searched_last"
        params = {}
        respose = dc.fetch_result_from_remote_server(task_name, params)
        if "return" in respose:
            if "result" in respose["return"]:
                if respose["return"]["result"] == "success":
                    df1 = pd.DataFrame(respose["return"]["data"])
    if df1 is not None:
        # ag-Grid 옵션 설정
        gb1 = GridOptionsBuilder.from_dataframe(df1[["pattern", "code", "market", "name", "price", "stotprice"]])
        # configure selection
        gb1.configure_selection(selection_mode="single", use_checkbox=True)
        gb1.configure_column("pattern", header_name="검색그룹", width=140)
        gb1.configure_column("code", header_name="코드", width=70)
        gb1.configure_column("market", header_name="시장", width=70)
        gb1.configure_column("name", header_name="종목명", width=140)
        gb1.configure_column("price", header_name="현재가(원)", width=80
                            , type=["numericColumn","numberColumnFilter"]
                            , valueGetter="data.price.toLocaleString('ko-KR')", precision=0)
        gb1.configure_column("stotprice", header_name="시총(억)", width=80
                            , type=["numericColumn","numberColumnFilter"]
                            , valueGetter="data.stotprice.toLocaleString('ko-KR')", precision=0)
        #gb.configure_side_bar()
        grid1_options = gb1.build()

        # Streamlit에서 ag-Grid를 표시
        grid1 = AgGrid(df1,
                       height=330, #760
                       width="100%",
                       gridOptions=grid1_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       key="grid1",
                       reload_data=True,
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid1
                
def display_add_button(col12):
    global grid1

    grid1_selected_rows = None
    if not grid1:
        return None
    
    if 'selected_rows' in grid1:
        grid1_selected_rows = grid1['selected_rows']

    with col12:
        if grid1_selected_rows:
            button_add = st.button(label="▼ 관심추가", type="primary", on_click=on_click_add)
        else:
            button_add = st.button(label="▼ 관심추가", type="secondary", on_click=on_click_add)

    return button_add

def on_click_add():
    global grid1

    if grid1:
        if "selected_rows" in grid1:
            selected_rows = grid1["selected_rows"]
            if selected_rows:
                task_name = "insert_stock_interest"
                params = {'uidx': 1,
                          'market': f"{selected_rows[0]['market']}",
                          'code': f"{selected_rows[0]['code']}",
                          'name': f"{selected_rows[0]['name']}",
                          'pattern': f"{selected_rows[0]['pattern']}",
                          'description':""}
                respose = dc.fetch_result_from_remote_server(task_name, params)
                if "return" in respose:
                    if "result" in respose["return"]:
                        if respose["return"]["result"] == "error":
                            st.error(respose["return"]["data"])

def display_remove_button(col13):
    global grid2

    grid2_selected_rows = None
    if 'selected_rows' in grid2:
        grid2_selected_rows = grid2['selected_rows']

    with col13:
        if grid2_selected_rows:
            button_del = st.button(label="▲ 관심삭제", type="primary", on_click=on_click_remove)
        else:
            button_del = st.button(label="▲ 관심삭제", type="secondary", on_click=on_click_remove)

    return button_del

def on_click_remove():
    global grid2

    if grid2:
        if "selected_rows" in grid2:
            selected_rows = grid2["selected_rows"]
            if selected_rows:
                task_name = "delete_stock_interest"
                params = {'uidx': 1,
                          'market': f"{selected_rows[0]['market']}",
                          'code': f"{selected_rows[0]['code']}"}
                respose = dc.fetch_result_from_remote_server(task_name, params)
                if "return" in respose:
                    if "result" in respose["return"]:
                        if respose["return"]["result"] == "error":
                            st.error(respose["return"]["data"])

def display_interest_list():

    # 관심종목
    # Add subtitle
    col_subtitle2 = scr.create_column_subtitle('관심종목')
    st.markdown(col_subtitle2, unsafe_allow_html=True)

    df2 = {}

    task_name = "get_stocklist_interest"
    params = {}
    respose = dc.fetch_result_from_remote_server(task_name, params)
    if "return" in respose:
        if "result" in respose["return"]:
            if respose["return"]["result"] == "success":
                df2 = pd.DataFrame(respose["return"]["data"])

    if df2 is not None:                

        # ag-Grid 옵션 설정
        gb2 = GridOptionsBuilder.from_dataframe(df2[["pattern", "code", "market", "name", "description"]])
        # configure selection
        gb2.configure_selection(selection_mode="single", use_checkbox=True)
        gb2.configure_column("pattern", header_name="검색그룹", width=140)
        gb2.configure_column("code", header_name="코드", width=70)
        gb2.configure_column("market", header_name="시장", width=70)
        gb2.configure_column("name", header_name="종목명", width=140)
        gb2.configure_column("description", header_name="메모", width=160)

        #gb.configure_side_bar()
        grid2_options = gb2.build()

        # Streamlit에서 ag-Grid를 표시
        grid2 = AgGrid(df2,
                       height=330, #760
                       width="100%",
                       gridOptions=grid2_options,
                       allow_unsafe_jscode=True,
                       custom_css={"#gridToolBar":{"padding-bottom":"0px!important",}},
                       key="grid2",
                       reload_data=True, 
                       update_mode=(GridUpdateMode.MODEL_CHANGED|GridUpdateMode.GRID_CHANGED|GridUpdateMode.SELECTION_CHANGED|GridUpdateMode.VALUE_CHANGED))
        return grid2

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
    ss.check_session('pages/page_list.searched.py')
    sb.menu_with_redirect()
    sc.show_min_sidebar()

    global grid1, grid2

    grid1, grid2 = None, None
    stock_code1, stock_name1 = None, None
    stock_code2, stock_name2 = None, None
    selected_stock_code, selected_stock_name = None, None
    grid1_selected_rows, grid2_selected_rows = None, None

    col1, col2 = st.columns(2)

    with col1:
        grid1 = display_search_results()
        col11, col12, col13, col14 = st.columns(4)
        display_add_button(col12)
        grid2 = display_interest_list()
        display_remove_button(col13)

        if grid1:
            if 'selected_rows' in grid1:
                grid1_selected_rows = grid1['selected_rows']
                if grid1_selected_rows:
                    stock_code1 = grid1_selected_rows[0]['code'] + "." + grid1_selected_rows[0]['market']
                    stock_name1 = grid1_selected_rows[0]['name']
        if grid2:
            if 'selected_rows' in grid2:
                grid2_selected_rows = grid2['selected_rows']
                if grid2_selected_rows:
                    stock_code2 = grid2_selected_rows[0]['code'] + "." + grid2_selected_rows[0]['market']
                    stock_name2 = grid2_selected_rows[0]['name']

    with col2:

        if grid1_selected_rows and grid2_selected_rows:
            selected_stock_name, selected_stock_code = None, None
            st.write(f"{stock_name1}과 {stock_name2} 중 하나만 선택하세요.")
            return
        elif grid1_selected_rows:
            selected_stock_name, selected_stock_code = stock_name1, stock_code1
        elif grid2_selected_rows:
            selected_stock_name, selected_stock_code = stock_name2, stock_code2

        indicators_params_dy = {
            'ema': {
                'EMA_A': {'length': 33, 'color': 'red', 'linewidth': 1},
                'EMA_B': {'length': 112, 'color': 'orange', 'linewidth': 1},
                'EMA_C': {'length': 224, 'color': 'black', 'linewidth': 2},
            },
        }
        display_stock_charts(selected_stock_name, selected_stock_code, indicators_params=indicators_params_dy, cycle="일봉", period="5y", interval="1d", height=350)
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
