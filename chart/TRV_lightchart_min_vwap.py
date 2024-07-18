import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts_ntf import renderLightweightCharts

import time
from datetime import datetime
import pytz

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import zigzag as zz
from util import vwapchain as vwc


#COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
#COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
COLOR_BULL = 'rgba(231,25,9,0.8)' # #26a69a
COLOR_BEAR = 'rgba(17,91,203,0.8)'  # #ef5350


def dataToJSON(df, column, slice=0, color=None):
    data = df[['time', column, 'color']].copy()
    data = data.rename(columns={column: "value"})
    if color is None:
        data.drop('color', axis=1, inplace=True)
    elif color != 'default':
        data['color'] = color
    if slice > 0:
        data = data.iloc[slice:, :]
    return json.loads(data.to_json(orient="records"))

def convertDataToJSON(df, column):
    data = df[['time', column]].copy()
    data = data.rename(columns={column: "value"})
    return json.loads(data.to_json(orient="records"))


def get_candlestick_string(title, data=None, pane=0):
    return {
            "type": 'Candlestick',
            "title": f"{title}",
            "data": data,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR,
                "pane": pane
            }
    }


def get_series_line_string(title, data=None, color='black', linewidth=1, linestyle=0, pane=0):
    return {
            "type": 'Line',
            "title": f"{title}",
            "data": data,
            "options": {
                "color": f"{color}",
                "lineWidth": linewidth,
                "lineStyle": linestyle, # LineStyle: 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
                "crosshairMarkerVisible": 0,
                "priceLineVisible": 0,
                "lineVisible": 0,
                "pane": pane,
            }
    }

def get_series_histogram_string(title, data=None, pane=1):
    return {
            "type": 'Histogram',
            "title": f"{title}",
            "data": data,
            "options": {
                "priceFormat": {
                    "type": 'volume',
                },
                "pane": pane,
            }
    }

VWAP_BAND = ['STD+1', 'STD-1', 'STD+2', 'STD-2']
def set_vwap_indicators(series, indicators):
    # indicators : 지표명과 대응하는 계산하는 함수명을 담고 있는 dictionary
    for indicator, (func, *args) in indicators.items():
        vwap, color = func(*args)
        vwap = vwap.reset_index().rename(columns={'Date':'time'})
        vwap['time'] = vwap['time'].dt.strftime('%Y-%m-%d %H:%M:%S') # Date to string
        json_vwap = convertDataToJSON(vwap, indicator)
        series.append(get_series_line_string(title=indicator, data=json_vwap, color=color, linewidth=2, linestyle=0, pane=0))
        for band_name in VWAP_BAND:
            if band_name in vwap.columns:
                vwap_std1p = convertDataToJSON(vwap, band_name)
                series.append(get_series_line_string(title=indicator+band_name, data=vwap_std1p, color=color, linewidth=1, linestyle=0, pane=0))

    return series


def string_datetime_to_timestamp(value):
    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    utc_dt = pytz.utc.localize(dt)
    kst_dt = utc_dt.astimezone(pytz.timezone('Asia/Seoul'))
    timestamp = int(time.mktime(kst_dt.timetuple()))
    return timestamp

def get_stock_chart(symbol
                   , dataframe
                   , vwap_dataframe
                   , vwap_band_gap
                   , vwap_high1_dataframe
                   , vwap_high2_dataframe
                   , vwap_highest_dataframe
                   , vwap_1day_dataframe
                   , bollinger_dataframe
                   , indicators_params={}
                   , pane_name="multipane"
                   , time_minspacing=8
                   , show_volume=True
                   , chart_height=350):
    
    # Parameters:
    #     symbol (str): 주식 기호.
    #     dataframe (pd.DataFrame): 주식 데이터가 포함된 데이터프레임.
    #     indicators_params (dict): 지표에 대한 파라미터 딕셔너리. (예: {'ema': {'A': 33, 'B': 112, ...}, 'vwap': {'VWAP_1': {'dt': ..., 'mt1': ..., 'mt2': ...}, ...}})
    #     pane_name (str): 패널 이름 (기본값: "multipane").
    #     time_minspacing (int): 시간 간격 (기본값: 8).
    #     show_volume (bool): 거래량 표시 여부 (기본값: True).
    #     chart_height (int): 차트 높이 (기본값: 350).
        
    # NaN 삭제
    dataframe.dropna(inplace=True)
    # open, high, low 값이 0이고 close 값이 0보다 클 경우, open, high, low 값을 close 값으로 세팅
    condition = (dataframe['open'] <= 1) & (dataframe['high']<= 1) & (dataframe['low'] <= 1) & (dataframe['close'] > 0)
    dataframe.loc[condition, ['open', 'high', 'low']] = dataframe.loc[condition, 'close']

    # 캔들
    df = dataframe.reset_index()
    #df.columns = ['time','open','high','low','close','volume']                  # rename columns
    #df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')                             # Date to string
    df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear
    df['time'] = df['time'].apply(string_datetime_to_timestamp)
    candles = json.loads(df.to_json(orient = "records"))
    
    # Create the initial seriesMultipaneChart
    seriesMultipaneChart = [get_candlestick_string(title="Main Chart", data=candles, pane=0)]

    # EMA 계산
    ema_params = indicators_params.get('ema', {})
    for ema_name, ema_param in ema_params.items():
        if ema_name:
            ema_length = ema_param.get('length')
            ema_color = ema_param.get('color')
            ema_linewidth = ema_param.get('linewidth')

            #df[f"EMA_{ema_length}"] = df['close'].rolling(window=ema_length).mean() # MA
            df[f"EMA_{ema_length}"] = df['close'].ewm(span=ema_length, adjust=False).mean() # EMA
            if f"EMA_{ema_length}" in df.columns:
                ema_data = dataToJSON(df, f"EMA_{ema_length}", ema_length, ema_color)  # 예시로 red 사용
                seriesMultipaneChart.append(get_series_line_string(title=f"EMA_{ema_length}", data=ema_data, color=ema_color, linewidth=ema_linewidth, pane=0))

    # # VWAP 계산
    # # VWAP 은 계산과정이 많아 df 가 아닌 dataframe 을 사용해서 계산
    # stock_indicators_options = {}
    # vwap_params = indicators_params.get('vwap', {})
    # for vwap_name, vwap_param in vwap_params.items():
    #     if vwap_name:
    #         vwap_dt = vwap_param.get('dt')
    #         vwap_mt1 = vwap_param.get('mt1')
    #         vwap_mt2 = vwap_param.get('mt2')
    #         vwap_color = vwap_param.get('color')

    #         if vwap_dt is not None:
    #             stock_indicators_options.update({
    #                 vwap_name: (calculate_vwap_bands, dataframe, vwap_dt, vwap_name, vwap_mt1, vwap_mt2, vwap_color),
    #             })
    # seriesMultipaneChart = set_vwap_indicators(series=seriesMultipaneChart, indicators=stock_indicators_options)

    # ZigZag
    #gap = (vwap_band_gap * 0.6) / 100
    gap = 0.01
    zigzag_data = zz.get_zigzag_threshold(dataframe, base_price='close', threshold=gap)

    zigzag_points = zigzag_data.copy()
    today = datetime.now(pytz.timezone('Asia/Seoul'))
    base_datetime = today.strftime("%Y-%m-%d 00:10:00") # 표준시로
    num, slopes, vwaps = vwc.get_near_2vwaps_current_position(zigzag_points=zigzag_points, input_data=dataframe, current_time=base_datetime)
    # st.text(slopes)
    # st.text(vwaps[0])
    # st.text(vwaps[1])
    vwap_high1_dataframe = vwaps[0]
    vwap_high2_dataframe = vwaps[1]

    #zigzag_data = zz.get_zigzag_lines(dataframe, base_price="close", window_size=10, std_threshold=0.01)
    zigzag_data['time'] = zigzag_data['time'].apply(string_datetime_to_timestamp)
    zigzag_data = zigzag_data.reset_index()
    #zigzag_data['time'] = zigzag_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    zigzag_line_data = convertDataToJSON(zigzag_data, "value")
    seriesMultipaneChart.append(get_series_line_string(title=f"ZIGZAG", data=zigzag_line_data, color="black", linewidth=2, pane=0))

    # VWAP
    if vwap_dataframe is not None and not vwap_dataframe.empty:
        vwap_dataframe = vwap_dataframe.reset_index()
        vwap_dataframe['time'] = vwap_dataframe['time'].apply(string_datetime_to_timestamp)

        vwap_vwap_df = vwap_dataframe[['time', 'vwap']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "vwap")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP", data=vwap_vwap_df, color="green", linewidth=2, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std1p']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std1p")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD1P", data=vwap_vwap_df, color="darkcyan", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std1m']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std1m")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD1M", data=vwap_vwap_df, color="darkcyan", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std2p']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std2p")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD2P", data=vwap_vwap_df, color="olive", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std2m']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std2m")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD2M", data=vwap_vwap_df, color="olive", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std3p']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std3p")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD3P", data=vwap_vwap_df, color="darkgreen", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std3m']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std3m")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD3M", data=vwap_vwap_df, color="darkgreen", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std4p']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std4p")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD4P", data=vwap_vwap_df, color="darkolivegreen", linewidth=1, pane=0))
        vwap_vwap_df = vwap_dataframe[['time', 'std4m']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "std4m")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_STD4M", data=vwap_vwap_df, color="darkolivegreen", linewidth=1, pane=0))

    # 고점 기준 VWAP
    if vwap_high1_dataframe is not None and not vwap_high1_dataframe.empty:
        vwap_high1_dataframe['time'] = vwap_high1_dataframe['time'].apply(string_datetime_to_timestamp)
        vwap_vwap_df = vwap_high1_dataframe[['time', 'vwap']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "vwap")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_HIGH1", data=vwap_vwap_df, color="blue", linewidth=2, pane=0))

    if vwap_high2_dataframe is not None and not vwap_high2_dataframe.empty:
        vwap_high2_dataframe['time'] = vwap_high2_dataframe['time'].apply(string_datetime_to_timestamp)
        vwap_vwap_df = vwap_high2_dataframe[['time', 'vwap']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "vwap")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_HIGH2", data=vwap_vwap_df, color="blue", linewidth=2, pane=0))

    # # 최고점 기준 VWAP
    # if vwap_highest_dataframe is not None and not vwap_highest_dataframe.empty:
    #     vwap_highest_dataframe['time'] = vwap_highest_dataframe['time'].apply(string_datetime_to_timestamp)
    #     vwap_vwap_df = vwap_highest_dataframe[['time', 'vwap']]
    #     vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "vwap")
    #     seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_HIGHEST", data=vwap_vwap_df, color="blue", linewidth=2, pane=0))


    # 직전저점 기준 일봉 VWAP
    if vwap_1day_dataframe is not None and not vwap_1day_dataframe.empty:
        vwap_1day_dataframe['time'] = vwap_1day_dataframe['time'].apply(string_datetime_to_timestamp)
        vwap_vwap_df = vwap_1day_dataframe[['time', 'vwap']]
        vwap_vwap_df = convertDataToJSON(vwap_vwap_df, "vwap")
        seriesMultipaneChart.append(get_series_line_string(title=f"VWAP_1D", data=vwap_vwap_df, color="red", linewidth=2, pane=0))

    # 볼린저밴드
    if bollinger_dataframe is not None and not bollinger_dataframe.empty:
        bollinger_dataframe = bollinger_dataframe.reset_index()
        bollinger_dataframe['time'] = bollinger_dataframe['time'].apply(string_datetime_to_timestamp)

        bl_df = bollinger_dataframe[['time', 'BL_std4p']]
        bl_df = convertDataToJSON(bl_df, "BL_std4p")
        seriesMultipaneChart.append(get_series_line_string(title=f"BL_STD4P", data=bl_df, color="darkblue", linewidth=2, pane=0))
        bl_df = bollinger_dataframe[['time', 'BL_std4m']]
        bl_df = convertDataToJSON(bl_df, "BL_std4m")
        seriesMultipaneChart.append(get_series_line_string(title=f"BL_STD4M", data=bl_df, color="darkblue", linewidth=2, pane=0))

    chartMultipaneOptions = [
        {
            "width": 0,
            "height": chart_height,
            "layout": {
                "background": {
                    "type": "solid",
                    "color": "white",
                },
                "textColor": "black"
            },
            "crosshair": {
                "mode": 0,  # 0-Normal, 1-Magnet, 2-Hidden
                "vertLine": { "visible": "false", "color": "white"},
                "horzLine": { "visible": "false", "color": "white" },
            },
            "grid": {
                "vertLines": {
                    "color": 'rgba(197, 203, 206, 0.4)',
                    "style": 3, # LineStyle: 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
                },
                "horzLines": {
                    "color": 'rgba(197, 203, 206, 0.4)',
                    "style": 3, # LineStyle: 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
                }
            },
            "rightPriceScale": {
                "borderColor": "rgba(197, 203, 206, 0.8)",
                "mode": 0, # PriceScaleMode: 0-Normal, 1-Logarithmic, 2-Percentage, 3-IndexedTo100
            },
            "timeScale": {
                "borderColor": "rgba(197, 203, 206, 0.8)",
                "barSpacing": 10,
                "timeVisible": True,
                "minBarSpacing": time_minspacing # Adjusted minBarSpacing for maximum zoom-out
            },
            "localization": {
                "dateFormat": "yyyy-MM-dd",
            }
        }
    ]

    # Render the multipane chart
    return renderLightweightCharts([{"chart": chartMultipaneOptions[0], "series": seriesMultipaneChart}], pane_name)

# Example usage:
#click_events_dy = get_stock_chart(symbol="010600.KS", period="3y", interval="1d", ema_A_length=33, ema_B_length=112, ema_C_length=224, pane_name="pane_daily", time_minspacing=5)