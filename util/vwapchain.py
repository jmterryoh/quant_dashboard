import sqlite3
import numpy as np
import pandas as pd
from util import zigzag as zz

# 가장 최근의 zigzag 피크와 밸리 찾기
def find_recent_zigzag_points(zigzag_df, current_time):
    current_time = pd.to_datetime(current_time)
    zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])
    prior_zigzag_points = zigzag_df[zigzag_df['time'] < current_time].sort_values(by='time', ascending=False)
    recent_points = []
    for _, row in prior_zigzag_points.iterrows():
        if row['pivot'] == 'peak' or row['pivot'] == 'valley':
            recent_points.append(row)
        if len(recent_points) == 2:
            break
    return pd.DataFrame(recent_points)

# Anchored VWAP 계산 함수
def calculate_vwap(df, anchor_time, base_price):
    df = df.copy()
    df = df[df['time'] >= anchor_time]
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_vol_price'] = (df['volume'] * df[f"{base_price}"]).cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    return df

# 가격과 VWAP의 이격 계산 함수
def calculate_vwap_deviation(minute_data, anchor_time, end_time, base_price):
    # vwap 계산
    vwap_df = calculate_vwap(minute_data, anchor_time, base_price=base_price)

    # slope 계산
    temp_vwap_df = vwap_df[(vwap_df['time'] >= anchor_time) & (vwap_df['time'] <= end_time)].copy()
    temp_vwap_df.set_index('time', inplace=True)
    deviations = []

    zip_df = zip(temp_vwap_df.index, temp_vwap_df[f"{base_price}"])
    for time, price in zip_df:
        vwap_price = temp_vwap_df.loc[time, 'vwap']
        deviation = price - vwap_price
        deviations.append({"time": time, "deviation": deviation})
    deviations_df = pd.DataFrame(deviations)
    deviations_df['time'] = pd.to_datetime(deviations_df['time'])
    return deviations_df, vwap_df

# 기울기 계산 함수 (정규화 포함)
# Slope가 양수: 가격과 Anchored VWAP의 이격이 증가, 가격이 vwap과 멀어지고 있음
# Slope가 음수: 가격과 Anchored VWAP의 이격이 감소, 가격이 vwap과 가까워지고 있음
def calculate_slope(deviations):
    deviations['time_diff'] = (deviations['time'] - deviations['time'].min()).dt.total_seconds() / 60
    mean_deviation = deviations['deviation'].mean()
    std_deviation = deviations['deviation'].std()
    if std_deviation == 0:
        return 0
    deviations['normalized_deviation'] = (deviations['deviation'] - mean_deviation) / std_deviation
    
    try:
        slope = np.polyfit(deviations['time_diff'], deviations['normalized_deviation'], 1)[0]
    except np.linalg.LinAlgError:
        slope = 0
    return slope

def vwap_trend(df, end_time, window=5):
    # 최근 window 기간 동안의 VWAP 변화를 비교하여 추세 판단
    vwap_df = df[(df['time'] <= end_time)].copy()
    vwap_df['vwap_change'] = vwap_df['vwap'].diff()
    recent_changes = vwap_df['vwap_change'].tail(window)
    if recent_changes.mean() > 0:
        trend = 'uptrend'
    elif recent_changes.mean() < 0:
        trend = 'downtrend'
    else:
        trend = 'sideways'
    return trend

# 전체 프로세스 실행 함수
def get_near_2vwaps_current_position(zigzag_points, input_data, current_time):
    # Zigzag 정보 읽기
    #zigzag_points = zz.get_zigzag_threshold(minute_data, base_price="close", threshold=0.01)

    # 가장 최근의 zigzag 피크와 밸리 찾기
    recent_zigzag_points = find_recent_zigzag_points(zigzag_points, current_time)
    if recent_zigzag_points.empty or len(recent_zigzag_points) < 2:
        print("Not enough zigzag points found before the current time.")
        return pd.DataFrame()

    # 현재시점 종가
    currnet_close = 0
    try:
        current_close = input_data[input_data.index == current_time]['close']
        current_close = current_close.reset_index()
        current_close = current_close.iloc[0]['close']
    except Exception as e:
        if str(e) == 'single positional indexer is out-of-bounds':
            current_close = input_data.tail(1)
            current_close = current_close.reset_index()
            current_close = current_close.iloc[0]['close']

    # reset index
    minute_data = input_data.copy()
    minute_data.reset_index(inplace=True)
    minute_data['time'] = pd.to_datetime(minute_data['time'])

    # Anchored VWAP 계산 및 이격 계산
    num = 0
    vwap_list = []
    slope_results = []
    for index, row in recent_zigzag_points.iterrows():
        anchor_time = pd.to_datetime(row["time"])
        end_time = pd.to_datetime(current_time)
        pivot = row['pivot']

        deviation_df = pd.DataFrame()
        vwap_df = pd.DataFrame()
        if pivot == "peak":
            deviation_df, vwap_df = calculate_vwap_deviation(minute_data, anchor_time=anchor_time, end_time=end_time, base_price="close")
        elif pivot == "valley":
            deviation_df, vwap_df = calculate_vwap_deviation(minute_data, anchor_time=anchor_time, end_time=end_time, base_price="low")

        trend = vwap_trend(df=vwap_df, end_time=end_time, window=3)

        vwap_df['time'] = vwap_df['time'].astype(str)
        vwap = 0
        try:
            vwap = vwap_df[(vwap_df['time'] == current_time)]['vwap']
            vwap = vwap.reset_index()
            vwap = vwap.iloc[0]['vwap']
        except Exception as e:
            if str(e) == 'single positional indexer is out-of-bounds':
                vwap = vwap_df.tail(1)
                vwap = vwap.reset_index()
                vwap = vwap.iloc[0]['vwap']

        above_vwap = False
        if current_close > vwap:
            above_vwap = True

        slope = calculate_slope(deviation_df)
        slope_results.append({
            "anchor_time": row['time'],
            "pivot": pivot,
            "slope": slope,
            "current_close_greater_than_vwap": above_vwap,
            "vwap_trend": trend
        })
        num += 1

        vwap_list.append(vwap_df)

    # 결과를 DataFrame으로 반환 및 time 컬럼을 문자열로 변환
    slope_results = pd.DataFrame(slope_results)
    slope_results['anchor_time'] = slope_results['anchor_time'].astype(str)

    return num, slope_results, vwap_list