import sqlite3
import numpy as np
import pandas as pd
from util import zigzag as zz


# 가장 최근의 zigzag 피크와 밸리 찾기
def find_recent_zigzag_2points(zigzag_df, current_time):
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

# 가장 최근의 peak, valley 를 찾고 vwap 을 계산하는 메인 함수
def get_near_2vwaps_current_position(zigzag_points, input_data, current_time):
    # Zigzag 정보 읽기
    #zigzag_points = zz.get_zigzag_threshold(minute_data, base_price="close", threshold=0.01)

    # 가장 최근의 zigzag 피크와 밸리 찾기
    recent_zigzag_points = find_recent_zigzag_2points(zigzag_points, current_time)
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

# 가장 최근의 peak, valley 를 찾고 vwap 을 계산하는 메인 함수
def get_day_open_2vwaps(zigzag_points, input_data, open_time, current_time):

    peak_datetime = ""
    valley_datetime = ""

    open_time = pd.to_datetime(open_time)

    # 가장 최근의 zigzag 피크와 밸리 찾기
    recent_zigzag_points = find_recent_zigzag_2points(zigzag_points, current_time)
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
            # 장시초일시보다 이전일 경우, 장시초로 설정
            if anchor_time < open_time:
                anchor_time = open_time
            deviation_df, vwap_df = calculate_vwap_deviation(minute_data, anchor_time=anchor_time, end_time=end_time, base_price="close")
            peak_datetime = anchor_time
        elif pivot == "valley":
            # 장시초일시보다 이전일 경우, 장시초로 설정
            if anchor_time < open_time:
                anchor_time = open_time
            deviation_df, vwap_df = calculate_vwap_deviation(minute_data, anchor_time=anchor_time, end_time=end_time, base_price="low")
            valley_datetime = anchor_time

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

    peak_datetime = peak_datetime.strftime("%Y-%m-%d %H:%M:%S")
    valley_datetime = valley_datetime.strftime("%Y-%m-%d %H:%M:%S")

    return num, slope_results, vwap_list, peak_datetime, valley_datetime


# 가장 최근의 zigzag 피크와 밸리 찾기
def find_recent_zigzag_points(zigzag_df, start_time, end_time):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])
    recent_points = zigzag_df[(zigzag_df['time'] >= start_time) & (zigzag_df['time'] <= end_time)]
    return recent_points

# 정규화된 VWAP 기울기 계산 함수
def calculate_normalized_vwap_slope(vwap_df):
    # Use only the last 5 entries
    vwap_len = len(vwap_df)
    if vwap_len > 5:
        vwap_len = 5
    recent_vwap_df = vwap_df.iloc[vwap_len * (-1):]
    
    # Calculate time difference in minutes
    recent_vwap_df['time_diff'] = (recent_vwap_df.index - recent_vwap_df.index.min()).total_seconds() / 60
    
    # Normalize VWAP values
    mean_vwap = recent_vwap_df['vwap'].mean()
    std_vwap = recent_vwap_df['vwap'].std()
    recent_vwap_df['normalized_vwap'] = (recent_vwap_df['vwap'] - mean_vwap) / std_vwap
    
    # Fit a linear model to the normalized VWAP values
    slope = np.polyfit(recent_vwap_df['time_diff'], recent_vwap_df['normalized_vwap'], 1)[0]
    return slope

# VWAP 터치 조건 판단 함수
def check_vwap_touch(valley_time, minute_data, vwap_df, base_price, tolerance=0.005): # 0.3%
    vwap_price = vwap_df.loc[valley_time, 'vwap']
    valley_price = minute_data.loc[valley_time, 'close']
    # if valley_price > vwap_price:
    #     return True
    return abs(valley_price - vwap_price) / vwap_price <= tolerance #and valley_price >= vwap_price

# peak_time 이후 VWAP 값보다 close 값이 큰 최초의 시간을 찾는 함수
def find_first_above_vwap_time(minute_data, vwap_df, start_time, base_price):
    for time in vwap_df.index:
        if time >= start_time and minute_data.loc[time, f'{base_price}'] > vwap_df.loc[time, 'vwap']:
            return time
    return None

# 가장 가까운 이전 시간을 찾는 함수
def find_nearest_index(dataframe, target_time):
    target_time = pd.to_datetime(target_time)
    nearest_time = dataframe.index.asof(target_time)
    return nearest_time

# 단계 판단 함수
def determine_market_phase(recent_zigzag_points):
    if recent_zigzag_points.empty:
        return "Not enough data"

    recent_zigzag_points = recent_zigzag_points.sort_values(by='time')
    
    if len(recent_zigzag_points) < 6:
        return "Not enough zigzag points"

    last_points = recent_zigzag_points.iloc[-6:]

    # Consolidation 판단
    price_range = last_points['value'].max() - last_points['value'].min()
    if price_range / last_points['value'].mean() < 0.05:  # 5% 범위 내에서 횡보하면 Consolidation으로 판단
        return "Consolidation"

    # 최근 6개의 포인트 분석
    if all(last_points.iloc[i]['pivot'] == 'valley' for i in [0, 2, 4]) and all(last_points.iloc[i]['pivot'] == 'peak' for i in [1, 3, 5]):
        if last_points.iloc[4]['value'] > last_points.iloc[0]['value'] and last_points.iloc[5]['value'] > last_points.iloc[1]['value']:
            return "Accumulation"
        else:
            return "Decline"
    elif all(last_points.iloc[i]['pivot'] == 'peak' for i in [0, 2, 4]) and all(last_points.iloc[i]['pivot'] == 'valley' for i in [1, 3, 5]):
        if last_points.iloc[4]['value'] > last_points.iloc[0]['value'] and last_points.iloc[5]['value'] > last_points.iloc[1]['value']:
            return "Markup"
        else:
            return "Distribution"
    else:
        return "Not enough data"

# peak부터 시작한 VWAP의 기울기가 정규화된 상태에서 상승이고, peak 이후 valley 값이 상승하며, 최근 valley 값이 VWAP을 위에서 아래로 터치한 경우
def find_vwap_support_points(zigzag_points, input_data, peak_time, current_time):

    base_price = 'close'

    # 분봉데이터 추출
    minute_data = input_data[input_data.index >= peak_time].copy()
    minute_data.reset_index(inplace=True)
    minute_data['time'] = pd.to_datetime(minute_data['time'])

    anchor_time = pd.to_datetime(peak_time)
    
    # slope 계산
    vwap_df = calculate_vwap(minute_data, anchor_time, base_price=base_price)

    # index to time
    vwap_df.set_index('time', inplace=True)
    minute_data.set_index('time', inplace=True)

    # 최근 zigzag 피크와 밸리 찾기
    subsequent_zigzag_points = find_recent_zigzag_points(zigzag_points, peak_time, current_time)

    #  vwap 값 위치하는 시점 최초 찾기
    first_valley_time = None
    if not subsequent_zigzag_points[subsequent_zigzag_points['pivot'] == 'valley'].empty:
        first_valley_time = subsequent_zigzag_points[subsequent_zigzag_points['pivot'] == 'valley'].iloc[0]['time']
    
    first_above_vwap_time = None
    if first_valley_time:
        first_above_vwap_time = find_first_above_vwap_time(minute_data=minute_data, vwap_df=vwap_df, start_time=first_valley_time, base_price=base_price)

    results = []
    valid_valleys = subsequent_zigzag_points[(subsequent_zigzag_points['pivot'] == 'valley')]
    for _, row in valid_valleys.iterrows():
        if row['pivot'] == 'valley':
            peak_valley_time = row['time']
            if check_vwap_touch(peak_valley_time, minute_data, vwap_df, base_price):

                temp_vwap_df = vwap_df[vwap_df.index <= peak_valley_time].copy()
                vwap_slope = calculate_normalized_vwap_slope(temp_vwap_df)

                results.append({
                    "peak_time": peak_time,
                    "valley_time": peak_valley_time,
                    "valley_value": row['value'],
                    "vwap_slope": vwap_slope,
                    "touch_condition_met": True
                })

    vwap_df = vwap_df.reset_index()
    vwap_df['time'] = vwap_df['time'].astype(str)

    results_df = pd.DataFrame(results)
    return vwap_df, results_df, first_above_vwap_time
