import pandas as pd
import pytz
from datetime import datetime, timedelta
from hts import TV_rest_api as tv


# Zig-Zag
def calculate_zigzag(base_prices):
    pivot_points = []
    current_peak = float('-inf')
    current_valley = float('inf')
    current_pivot_type = None

    for i in range(len(base_prices) - 1):
        current_close = base_prices[i]
        next_close = base_prices[i + 1]

        if current_pivot_type is None:
            current_peak = current_valley = current_close
            current_peak_i = current_valley_i = i
            current_pivot_type = 'peak' if current_close > next_close else 'valley'
        elif current_pivot_type == 'peak':
            if current_close >= current_peak:
                current_peak_i = i
                current_peak = current_close
            elif current_close < current_peak:# - thresholds[min(i, len(thresholds)-1)]:
                pivot_points.append((current_peak_i, current_peak, 'peak'))
                current_valley_i = i
                current_valley = current_close
                current_pivot_type = 'valley'
        elif current_pivot_type == 'valley':
            if current_close <= current_valley:
                current_valley_i = i
                current_valley = current_close
            elif current_close > current_valley:# + thresholds[min(i, len(thresholds)-1)]:
                pivot_points.append((current_valley_i, current_valley, 'valley'))
                current_peak_i = i
                current_peak = current_close
                current_pivot_type = 'peak'
    pivot_points.append((i, base_prices[i], current_pivot_type))

    return pivot_points


def get_zigzag(df, base_price="Close"):
    # Close values 
    data_df = df[f"{base_price}"].values
    dates = df.index

    zigzag_pivots = calculate_zigzag(base_prices=data_df)
    zigzag_lines_data = []
    for pivot in zigzag_pivots:
        zigzag_lines_data.append({"time": dates[pivot[0]], "value": pivot[1], "pivot": pivot[2]})

    # 마지막 pivot 값 추가
    # 마지막 time, value 값 추가
    zigzag_lines_data.append({"time":df.index[-1], "value":df.iloc[-1][f'{base_price}'], "pivot": "last"})

    return pd.DataFrame(zigzag_lines_data)


def get_zigzag_lines(dataframe, base_price="Close", window_size=10, std_threshold=0.01):

    # Close values 
    stock_prices = dataframe[f"{base_price}"].values
    dates = dataframe.index

    # Set the number of periods for calculating the standard deviation
    # Calculate the rolling standard deviation of close prices
    #rolling_std = np.std([stock_prices[i-window_size:i] for i in range(window_size, len(stock_prices))], axis=1)

    # Set threshold dynamically based on rolling standard deviation
    #thresholds = rolling_std * std_threshold

    # Calculate Zigzag pivot points
    #zigzag_pivots = calculate_zigzag(stock_prices, thresholds)
    zigzag_pivots = calculate_zigzag(stock_prices)

    zigzag_lines_data = []
    for pivot in zigzag_pivots:
        zigzag_lines_data.append({"time": dates[pivot[0]], "value": pivot[1]})
    # 마지막 pivot 값 추가
    #zigzag_lines_data.append({"time": dates[pivot[2]], "value": pivot[3]})
    # 마지막 time, value 값 추가
    zigzag_lines_data.append({"time":dataframe.index[-1], "value":dataframe.iloc[-1][f"{base_price}"]})

    return pd.DataFrame(zigzag_lines_data)


# 직전 valley 일자를 찾는 함수
def find_recent_valley_before_date(zigzag_df, input_date, days_before=30):
    # input_date 형식 변환
    target_date = datetime.strptime(str(input_date), '%Y%m%d')
    
    # # 입력된 날짜의 value 찾기
    # target_value = None
    # zigzag_df = zigzag_df.iloc[::-1]
    # for index, row in zigzag_df.iterrows():
    #     entry_date = row['time'].date()
    #     if entry_date == target_date and row['pivot'] == 'valley':
    #         target_value = row['value']
    #         break
    
    # # 해당 valley 값의 -1% 아래값
    # if target_value is None:
    #     return None    
    # target_value = int(target_value * 0.99)

    # 날짜 설정 (한국 시간대)
    one_month_ago = target_date - timedelta(days=days_before)
    zigzag_df = zigzag_df.iloc[::-1]

    # 입력된 날짜 이전의 가장 최근 valley의 날짜 찾기
    recent_valley_datetime = None
    recent_valley_value = None
    for index, row in zigzag_df.iterrows():
        entry_datetime = row['time']
        entry_value = row['value']
        #print(entry_date, row['pivot'], target_date)
        if one_month_ago <= entry_datetime < target_date and row['pivot'] == 'valley':# and row['value'] < target_value:
            recent_valley_datetime = entry_datetime
            recent_valley_value = entry_value
            break

    # 해당일에서 최소price 인 valley의 시간
    min_recent_valley_datetime = None
    if recent_valley_datetime:
        min_recent_valley_datetime = recent_valley_datetime
        min_recent_valley_value = recent_valley_value
        start_datetime = recent_valley_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        end_datetime = recent_valley_datetime.replace(hour=15, minute=30, second=0, microsecond=0)
        for index, row in zigzag_df.iterrows():
            entry_datetime = row['time']
            entry_value = row['value']
            if start_datetime <= entry_datetime <= end_datetime and row['pivot'] == 'valley':# and row['value'] < target_value:
                if entry_value < min_recent_valley_value:
                    min_recent_valley_datetime = entry_datetime
                    min_recent_valley_value = entry_value
    
    if min_recent_valley_datetime is None:
        return None

    return min_recent_valley_datetime.strftime('%Y%m%d%H%M%S')


def get_previous_valley_datetime(stock_code, stock_name, vdt, base_price="close", days_before=30):

    try:
        # vdt
        #vdt_date = pytz.timezone('Asia/Seoul').localize(datetime.strptime(vdt, "%Y%m%d"))
        price_1day_df = tv.get_tvdata_from_vdt_days_before(stock_code=stock_code, stock_name=stock_name, selected_minutes="5분", vdt=vdt, days_before=days_before)
        if price_1day_df is not None and not price_1day_df.empty:
            zigzag_df = get_zigzag(price_1day_df, base_price=base_price)
            if zigzag_df is not None and not zigzag_df.empty:
                previous_vdt = find_recent_valley_before_date(zigzag_df, vdt, days_before=days_before)
                if previous_vdt is None:
                    return vdt, price_1day_df
                return previous_vdt, price_1day_df

    except Exception as e:
        print(f"Error in get_previous_valley_date: {str(e)}")

    return None, None


def calculate_zigzag_threshold(base_prices, threshold=0.01, delay=1):
    pivot_points = []
    current_peak = float('-inf')
    current_valley = float('inf')
    current_pivot_type = None

    for i in range(len(base_prices) - delay):
        current_close = base_prices[i]
        next_closes = base_prices[i + 1:i + delay + 1]

        if current_pivot_type is None:
            current_peak = current_valley = current_close
            current_peak_i = current_valley_i = i
            current_pivot_type = 'peak' if current_close > next_closes[0] else 'valley'
        elif current_pivot_type == 'peak':
            if current_close >= current_peak:
                current_peak_i = i
                current_peak = current_close
            elif all(next_close < current_peak * (1 - threshold) for next_close in next_closes):
                pivot_points.append((current_peak_i, current_peak, 'peak'))
                current_valley_i = i + 1
                current_valley = next_closes[0]
                current_pivot_type = 'valley'
        elif current_pivot_type == 'valley':
            if current_close <= current_valley:
                current_valley_i = i
                current_valley = current_close
            elif all(next_close > current_valley * (1 + threshold) for next_close in next_closes):
                pivot_points.append((current_valley_i, current_valley, 'valley'))
                current_peak_i = i + 1
                current_peak = next_closes[0]
                current_pivot_type = 'peak'
    
    if current_pivot_type == 'peak':
        pivot_points.append((current_peak_i, current_peak, 'peak'))
    elif current_pivot_type == 'valley':
        pivot_points.append((current_valley_i, current_valley, 'valley'))
    
    return pivot_points

def get_zigzag_threshold(df, base_price="Close", threshold=0.01, delay=1):
    """
    Get Zig-Zag pivot points and return them in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the price data with a DateTime index.
        base_price (str): Column name for the base price.
        threshold (float): Minimum percentage change required to identify a new pivot.

    Returns:
        pd.DataFrame: DataFrame with Zig-Zag pivot points and their timestamps.
    """
    data_df = df[f"{base_price}"].values
    dates = df.index

    zigzag_pivots = calculate_zigzag_threshold(base_prices=data_df, threshold=threshold, delay=delay)
    zigzag_lines_data = []
    for pivot in zigzag_pivots:
        zigzag_lines_data.append({"time": dates[pivot[0]], "value": pivot[1], "pivot": pivot[2]})

    # Add the last pivot point manually if not already added
    if zigzag_lines_data[-1]["time"] != dates[-1]:
        zigzag_lines_data.append({"time": dates[-1], "value": df.iloc[-1][f"{base_price}"], "pivot": "last"})

    return pd.DataFrame(zigzag_lines_data)


# 가장 최근의 zigzag 피크와 밸리 찾기
def find_recent_zigzag_points(zigzag_df, start_time, end_time, prior_points_count=2):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])

    # Find the specified number of Zig-Zag points immediately before the start time
    prior_points = zigzag_df[(zigzag_df['time'] <= start_time) & (zigzag_df['pivot'] != 'last')].tail(prior_points_count)

    # Find the Zig-Zag points within the specified time range
    recent_points = zigzag_df[(zigzag_df['time'] > start_time) & (zigzag_df['time'] <= end_time)]
    
    # Combine the prior points and recent points
    combined_points = pd.concat([prior_points, recent_points]).reset_index(drop=True)

    return combined_points    


# zigzag로 Stage 1 ~ Stage 4 단계 단순판단 함수
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


def backtest_zigzag(df, base_price="Close", threshold=0.01, delay_values=range(1, 20)):
    results = []
    data_df = df[f"{base_price}"].values
    dates = df.index

    for delay in delay_values:
        zigzag_pivots = calculate_zigzag_threshold(base_prices=data_df, threshold=threshold, delay=delay)
        # Calculate performance based on zigzag_pivots (this is a simplified example)
        profit = 0
        for i in range(1, len(zigzag_pivots)):
            if zigzag_pivots[i-1][2] == 'valley' and zigzag_pivots[i][2] == 'peak':
                profit += zigzag_pivots[i][1] - zigzag_pivots[i-1][1]
            elif zigzag_pivots[i-1][2] == 'peak' and zigzag_pivots[i][2] == 'valley':
                profit += zigzag_pivots[i-1][1] - zigzag_pivots[i][1]
        results.append((delay, profit))
    
    best_delay = max(results, key=lambda x: x[1])[0]
    return best_delay, results