import pandas as pd

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


def get_zigzag(df):
    # Close values 
    data_df = df['close'].values
    dates = df.index

    zigzag_pivots = calculate_zigzag(base_prices=data_df)
    zigzag_lines_data = []
    for pivot in zigzag_pivots:
        zigzag_lines_data.append({"time": dates[pivot[0]], "value": pivot[1], "pivot": pivot[2]})

    # 마지막 pivot 값 추가
    # 마지막 time, value 값 추가
    zigzag_lines_data.append({"time":df.index[-1], "value":df.iloc[-1]['close'], "pivot": "last"})

    return pd.DataFrame(zigzag_lines_data)


def get_zigzag_lines(dataframe, window_size=10, std_threshold=0.01):

    # Close values 
    stock_prices = dataframe['close'].values
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
    zigzag_lines_data.append({"time":dataframe.index[-1], "value":dataframe.iloc[-1]['close']})

    return pd.DataFrame(zigzag_lines_data)


# Input : df = Dataframe(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
#         anchor_date = anchord datetime
#         multiplier1 = vwap band 1
#         multiplier2 = vwap band 2 
# Output : Dataframe(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'STD+1', 'STD-1', 'STD+2', 'STD-2'])
def calculate_vwap_bands(df, anchor_date, vwap_name, multiplier1, multiplier2, vwap_color):    
    # Use .loc[] to avoid setting values on a slice of DataFrame
    df1 = df.loc[df.index >= anchor_date].copy()

    # VWAP Formula : ACC (Close * Volume) / ACC (Volume)
    data = pd.DataFrame({
        'close': df1['close'],  # Close value base
        'volume': df1['volume'],
        'TP': (df1['close'] * df1['volume']),
        vwap_name: (df1['close'] * df1['volume']).cumsum() / df1['volume'].cumsum()
    })
    df1[vwap_name] = data[vwap_name]

    if multiplier1 is not None or multiplier2 is not None:
        # data['TPP'] = (df1['Close'] * df1['Close'] * df1['Volume']).cumsum() / df1['Volume'].cumsum()
        # STD Formula : sqrt( (ACC(Close X Close * Volume) / ACC(Volume)) - VWAP * VWAP )
        # multiple 1: 1.28, mutiple 2: 2.01, multiple 3: 2.51
        data['TPP'] = (data['close'] * data['TP']).cumsum() / data['volume'].cumsum() # Volume!! 
        data['VW2'] = data[vwap_name] * data[vwap_name]
        data['STD'] = (data['TPP'] - data['VW2']) ** 0.5
        if multiplier1 is not None:
            df1.loc[:, 'STD+1'] = data[vwap_name] + data['STD'] * multiplier1
            df1.loc[:, 'STD-1'] = data[vwap_name] - data['STD'] * multiplier1
        if multiplier2 is not None:
            df1.loc[:, 'STD+2'] = data[vwap_name] + data['STD'] * multiplier2
            df1.loc[:, 'STD-2'] = data[vwap_name] - data['STD'] * multiplier2

    return df1, vwap_color