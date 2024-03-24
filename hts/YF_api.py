import yfinance as yf

def fetch_stock_data(symbol, period, interval):
    # Fetch historical data using yfinance
    # Request historic pricing data via finance.yahoo.com API
    return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
    
