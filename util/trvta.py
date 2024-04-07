# https://python-tradingview-ta.readthedocs.io/en/latest/usage.html#instantiating-ta-handler
from tradingview_ta import *

def get_tradingview_ta(stock_codes, interval=Interval.INTERVAL_1_DAY):
    recommendation_list = []
    krx_symbols = ['KRX:' + symbol for symbol in stock_codes]
    if krx_symbols:
        analysis = get_multiple_analysis(screener="korea", interval=interval, symbols=krx_symbols)
        for symbol in analysis:
            if analysis[symbol]:
                recommendation = analysis[symbol].oscillators['RECOMMENDATION']
                recommendation = recommendation.replace('STRONG_BUY','강력매수').replace('STRONG_SELL','강력매도').replace('BUY','매수').replace('SELL','매도').replace('NEUTRAL','중립')
                indicator = f"+{analysis[symbol].oscillators['BUY']},-{analysis[symbol].oscillators['SELL']},({analysis[symbol].oscillators['NEUTRAL']})"
                recommendation_list.append({"code":symbol.split(':')[1], "recommendation":recommendation, "indicator":indicator})

    return recommendation_list
