
def format_price_without_commas(value):
    return '{:.0f}'.format(float(value))
def format_price_with_commas(value):
    return '{:,.0f}'.format(float(value))
def format_stotprice_with_commas(value):
    return '{:,.0f}억'.format(float(value))
