# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:32:53 2018

@author: marchims
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.dates as plt_date
import datetime

def plotDepth(client,pair='NANOBTC'):
    depth = client.get_order_book(symbol=pair)
    
    price_bid = []
    amt_bid = []
    for item in depth['bids']:
        price_bid.append(float(item[0]))
        amt_bid.append(float(item[1]))
    
    price_ask = []
    amt_ask = []
    for item in depth['asks']:
        price_ask.append(float(item[0]))
        amt_ask.append(float(item[1]))
    plt.plot(price_bid,np.cumsum(amt_bid),'g-',price_ask,np.cumsum(amt_ask),'r-')

    plt.xlabel('Price')
    plt.ylabel('Amount')
    plt.title('Order Book for {0}'.format(pair))
    
    plt.show()
    
def plotTrades(client,pair='NANOBTC'):

    trades = client.get_historical_trades(symbol=pair)
    
    price = []
    amt = []
    time = []
    for item in trades:
        price.append(float(item['price']))
        amt.append(float(item['qty']))
        time.append(datetime.datetime.fromtimestamp(item['time']/1000.0).strftime("%Y-%m-%d %H:%M:%S.%f"))
    
    plt.plot(time,price,'g-')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Trade History for {0}'.format(pair))
    
    plt.show()