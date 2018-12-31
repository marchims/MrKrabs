# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 01:42:58 2018

@author: marchims
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from binance.client import Client
import xlsxwriter


client = Client("a9jZULWZPvoYrwVQC5R2oJBlALi9SwWbjxOSB6otHVGgbLd37M0JIzFmCj8vxGX7","wg3ARO6AhBUO1I3fUHoQMMLo2bAWHTHbrwQ4E5mvZRDMrElp4NQUxVBgYFDYYDUa")
pair = 'NANOBTC'

def read_depth():
    depth = client.get_order_book(symbol=pair)
    
    price_bid = []
    amt_bid = []
    date = [datetime.datetime.now()]
    for item in depth['bids']:
        price_bid.append(float(item[0]))
        amt_bid.append(float(item[1]))
    
    price_ask = []
    amt_ask = []
    for item in depth['asks']:
        price_ask.append(float(item[0]))
        amt_ask.append(float(item[1]))
        
    return date+price_bid+amt_bid+price_ask+amt_ask


def read_depth_slope():
    depth = client.get_order_book(symbol=pair)
    
    price_bid = np.zeros((len(depth['bids']),1))
    amt_bid = np.zeros((len(depth['bids']),1))
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    for idx,item in enumerate(depth['bids']):
        price_bid[idx] = float(item[0])
        amt_bid[idx] = float(item[1])
    
    price_ask = np.zeros((len(depth['asks']),1))
    amt_ask = np.zeros((len(depth['asks']),1))
    for idx,item in enumerate(depth['asks']):
        price_ask[idx] = float(item[0])
        amt_ask[idx] = float(item[1])
    
    sum_bid = np.sum(amt_bid,axis=0)
    sum_ask = np.sum(amt_ask,axis=0)

    slope_bid = -(sum_bid - amt_bid[0])/((price_bid[-1]-price_bid[0])/price_bid[0]*100)
    slope_ask = (sum_ask - amt_ask[0])/((price_ask[-1]-price_ask[0])/price_ask[0]*100)
    
    return [date, slope_bid, slope_ask, sum_bid, sum_ask, price_bid[0], price_ask[0], price_bid[-1], price_ask[-1]]

def read_hist_data():
    klines = client.get_historical_klines("NANOBTC", Client.KLINE_INTERVAL_1MINUTE, "8 hours ago UTC")
    for item in klines:
        item[0] = datetime.datetime.fromtimestamp(item[0]/1000.0).strftime("%Y-%m-%d %H:%M:%S.%f")
        item[1] = float(item[1])
        item[2] = float(item[2])
        item[3] = float(item[3])
        item[4] = float(item[4])
        item[5] = float(item[5])
        item[6] = datetime.datetime.fromtimestamp(item[6]/1000.0).strftime("%Y-%m-%d %H:%M:%S.%f")
        item[7] = float(item[7])
        item[8] = float(item[8])
        item[9] = float(item[9])
        item[10] = float(item[10])
    return klines
  
'''
data = []
#12960
for i in range(120*36):
    data.append(read_depth_slope())
    time.sleep(30.0)

workbook = xlsxwriter.Workbook('NANO_13.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write_row(0, 0, ['Date','Bid Slope','Ask Slope','Buy Vol','Ask Vol','High Bid','Low Ask','Low Bid','High Ask'])
for row, row_data in enumerate(data):
    worksheet.write_row(row+1, 0, row_data)

workbook.close()
'''

kdata = read_hist_data()

workbook = xlsxwriter.Workbook('NANO_candles_TestSet.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write_row(0, 0, ['Open Date','Open','High','Low','Close','Volume','Close Date','Quote Asset Volume','Number of Trades'])
for row, row_data in enumerate(kdata):
    worksheet.write_row(row+1, 0, row_data)

workbook.close()
