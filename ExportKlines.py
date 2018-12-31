# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:03:59 2018

@author: marchims
"""
import numpy as np
import pandas as pd
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
from twisted.internet import reactor
import matplotlib.pyplot as plt
import time
import pdb
from threading import Timer
import importlib
import CryptoTrader

importlib.reload(CryptoTrader)
mybot = CryptoTrader.Rebalancer("ETH","C:\\Users\\marchims\\Documents\\Portfolio balances.xlsx")

kline_data = {}
for coin in mybot.coin_data:
    if 'market_name' in mybot.coin_data[coin]:
        for market in mybot.coin_data[coin]['market_name']:
            print('Getting data for market {}...'.format(market))
            kline_data[market] = mybot.client.get_historical_klines(market, Client.KLINE_INTERVAL_1MINUTE, '{} days ago UTC'.format(30))
            kline_data[market]= np.array(kline_data[market])
            kline_data[market] = kline_data[market].astype('float')
            print('Writing data to csv...')
            toWrite = pd.DataFrame(data = kline_data[market])
            toWrite.to_csv('{}.csv'.format(market),header = False,index = False)
            time.sleep(60)