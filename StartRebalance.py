# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:17:52 2018

@author: marchims
"""

import importlib
import CryptoTrader
importlib.reload(CryptoTrader)
import pandas as pd

keys = pd.read_csv("/home/pi/Crypto/keys.csv",header=None).values

mybot = CryptoTrader.Rebalancer(keys,"ETH","/home/pi/Crypto/crypto/Portfolio balances.xlsx")

mybot.trading_enabled = False
mybot.total_fees = 0.021500653
mybot.rebalance()
mybot.stop()