# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:17:52 2018

@author: marchims
"""

print("Starting rebalance bot!")
import importlib
import CryptoTrader
importlib.reload(CryptoTrader)
import pandas as pd

keys = pd.read_csv("/home/pi/Crypto/config/keys.csv",header=None).values

mybot = CryptoTrader.Rebalancer(keys,"ETH","/home/pi/Crypto/config/Portfolio balances.xlsx")
#mybot.total_fees = 0.023766013627231998
mybot.trading_enabled = True

mybot.rebalance()