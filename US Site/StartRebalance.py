# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:17:52 2018

@author: marchims
"""

print("Starting rebalance bot!")
import importlib
import CryptoTraderUS
importlib.reload(CryptoTraderUS)
import pandas as pd

keys = pd.read_csv("/home/pi/Crypto/config/keys_us.csv",header=None).values

mybot = CryptoTraderUS.Rebalancer(keys,"USD","/home/pi/Crypto/config/Portfolio balances US.xlsx")
#mybot.total_fees = 0.023766013627231998
mybot.trading_enabled = True

mybot.rebalance()