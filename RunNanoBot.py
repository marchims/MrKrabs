# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 06:08:35 2018

@author: marchims
"""

print("Starting nano bot!")
import importlib
import CryptoTrader
importlib.reload(CryptoTrader)
import pandas as pd

keys = pd.read_csv("/home/pi/Crypto/config/keys.csv",header=None).values


mybot = CryptoTrader.MrKrabs2(keys,'NANOBTC')
mybot.loadNetwork('/home/pi/Crypto/MrKrabs/Network Config/BuyModel_V20.xlsx','buy',4)
mybot.loadNetwork('/home/pi/Crypto/MrKrabs/Network Config/SellModel_V20.xlsx','sell',4)
mybot.run()
#mybot.enable_trading()
#mybot.getValue(0.19392632+0.08,577.9499342)