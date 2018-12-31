# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:47:38 2018

@author: marchims
"""

#import numpy as np
import matplotlib.pyplot as plt
import datetime
#from twisted.internet import reactor
from binance.websockets import BinanceSocketManager
from binance.client import Client

def process_trades(msg):
    if msg['e'] == 'error':
        # close and restart the socket
        print('There was an error and the socket will be closed')
        #bm.stop_socket(conn_key)
        #bm.close()
        #reactor.stop()
    else:
        # process message normally
        trade_date.append(datetime.datetime.fromtimestamp(msg['T']/1000.0))
        trade_price.append(float(msg['p']))
        ax.plot(trade_date,trade_price,'b-')
        fig.canvas.draw()


client = Client("a9jZULWZPvoYrwVQC5R2oJBlALi9SwWbjxOSB6otHVGgbLd37M0JIzFmCj8vxGX7","wg3ARO6AhBUO1I3fUHoQMMLo2bAWHTHbrwQ4E5mvZRDMrElp4NQUxVBgYFDYYDUa")
bm = BinanceSocketManager(client)
# start any sockets here, i.e a trade socket
diff_key = bm.start_aggtrade_socket('NANOBTC', process_trades)

trade_date = []
trade_price = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# then start the socket manager
bm.start()