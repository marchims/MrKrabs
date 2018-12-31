# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:32:26 2018

@author: marchims
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:38:12 2018

@author: marchims
"""

import numpy as np
from binance.client import Client
from binance.websockets import BinanceSocketManager
import matplotlib.pyplot as plt
import time

class MrKrabs2:
    
    buffer_time = 1000 #seconds
    ema_len = 1/60
    
    def __init__(self,pair):
        self.client = Client("a9jZULWZPvoYrwVQC5R2oJBlALi9SwWbjxOSB6otHVGgbLd37M0JIzFmCj8vxGX7","wg3ARO6AhBUO1I3fUHoQMMLo2bAWHTHbrwQ4E5mvZRDMrElp4NQUxVBgYFDYYDUa")
        self.pair = pair
        # start any sockets here, i.e a trade socket
        self.trade_buffer = np.zeros((0))
        self.ask_buffer = np.zeros((0))
        self.bid_buffer = np.zeros((0))
        self.time_buffer = np.zeros((0))
        self.time_buffer_depth = np.zeros((0))
        self.filt_ema_buffer = np.zeros((0))
        self.ask_buffer_ema = np.zeros((0))
        self.bid_buffer_ema = np.zeros((0))
        self.prev_sample = np.nan
        self.data = np.zeros((0,8))
    
    def update_trades(self,msg):
        if msg['e'] == 'trade':
            self.time_buffer = np.append(self.time_buffer,np.array(msg['T']/1000).reshape((1)),axis=0)
            self.trade_buffer = np.append(self.trade_buffer,np.array(float(msg['p'])).reshape((1)),axis=0)
            if self.time_buffer.shape[0] >= 2:
                dt = self.time_buffer[-1] - self.time_buffer[-2]
                if dt == 0:
                    dt = 0.01
                a = dt / (1/self.ema_len)
                u = np.exp(-a)
                v = (1.0-u)/a
                next_ema = u*self.filt_ema_buffer[-1] + (v-u)*self.prev_sample + (1.0-v)*self.trade_buffer[-1]
                self.filt_ema_buffer = np.append(self.filt_ema_buffer,next_ema.reshape((1)),axis=0)
            else:
                self.filt_ema_buffer = np.zeros((1))
                self.filt_ema_buffer[0] = self.trade_buffer[-1] 
                
            valid = self.time_buffer >= self.time_buffer[-1] - self.buffer_time
            self.prev_sample = self.trade_buffer[-1]
            self.time_buffer = self.time_buffer[valid]
            self.trade_buffer = self.trade_buffer[valid]
            self.filt_ema_buffer = self.filt_ema_buffer[valid]
            
        elif msg['e'] == '24hrTicker':
            self.time_buffer_depth = np.append(self.time_buffer_depth,np.array(msg['E']/1000).reshape((1)),axis=0)
            self.bid_buffer = np.append(self.bid_buffer,np.array(float(msg['b'])).reshape((1)),axis=0)
            self.ask_buffer = np.append(self.ask_buffer,np.array(float(msg['a'])).reshape((1)),axis=0)
            if self.time_buffer_depth.shape[0] >= 2:
                dt = self.time_buffer_depth[-1] - self.time_buffer_depth[-2]
                if dt == 0:
                    dt = 0.01
                a = dt / (1/self.ema_len)
                u = np.exp(-a)
                v = (1.0-u)/a
                next_ema = u*self.ask_buffer_ema[-1] + (v-u)*self.ask_buffer[-2] + (1.0-v)*self.ask_buffer[-1]
                self.ask_buffer_ema = np.append(self.ask_buffer_ema,next_ema.reshape((1)),axis=0)
                
                next_ema = u*self.bid_buffer_ema[-1] + (v-u)*self.bid_buffer[-2] + (1.0-v)*self.bid_buffer[-1]
                self.bid_buffer_ema = np.append(self.bid_buffer_ema,next_ema.reshape((1)),axis=0)
                
            else:
                self.ask_buffer_ema = np.zeros((1))
                self.ask_buffer_ema[0] = self.ask_buffer[-1]
                self.bid_buffer_ema = np.zeros((1))
                self.bid_buffer_ema[0] = self.bid_buffer[-1]
            

            bid_qty = np.array(float(msg['B'])).reshape((1,1))
            ask_qty = np.array(float(msg['A'])).reshape((1,1))
            if self.filt_ema_buffer.shape[0] > 1:
                self.data = np.append(self.data,np.concatenate((self.time_buffer_depth[-1].reshape((1,1)),self.filt_ema_buffer[-1].reshape((1,1)),
                                                                self.bid_buffer[-1].reshape((1,1)),self.ask_buffer[-1].reshape((1,1)),
                                                                self.bid_buffer_ema[-1].reshape((1,1)),self.ask_buffer_ema[-1].reshape((1,1)),bid_qty,ask_qty),axis=1),axis=0)
                
            valid = self.time_buffer_depth >= self.time_buffer_depth[-1] - self.buffer_time
            self.time_buffer_depth = self.time_buffer_depth[valid]
            self.bid_buffer = self.bid_buffer[valid]
            self.ask_buffer = self.ask_buffer[valid]
            self.bid_buffer_ema = self.bid_buffer_ema[valid]
            self.ask_buffer_ema = self.ask_buffer_ema[valid]
            
            plt.gcf().clear()
            plt.plot(self.time_buffer,self.trade_buffer,'bx')
            plt.plot(self.time_buffer,self.filt_ema_buffer,'k-')
            plt.plot(self.time_buffer_depth,self.bid_buffer,'g.',self.time_buffer_depth,self.ask_buffer,'r.')
            plt.plot(self.time_buffer_depth,self.bid_buffer_ema,'g--',self.time_buffer_depth,self.ask_buffer_ema,'r--')
            #plt.pause(0.05)
            plt.draw()
        elif msg['e'] == 'error':
            print(msg)
            self.stop()
        
        if self.data.shape[0] >= 1000000:
            self.stop()
        
    
    def run(self,duration=np.inf):
        self.socket_manager = BinanceSocketManager(self.client)
        self.socket_key = self.socket_manager.start_symbol_ticker_socket(self.pair, lambda x: self.update_trades(x))
        self.socket_key_trade = self.socket_manager.start_trade_socket(self.pair, lambda x: self.update_trades(x))
        self.fig = plt.figure(1)
        plt.ion()
        # then start the socket manager
        
        self.socket_manager.start()
    
    def stop(self):
        self.socket_manager.stop_socket(self.socket_key)
        self.socket_key = None
        self.socket_manager.stop_socket(self.socket_key_trade)
        self.socket_key_trade = None
        self.socket_manager.close()
        self.socket_manager = None