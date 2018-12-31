# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:38:12 2018

@author: marchims
"""

import numpy as np
from binance.enums import *
import datetime
from binance.client import Client
import xlsxwriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import operator
import DataFormatting
import time

class MrKrabs:
    
    windows = [3*1,6*1,12*1,24]
    max_perc_trade_base = 0.25
    qsum_decay_rate = 0.97
    ki = 0.20
    fee = 0.1/100
    cooldown = 30
    factor = 0.9
    
    def __init__(self,weight,bias,min_x,max_x,pair):
        self.W = weight
        self.b = bias
        self.min_x = min_x
        self.max_x = max_x
        self.client = Client("a9jZULWZPvoYrwVQC5R2oJBlALi9SwWbjxOSB6otHVGgbLd37M0JIzFmCj8vxGX7","wg3ARO6AhBUO1I3fUHoQMMLo2bAWHTHbrwQ4E5mvZRDMrElp4NQUxVBgYFDYYDUa")
        self.get_pair_info(pair)
        self.last_delta = np.nan
        self.last_trade_price = np.nan
        self.last_cross_price = np.nan
        self.last_trade_time = 0
        self.last_trade_amt = 0
        self.last_guidance = np.nan
        self.qsum = 0
        self.num_trades = 0
        self.num_correct = 0
        self.trading_enabled = False
        self.hist_time = []
        self.hist_price = []
        self.hist_amt_base = []
        self.max_perc_trade = self.max_perc_trade_base
    
    
    def enable_trading(self):
        self.trading_enabled = True
    
    def disable_trading(self):
        self.trading_enabled = False
    
    def get_pair_info(self,pair):
        self.pair = pair
        self.base_coin = pair[-3:]
        self.trade_coin = pair[:-3]
        info = self.client.get_symbol_info(self.pair)
        self.coin_precision = float(info['filters'][1]['stepSize'])
        self.base_precision = float(info['filters'][2]['minNotional'])
        
    
    def update_wallet(self):
        temp = self.client.get_asset_balance(asset=self.base_coin)
        self.base_amt = float(temp['free'])
        temp = self.client.get_asset_balance(asset=self.trade_coin)
        self.coin_amt = float(temp['free'])
    
    def run(self,duration=np.inf):
        
        plt.figure(1)
        plt.ion()
        '''
        plt.figure(2)
        plt.ion()
        '''
        self.update_wallet()
        checkpoint_last_cross = np.nan
        start_amt_base = self.base_amt
        start_amt_coin = self.coin_amt
        filt_model = np.nan
        last_state = np.nan
        while True:
            kline = self.read_candle_data('{} minutes ago UTC'.format(np.max(self.windows)+21))
            
            isCross = self.check_cross(kline)
            
            model_out,raw = self.get_guidance(kline)
            confidence = max(0,(1.0/0.56)*(np.max(raw[0,:])-0.56))
            '''
            trades = self.client.get_recent_trades(symbol=self.pair)
            trade_time = trades[-1]['time']/1000
            trade_price = float(trades[-1]['price'])
            '''
            
            '''
            plt.figure(2)
            ax2 = plt.subplot(211)
            plt.plot(trade_time,price_bid,'b.',trade_time,price_ask,'b.')
            plt.subplot(212,sharex=ax2)
            plt.plot(trade_time,raw[0,0],'g.',trade_time,raw[0,1],'r.')
            plt.pause(0.05)
            plt.draw()
            '''
            '''
            if np.isnan(filt_model):
                filt_model = model_out
                last_state = model_out
            else:
                filt_model = filt_model*self.factor + (1.0-self.factor)*model_out

            if filt_model >= 0.6 and last_state == 0:
                isCross = True
                last_state = 1
                print('Inflection')
            elif filt_model <= 0.4 and last_state == 1:
                isCross = True
                last_state = 0
                print('Inflection')
            '''
            if isCross:
                self.max_perc_trade = self.max_perc_trade_base * confidence
                trade_time,price_bid,price_ask = self.get_depth()
                if trade_time - self.last_trade_time >= self.cooldown:
                    value = self.base_amt + self.coin_amt*price_bid
                    successFlag = True
                    
                    if model_out == 0:
                        # buy
                        s = 'Bought'
                        trade_price = price_ask
                        trade_amt_base = max(self.base_precision,min(self.base_amt,self.ki*self.qsum+(1)*self.max_perc_trade*value))
                        trade_amt_coin = float(math.floor(trade_amt_base/trade_price/self.coin_precision))*self.coin_precision
                        trade_amt_base = trade_amt_coin*trade_price
                        if self.trading_enabled:
                            None
                        else:
                            if self.base_amt >= trade_amt_base and trade_amt_base >= self.base_precision:
                                self.base_amt -= trade_amt_base
                                self.coin_amt += (1.0-self.fee)*trade_amt_coin
                                self.qsum -= trade_amt_base
                                self.hist_time.append(trade_time)
                                self.hist_price.append(trade_price)
                                self.hist_amt_base.append(trade_amt_base)
                            else:
                                print('Cannot buy {} {}!'.format(trade_amt_coin,self.trade_coin))
                                successFlag = False
                    elif model_out == 1:
                        # sell
                        s = 'Sold'
                        trade_price = price_bid
                        trade_amt_base = max(self.base_precision,min(self.coin_amt*trade_price,-self.ki*self.qsum+(1)*self.max_perc_trade*value))
                        trade_amt_coin = float(math.floor(trade_amt_base/trade_price/self.coin_precision))*self.coin_precision
                        trade_amt_base = trade_amt_coin*trade_price
                        if self.trading_enabled:
                            None
                        else:
                            if self.coin_amt >= trade_amt_coin and trade_amt_base >= self.base_precision:
                                self.base_amt += (1.0-self.fee)*trade_amt_base
                                self.coin_amt -= trade_amt_coin
                                self.qsum += trade_amt_base
                                self.hist_time.append(trade_time)
                                self.hist_price.append(trade_price)
                                self.hist_amt_base.append(trade_amt_base)
                            else:
                                print('Cannot sell {} {}!'.format(trade_amt_coin,self.trade_coin))
                                successFlag = False
                    else:
                        s = 'Held'
                        trade_price = (price_bid + price_ask)/2
                        trade_amt_coin = 0.0
                        trade_amt_base = 0.0
                    self.last_cross_price = trade_price
                    if np.isnan(self.last_guidance):
                        None
                        acc = 0.0
                    else:
                        #print('{} {}'.format(self.last_cross_price,checkpoint_last_cross))
                        self.num_trades += 1
                        if self.last_cross_price >= checkpoint_last_cross and self.last_guidance == 0:
                            self.num_correct += 1
                        elif self.last_cross_price < checkpoint_last_cross and self.last_guidance == 1:
                            self.num_correct += 1
                        acc = self.num_correct/self.num_trades
                    
                    if self.trading_enabled:
                        self.update_wallet()
                    
                    self.last_trade_price = trade_price
                    self.last_trade_time = trade_time
                    self.last_guidance = model_out
                    self.qsum *= self.qsum_decay_rate
                    checkpoint_last_cross = self.last_cross_price
                    if successFlag:
                        print('{} {} {} at time: {} \t Current trade Price: {} \t Model Total Acc: {:.3f} \t Portfolio Value: {} \t Coin Amt: {} Base Amt: {}'.format(s,
                              trade_amt_coin,
                              self.trade_coin,datetime.datetime.fromtimestamp(trade_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
                              trade_price,acc,value/(start_amt_base+start_amt_coin*price_bid)*100,
                              self.coin_amt,self.base_amt))
                    
                    plt.figure(1)
                    ax1 = plt.subplot(311)
                    plt.plot(trade_time,trade_price,'g.',trade_time,self.last_cross_price,'rx')
                    plt.subplot(312,sharex=ax1)
                    plt.plot(trade_time,self.coin_amt,'g.')
                    plt.subplot(313,sharex=ax1)
                    plt.plot(trade_time,value/(start_amt_base+start_amt_coin*price_bid),'g.')
                    plt.pause(0.05)
                    plt.draw()
                    '''
                    plt.figure(2)
                    ax2 = plt.subplot(211)
                    plt.plot(trade_time,trade_price,'g.',trade_time,self.last_cross_price,'rx')
                    plt.pause(0.05)
                    plt.draw()
                    '''
            
            time.sleep(5)
            
    
    def get_guidance(self,kline):
        inputs,self.last_cross_price = DataFormatting.format_live_data(kline,self.windows,self.last_cross_price)
        inputs_scaled,_,_ = DataFormatting.scale_inputs(inputs,self.min_x,self.max_x)
        model_out = self.eval_model(inputs_scaled)
        model_index = np.zeros(model_out.shape[0])
        for  i in range(model_out.shape[0]):
            model_index[i], _ = max(enumerate(model_out[i]), key=operator.itemgetter(1))
        return model_index[-1],model_out
    
    def get_depth(self):
        depth = self.client.get_order_book(symbol=self.pair)
    
        price_bid = []
        date = self.client.get_server_time()
        date = date['serverTime']/1000.0
        price_bid = float(depth['bids'][0][0])
        
        price_ask = []
        price_ask = float(depth['asks'][0][0])
        return date,price_bid,price_ask
    
    def check_cross(self,kline):
        
        mean_price = np.mean(kline[:,1:4],axis=1,keepdims=True);
        mean_price_cusum = np.cumsum(mean_price)
        p1 = mean_price_cusum[self.windows[0]:]
        p2 = mean_price_cusum[:-self.windows[0]]
        mean1 = (p1-p2)/self.windows[0]
        p1 = mean_price_cusum[self.windows[1]:]
        p2 = mean_price_cusum[:-self.windows[1]]
        mean2 = (p1-p2)/self.windows[1]
        
        if np.isnan(self.last_delta):
            self.last_delta = mean1[-1]-mean2[-1]
            return False
        
        if np.sign(self.last_delta) != np.sign(mean1[-1]-mean2[-1]):
            # then it's a crossing
            self.last_delta = mean1[-1]-mean2[-1]
            return True
        
        self.last_delta = mean1[-1]-mean2[-1]
        return False
        
    
    def read_candle_data(self,t1,*args):
        if len(args)>0:
            klines = self.client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, t1,args[0])
        else:
            klines = self.client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, t1)
        klines = np.array(klines)
        klines = klines.astype('float')
        return klines
    
    def eval_model(self,x):
        z = {}
        a = {}
        z['layer0'] = np.matmul(x,self.W[0]) + self.b[0]
        a['layer0'] = np.tanh(z['layer0'])
        for L in range(1,len(self.W)-1):
            z['layer{}'.format(L)] = np.matmul(a['layer{}'.format(L-1)], self.W[L]) + self.b[L]
            a['layer{}'.format(L)] = np.tanh(z['layer{}'.format(L)])
        
        z['layer{}'.format(len(self.W)-1)] = np.matmul(a['layer{}'.format(len(self.W)-2)], self.W[len(self.W)-1]) + self.b[len(self.W)-1]
        a['layer{}'.format(len(self.W)-1)] = z['layer{}'.format(len(self.W)-1)]
        output = MrKrabs.softmax(a['layer{}'.format(len(self.W)-1)])
        return output
    
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
        return e_x / np.sum(e_x,axis=1,keepdims=True)