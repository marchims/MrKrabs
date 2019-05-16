# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:59:29 2018

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
from twisted.internet import reactor
import json
import sys
from pathlib import Path
import os

class Rebalancer:

    interval = 3/60 # hours
    
    fudge_factor = 1.01 # percent extra 
    
    fee_amt = 0.75*0.10/100 # fee amt
    margin = 0.01-(0.75*0.10/100) # amount over the fee amount that will trigger a trade (0.05%)
    
    norm_margin = 0.04 # tolerance normalized
    
    coin_data = {}

    def __init__(self,keys,base_currency,ratio_worksheet):
        self.client = Client(keys[0][0],keys[0][1])
        self.base_coin = base_currency
        self.wallet_updated = False
        
        self.get_exchange_info()
        
        self.last_rebalance = 0
        self.init_coin_amt = []
        self.norm_gain = 0
        self.total_gain = 0
        
        self.total_fees = 0
        self.log_file = "../logs/rebablance_state.json"
        self.trade_log_file = '../logs/rebalance_log.csv'
        self.archive_dir = '../logs/archive'
        self.archive_logs()
        
        # load the worksheet with the ratios
        data = pd.read_excel(ratio_worksheet,header=None)
        names = data.loc[0,1:]
        for i in range(1,len(names)+1):
            coin = names.loc[i]
            self.coin_data[coin] = {}
            self.coin_data[coin]['weight'] = data.loc[1,i]
            self.init_coin_amt.append(data.loc[2,i])
            if coin != self.base_coin:
                self.get_market_info(coin)
            else:
                
                self.coin_data[coin]['min_trade_amt'] = [0.00001]
               
        self.num_coins = len(names)
        self.init_trade_log()
        self.init_total_value = 0
        self.get_values(verbose=False)
        self.trading_enabled = False
            
        self.init_total_value = data.loc[3,1]
        self.load_state()
        
    def load_state(self):
        try:
            fid = open(self.log_file,'r')
        except:
            return
        load_struct = json.load(fid)
        fid.close()
        
        self.total_fees = load_struct['total_fees']
        self.interval = load_struct['interval']
        self.fudge_factor = load_struct['fudge_factor']
        self.fee_amt = load_struct['fee_amt']
        self.margin = load_struct['margin']
        self.norm_margin = load_struct['norm_margin']
        
    def save_state(self):
        save_struct = {}
        save_struct['total_fees'] = self.total_fees
        save_struct['interval'] = self.interval
        save_struct['fudge_factor'] = self.fudge_factor
        save_struct['fee_amt'] = self.fee_amt
        save_struct['margin'] = self.margin
        save_struct['norm_margin'] = self.norm_margin
 
        fid = open(self.log_file,'w')
        json.dump(save_struct,fid)
        fid.close()
        
    def init_trade_log(self):
        self.trade_data = np.zeros((0,self.num_coins+4))
        
    def archive_logs(self):
        f = Path(self.trade_log_file)
        a = Path(self.archive_dir)
        if f.is_file():
            if not a.is_dir():
                os.mkdir(self.archive_dir)
            os.rename(self.trade_log_file,'{}/rebalance_log_{}.csv'.format(self.archive_dir,time.strftime('%m_%d_%y_%H_%M_%S')))
        
    def stop(self):
        self.timer.cancel()
    
    def run(self):
        self.timer = Timer(self.interval*3600,lambda: self.rebalance())
        self.timer.start()
        
    def rebalance(self):
        # get current values
        total_value = self.get_values(verbose=False)
        if self.wallet_updated == False:
            print('Wallet not updated. Skipping')
            self.timer = Timer(self.interval*3600,lambda: self.rebalance())
            self.timer.start()
            return
            
        
        # get current ratios
        self.get_ratios(total_value)
        
        # find out how much base currency needs to be bought/sold for each asset
        self.get_targets(total_value)
        
        # sort by smallest purchase to min notional ratio
        target = self.get_property_list('target_amt')
        current = self.get_property_list('last_value')
        min_trade = self.get_property_list('min_trade_amt')
        markets = self.get_property_list('market_name')
        
        coin_names = list(self.coin_data.keys())
        
        delta = []
        norm_delta = []
        score = []
        for i in range(len(target)):
            m = min_trade[i][0]
            raw_delta = target[i]-current[i]
            abs_delta = abs(raw_delta)
            for j in range(1,len(min_trade[i])):
                m *= self.fudge_factor
                if raw_delta > 0:
                    # need to buy, will be selling bnb
                    m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'buy'))
                else:
                    # need to sell coin, will be buying bnb
                    m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'sell'))
                
            score.append(abs_delta/m)
            if abs_delta >= m and self.base_coin != coin_names[i]:
                delta.append(abs_delta/total_value)
                norm_delta.append(abs_delta/target[i])
                #print(coin_names[i])
                #print('{} {}'.format(target[i],current[i]))
            else:
                delta.append(0)
                norm_delta.append(0)
        
        rebalance_order = [x for _,x in sorted(zip(score,coin_names))]        
        
        delta = np.array(delta)
        norm_delta = np.array(norm_delta)
        
        
        # begin trading one by one in this order
        if np.any(delta>(self.fee_amt+self.margin)) or np.any(norm_delta > self.norm_margin):
            
            print("Rebalance needed ({:.3f}% error, {:.3f}% norm).".format(np.max(delta)*100,np.max(norm_delta)*100))
            if self.trading_enabled == False:
                print('These trades are SIMULATED!')
                
            bucket = total_value
            completed = []
            for coin in rebalance_order:
                if coin != self.base_coin:
                    bucket,completed = self.make_trade(coin,bucket,black_list=completed)
            print("{}: {}".format(self.base_coin,bucket))
            self.log_values()
            
        else:
            print("No rebalance needed ({:.3f}% error {:.3f}% norm).".format(np.max(delta)*100,np.max(norm_delta)*100))
        self.timer = Timer(self.interval*3600,lambda: self.rebalance())
        self.timer.start()
        self.save_state()
        
    def make_trade(self,coin,bucket,black_list):
        target = self.get_target_ratio(coin,black_list) * bucket
        trade_amt = target - self.coin_data[coin]['last_value']

        m = self.coin_data[coin]['min_trade_amt'][0]
        for j in range(1,len(self.coin_data[coin]['min_trade_amt']),):
            if trade_amt > 0:
                m = max(self.coin_data[coin]['min_trade_amt'][j],self.coin_data[coin]['min_trade_amt'][j-1]*
                        self.lookup_price(self.coin_data[coin]['market_name'][j],'buy'))
            else:
                m = max(self.coin_data[coin]['min_trade_amt'][j],self.coin_data[coin]['min_trade_amt'][j-1]*
                        self.lookup_price(self.coin_data[coin]['market_name'][j],'sell'))


        if len(self.coin_data[coin]['market_name']) > 1:
            use_factor = True
        else:
            use_factor = False

        if trade_amt > 0:
            trade_side = SIDE_BUY
            
            r = range(len(self.coin_data[coin]['market_name'])-1, -1, -1)
        else:
            trade_side = SIDE_SELL
            trade_amt = -trade_amt
            r = range(len(self.coin_data[coin]['market_name']))
        min_trade_flag = False
        if trade_amt < m:
            trade_amt = 0
            min_trade_flag = True

        traded_coin_amt_final = 0
        
        if use_factor:
            trade_amt *= self.fudge_factor
    
        for i in r:
            
            for j in range(i+1,len(self.coin_data[coin]['market_name'])):
                trade_amt /= self.lookup_price(self.coin_data[coin]['market_name'][j],trade_side)

            coin_amt = self.get_coin_amt_from_market(self.coin_data[coin]['market_name'][i],trade_amt,trade_side,update_flag=True,factor=use_factor)
            use_factor = False
            
            if coin_amt > 0:
                
                if self.trading_enabled == True:
                    try:
                        result = self.client.create_order(symbol=self.coin_data[coin]['market_name'][i],side=trade_side,type=ORDER_TYPE_MARKET ,quantity=coin_amt,newOrderRespType=ORDER_RESP_TYPE_FULL)
                        qty = float(result['executedQty'])
                        avg_price = 0
                        for j in range(len(result['fills'])):
                            avg_price += float(result['fills'][j]['price'])*float(result['fills'][j]['qty'])/qty
                            
                            
                        if trade_side == SIDE_SELL:
                            trade_amt = qty*avg_price
                        else:
                            trade_amt = qty*avg_price
                        
                        if len(self.coin_data[coin]['market_name']) > 1:
                            if i == 0 and trade_side == SIDE_SELL:
                                self.total_fees += trade_amt*self.fee_amt*self.lookup_price(self.coin_data[coin]['market_name'][1],trade_side)
                            elif i == 0 and trade_side == SIDE_BUY:
                                self.total_fees += trade_amt*self.fee_amt*self.lookup_price(self.coin_data[coin]['market_name'][1],trade_side)
                        else:
                            self.total_fees += trade_amt*self.fee_amt
                            
                        if i < len(self.coin_data[coin]['market_name'])-1:
                            trade_amt *= self.lookup_price(self.coin_data[coin]['market_name'][i+1],trade_side)
                            
                        if trade_side == SIDE_BUY:
                            print('Buy {} {}@{}'.format(qty,self.coin_data[coin]['market_name'][i],avg_price))
                        else:
                            print('Sell {} {}@{}'.format(qty,self.coin_data[coin]['market_name'][i],avg_price))
                        time.sleep(1)
                    except:
                        print("Error making trade for {} {}@{}".format(self.coin_data[coin]['market_name'][i],coin_amt,self.lookup_price(self.coin_data[coin]['market_name'][i],trade_side)))
                        qty = 0
                else:
                    try:
                        self.client.create_test_order(symbol=self.coin_data[coin]['market_name'][i],side=trade_side,type=ORDER_TYPE_MARKET ,quantity=coin_amt)
                        qty = coin_amt
                        avg_price = self.lookup_price(self.coin_data[coin]['market_name'][i],trade_side)    
                        
                        if trade_side == SIDE_SELL:
                            trade_amt = qty*avg_price
                        else:
                            trade_amt = qty*avg_price
                        
                        if i < len(self.coin_data[coin]['market_name'])-1:
                            trade_amt *= self.lookup_price(self.coin_data[coin]['market_name'][i+1],trade_side)
                        
                        if trade_side == SIDE_BUY:
                            print('Buy {} {}@{}'.format(qty,self.coin_data[coin]['market_name'][i],avg_price))
                        else:
                            print('Sell {} {}@{}'.format(qty,self.coin_data[coin]['market_name'][i],avg_price))
                    except:
                        print("Error making trade for {} {}@{}".format(self.coin_data[coin]['market_name'][i],coin_amt,self.lookup_price(self.coin_data[coin]['market_name'][i],trade_side)))
                        qty = 0
                       
                if i == 0:
                    traded_coin_amt_final = qty
            else:
                trade_amt = 0
                print("No trade for {}".format(coin))
        if trade_side == SIDE_BUY:
            new_amt = traded_coin_amt_final+self.coin_data[coin]['amt']            
        else:
            new_amt = -traded_coin_amt_final+self.coin_data[coin]['amt']
        
##        if min_trade_flag == True:
##            new_val = target
##        else:
        new_val = self.get_coin_value(coin,amt=new_amt)
        bucket -= new_val
        print("Target: {:.5f} Actual: {:.5f} Error: {:.5f}".format(target,new_val,target-new_val))
        
        black_list.append(coin)
        return bucket,black_list        
        
    def get_coin_amt_from_market(self,market,base_amt,side,update_flag=False,factor=False):
        if update_flag == True:
            self.update_price(market)
        
        # get market info
        for market2 in self.exchange_info['symbols']:
            if market2['symbol'].lower() == market.lower():
                prec = float(market2['filters'][2]['stepSize'])
                min_amt = float(market2['filters'][3]['minNotional'])
                break
        
        if factor:
            trade_amt_coin = np.ceil(base_amt/self.lookup_price(market,side)/prec)*prec
        else:
            trade_amt_coin = np.floor(base_amt/self.lookup_price(market,side)/prec)*prec
        f = 1.0
        if factor:
            f = self.fudge_factor
        if trade_amt_coin*self.lookup_price(market,side) < min_amt*f:
            #print("{} {}".format(trade_amt_coin*self.lookup_price(market,side),min_amt))
            return 0
        
        
        return trade_amt_coin
        
        
        
    def get_market_info(self,coin):
        des_market_name = coin.lower()+self.base_coin.lower()
        
        for market in self.exchange_info['symbols']:
            if market['symbol'].lower() == des_market_name:
                self.coin_data[coin]['coin_precision'] = [float(market['filters'][2]['stepSize'])]
                self.coin_data[coin]['min_trade_amt'] = [float(market['filters'][3]['minNotional'])]
                self.coin_data[coin]['market_name'] = [market['symbol']]
                return
        
        # if it's here then it's not a direct market
        des_market_name = coin.lower()+'bnb'
        for market in self.exchange_info['symbols']:
            if market['symbol'].lower() == des_market_name:
                self.coin_data[coin]['coin_precision'] = [float(market['filters'][2]['stepSize'])]
                self.coin_data[coin]['min_trade_amt'] = [float(market['filters'][3]['minNotional'])]
                self.coin_data[coin]['market_name'] = [market['symbol']]
                break
         
        des_market_name = 'bnb'+self.base_coin.lower()
        for market in self.exchange_info['symbols']:
            if market['symbol'].lower() == des_market_name:
                self.coin_data[coin]['coin_precision'].append(float(market['filters'][2]['stepSize']))
                self.coin_data[coin]['min_trade_amt'].append(float(market['filters'][3]['minNotional']))
                self.coin_data[coin]['market_name'].append(market['symbol'])
                break
        


    def get_exchange_info(self):
        self.exchange_info = self.client.get_exchange_info()
    
    def update_prices(self):
        self.price_info = self.client.get_orderbook_tickers()
        
    def update_price(self,market):
        new_data = self.client.get_orderbook_ticker(symbol=market)
        for i in range(len(self.price_info)):
            if self.price_info[i]['symbol'] == market:
                self.price_info[i] = new_data
                break

    def get_property_list(self,name):
        values = []
        for coin in self.coin_data:
            if name in self.coin_data[coin]:
                values.append(self.coin_data[coin][name])
            else:
                values.append(None)
        return values

    def lookup_amt(self,coin):
        if 'amt' in self.coin_data[coin]:
            return self.coin_data[coin]['amt']
        else:
            return None
    
    def lookup_price(self,market,side):
        for i in range(len(self.price_info)):
            if self.price_info[i]['symbol'] == market:
                if side == 'buy' or side == SIDE_BUY:
                    price = float(self.price_info[i]['askPrice'])
                else:
                    price = float(self.price_info[i]['bidPrice'])
                break
        return price
    
    def set_weight(self,coin,weight):
        self.coin_data[coin]['weight'] = weight
    
    
    def get_coin_value(self,coin,amt=-1):
        if amt < 0:
            amt = self.lookup_amt(coin)
        value = amt
        if coin != self.base_coin:
            for i in range(len(self.coin_data[coin]['market_name'])):
                value *= 0.5*self.lookup_price(self.coin_data[coin]['market_name'][i],'sell') + 0.5*self.lookup_price(self.coin_data[coin]['market_name'][i],'buy')
        return value
    
    def get_values(self,verbose = True):
        self.update_wallet()
        if not self.wallet_updated:
            return -1
        self.update_prices()
        total_value = -self.total_fees
        for coin in self.coin_data:
            if 'amt' in self.coin_data[coin]:
                value = self.get_coin_value(coin,self.lookup_amt(coin))
                if verbose:
                    print("{}: {}".format(coin,value))
                self.coin_data[coin]['last_value'] = value
                total_value = total_value + value
        if self.init_total_value != 0:
            idx = 0
            hodl_portfolio_value = 0
            for coin in self.coin_data:
                hodl_portfolio_value += self.get_coin_value(coin,self.init_coin_amt[idx])
                idx += 1
            self.norm_gain = (total_value-hodl_portfolio_value)/hodl_portfolio_value*100
            self.total_gain = (total_value-self.init_total_value)/self.init_total_value*100
            print('Total Value: {} (Raw:{:.2f}% Norm:{:.2f}%)'.format(total_value,self.total_gain,self.norm_gain))
        else:
            print('Total Value: {}'.format(total_value))
        return total_value

    def get_ratios(self,total_value):
        for coin in self.coin_data:
            if 'last_value' in self.coin_data[coin]:
                self.coin_data[coin]['last_ratio'] = self.coin_data[coin]['last_value'] / total_value
            else:
                self.coin_data[coin]['last_ratio'] = -1
                
    def get_target_ratio(self,coin,black_list=[]):
        total_weight = 0
        for c in self.coin_data:
            if 'amt' in self.coin_data[c] and c not in black_list:
                total_weight = total_weight + self.coin_data[c]['weight']
        return self.coin_data[coin]['weight'] / total_weight
         

    def get_targets(self,total_value):
        
        min_trade = self.get_property_list('min_trade_amt')
        markets = self.get_property_list('market_name')
        current = self.get_property_list('last_value')
        coin_names = list(self.coin_data.keys())
        state_change = True
        black_list = []
        deduct_value = 0
        while state_change:
            state_change = False
            target = []
            for i in range(len(coin_names)):
                if coin_names[i] in black_list:
                    target.append(current[i])
                else:
                    target.append((total_value-deduct_value) * self.get_target_ratio(coin_names[i],black_list))
                    m = min_trade[i][0]
                    raw_delta = target[i]-current[i]
                    abs_delta = abs(raw_delta)
                    for j in range(1,len(min_trade[i])):
                        m *= self.fudge_factor
                        if raw_delta > 0:
                            # need to buy, will be selling bnb
                            m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'buy'))
                        else:
                            # need to sell coin, will be buying bnb
                            m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'sell'))
                        
                    if abs_delta < m:
                        black_list.append(coin_names[i])
                        deduct_value += current[i]
                        state_change = True
                        break
        for i in range(len(coin_names)):      
            self.coin_data[coin_names[i]]['target_amt'] = target[i]
        

    def log_values(self):
        self.trade_data = np.append(self.trade_data,
                                    np.concatenate((np.array(time.time()).reshape((1,1)),
                                    np.array(self.get_property_list('last_value')).reshape((1,self.num_coins)),
                                    np.array([self.norm_gain,self.total_gain,self.total_fees]).reshape((1,3))),axis=1),axis=0)
        try:
            toWrite = pd.DataFrame(data = self.trade_data)
            toWrite.to_csv(self.trade_log_file,header = False,index = False)
        except:
            print('Error writing to log file!')
        

    def update_wallet(self):
        self.wallet_updated = False
        try:
            account_data = self.client.get_account()
        except:
            return    
        
        coin_updates = {}
        for coin in self.coin_data:
            coin_updates[coin] = False
            
        # find asset balance in list of balances
        if "balances" in account_data:
            for bal in account_data['balances']:
                for coin in self.coin_data: 
                    if bal['asset'].lower() == coin.lower():
                        self.coin_data[coin]['amt'] = float(bal['free']) + float(bal['locked'])
                        coin_updates[coin] = True
 
        update_success = True
        for coin in self.coin_data:
            if coin_updates[coin] == False:
                update_success = False
                break
    
        self.wallet_updated = update_success
        if self.wallet_updated == True:
            print('Wallet successfully updated')



class MrKrabs2:
    
    buffer_time = 1000 #seconds
    ema_len = 1/60
    
    max_hist_len = 10
    reset_timeout = 30000 # seconds since last update
    timeout_thresh = 30
    trade_socket_timeout = 120 # seconds without a trade
    # model params
    #margin = 0.45/100
    fee_amt = 0.075/100
    base_trade_amt = 0.03 # was 0.05 on 8-14-18
    bias_limit = 0.4064
    buyback_ratio = 0.8  # was 0.75 on 8-14-18
    variance_ff = [1/120,1/300] # was 1200 for both on 8-24-18
    ema_filts = [1/1600,1/200,1/20]
    slope_levels = [0.05,0.1,0.2,0.5,1] # percent
    
    margin_scale = 0.36  # was 0.9 on 8-14-18
    margin_exp = 0.9
    margin_min = 0.05
    margin_max = 0.9 # dropped from 1.5 on 8-24-18
    
    # bnb things
    min_bnb = 0.2
    bnb_buy_amt = 0.85 # raised from 0.75 due to being below 0.001 BTC
    
    cooldown_reset_thresh = 15
    buyback_enable_thresh = 2
    
    refresh_rate = 1.5
    wallet_timeout_loops = 3
    
    def __init__(self,keys,pair):
        self.client = Client(keys[0][0],keys[0][1])
        self.get_pair_info(pair)
        self.fee_coin_amt = 0
        self.log_file = "../logs/{}_state.json".format(pair)
        self.trade_log_file = '../logs/{}_log.csv'.format(pair)
        self.archive_dir = '../logs/archive_{}'.format(pair)
        
        self.init_coin_amt = 0
        self.init_base_amt = 0
        
        self.trade_buffer = np.zeros((0))
        self.ask_buffer = np.zeros((0))
        self.bid_buffer = np.zeros((0))
        self.time_buffer = np.zeros((0))
        self.time_buffer_depth = np.zeros((0))
        self.filt_ema_buffer = np.zeros((0))
        self.ask_buffer_ema = np.zeros((0))
        self.bid_buffer_ema = np.zeros((0))
        self.prev_sample = np.nan
        self.prev_sample_vol_buy = np.nan
        self.prev_sample_vol_sell = np.nan
        self.trade_buffer_vol_buy = 0
        self.trade_buffer_vol_sell = 0
        self.filt_ema_buffer_vol_buy = np.zeros((0))
        self.filt_ema_buffer_vol_sell = np.zeros((0))
        
        self.depth_ask_price = np.zeros((20))
        self.depth_bid_price = np.zeros((20))
        self.depth_ask_qty = np.zeros((20))
        self.depth_bid_qty = np.zeros((20))
        self.depth_bid_vol = np.zeros((1,len(self.slope_levels)))
        self.depth_ask_vol = np.zeros((1,len(self.slope_levels)))
        
        
        
        self.data = np.zeros((0,8+2*len(self.slope_levels)+2))
        
        self.models = {}
        self.hist_data = np.zeros((0,58))
        self.ema_bids_last = [np.nan,np.nan,np.nan]
        self.trade_data = np.zeros((0,7))
        self.trading_enabled = False
        self.wallet_up_to_date = False
        self.cooldown_reset= True
        self.logging_enabled = False
        self.plots = False
        
        self.last_action = 'none'
        
        # Algorithm counters
        self.mean_buy_price = 0
        self.mean_sell_price = 0
        self.bias = 0
        self.buy_surplus = 0
        self.sell_surplus = 0
        self.wallet_timeout_count = 0
        self.dt_hist = np.zeros((0,2))
        self.var_est_last = np.nan
        self.buyback_enable = False
        
        self.target_mix = 0.50
        self.buy_model_installed = False
        self.sell_model_installed = False
        
        self.load_state()
        self.archive_logs()
        
        
    def load_state(self):
        try:
            fid = open(self.log_file,'r')
        except:
            return
        load_struct = json.load(fid)
        fid.close()
        
        self.last_action = load_struct['last_action']
        self.cooldown_reset = load_struct['cooldown_reset']
        self.buyback_enable = load_struct['buyback_enable']
        self.mean_buy_price = load_struct['mean_buy_price']
        self.mean_sell_price = load_struct['mean_sell_price']
        self.buffer_time = load_struct['buffer_time']
        self.ema_len = load_struct['ema_len']
        self.max_hist_len = load_struct['max_hist_len']
        self.reset_timeout = load_struct['reset_timeout']
        self.timeout_thresh = load_struct['timeout_thresh']
        self.trade_socket_timeout = load_struct['trade_socket_timeout']
        self.fee_amt = load_struct['fee_amt']
        self.base_trade_amt = load_struct['base_trade_amt']
        self.bias_limit = load_struct['bias_limit']
        self.buyback_ratio = load_struct['buyback_ratio']
        self.variance_ff = load_struct['variance_ff']
        self.ema_filts = load_struct['ema_filts']
        self.slope_levels = load_struct['slope_levels']
        self.margin_scale = load_struct['margin_scale']
        self.margin_exp = load_struct['margin_exp']
        self.margin_min = load_struct['margin_min']
        self.margin_max = load_struct['margin_max']
        self.min_bnb = load_struct['min_bnb']
        self.bnb_buy_amt = load_struct['bnb_buy_amt']
        self.cooldown_reset_thresh = load_struct['cooldown_reset_thresh']
        self.buyback_enable_thresh = load_struct['buyback_enable_thresh']
        self.refresh_rate = load_struct['refresh_rate']
        self.init_coin_amt = load_struct['init_coin_amt']
        self.init_base_amt = load_struct['init_base_amt']
        self.wallet_timeout_loops = load_struct['wallet_timeout_loops']
        
    def save_state(self):
        save_struct = {}
        save_struct['last_action'] = self.last_action
        save_struct['cooldown_reset'] = self.cooldown_reset
        save_struct['buyback_enable'] = self.buyback_enable
        save_struct['mean_buy_price'] = self.mean_buy_price
        save_struct['mean_sell_price'] = self.mean_sell_price
        
        
        save_struct['buffer_time'] = self.buffer_time
        save_struct['ema_len'] = self.ema_len
        save_struct['max_hist_len'] = self.max_hist_len
        save_struct['reset_timeout'] = self.reset_timeout
        save_struct['timeout_thresh'] = self.timeout_thresh
        save_struct['trade_socket_timeout'] = self.trade_socket_timeout
        save_struct['fee_amt'] = self.fee_amt
        save_struct['base_trade_amt'] = self.base_trade_amt
        save_struct['bias_limit'] = self.bias_limit
        save_struct['buyback_ratio'] = self.buyback_ratio
        save_struct['variance_ff'] = self.variance_ff
        save_struct['ema_filts'] = self.ema_filts
        save_struct['slope_levels'] = self.slope_levels
        save_struct['margin_scale'] = self.margin_scale
        save_struct['margin_exp'] = self.margin_exp
        save_struct['margin_min'] = self.margin_min
        save_struct['margin_max'] = self.margin_max
        save_struct['min_bnb'] = self.min_bnb
        save_struct['bnb_buy_amt'] = self.bnb_buy_amt
        save_struct['cooldown_reset_thresh'] = self.cooldown_reset_thresh
        save_struct['buyback_enable_thresh'] = self.buyback_enable_thresh
        save_struct['refresh_rate'] = self.refresh_rate
        save_struct['init_coin_amt'] = self.init_coin_amt
        save_struct['init_base_amt'] = self.init_base_amt
        save_struct['wallet_timeout_loops'] = self.wallet_timeout_loops
 
        fid = open(self.log_file,'w')
        json.dump(save_struct,fid)
        fid.close()
    
    def archive_logs(self):
        f = Path(self.trade_log_file)
        a = Path(self.archive_dir)
        if f.is_file():
            if not a.is_dir():
                os.mkdir(self.archive_dir)
            os.rename(self.trade_log_file,'{}/trade_log_{}.csv'.format(self.archive_dir,time.strftime('%m_%d_%y_%H_%M_%S')))
            
    def enable_trading(self):
        self.trading_enabled = True
    
    def disable_trading(self):
        self.trading_enabled = False
    
    def get_pair_info(self,pair):
        self.pair = pair
        self.base_coin = pair[-3:]
        self.trade_coin = pair[:-3]
        info = self.client.get_symbol_info(self.pair)
        self.coin_precision = float(info['filters'][2]['stepSize'])
        self.min_trade_amt = float(info['filters'][3]['minNotional'])

    def account_update(self,msg):
        if msg['e'] == 'outboundAccountInfo':
            base_updated = False
            coin_updated = False
            # find asset balance in list of balances
            if "B" in msg:
                for bal in msg['B']:
                    if bal['a'].lower() == self.base_coin.lower():
                        self.base_amt = float(bal['f'])
                        base_updated = True
                    if bal['a'].lower() == self.trade_coin.lower():
                        self.coin_amt = float(bal['f'])
                        coin_updated = True
                    if bal['a'].lower() == "bnb":
                        self.fee_coin_amt = float(bal['f'])
     
            self.wallet_up_to_date = coin_updated and base_updated
            if self.wallet_up_to_date == True:
                print('{}  Wallet update received'.format(time.strftime('%H:%M:%S')))

    def update_wallet(self):
        account_data = self.client.get_account()
        self.wallet_up_to_date = False
        base_updated = False
        coin_updated = False
        # find asset balance in list of balances
        if "balances" in account_data:
            for bal in account_data['balances']:
                if bal['asset'].lower() == self.base_coin.lower():
                    self.base_amt = float(bal['free'])
                    base_updated = True
                if bal['asset'].lower() == self.trade_coin.lower():
                    self.coin_amt = float(bal['free'])
                    coin_updated = True
                if bal['asset'].lower() == "bnb":
                    self.fee_coin_amt = float(bal['free'])
 
        self.wallet_up_to_date = coin_updated and base_updated
        if self.wallet_up_to_date == True:
            print('{}  Wallet manually updated'.format(time.strftime('%H:%M:%S')))
        # temporary amounts for testing
        #self.base_amt = 0.1
        #self.coin_amt = 100
        
    def playback(self,inputs,base,coin,compare):
        
        self.base_amt = base
        self.coin_amt = coin
        
        m,n = inputs.shape
        self.data = np.zeros((0,n))
        self.hist_data = np.zeros((0,58))
        self.ema_bids_last = [np.nan,np.nan,np.nan]
        self.var_est_last = np.nan
        self.trade_data = np.zeros((0,5))
        self.trading_enabled = False
        
        # Algorithm counters
        self.mean_buy_price = 0
        self.mean_sell_price = 0
        self.bias = 0
        self.buy_surplus = 0
        self.target_mix = base/(base + inputs[0,2]*coin)
        self.sell_surplus = 0
        self.buyback_enable = False
        formatted_inputs = np.zeros((0,58))
        for i in range(50000,m):
            self.data = np.append(self.data,inputs[i,:].reshape((1,n)),axis = 0)
            self.processLatestData(playbackFlag=True)
            formatted_inputs = np.append(formatted_inputs,self.hist_data[-1,:].reshape((1,58)),axis=0)
            if max(abs(self.hist_data[-1,:] - compare[i,:]))>1e-7:
                print("Inputs don't match at {}! {}".format(i,max(abs(self.hist_data[-1,:] - compare[i,:]))))
            self.trade_model(trade_time=inputs[i,0])
            self.validate_trades()
        if self.plots:
            plt.figure()
            plt.plot(self.data[:,0],self.data[:,1],'k-')
            plt.plot(self.data[:,0],self.data[:,2],'g.',self.data[:,0],self.data[:,3],'r.')
            plt.plot(self.trade_data[:,0],self.trade_data[:,2],'ko',markersize = 12)
        
        return formatted_inputs
    
    def update_trades(self,msg):
        if msg['e'] == 'trade':
            self.time_buffer = np.append(self.time_buffer,np.array(msg['T']/1000).reshape((1)),axis=0)
            self.trade_buffer = np.append(self.trade_buffer,np.array(float(msg['p'])).reshape((1)),axis=0)
            if self.data.shape[0]>1:
                if float(msg['p']) <= self.data[-1,2]:
                    self.trade_buffer_vol_sell = self.trade_buffer_vol_sell+float(msg['q'])*float(msg['p'])
                elif float(msg['p']) >= self.data[-1,3]:
                    self.trade_buffer_vol_buy = self.trade_buffer_vol_buy+float(msg['q'])*float(msg['p'])
                else:
                    self.trade_buffer_vol_buy = self.trade_buffer_vol_buy+float(msg['q'])*float(msg['p'])/2
                    self.trade_buffer_vol_sell = self.trade_buffer_vol_sell+float(msg['q'])*float(msg['p'])/2
            if self.time_buffer.shape[0] >= 2:
                dt = self.time_buffer[-1] - self.time_buffer[-2]
                if dt == 0:
                    dt = 0.01
                a = dt / (1/self.ema_len)
                u = np.exp(-a)
                v = (1.0-u)/a
                # Price
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
                if dt < 0.1:
                    dt = 0.10
                a = dt / (1/self.ema_len)
                u = np.exp(-a)
                v = (1.0-u)/a
                next_ema = u*self.ask_buffer_ema[-1] + (v-u)*self.ask_buffer[-2] + (1.0-v)*self.ask_buffer[-1]
                self.ask_buffer_ema = np.append(self.ask_buffer_ema,next_ema.reshape((1)),axis=0)
                
                next_ema = u*self.bid_buffer_ema[-1] + (v-u)*self.bid_buffer[-2] + (1.0-v)*self.bid_buffer[-1]
                self.bid_buffer_ema = np.append(self.bid_buffer_ema,next_ema.reshape((1)),axis=0)

                # Volume
                next_ema = u*self.filt_ema_buffer_vol_buy[-1] + (v-u)*self.prev_sample_vol_buy + (1.0-v)*self.trade_buffer_vol_buy
                self.filt_ema_buffer_vol_buy = np.append(self.filt_ema_buffer_vol_buy,next_ema.reshape((1)),axis=0)
                
                next_ema = u*self.filt_ema_buffer_vol_sell[-1] + (v-u)*self.prev_sample_vol_sell + (1.0-v)*self.trade_buffer_vol_sell
                self.filt_ema_buffer_vol_sell = np.append(self.filt_ema_buffer_vol_sell,next_ema.reshape((1)),axis=0)
                
            else:
                self.ask_buffer_ema = np.zeros((1))
                self.ask_buffer_ema[0] = self.ask_buffer[-1]
                self.bid_buffer_ema = np.zeros((1))
                self.bid_buffer_ema[0] = self.bid_buffer[-1]
                self.filt_ema_buffer_vol_buy = np.zeros((1))
                self.filt_ema_buffer_vol_buy[0] = self.trade_buffer_vol_buy
                self.filt_ema_buffer_vol_sell = np.zeros((1))
                self.filt_ema_buffer_vol_sell[0] = self.trade_buffer_vol_sell
            
            self.prev_sample_vol_buy = self.trade_buffer_vol_buy
            self.prev_sample_vol_sell = self.trade_buffer_vol_sell
            self.trade_buffer_vol_buy = 0
            self.trade_buffer_vol_sell = 0
            bid_qty = np.array(float(msg['B'])).reshape((1,1))
            ask_qty = np.array(float(msg['A'])).reshape((1,1))
            if self.filt_ema_buffer.shape[0] > 1:
                self.data = np.append(self.data,np.concatenate((self.time_buffer_depth[-1].reshape((1,1)),self.filt_ema_buffer[-1].reshape((1,1)),
                                                                self.bid_buffer[-1].reshape((1,1)),self.ask_buffer[-1].reshape((1,1)),
                                                                self.bid_buffer_ema[-1].reshape((1,1)),self.ask_buffer_ema[-1].reshape((1,1)),bid_qty,ask_qty,
                                                                self.depth_bid_vol.reshape((1,len(self.depth_bid_vol))),self.depth_ask_vol.reshape((1,len(self.depth_ask_vol))),
                                                                self.filt_ema_buffer_vol_sell[-1].reshape((1,1)),self.filt_ema_buffer_vol_buy[-1].reshape((1,1))),axis=1),axis=0)
                self.processLatestData()
                #self.trade_model()
                
                #self.validate_trades()
                
                
                
                
            valid = self.time_buffer_depth >= self.time_buffer_depth[-1] - self.buffer_time
            self.time_buffer_depth = self.time_buffer_depth[valid]
            self.bid_buffer = self.bid_buffer[valid]
            self.ask_buffer = self.ask_buffer[valid]
            self.bid_buffer_ema = self.bid_buffer_ema[valid]
            self.ask_buffer_ema = self.ask_buffer_ema[valid]
            self.filt_ema_buffer_vol_buy = self.filt_ema_buffer_vol_buy[valid]
            self.filt_ema_buffer_vol_sell = self.filt_ema_buffer_vol_sell[valid]
            
            if len(self.time_buffer) >= 1 and len(self.time_buffer_depth) >= 1 and self.plots:
                plt.figure(2).clear()
                plt.plot(-np.array(self.slope_levels),self.depth_bid_vol,'g.-')
                plt.plot(np.array(self.slope_levels),self.depth_ask_vol,'r.-')
                plt.draw()
                
                plt.figure(1).clear()
                #plt.plot(self.time_buffer_depth,self.filt_ema_buffer_vol_sell,'r-')
                #plt.plot(self.time_buffer_depth,self.filt_ema_buffer_vol_buy,'g-')
                plt.plot(self.time_buffer,self.trade_buffer,'bx')
                plt.plot(self.time_buffer,self.filt_ema_buffer,'k-')
                plt.plot(self.time_buffer_depth,self.bid_buffer,'g.',self.time_buffer_depth,self.ask_buffer,'r.')
                #plt.plot(self.time_buffer_depth,self.bid_buffer_ema,'g--',self.time_buffer_depth,self.ask_buffer_ema,'r--')
            #elif len(self.time_buffer) >= 1 and self.plots == False:
                #print("Bid:{}\t\tAsk:{}".format(self.data[-1,2],self.data[-1,3]))    
            
            
            if self.trade_data.shape[0] >= 1 and self.plots:
                valid = np.where(self.trade_data[:,0] >= self.time_buffer_depth[-1] - self.buffer_time)
                if np.any(valid):
                    plt.plot(self.trade_data[valid,0],self.trade_data[valid,2],'ko',markersize = 12)   
            
            if len(self.time_buffer) >= 1 and len(self.time_buffer_depth) >= 1 and self.plots: 
                plt.draw()
            
            #plt.pause(0.05)
            
        elif msg['e'] == 'error':
            print(msg)
            print('Stopping')
            self.stop()
        
        # Auto Save
        if self.data.shape[0] >= 100:
##            print('{}  Resetting data'.format(time.strftime('%H_%M_%S')))
            if self.logging_enabled:
                toWrite = pd.DataFrame(data = self.data)
                toWrite.to_csv('nano_hi_res_test_{}.csv'.format(time.strftime('%H_%M_%S %m_%d_%y')),header = False,index = False)
                
            
            self.data = self.data[-3:,:].reshape((3,8+2*len(self.slope_levels)+2))
            
            
        
    def update_depth(self,msg):
        for i in range(len(msg['bids'])):
            self.depth_bid_price[i] = float(msg['bids'][i][0])
            self.depth_bid_qty[i] = float(msg['bids'][i][1])
        
        for i in range(len(msg['asks'])):
            self.depth_ask_price[i] = float(msg['asks'][i][0])
            self.depth_ask_qty[i] = float(msg['asks'][i][1])
        
        # calculate slopes at each price point in slope_levels
        self.slopes_ask = np.zeros((len(self.slope_levels)))
        self.slopes_bid = np.zeros((len(self.slope_levels)))
        
        curr_price_bid = self.depth_bid_price[0]
        curr_price_ask = self.depth_ask_price[0]
        
        norm_perc_ask = (self.depth_ask_price-curr_price_ask)/curr_price_ask*100
        norm_perc_bid = -(self.depth_bid_price-curr_price_bid)/curr_price_bid*100
    
        depth_asks = np.cumsum(np.multiply(self.depth_ask_qty,self.depth_ask_price))
        ask_req_qty = np.interp(self.slope_levels,norm_perc_ask,depth_asks)
        max_norm_perc = np.max(norm_perc_ask)
        for i in range(len(ask_req_qty)):
            if self.slope_levels[i]>max_norm_perc:
                ask_req_qty[i] = (ask_req_qty[i]-ask_req_qty[0])*self.slope_levels[i]/max_norm_perc+ask_req_qty[0]
        
        depth_bids = np.cumsum(np.multiply(self.depth_bid_qty,self.depth_bid_price))
        bid_req_qty = np.interp(self.slope_levels,norm_perc_bid,depth_bids)
        max_norm_perc = np.max(norm_perc_bid)
        for i in range(len(bid_req_qty)):
            if self.slope_levels[i]>max_norm_perc:
                bid_req_qty[i] = (bid_req_qty[i]-bid_req_qty[0])*self.slope_levels[i]/max_norm_perc+bid_req_qty[0]
            
        self.depth_bid_vol = bid_req_qty
        self.depth_ask_vol = ask_req_qty
        
        
    
    def run(self,duration=np.inf):
        self.socket_manager = BinanceSocketManager(self.client)
        self.socket_key = self.socket_manager.start_symbol_ticker_socket(self.pair, lambda x: self.update_trades(x))
        self.socket_key_trade = self.socket_manager.start_trade_socket(self.pair, lambda x: self.update_trades(x))
        self.socket_key_orders = self.socket_manager.start_depth_socket(self.pair, lambda x: self.update_depth(x), depth=BinanceSocketManager.WEBSOCKET_DEPTH_20)
        self.socket_key_account = self.socket_manager.start_user_socket(lambda x: self.account_update(x))
        # Get latest wallets
        self.update_wallet()
        
        if self.plots:
            self.fig = plt.figure(1)
            plt.ion()
            self.fig_depth = plt.figure(2)
            plt.ion()
        # then start the socket manager
        self.socket_manager.start()
        self.timer = Timer(self.refresh_rate,lambda: self.trade_model())
        self.timer.start()

    def stop(self):
        self.timer.cancel()
        self.socket_manager.close()
        #self.socket_manager.stop_socket(self.socket_key)
        self.socket_key = None
        #self.socket_manager.stop_socket(self.socket_key_trade)
        self.socket_key_trade = None
        #self.socket_manager.stop_socket(self.socket_key_orders)
        self.socket_key_orders = None
        #self.socket_manager.stop_socket(self.socket_key_account)
        self.socket_key_account = None
        reactor.stop()
        self.socket_manager = None
        sys.exit()
    
    '''
    ========================================================================
    Model functions
    ========================================================================
    '''
    def loadNetwork(self,filename,modelname,layers):
        self.models[modelname] = {}
        temp = pd.read_excel(filename,sheet_name='InputConfig',header=None).values
        self.models[modelname]['in_gain'] = temp[:,0].reshape((temp.shape[0],1))
        self.models[modelname]['in_offset'] = temp[:,1].reshape((temp.shape[0],1))
        self.models[modelname]['in_ymin'] = -1
        layer_data = {}
        for i in range(layers):
            layer_data[i] = {}
            sheetname = 'Layer{:0>2n}'.format(i+1)
            temp = pd.read_excel(filename,sheet_name=sheetname,header=None).values
            layer_data[i]['weight'] = temp[:,:-1]
            layer_data[i]['bias'] = temp[:,-1].reshape((temp.shape[0],1))
            if i == layers-1:
                layer_data[i]['fcn'] = 'softmax'
            else:
                layer_data[i]['fcn'] = 'tanh'

        self.models[modelname]['layers'] = layer_data
        if modelname == "buy":
            self.buy_model_installed = True
        elif modelname == "sell":
            self.sell_model_installed = True
    
    def getValue(self):
        bid = self.data[-1,2]
        value = bid*self.coin_amt + self.base_amt
        base_value = bid*self.init_coin_amt + self.init_base_amt

        return (value - base_value)/base_value*100
    
    def validate_trades(self):
        if self.trade_data.shape[0] == 0:
            return
        
        return
        '''
        t = self.data[-1,0]
        valid = np.where(t - self.trade_data[:,0] < 300)
        for i in range(len(valid)):
            if self.trade_data[valid[i],-1] == False:
                if self.trade_data[valid[i],1] == 1:
                    # buy
                    if (self.data[-1,2] - self.trade_data[valid[i],2])/self.trade_data[valid[i],2] > self.fee_amt:
                        self.trade_data[valid[i],-1] = True
                else:
                    # sell
                    if (self.trade_data[valid[i],2] - self.data[-1,3])/self.trade_data[valid[i],2] > self.fee_amt:
                        self.trade_data[valid[i],-1] = True
        '''
        
        
    def trade_model(self):
        flag = 0
        self.timer = Timer(self.refresh_rate,lambda: self.trade_model())
        self.timer.start()
        if self.data.shape[0] == 0:
            return
        
        if self.trading_enabled == False and flag == 0:
            trade_time = self.client.get_server_time()
            trade_time = trade_time['serverTime']/1000
            
        if self.trading_enabled == True and self.trade_data.shape[0] > 0:
            if time.time() - self.trade_data[-1,0] < 0.25:
                print('{}  Updating too fast. Ignoring loop'.format(time.strftime('%H:%M:%S')))
                return
            
            
        if self.trading_enabled == True and self.trade_data.shape[0] > 0:
            if self.trade_data[-1,-3] == True: 
                if time.time() - self.trade_data[-1,0] > self.buyback_enable_thresh and self.buyback_enable == False:
                    print('{}  Enabling buyback'.format(time.strftime('%H:%M:%S')))
                    self.buyback_enable= True
            else:
                self.buyback_enable= True
        else:
            self.buyback_enable= True
            
            
        if self.trading_enabled == True and self.trade_data.shape[0] > 0:
            if self.trade_data[-1,-3] == False: 
                if time.time() - self.trade_data[-1,0] > self.cooldown_reset_thresh and self.cooldown_reset == False:
                    if self.cooldown_reset == False:
                        print('{}  Disabling cooldown'.format(time.strftime('%H:%M:%S')))
                        self.cooldown_reset= True
            else:
                if self.cooldown_reset == False:
                    print('{}  Disabling cooldown'.format(time.strftime('%H:%M:%S')))
                    self.cooldown_reset= True
        else:
            self.cooldown_reset= True
        
        #p = np.random.rand()
        buy_guidance = False
        sell_guidance = False
        need_log = False
        
        '''
        p = np.random.rand()
        if p > 0.9:
            buy_guidance = True
        elif p < 0.10:
            sell_guidance = True
        '''    
        if time.time() - self.data[-1,0] > self.timeout_thresh and time.time() - self.data[-1,0] < self.trade_socket_timeout and flag == 0:
            print('{}  Large time since server update. {:.2f} seconds'.format(time.strftime('%H:%M:%S'),time.time() - self.data[-1,0]))
            return
        elif time.time() - self.data[-1,0] >= self.trade_socket_timeout and flag == 0:
            print('{}  Restarting bot'.format(time.strftime('%H:%M:%S'),time.time() - self.data[-1,0]))
            self.stop()
            return
        else:
            if self.buy_model_installed and (self.cooldown_reset == True or (self.cooldown_reset == False and self.last_action == 'buy')):
                buy_guidance = self.eval_model('buy')
            if self.sell_model_installed and (self.cooldown_reset == True or (self.cooldown_reset == False and self.last_action == 'sell')):
                sell_guidance = self.eval_model('sell')

        bid = self.data[-1,2]
        ask = self.data[-1,3]
        
        if self.trading_enabled == True:
            if self.wallet_up_to_date == False:
                self.wallet_timeout_count += 1
                if self.wallet_timeout_count >= self.wallet_timeout_loops:
                    self.update_wallet()
                    if self.wallet_up_to_date == False:
                        print('{}  Wallet not updated. Shutting Down'.format(time.strftime('%H:%M:%S')))
                        # restart websockets
                        self.stop()
#                self.update_wallet()
#                self.socket_manager.stop_socket(self.socket_key_account)
#                self.socket_key_account = self.socket_manager.start_user_socket(lambda x: self.account_update(x))
                return
            else:
                self.wallet_timeout_count = 0
            self.bias = self.base_amt - float(self.target_mix*(bid*self.coin_amt + self.base_amt))
            self.sell_surplus = max(0,self.bias);
            self.buy_surplus = -min(0,self.bias);
        
        # Margin calculations
        norm_price = np.mean(self.data[-1,4:6].reshape((1,2)),axis=1).reshape((1,1))
        margin_unscaled = float(self.margin_scale*np.power(self.var_est_last/norm_price*100,self.margin_exp))/100.0
        self.margin = min(max(margin_unscaled,self.margin_min/100),self.margin_max/100)
        
        if abs(self.bias) < self.min_trade_amt:
            self.mean_sell_price = bid
            self.mean_buy_price = ask
            
        bid_qty = self.data[-1,6]
        ask_qty = self.data[-1,7]
        value = bid*self.coin_amt + self.base_amt
        #print('Gain: {:.3f}'.format((value-(bid*100 + 0.1))/(bid*100 + 0.1)*100))
        if sell_guidance == True and buy_guidance == False:
            # sell coin
            print('{}  Sell decision detected'.format(time.strftime('%H:%M:%S')))
            max_sell = min(self.bias_limit*value-self.bias,self.base_trade_amt*value)
            amt = min(max_sell,bid_qty*bid)
            if amt*(1-self.fee_amt) >= self.min_trade_amt:
                

                if self.trading_enabled == False:
                    if self.bias + amt*(1-self.fee_amt) > 0:
                        self.mean_sell_price = self.mean_sell_price*self.sell_surplus/(self.sell_surplus+amt*(1-self.fee_amt)) + bid*amt*(1-self.fee_amt)/(self.sell_surplus+amt*(1-self.fee_amt))
                    result = self.client.create_test_order(symbol=self.pair,side=SIDE_SELL,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/bid,2),price=str(bid),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                    self.coin_amt = self.coin_amt - amt/bid
                    self.base_amt = self.base_amt + amt*(1-self.fee_amt)
                    self.trade_data = np.append(self.trade_data,np.array([trade_time,-1,bid,amt,False,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                    print('Sell successful: {}@{}!'.format(round(amt/bid,2),bid))
                    self.cooldown_reset = False
                    self.buyback_enable= True
                    self.last_action = 'sell'
                else:
                    # Do something like initiate a trade
                    result = self.client.create_order(symbol=self.pair,side=SIDE_SELL,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/bid,2),price=str(bid),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                    coin_amt = float(result['executedQty'])
                    if coin_amt > 0:
                        amt = coin_amt*float(result['price'])
                        if self.bias + amt*(1-self.fee_amt) > 0:
                            self.mean_sell_price = self.mean_sell_price*self.sell_surplus/(self.sell_surplus+amt*(1-self.fee_amt)) + bid*amt*(1-self.fee_amt)/(self.sell_surplus+amt*(1-self.fee_amt))
                        print('{}  Sell successful: {}@{}!'.format(time.strftime('%H:%M:%S'),result['executedQty'],result['price']))
                        self.trade_data = np.append(self.trade_data,np.array([result['transactTime']/1000,-1,float(result['price']),float(result['executedQty']),False,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                        self.wallet_up_to_date = False
                        self.cooldown_reset = False
                        self.buyback_enable= True
                        need_log = True
                        self.last_action = 'sell'
                        self.base_amt = self.base_amt + coin_amt*float(result['price'])
                        self.coin_amt = self.coin_amt - coin_amt

        elif sell_guidance == False and buy_guidance == True:
            # buy coin
            print('{}  Buy decision detected'.format(time.strftime('%H:%M:%S')))
            max_buy = min(self.bias_limit*value+self.bias,self.base_trade_amt*value)
            amt = min(max_buy,ask_qty*ask)
            if amt >= self.min_trade_amt:
                
                    
                if self.trading_enabled == False:
                    if self.bias - amt < 0:
                        self.mean_buy_price = self.mean_buy_price*self.buy_surplus/(self.buy_surplus+amt) + ask*amt/(self.buy_surplus+amt)
                    result = self.client.create_test_order(symbol=self.pair,side=SIDE_BUY,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/ask,2),price=str(ask),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                    self.base_amt = self.base_amt - amt
                    self.coin_amt = self.coin_amt + amt/ask*(1-self.fee_amt)
                    self.trade_data = np.append(self.trade_data,np.array([trade_time,1,ask,amt,False,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                    print('Buy successful {}@{}!'.format(round(amt/ask,2),ask))
                    self.cooldown_reset = False
                    self.buyback_enable= True
                    self.last_action = 'buy'
                else:
                    # Do something like initiate a trade
                    result = self.client.create_order(symbol=self.pair,side=SIDE_BUY,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/ask,2),price=str(ask),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                    coin_amt = float(result['executedQty'])
                    if coin_amt > 0:
                        amt = coin_amt*float(result['price'])
                        if self.bias - amt < 0:
                            self.mean_buy_price = self.mean_buy_price*self.buy_surplus/(self.buy_surplus+amt) + ask*amt/(self.buy_surplus+amt)
                        print('{}  Buy successful: {}@{}!'.format(time.strftime('%H:%M:%S'),result['executedQty'],result['price']))
                        self.trade_data = np.append(self.trade_data,np.array([result['transactTime']/1000,1,float(result['price']),float(result['executedQty']),False,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                        self.wallet_up_to_date = False
                        self.cooldown_reset = False
                        self.buyback_enable= True
                        need_log = True
                        self.last_action = 'buy'
                        self.base_amt = self.base_amt - coin_amt*float(result['price'])
                        self.coin_amt = self.coin_amt + coin_amt
        elif self.buyback_enable == True:
            # do the partial buyback
            if self.bias < 0:
                # need to sell coin
                if (bid-self.mean_buy_price)/self.mean_buy_price >= self.margin + self.fee_amt:
                    # sell coin
                    max_sell = -min(-self.min_trade_amt,self.buyback_ratio*self.bias)
                    amt = min(max_sell,bid_qty*bid)
                    if amt*(1-self.fee_amt) >= self.min_trade_amt:
                        
                        if self.trading_enabled == False:
                            if self.bias + amt*(1-self.fee_amt) > 0:
                                self.mean_sell_price = self.mean_sell_price*self.sell_surplus/(self.sell_surplus+amt*(1-self.fee_amt)) + bid*amt*(1-self.fee_amt)/(self.sell_surplus+amt*(1-self.fee_amt))

                            result = self.client.create_test_order(symbol=self.pair,side=SIDE_SELL,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/bid,2),price=str(bid),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                            self.coin_amt = self.coin_amt - amt/bid
                            self.base_amt = self.base_amt + amt*(1-self.fee_amt)
                            self.trade_data = np.append(self.trade_data,np.array([trade_time,-1,bid,amt,True,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                            print('Sellback successful: {}@{}!'.format(round(amt/bid,2),bid))
                        else:
                            # Do something like initiate a trade
                            result = self.client.create_order(symbol=self.pair,side=SIDE_SELL,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/bid++self.coin_precision,2),price=str(bid),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                            coin_amt = float(result['executedQty'])
                            if coin_amt > 0:
                                amt = coin_amt*float(result['price'])
                                if self.bias + amt*(1-self.fee_amt) > 0:
                                    self.mean_sell_price = self.mean_sell_price*self.sell_surplus/(self.sell_surplus+amt*(1-self.fee_amt)) + bid*amt*(1-self.fee_amt)/(self.sell_surplus+amt*(1-self.fee_amt))
                                    
                                print('{}  Sellback successful: {}@{}!'.format(time.strftime('%H:%M:%S'),result['executedQty'],result['price']))
                                self.trade_data = np.append(self.trade_data,np.array([result['transactTime']/1000,-1,float(result['price']),float(result['executedQty']),True,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                                self.buyback_enable = False           
                                self.wallet_up_to_date = False
                                need_log = True
                                self.base_amt = self.base_amt + coin_amt*float(result['price'])
                                self.coin_amt = self.coin_amt - coin_amt
                        

            elif self.bias > 0:
                # need to buy coin
                if (self.mean_sell_price-ask)/self.mean_sell_price >=self.margin + self.fee_amt:
                    # buy coin
                    
                    max_buy = max(self.min_trade_amt,self.buyback_ratio*self.bias)
                    amt = min(max_buy,ask_qty*ask)
                    
                    if amt >= self.min_trade_amt:
                        if self.trading_enabled == False:
                            if self.bias - amt < 0:
                                self.mean_buy_price = self.mean_buy_price*self.buy_surplus/(self.buy_surplus+amt) + ask*amt/(self.buy_surplus+amt)
                            
                            result = self.client.create_test_order(symbol=self.pair,side=SIDE_BUY,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/ask,2),price=str(ask),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                            self.base_amt = self.base_amt - amt
                            self.coin_amt = self.coin_amt + amt/ask*(1-self.fee_amt)
                            self.trade_data = np.append(self.trade_data,np.array([trade_time,1,ask,amt,True,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                            print('Buyback successful {}@{}!'.format(round(amt/ask,2),ask))
                        else:
                            # Do something like initiate a trade
                            result = self.client.create_order(symbol=self.pair,side=SIDE_BUY,type=ORDER_TYPE_LIMIT,timeInForce=TIME_IN_FORCE_IOC,quantity=round(amt/ask+self.coin_precision,2),price=str(ask),newOrderRespType=ORDER_RESP_TYPE_RESULT)
                            coin_amt = float(result['executedQty'])
                            
                            if coin_amt > 0:
                                amt = coin_amt*float(result['price'])
                                if self.bias - amt < 0:
                                    self.mean_buy_price = self.mean_buy_price*self.buy_surplus/(self.buy_surplus+amt) + ask*amt/(self.buy_surplus+amt)
                                    
                                print('{}  Buyback successful: {}@{}!'.format(time.strftime('%H:%M:%S'),result['executedQty'],result['price']))
                                self.trade_data = np.append(self.trade_data,np.array([result['transactTime']/1000,1,float(result['price']),float(result['executedQty']),True,self.coin_amt,self.base_amt]).reshape((1,7)),axis=0)
                                self.buyback_enable = False
                                self.wallet_up_to_date = False
                                need_log = True
                                self.base_amt = self.base_amt - coin_amt*float(result['price'])
                                self.coin_amt = self.coin_amt + coin_amt
        
        # BNB purchase if the amt in the wallet is lower than a specific amount
        if self.fee_coin_amt < self.min_bnb and self.trading_enabled:
            # buy more BNB with BTC
            result = self.client.order_market_buy(symbol='BNBBTC',quantity=self.bnb_buy_amt)
            amt = float(result['executedQty'])
            if amt > 0:
                print('{}  Bought BNB: {}@{}!'.format(time.strftime('%H:%M:%S'),result['executedQty'],result['price']))
                self.wallet_up_to_date = False
        
        self.save_state()            
        if need_log == True:
            toWrite = pd.DataFrame(data = self.trade_data)
            toWrite.to_csv(self.trade_log_file,header = False,index = False)
            print('{}  Total gain: {:.2f}%'.format(time.strftime('%H:%M:%S'),self.getValue()))
        
                         
    def processLatestData(self,playbackFlag = False):
        row_data = self.data[-1,2:6].reshape((1,4))
        in_size = 58
        # Check time since last data point. If more than        
        if self.data.shape[0] > 1:
            dt = self.data[-1,0] - self.data[-2,0]
        else:
            dt = 1000
        if dt > self.reset_timeout and playbackFlag == False:
            print('Timeout detected a large time gap since last update. Historical data will be reset.')
            self.hist_data = np.zeros((0,in_size))
        
        num_hist = self.hist_data.shape[0]
        
        # Do the EMA for the data in row_data
        
        curr_mean_price = self.data[-1,1].reshape((1,1))
        norm_price = np.mean(self.data[-1,4:6].reshape((1,2)),axis=1).reshape((1,1))
        if num_hist >= 1:
            last_mean_price = self.data[-2,1].reshape((1,1))
        else:
            last_mean_price = np.nan
        
        ff = self.ema_filts
        temp = np.zeros((1,len(ff)))
        for i in range(len(ff)):
            if num_hist >= 1:
                if dt < 0.1:
                    dt = 0.1
                a = dt / (1/ff[i])
                u = np.exp(-a)
                v = (1.0-u)/a
                last_sample = self.ema_bids_last[i]
                self.ema_bids_last[i] = u*self.ema_bids_last[i] + (v-u)*last_mean_price + (1.0-v)*curr_mean_price
                temp[0,i] = (v-u)*last_mean_price + (1.0-v)*curr_mean_price
                row_data = np.append(row_data,self.ema_bids_last[i],axis=1)
                
                if i == 1:
                    curr_diff = abs(curr_mean_price - self.ema_bids_last[i])
                    last_diff = abs(last_mean_price - last_sample)
                    
                    if curr_diff > self.var_est_last:
                        a = dt / (1/self.variance_ff[0])
                    else:
                        a = dt / (1/self.variance_ff[1])
                        
                    u = np.exp(-a)
                    v = (1.0-u)/a
                    self.var_est_last = u*self.var_est_last + (v-u)*last_diff + (1.0-v)*curr_diff
                
                
            else:
                row_data = np.append(row_data,curr_mean_price,axis=1)
                self.ema_bids_last[i] = curr_mean_price
                
                if i == 1:
                    self.var_est_last = 0

        
        n = row_data.shape[1]
        for i in range(n):
            for j in range(i+1,n):
                row_data = np.append(row_data,row_data[0,i].reshape((1,1))-row_data[0,j].reshape((1,1)),axis=1)
        # normalization
        row_data = np.divide(row_data,norm_price)
        # add the variance data
        row_data = np.append(row_data,self.var_est_last/norm_price*100.0,axis=1)
        
        n2 = row_data.shape[1]
        for i in range(n2):
            if num_hist >= 1:
                row_data = np.append(row_data,(row_data[0,i].reshape((1,1))-self.hist_data[-1,i].reshape((1,1)))/dt,axis=1)
            else:
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)

        
        ''' Derivative stuff
        n2 = row_data.shape[1]
        for i in range(n2):
            if num_hist >= 1:
                row_data = np.append(row_data,row_data[0,i].reshape((1,1))-self.hist_data[-1,i].reshape((1,1)),axis=1)
                row_data = np.append(row_data,row_data[0,-1].reshape((1,1))-self.hist_data[-1,row_data.shape[1]-1].reshape((1,1)),axis=1)
            else:
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)
            
            if num_hist >= 10:
                row_data = np.append(row_data,row_data[0,i].reshape((1,1))-self.hist_data[-10,i].reshape((1,1)),axis=1)
            else:
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)
            
            if num_hist >= 30:
                row_data = np.append(row_data,row_data[0,i].reshape((1,1))-self.hist_data[-30,i].reshape((1,1)),axis=1)
            else:
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)
                
            if num_hist >= 100:
                row_data = np.append(row_data,row_data[0,i].reshape((1,1))-self.hist_data[-100,i].reshape((1,1)),axis=1)
            else:
                row_data = np.append(row_data,np.array(np.nan).reshape((1,1)),axis=1)
        
        
        '''   
        
        if num_hist == self.max_hist_len:
            self.hist_data = np.append(self.hist_data[1:,:].reshape((self.max_hist_len-1,in_size)),row_data,axis=0)
        else:
            self.hist_data = np.append(self.hist_data,row_data,axis=0)
        
    
    def saveSnapshot(self):
        None

    
    def eval_model(self,model_name):
        num_layers = len(self.models[model_name]['layers'])
        if self.hist_data.shape[0] < 2:
            return False
        #print(self.hist_data.shape)
        a = self.hist_data[-1,:]
        #pdb.set_trace()
        a = a.transpose().reshape((58,1))
        # Enable conditions for evaluating the buy or sell models

        #pdb.set_trace()
        if np.any(np.isnan(a)):
            return False
        if model_name == 'buy':
            # last ask was lower than previous ask and ema ask is decreasing
            if self.data[-1,3] > self.data[-2,3] or self.data[-1,5] >= self.data[-2,5]:
                return False
        elif model_name == 'sell':
            if self.data[-1,2] < self.data[-2,2] or self.data[-1,4] <= self.data[-2,4]:
                return False
        #print('Running {} model...'.format(model_name))
        
        # scale inputs
        a = np.multiply(np.subtract(a,self.models[model_name]['in_offset']),self.models[model_name]['in_gain']) + self.models[model_name]['in_ymin']
        for L in range(0,num_layers):
            z = np.matmul(self.models[model_name]['layers'][L]['weight'],a) + self.models[model_name]['layers'][L]['bias']
            if self.models[model_name]['layers'][L]['fcn'] == 'tanh':
                a = np.tanh(z)
            elif self.models[model_name]['layers'][L]['fcn'] == 'softmax':
                a = MrKrabs2.softmax(z)
        #print('{} decision is {}'.format(model_name,a[1,0] > a[0,0]))
        return a[1,0] > a[0,0]
        
    def eval_model_playback(self,model_name,data):
        num_layers = len(self.models[model_name]['layers'])
        if data.shape[0] < 2:
            return False
        #print(self.hist_data.shape)
        a = data
        a = a.transpose()

        # scale inputs
        a = np.multiply(np.subtract(a,self.models[model_name]['in_offset']),self.models[model_name]['in_gain']) + self.models[model_name]['in_ymin']
        for L in range(0,num_layers):
            z = np.matmul(self.models[model_name]['layers'][L]['weight'],a) + self.models[model_name]['layers'][L]['bias']
            if self.models[model_name]['layers'][L]['fcn'] == 'tanh':
                a = np.tanh(z)
            elif self.models[model_name]['layers'][L]['fcn'] == 'softmax':
                a = MrKrabs2.softmax(z)
        #print('{} decision is {}'.format(model_name,a[1,0] > a[0,0]))
        return a
    
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x,axis=0,keepdims=True))
        return e_x / np.sum(e_x,axis=0,keepdims=True)