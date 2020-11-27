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
from decimal import *
import os

class Rebalancer:

    interval = 1/60 # hours
    
    fudge_factor = 1.01 # percent extra 
    
    fee_amt = 0.10/100 # fee amt
    margin = 0.01-(0.10/100) # amount over the fee amount that will trigger a trade (0.05%)
    
    max_margin = 1 # was 0.01
    norm_margin = 0.04 # tolerance normalized
    
    coin_data = {}

    def __init__(self,keys,base_currency,ratio_worksheet):
        Client.API_URL = 'https://api.binance.us/api'
        Client.PUBLIC_API_VERSION = 'v3'
        Client.WITHDRAW_API_URL = 'https://api.binance.us/wapi'
        Client.WEBSITE_URL = 'https://www.binance.us'
        
        self.client = Client(keys[0][0],keys[0][1])

	# some US specific things
        #self.client.API_URL = 'https://api.binance.us/api'
        #self.client.WITHDRAW_API_URL = 'https://api.binance.us/wapi'
        #self.client.WEBSITE_URL = 'https://www.binance.us'

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
        self.text_log = '../logs/rebalance_summary.txt'
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
        current_buy = self.get_property_list('last_value_buy')
        current_sell = self.get_property_list('last_value_sell')
        min_trade = self.get_property_list('min_trade_amt')
        markets = self.get_property_list('market_name')
        
        coin_names = list(self.coin_data.keys())
        
        delta = []
        norm_delta = []
        score = []
        for i in range(len(target)):
            m = min_trade[i][0]
            raw_delta = 0
            if target[i] > current_buy[i]:
                raw_delta = target[i]-current_buy[i]
            elif target[i] < current_sell[i]:
                raw_delta = target[i]-current_sell[i]
            abs_delta = abs(raw_delta)
            for j in range(1,len(min_trade[i])):
                m *= self.fudge_factor
                if raw_delta > 0:
                    # need to buy, will be selling bnb
                    m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'buy'))
                else:
                    # need to sell coin, will be buying bnb
                    m = max(min_trade[i][j],min_trade[i][j-1]*self.lookup_price(markets[i][j],'sell'))
                
            score.append(raw_delta/m)
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
        self.write_text_file(total_value)
        
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
            flag = False;
            sell = self.lookup_price(self.coin_data[coin]['market_name'][i],SIDE_SELL)
            buy = self.lookup_price(self.coin_data[coin]['market_name'][i],SIDE_BUY)
            if (buy-sell)/sell > self.max_margin:
                print('Trade margins too high. Skipping...')
                flag = True
            for j in range(i+1,len(self.coin_data[coin]['market_name'])):
                sell = self.lookup_price(self.coin_data[coin]['market_name'][j],SIDE_SELL)
                buy = self.lookup_price(self.coin_data[coin]['market_name'][j],SIDE_BUY)
                if (buy-sell)/sell > self.max_margin:
                    print('Trade margins too high. Skipping...')
                    flag = True
                trade_amt /= self.lookup_price(self.coin_data[coin]['market_name'][j],trade_side)

            coin_amt = self.get_coin_amt_from_market(self.coin_data[coin]['market_name'][i],trade_amt,trade_side,update_flag=True,factor=use_factor)
            use_factor = False
            
            if coin_amt > 0 and flag == False:
                
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
                prec = Decimal(market2['filters'][2]['stepSize'].rstrip('0'))
                min_amt = Decimal(market2['filters'][3]['minNotional'].rstrip('0'))
                break
        
        if factor:
            trade_amt_coin = Decimal(base_amt/self.lookup_price(market,side)).quantize(prec,ROUND_UP)
        else:
            trade_amt_coin = Decimal(base_amt/self.lookup_price(market,side)).quantize(prec,ROUND_DOWN)
        f = Decimal(1.0)
        if factor:
            f = Decimal(self.fudge_factor)
        if trade_amt_coin*Decimal(self.lookup_price(market,side)) < min_amt*f:
            #print("{} {}".format(trade_amt_coin*self.lookup_price(market,side),min_amt))
            return 0
        
        return float(trade_amt_coin)
        
        
        
    def get_market_info(self,coin):
        des_market_name = coin.lower()+self.base_coin.lower()
        print(coin);
        for market in self.exchange_info['symbols']:
            if market['symbol'].lower() == des_market_name:
                self.coin_data[coin]['coin_precision'] = [float(market['filters'][2]['stepSize'])]
                self.coin_data[coin]['min_trade_amt'] = [float(market['filters'][3]['minNotional'])]
                self.coin_data[coin]['market_name'] = [market['symbol']]
                return
        
        # if it's here then it's not a direct market
        des_market_name = coin.lower()+'usdt'
        for market in self.exchange_info['symbols']:
            if market['symbol'].lower() == des_market_name:
                self.coin_data[coin]['coin_precision'] = [float(market['filters'][2]['stepSize'])]
                self.coin_data[coin]['min_trade_amt'] = [float(market['filters'][3]['minNotional'])]
                self.coin_data[coin]['market_name'] = [market['symbol']]
                break
         
        des_market_name = 'usdt'+self.base_coin.lower()
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
    def get_coin_value_buy(self,coin,amt=-1):
        if amt < 0:
            amt = self.lookup_amt(coin)
        value = amt
        if coin != self.base_coin:
            for i in range(len(self.coin_data[coin]['market_name'])):
                value *= self.lookup_price(self.coin_data[coin]['market_name'][i],'buy')
        return value
    def get_coin_value_sell(self,coin,amt=-1):
        if amt < 0:
            amt = self.lookup_amt(coin)
        value = amt
        if coin != self.base_coin:
            for i in range(len(self.coin_data[coin]['market_name'])):
                value *= self.lookup_price(self.coin_data[coin]['market_name'][i],'sell')
        return value
    
    def get_values(self,verbose = True):
        self.update_wallet()
        if not self.wallet_updated:
            return -1
        self.update_prices()
        total_value = 0
        for coin in self.coin_data:
            if 'amt' in self.coin_data[coin]:
                value = self.get_coin_value(coin,self.lookup_amt(coin))
                buy_val = self.get_coin_value_buy(coin,self.lookup_amt(coin))
                sell_val = self.get_coin_value_sell(coin,self.lookup_amt(coin))
                if verbose:
                    print("{}: {}".format(coin,value))
                self.coin_data[coin]['last_value'] = value
                self.coin_data[coin]['last_value_buy'] = buy_val
                self.coin_data[coin]['last_value_sell'] = sell_val
                total_value = total_value + value
        if self.init_total_value != 0:
            idx = 0
            hodl_portfolio_value = 0
            for coin in self.coin_data:
                hodl_portfolio_value += self.get_coin_value(coin,self.init_coin_amt[idx])
                idx += 1
            self.norm_gain = (total_value-self.total_fees-hodl_portfolio_value)/hodl_portfolio_value*100
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
            
    def write_text_file(self,total_value):
        fid = open(self.text_log,'w+')
        fid.write('Total:       $%.2f\n' % total_value)
        fid.write('Gain:         %.2f%%\n' % self.total_gain)
        fid.write('Norm. Gain:   %.3f%%\n' % self.norm_gain)
        fid.close()

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

