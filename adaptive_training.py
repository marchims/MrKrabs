# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:51:54 2018

@author: marchims
"""

import numpy as np
import pandas as pd
import datetime
from binance.client import Client
import xlsxwriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import operator
import DataFormatting

def read_hist_data(pair,minutes):
    klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, "{} minutes ago UTC".format(minutes))
    klines = np.array(klines)
    klines = klines.astype('float')
    return klines

def read_training_data(pair,t1,*args):
    if len(args)>0:
        klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, t1,args[0])
    else:
        klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, t1)
    klines = np.array(klines)
    klines = klines.astype('float')
    return klines

def initialize_weights(num_in,num_hidden,num_out):
    W = {}
    b = {}
    W['layer0'] = tf.Variable(tf.random_normal([num_in,num_hidden[0]], seed=seed, stddev=math.sqrt(2/num_in)))
    b['layer0'] = tf.Variable(tf.zeros([1,num_hidden[0]]))
    for L in range(1,len(num_hidden)):
        W['layer{}'.format(L)] = tf.Variable(tf.random_normal([num_hidden[L-1],num_hidden[L]], seed=seed, stddev=math.sqrt(2/num_hidden[L-1])))
        b['layer{}'.format(L)] = tf.Variable(tf.zeros([1,num_hidden[L]]))
    W['layer{}'.format(len(num_hidden))] = tf.Variable(tf.random_normal([num_hidden[-1],num_out], seed=seed, stddev=math.sqrt(2/num_hidden[-1])))
    b['layer{}'.format(len(num_hidden))] = tf.Variable(tf.zeros([1,num_out]))
    return W,b

def forward_prop(x,W,b,keep_prob):
    z = {}
    a = {}
    z['layer0'] = tf.add(tf.matmul(x, W['layer0']), b['layer0'])
    a['layer0'] = tf.nn.tanh(z['layer0'])
    a['layer0'] = tf.nn.dropout(a['layer0'], keep_prob,seed=seed)
    for L in range(1,len(W)-1):
        z['layer{}'.format(L)] = tf.add(tf.matmul(a['layer{}'.format(L-1)], W['layer{}'.format(L)]), b['layer{}'.format(L)])
        a['layer{}'.format(L)] = tf.nn.tanh(z['layer{}'.format(L)])
        a['layer{}'.format(L)] = tf.nn.dropout(a['layer{}'.format(L)],keep_prob,seed=seed)
    
    z['layer{}'.format(len(W)-1)] = tf.add(tf.matmul(a['layer{}'.format(len(W)-2)], W['layer{}'.format(len(W)-1)]), b['layer{}'.format(len(W)-1)])
    a['layer{}'.format(len(W)-1)] = z['layer{}'.format(len(W)-1)]
    return z,a

def getWeights(W,b):
    with sess.as_default():
        weight = []
        bias = []
        for i in range(len(W)):
            weight.append( W['layer{}'.format(i)].eval())
            bias.append(b['layer{}'.format(i)].eval())
            
    return weight,bias

def eval_model(x,W,b):
    z = {}
    a = {}
    z['layer0'] = np.matmul(x, W[0]) + b[0]
    a['layer0'] = np.tanh(z['layer0'])
    for L in range(1,len(W)-1):
        z['layer{}'.format(L)] = np.matmul(a['layer{}'.format(L-1)], W[L]) + b[L]
        a['layer{}'.format(L)] = np.tanh(z['layer{}'.format(L)])
    
    z['layer{}'.format(len(W)-1)] = np.matmul(a['layer{}'.format(len(W)-2)], W[len(W)-1]) + b[len(W)-1]
    a['layer{}'.format(len(W)-1)] = z['layer{}'.format(len(W)-1)]
    
    output = a['layer{}'.format(len(W)-1)]
    return output
    
def train_network(sess,input_scaled,targets_index,train_idx,val_idx):

    with sess.as_default():
        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize
        min_val_cost = np.inf
        val_count = 0
        for epoch in range(epochs):
        
            _, avg_cost = sess.run([optimizer, cost], feed_dict = {x: input_scaled[train_idx], y: targets_index[train_idx]})
            
            # Validation
            val_cost = cost.eval({x: input_scaled[val_idx], y: targets_index[val_idx]})
            if val_cost < min_val_cost:
                min_val_cost = val_cost
                val_count = 0
            else:
                val_count += 1
                if val_count >= val_count_max:
                    #print("Epoch:{} \t\t Cost:{:.9f} \t\t Val Cost:{:.9f}".format(epoch+1,avg_cost,min_val_cost))
                    print("\nValidation minimized!")
                    break
            
            
            if epoch%1000==0:
                pred_temp = tf.equal(tf.argmax(output_layer, 1), y)
                accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
                train_acc = accuracy.eval({x: input_scaled[train_idx], y: targets_index[train_idx]})
                val_acc = cost.eval({x: input_scaled[val_idx], y: targets_index[val_idx]})
                print("Epoch:{} \t\t Cost:{:.4f} \t\t Training Accuracy:{:.4f} \t\t Validation Accuracy:{:.4f}".format(epoch+1,avg_cost,train_acc,val_acc))
                #print("Epoch:{} \t\t Cost:{:.9f} \t\t Val Cost:{:.9f} \t\t Count:{}".format(epoch+1,avg_cost,val_cost,val_count))
    
    print("\nTraining complete!")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return e_x / np.sum(e_x,axis=1,keepdims=True)

def test_model(inputs_scaled,targets_buy,targets_sell,actual_price,max_price,min_price,time_ax,weight_buy,bias_buy,weight_sell,bias_sell):

    input_filter = 0.1
    
    targets_index_buy = np.zeros(len(targets_buy))
    targets_index_sell = np.zeros(len(targets_sell))
    for  i in range(len(targets_buy)):
        targets_index_buy[i], _ = max(enumerate(targets_buy[i]), key=operator.itemgetter(1))
        targets_index_sell[i], _ = max(enumerate(targets_sell[i]), key=operator.itemgetter(1))

    batch_size = int(t_forecast*60.0/dt)
    input_row_buy = np.copy(inputs_scaled[0,:])
    input_row_buy = input_row_buy.reshape((1,len(input_row_buy)))
    #input_row_buy[0,:7] = np.zeros((1,7))
    #input_row_buy[0,0] = 0
    input_row_sell = np.copy(inputs_scaled[0,:])
    input_row_sell = input_row_sell.reshape((1,len(input_row_sell)))
    #input_row_sell[0,:7] = np.zeros((1,7))
    #input_row_sell[0,0] = 0
    L = targets_buy.shape[0]
    est_perc_buy = np.zeros((L,1))
    est_perc_sell = np.zeros((L,1))
    p_arr_buy = np.array(p_ax)
    p_arr_sell = -np.array(p_ax)
    index = 0
    plt.figure(2)
    plt.ion()
    ax1 = plt.subplot(311)
    plt.plot(time_ax/1000,min_price,'r.',time_ax/1000,max_price,'g.')
    ax2 = plt.subplot(312,sharex=ax1)
    ax3 = plt.subplot(313,sharex=ax1)
    i = 0
    batch = 0
    closed_loop_err_buy = []
    closed_loop_err_sell = []
    while i < L:
        batch_stop = min(i+batch_size,L)
        r = range(i,batch_stop)
        index = 0
        model_out_buy = np.zeros((len(r),num_out))
        model_out_sell = np.zeros((len(r),num_out))
        while i < batch_stop:
            model_out_buy[index,:] = softmax(eval_model(input_row_buy,weight_buy,bias_buy))
            model_out_sell[index,:] = softmax(eval_model(input_row_sell,weight_sell,bias_sell))
            if i == L-1:
                i += 1
                break
            input_row_buy[0,0:] = inputs_scaled[i+1,0:]
            input_row_sell[0,0:] = inputs_scaled[i+1,0:]
            '''         
            input_row_buy[0,1:] = np.copy(inputs_scaled[i+1,1:])
            input_row_sell[0,1:] = np.copy(inputs_scaled[i+1,1:])
            a = input_row_buy[0,0]*(max_x_buy[0,0]-min_x_buy[0,0])+min_x_buy[0,0]
            b = np.matmul(model_out_buy[index,:],p_arr_buy)
            c = (1-input_filter)*a + input_filter*b
            c1 = c-min_x_buy[0,0]
            c2 = c1/(max_x_buy[0,0]-min_x_buy[0,0])
            input_row_buy[0,0] = c2
            #print("{} \t {} \t {} \t {}".format(a,b,c,c2))
            '''
            '''
            a = input_row_sell[0,0]*(max_x_sell[0,0]-min_x_sell[0,0])+min_x_sell[0,0]
            b = np.matmul(model_out_sell[index,:],p_arr_sell)
            c = (1-input_filter)*a + input_filter*b
            c1 = c-min_x_sell[0,0]
            c2 = c1/(max_x_sell[0,0]-min_x_sell[0,0])
            input_row_sell[0,0] = c2
            '''
            '''
            input_row_buy[0,num_out:] = inputs_scaled[i+1,num_out:]
            input_row_sell[0,num_out:] = inputs_scaled[i+1,num_out:]
            input_row_buy[0,:num_out] = (1-input_filter)*input_row_buy[0,:num_out].reshape((1,num_out)) + input_filter*model_out_buy[index,:]
            input_row_sell[0,:num_out] = (1-input_filter)*input_row_sell[0,:num_out].reshape((1,num_out)) + input_filter*model_out_sell[index,:]
            '''
            i += 1
            index += 1
            
        est_perc_buy = np.matmul(model_out_buy,p_arr_buy)
        expected_perc_buy = np.matmul(targets_buy[r,:],p_arr_buy)
        closed_loop_err_buy.append(np.mean(np.power(est_perc_buy-expected_perc_buy,2)))
        
        est_perc_sell = np.matmul(model_out_sell,p_arr_sell)
        expected_perc_sell = np.matmul(targets_sell[r,:],p_arr_sell)
        closed_loop_err_sell.append(np.mean(np.power(est_perc_sell-expected_perc_sell,2)))
    
    
    
        print("Batch {}: \t\t Closed Loop Buy {:.4f} \t\t Closed Loop Sell {:.4f}".format(batch+1,closed_loop_err_buy[-1],closed_loop_err_sell[-1],))
        plt.subplot(312,sharex=ax1)
        plt.plot(time_ax[r]/1000,expected_perc_buy,'go',time_ax[r]/1000,est_perc_buy,'rx')
        plt.subplot(313,sharex=ax1)
        plt.plot(time_ax[r]/1000,expected_perc_sell,'go',time_ax[r]/1000,est_perc_sell,'rx')
        plt.pause(0.05)
        plt.draw()
        batch += 1
            
    return closed_loop_err_buy,closed_loop_err_sell

def train_adaptive(sess,inputs_scaled,targets,actual_price,max_price,time_ax,trainFlag,learning_rate):
    with sess.as_default():
        if trainFlag:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        targets_index = np.zeros(len(targets))
        for  i in range(len(targets)):
            targets_index[i], _ = max(enumerate(targets[i]), key=operator.itemgetter(1))
    
        batch_size = int(t_forecast*60.0/dt)
        input_row = inputs_scaled[0,:]
        input_row = input_row.reshape((1,len(input_row)))
        input_row[0,:7] = np.zeros((1,7))
        input_row[0,0] = 1
        L = targets.shape[0]
        est_perc = np.zeros((L,1))
        p_arr = np.array(p_ax)
        index = 0
        plt.figure(1)
        plt.ion()
        ax1 = plt.subplot(211)
        plt.plot(time_ax/1000,actual_price,'.',time_ax/1000,max_price,'.')
        ax2 = plt.subplot(212,sharex=ax1)
        i = 0
        batch = 0
        closed_loop_err = []
        open_loop_err = []
        while i < L:
            batch_stop = min(i+batch_size,L)
            r = range(i,batch_stop)
            weight,bias = getWeights(W,b)
            index = 0
            model_out = np.zeros((len(r),num_out))
            while i < batch_stop:
                model_out[index,:] = softmax(eval_model(input_row,weight,bias))
                if i == L-1:
                    i += 1
                    break
                input_row = inputs_scaled[i+1,:]
                input_row = input_row.reshape((1,len(input_row)))
                input_row[0,:num_out] = (input_row[0,:num_out].reshape((1,num_out))+model_out[index,:])/2
                i += 1
                index += 1
                
            est_perc = np.matmul(model_out,p_arr)
            expected_perc = np.matmul(targets[r,:],p_arr)
            closed_loop_err.append(np.mean(np.power(est_perc-expected_perc,2)))
            
            open_loop_err.append(cost.eval({x: inputs_scaled[r,:], y: targets_index[r]}))
            if trainFlag:
                _, avg_cost = sess.run([optimizer, cost], feed_dict = {x: inputs_scaled[r,:], y: targets_index[r]})
            
            print("Batch {}: \t\t Closed Loop Err {:.4f} \t\t Open Loop Err {:.4f}".format(batch+1,closed_loop_err[-1],open_loop_err[-1]))
            plt.plot(time_ax[r]/1000,expected_perc,'go')
            plt.plot(time_ax[r]/1000,est_perc,'rx')
            plt.pause(0.05)
            plt.draw()
            batch += 1
            
    return open_loop_err,closed_loop_err

def deploy_model(t0,input_row,min_x,max_x,W,b,batch_size=30):
    
    p_arr = np.array(p_ax)
    plt.figure()
    ax1 = plt.subplot(211)
    plt.title('Price')
    ax2 = plt.subplot(212,sharex=ax1)
    plt.title('30 minute price increase')
    while True:
        input_row = DataFormatting.scale_inputs(input_row,min_x,max_x)
        model_out = softmax(eval_model(input_row,W,b))


        # get latest data
        input_row = inputs_scaled[i+1,:]
        input_row = input_row.reshape((1,len(input_row)))
        input_row[0,:num_out] = (input_row[0,:num_out].reshape((1,num_out))+model_out[index,:])/2
        i += 1
        index += 1


def run_model_v2(inputs,outputs,time_ax,price_indexed,W,b,coin_start,base_start,ki=0.10):
    coin = coin_start
    base = base_start
    coin_best = coin_start
    base_best = base_start
    max_perc_trade = 0.25
    min_coin_perc = 0.0
    min_trade_amt = 0.001
    fee_perc = 0.1/100
    model_out = softmax(eval_model(inputs,W,b))
    m = inputs.shape[0]
    model_index = np.zeros(m)
    confidence =  np.zeros(m)
    for  i in range(m):
        model_index[i], _ = max(enumerate(model_out[i]), key=operator.itemgetter(1))
        confidence[i] = 1#max(0,(1.0/0.50)*(np.max(model_out[i,:])-0.50))
    
    print(confusion_matrix(outputs,model_index))
    
    plt.figure()
    ax1 = plt.subplot(511)
    plt.plot(mean_time,mean_price,'.-',time_ax,price_indexed,'rx')
    plt.subplot(512,sharex=ax1)
    plt.plot(time_ax,targets_index,'go',time_ax,model_index,'rx')
    plt.subplot(513,sharex=ax1)
    plt.subplot(514,sharex=ax1)
    plt.subplot(515,sharex=ax1)
    m = len(model_index)
    total = base + coin*price_indexed[i]
    total_best = base_best+coin_best*price_indexed[i]
    base_cusum = 0.0
    base_cusum_best = 0.0
    cusum_decay = 0.999
    acc_count = 0
    for i in range(m):
        
        if i == 0:
            perc_trade = max_perc_trade*confidence[i]
        else:
            perc_trade = max_perc_trade*confidence[i]
            #perc_trade = min(max_perc_trade,abs((price_indexed[i]-price_indexed[i-1])/price_indexed[i]))
        
        #trade_amt = 0.1*base
        if model_index[i] == outputs[i]:
            acc_count += 1
        if model_index[i]==0:
            # buy
            '''
            perc_trade = min(base/total,max_perc_trade)
            trade_amt =  max(min_trade_amt,1*perc_trade*total)
            '''
            trade_amt = max(min_trade_amt,ki*base_cusum+(1)*perc_trade*total)
            '''
            trade_amt = max(min_trade_amt,ki*base_cusum+perc_trade*base)
            '''
            if base >= trade_amt and trade_amt >= min_trade_amt:
                coin = trade_amt/price_indexed[i] + coin - fee_perc*trade_amt/price_indexed[i]
                base = base - trade_amt
                base_cusum -= trade_amt
            '''
            if base*max_trade >= 0.002:
                coin = max_trade*base/mean_price[i] + coin
                base = base - base*max_trade
            '''
        elif model_index[i]==1:
            '''
            perc_trade = min(max(0,coin*price_indexed[i]/total - min_coin_perc),max_perc_trade)
            trade_amt = max(min_trade_amt,1*perc_trade*total)
            '''
            trade_amt = max(min_trade_amt,-ki*base_cusum+(1)*perc_trade*total)
            if trade_amt/price_indexed[i] <= coin and trade_amt >= min_trade_amt:
                base = trade_amt + base - fee_perc*trade_amt
                coin = coin - trade_amt/price_indexed[i]
                base_cusum += trade_amt - fee_perc*trade_amt
            '''
            if max_trade*coin*mean_price[i] >= 0.002:
                base = max_trade*coin*mean_price[i] + base
                coin = coin - coin*max_trade
            '''
        
        #trade_amt = 0.1*base_best
        if outputs[i]==0:
            # buy
            '''
            perc_trade = min(base_best/total_best,max_perc_trade)
            trade_amt = max(min_trade_amt,1*perc_trade*total_best)
            '''
            trade_amt = max(min_trade_amt,ki*base_cusum_best+(1)*perc_trade*total_best)
            if base_best >= trade_amt and trade_amt >= min_trade_amt:
                coin_best = trade_amt/price_indexed[i] + coin_best - fee_perc*trade_amt/price_indexed[i]
                base_best = base_best - trade_amt
                base_cusum_best -= trade_amt
            '''
            if base_best*max_trade >= 0.002:
                coin_best = max_trade*base_best/mean_price[i] + coin_best
                base_best = base_best - base_best*max_trade
            '''
        elif outputs[i]==1:
            '''
            perc_trade = min(max(0,coin_best*price_indexed[i]/total_best-min_coin_perc),max_perc_trade)
            trade_amt = max(min_trade_amt,1*perc_trade*total_best)
            '''
            trade_amt = max(min_trade_amt,-ki*base_cusum_best+1*perc_trade*total_best)
            if trade_amt/price_indexed[i] <= coin_best and trade_amt >= min_trade_amt:
                base_best = trade_amt + base_best - fee_perc*trade_amt
                coin_best = coin_best - trade_amt/price_indexed[i]
                base_cusum_best += trade_amt - fee_perc*trade_amt
            '''
            if max_trade*coin_best*mean_price[i] >= 0.002:
                base_best = max_trade*coin_best*mean_price[i] + base_best
                coin_best = coin_best - coin_best*max_trade
            '''
        base_cusum *= cusum_decay
        base_cusum_best *= cusum_decay
        total = base + coin*price_indexed[i]
        total_best = base_best+coin_best*price_indexed[i]
        '''
        plt.subplot(513,sharex=ax1)
        plt.plot(time_ax[i],coin,'b.',time_ax[i],coin_best,'g.')
        plt.subplot(514,sharex=ax1)
        plt.plot(time_ax[i],model_out[i,0],'g.',time_ax[i],model_out[i,1],'r.')
        plt.subplot(515,sharex=ax1)
        plt.plot(time_ax[i],total,'b.',time_ax[i],total_best,'g.',time_ax[i],coin_start*price_indexed[i]+base_start,'r.')
        '''
        #print('Point: {} \t Model: {} \t Best: {}'.format(i,total,total_best))
    print('Accuracy: {}'.format(acc_count/m))
    return coin,base,coin_best,base_best,total,total_best

client = Client("a9jZULWZPvoYrwVQC5R2oJBlALi9SwWbjxOSB6otHVGgbLd37M0JIzFmCj8vxGX7","wg3ARO6AhBUO1I3fUHoQMMLo2bAWHTHbrwQ4E5mvZRDMrElp4NQUxVBgYFDYYDUa")
pair = 'VENBTC'
print('Fetching latest candle data...\n')

dt = 30; # seconds
t_forecast = 15 # minutes

train_kline = read_training_data(pair,'Dec 1, 2017','Mar 1, 2018')
# now format the inputs and targets matrix

print('Done!\n')
seed = 5
rng = np.random.RandomState(seed)

num_hidden = [18,12,12,10]
epochs = 5000000
batch_size = 128
learning_rate = 0.0000007
keep_prob = 1
val_ratio = 0.10
test_ratio = 0.0
val_counts = 0
val_count_max = 6000
p_ax = [-1,1]

# BUY network
plt.close('all')
print('Formatting inputs and outputs for network...\n')
inputs,outputs,time_ax,price_indexed,mean_time,mean_price,amt = DataFormatting.format_training_data_V3(train_kline,0.15)
inputs,min_x,max_x = DataFormatting.scale_inputs(inputs)

num_in = inputs.shape[1]
m = inputs.shape[0]
num_out = outputs.shape[1]
x = tf.placeholder(tf.float32, [None, num_in])
y = tf.placeholder(tf.int64, [None])
W,b = initialize_weights(num_in,num_hidden,num_out)

Z,A = forward_prop(x,W,b,keep_prob)
output_layer = A['layer{}'.format(len(W)-1)]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

test_idx = int((1-test_ratio)*m)
idx = np.random.permutation(test_idx)
test_idx = range(test_idx,m)
val_idx,train_idx = idx[:int(val_ratio*m)],idx[int(val_ratio*m):]

targets_index = np.zeros(m)
for  i in range(m):
    targets_index[i], _ = max(enumerate(outputs[i]), key=operator.itemgetter(1))

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
print('Training Buy network...')
train_network(sess,inputs,targets_index,train_idx,val_idx)

weight,bias = getWeights(W,b)
sess.close()
plt.close('all')


#pair = 'NANOBTC'
coin_amt = 1000
base_amt = 0.0
inputs,outputs,time_ax,price_indexed,mean_time,mean_price,amt = DataFormatting.format_training_data_V3(train_kline,0.15)
inputs,min_x,max_x = DataFormatting.scale_inputs(inputs,min_x,max_x)
m = inputs.shape[0]
targets_index = np.zeros(m)
for  i in range(m):
    targets_index[i], _ = max(enumerate(outputs[i]), key=operator.itemgetter(1))
nano_model,model_btc,best_nano,best_btc,total,total_best = run_model_v2(inputs,targets_index,time_ax,price_indexed,weight,bias,coin_amt,base_amt)

print('Total nano model: {}'.format(nano_model + model_btc/price_indexed[-1]))
print('Total nano target: {}'.format(best_nano + best_btc/price_indexed[-1]))
print('Total nano do nothing: {}'.format(coin_amt + base_amt/price_indexed[-1]))

pair = 'VENBTC'
coin_amt = 58
base_amt = 0.0
test_kline = read_training_data(pair,'Mar 1, 2018')
inputs,outputs,time_ax,price_indexed,mean_time,mean_price,amt = DataFormatting.format_training_data_V3(test_kline,0.15)
inputs,min_x,max_x = DataFormatting.scale_inputs(inputs,min_x,max_x)
m = inputs.shape[0]
targets_index = np.zeros(m)
for  i in range(m):
    targets_index[i], _ = max(enumerate(outputs[i]), key=operator.itemgetter(1))

nano_model,model_btc,best_nano,best_btc,total,total_best = run_model_v2(inputs,targets_index,time_ax,price_indexed,weight,bias,coin_amt,base_amt)

print('Total nano model: {}'.format(nano_model + model_btc/price_indexed[-1]))
print('Total nano target: {}'.format(best_nano + best_btc/price_indexed[-1]))
print('Total nano do nothing: {}'.format(coin_amt + base_amt/price_indexed[-1]))


'''

#inputs_buy,targets_buy,inputs_sell,targets_sell,actual_price,max_price,min_price,time_ax = DataFormatting.format_training_data_V2(train_kline,dt,t_forecast)
inputs_buy,min_x_buy,max_x_buy = DataFormatting.scale_inputs(inputs_buy)
inputs_sell,min_x_sell,max_x_sell = DataFormatting.scale_inputs(inputs_sell)
num_in = inputs_buy.shape[1]
m = inputs_buy.shape[0]
num_out = targets_buy.shape[1]
x = tf.placeholder(tf.float32, [None, num_in])
y = tf.placeholder(tf.int64, [None])
W,b = initialize_weights(num_in,num_hidden,num_out)

Z,A = forward_prop(x,W,b,keep_prob)
output_layer = A['layer{}'.format(len(W)-1)]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

test_idx = int((1-test_ratio)*m)
idx = np.random.permutation(test_idx)
test_idx = range(test_idx,m)
val_idx,train_idx = idx[:int(val_ratio*m)],idx[int(val_ratio*m):]

targets_index_buy = np.zeros(m)
targets_index_sell = np.zeros(m)
for  i in range(m):
    targets_index_buy[i], _ = max(enumerate(targets_buy[i]), key=operator.itemgetter(1))
    targets_index_sell[i], _ = max(enumerate(targets_sell[i]), key=operator.itemgetter(1))

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
print('Training Buy network...')
train_network(sess,inputs_buy,targets_index_buy,train_idx,test_idx)
weight_buy,bias_buy = getWeights(W,b)
print('Training Sell network...')
sess.run(init)
train_network(sess,inputs_sell,targets_index_sell,train_idx,test_idx)
weight_sell,bias_sell = getWeights(W,b)

sess.close()
closed_err_buy,closed_err_sell = test_model(inputs_buy[test_idx,:],targets_buy[test_idx,:],targets_sell[test_idx,:],actual_price[test_idx],max_price[test_idx],min_price[test_idx],time_ax[test_idx],weight_buy,bias_buy,weight_sell,bias_sell)
'''
'''


train_kline = read_training_data(pair,'1 minute ago')
# now format the inputs and targets matrix
inputs,targets_buy_targets_sell,actual_price,max_price,min_price,time_ax = DataFormatting.format_training_data(train_kline,dt,t_forecast)
inputs,min_x,max_x = DataFormatting.scale_inputs(inputs,min_x,max_x)

open_err,closed_err = train_adaptive(sess,inputs,targets,actual_price,max_price,time_ax,False,1e-6)

#sess.close()
'''