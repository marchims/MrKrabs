# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:58:12 2018

@author: marchims
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

def scale_inputs(x,*args):
    if len(args) == 0:
        min_x = np.min(x,axis=0,keepdims=True)
        max_x = np.max(x,axis=0,keepdims=True)
    else:
        min_x = args[0]
        max_x = args[1]
        
    valid = max_x-min_x>0
    x[:,valid[0,:]] = np.divide(np.subtract(x[:,valid[0,:]],min_x[valid]),max_x[valid]-min_x[valid])
    return x,min_x,max_x

def format_live_data_V2(live_kline,windows):
    
    mean_time = np.mean(live_kline[:,[0,6]],axis=1,keepdims=True)
    
    m,n = live_kline.shape
    n2 = len(windows)
    # create uninterpolated data
    price_means = np.zeros((m,n2))
    volume_means = np.zeros((m,n2))
    mean_price = np.mean(live_kline[:,1:4],axis=1,keepdims=True);
    volumes = live_kline[:,5]
    
    dp_bought = live_kline[:,2]-live_kline[:,1] + live_kline[:,4]-live_kline[:,3]
    dp_sold = live_kline[:,1]-live_kline[:,3] + live_kline[:,2]-live_kline[:,4]
    buy_ratio = np.divide(dp_bought,dp_bought+dp_sold)
    sell_ratio = 1-buy_ratio
    buy_vol = np.multiply(buy_ratio,volumes)
    sell_vol = np.multiply(sell_ratio,volumes)
    
    dp_dv_buy = np.divide(dp_bought,buy_vol)
    dp_dv_sell = np.divide(dp_sold,sell_vol)
    dp_dv_buy[np.isnan(dp_dv_buy)] = 0
    dp_dv_sell[np.isnan(dp_dv_sell)] = 0
    buy_ratio[np.isnan(buy_ratio)] = 0.5
    buy_vol[np.isnan(buy_vol)] = 0
    sell_vol[np.isnan(sell_vol)] = 0
    
    dp_dv_buy_smoothed = np.zeros((m,n2))
    dp_dv_sell_smoothed = np.zeros((m,n2))
    buy_ratio_smoothed = np.zeros((m,n2))
    buy_vol_smoothed = np.zeros((m,n2))
    sell_vol_smoothed = np.zeros((m,n2))
    
    mean_price_cusum = np.cumsum(mean_price)
    volume_cusum = np.cumsum(volumes)
    dp_dv_buy_cusum = np.cumsum(dp_dv_buy)
    dp_dv_sell_cusum = np.cumsum(dp_dv_sell)
    buy_ratio_cusum  = np.cumsum(buy_ratio)
    buy_cusum = np.cumsum(buy_vol)
    sell_cusum = np.cumsum(sell_vol)
    for i in range(n2):
        
        p1 = mean_price_cusum[windows[i]:]
        p2 = mean_price_cusum[:-windows[i]]
        price_means[windows[i]:,i] = (p1-p2)/windows[i]
        price_means[:windows[i],i] = np.nan
        
        p1 = volume_cusum[windows[i]:]
        p2 = volume_cusum[:-windows[i]]
        volume_means[windows[i]:,i] = (p1-p2)/windows[i]
        volume_means[:windows[i],i] = np.nan
        
        p1 = buy_ratio_cusum[windows[i]:]
        p2 = buy_ratio_cusum[:-windows[i]]
        buy_ratio_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_ratio_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_buy_cusum[windows[i]:]
        p2 = dp_dv_buy_cusum[:-windows[i]]
        dp_dv_buy_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_buy_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_sell_cusum[windows[i]:]
        p2 = dp_dv_sell_cusum[:-windows[i]]
        dp_dv_sell_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_sell_smoothed[:windows[i],i] = np.nan
        
        p1 = buy_cusum[windows[i]:]
        p2 = buy_cusum[:-windows[i]]
        buy_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_vol_smoothed[:windows[i],i] = np.nan
        buy_vol_smoothed[:,i] = np.divide(buy_vol_smoothed[:,i],volume_means[:,i])
        
        p1 = sell_cusum[windows[i]:]
        p2 = sell_cusum[:-windows[i]]
        sell_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        sell_vol_smoothed[:windows[i],i] = np.nan
        sell_vol_smoothed[:,i] = np.divide(sell_vol_smoothed[:,i],volume_means[:,i])
    
    price_gradient = np.zeros((m,n2))
    price_gradient[1:,:] = np.divide(np.diff(price_means,axis=0),price_means[1:,:])*100
    norm_price = np.divide(price_means-mean_price,mean_price)*100

    price_gradient_2 = np.zeros((m,n2))
    smooth_window = 20
    for i in range(n2):
        gradient_cusum = np.zeros((m))
        gradient_cusum[1:] = np.nancumsum(np.diff(price_means[:,i],axis=0))
        p1 = gradient_cusum[smooth_window:]
        p2 = gradient_cusum[:-smooth_window]
        price_gradient_2[smooth_window:,i] = (p1-p2)/smooth_window
        #price_gradient_2[:windows[i],i] = np.nan
        price_gradient_2[1:,i] = np.divide(np.diff(price_gradient_2[:,i],axis=0),price_means[1:,i])*100
    
    deltas_buy_ratio= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_gradient= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_gradient_2= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_ratio_gradient= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_norm = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    index = 0
    smooth_window = 2
    for i in range(n2):
        for j in range(i+1,n2):
            deltas[:,index] = price_means[:,i]-price_means[:,j]
            deltas_norm[:,index] = deltas[:,index]/price_means[:,j]
            deltas_gradient[1:,index] = np.divide(np.diff(deltas[:,index],axis=0),price_means[1:,j])*100
            
            temp = np.zeros((m))
            temp[1:] = np.nancumsum(np.diff(deltas[:,index],axis=0))
            p1 = temp[smooth_window:]
            p2 = temp[:-smooth_window]
            smoothed_gradient = np.zeros((m))
            smoothed_gradient[smooth_window:] = (p1-p2)/smooth_window
            
            
            deltas_gradient_2[1:,index] = np.divide(np.diff(smoothed_gradient,axis=0),price_means[1:,j])*100
            deltas_buy_vol[:,index] = buy_vol_smoothed[:,i]-buy_vol_smoothed[:,j]
            deltas_sell_vol[:,index] = sell_vol_smoothed[:,i]-sell_vol_smoothed[:,j]
            deltas_buy_ratio[:,index] = buy_ratio_smoothed[:,i]-buy_ratio_smoothed[:,j]
            deltas_buy_ratio_gradient[1:,index] = np.diff(deltas_buy_ratio[:,i])
            deltas_buy[:,index] = dp_dv_buy_smoothed[:,i]-dp_dv_buy_smoothed[:,j]
            deltas_sell[:,index] = dp_dv_sell_smoothed[:,i]-dp_dv_sell_smoothed[:,j]
            index += 1
    
    idx_cross = np.zeros((0,0))
    for i in range(1):
        # find the zero crosses
       # print(i)
        if idx_cross.shape[1]==0:
            idx_cross = np.array(np.where(np.logical_and(np.isfinite(deltas[:m-1,i]),np.diff(np.sign(deltas[:,i]))!=0))+1)
        else:
            idx_cross = np.unique(np.concatenate((idx_cross,np.array(np.where(np.logical_and(np.isfinite(deltas[:m-1,i]),np.diff(np.sign(deltas[:,i]))!=0)))),axis=1)+1)
            idx_cross = idx_cross.reshape((1,idx_cross.shape[0]))
            
    
    mean_price_indexed = price_means[idx_cross[0,:],0].reshape((idx_cross.shape[1]),1)
    price_diff = np.zeros((len(mean_price_indexed),1))
    price_diff[1:] = np.divide((mean_price_indexed[1:] - mean_price_indexed[:-1]),mean_price_indexed[1:])
    inputs = np.concatenate((price_diff,norm_price[idx_cross[0,:],:],deltas_norm[idx_cross[0,:],:],price_gradient[idx_cross[0,:],:],deltas_gradient[idx_cross[0,:],:],deltas_gradient_2[idx_cross[0,:],:],price_gradient_2[idx_cross[0,:],:]),axis=1)
        
    valid = np.all(np.isfinite(inputs),axis=1)
    
    print(valid)
    inputs = inputs[valid,:]
    inputs = inputs[-1,:]
    
    return inputs

def format_live_data(live_kline,windows,last_mean_price):

    m,n = live_kline.shape
    n2 = len(windows)
    # create uninterpolated data
    price_means = np.zeros((m,n2))
    volume_means = np.zeros((m,n2))
    mean_price = np.mean(live_kline[:,1:4],axis=1,keepdims=True);
    volumes = live_kline[:,5]

    
    mean_price_cusum = np.cumsum(mean_price)
    volume_cusum = np.cumsum(volumes)

    for i in range(n2):
        
        p1 = mean_price_cusum[windows[i]:]
        p2 = mean_price_cusum[:-windows[i]]
        price_means[windows[i]:,i] = (p1-p2)/windows[i]
        price_means[:windows[i],i] = np.nan
        
        p1 = volume_cusum[windows[i]:]
        p2 = volume_cusum[:-windows[i]]
        volume_means[windows[i]:,i] = (p1-p2)/windows[i]
        volume_means[:windows[i],i] = np.nan
        
    
    price_gradient = np.zeros((m,n2))
    price_gradient[1:,:] = np.divide(np.diff(price_means,axis=0),price_means[1:,:])*100
    norm_price = np.divide(price_means-mean_price,mean_price)*100

    price_gradient_2 = np.zeros((m,n2))
    smooth_window = 5
    for i in range(n2):
        gradient_cusum = np.zeros((m))
        gradient_cusum[1:] = np.nancumsum(np.diff(price_means[:,i],axis=0))
        p1 = gradient_cusum[smooth_window:]
        p2 = gradient_cusum[:-smooth_window]
        price_gradient_2[smooth_window:,i] = (p1-p2)/smooth_window
        #price_gradient_2[:windows[i],i] = np.nan
        price_gradient_2[1:,i] = np.divide(np.diff(price_gradient_2[:,i],axis=0),price_means[1:,i])*100
    
    n3 = int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))
    deltas = np.zeros((m,n3))
    deltas_gradient= np.zeros((m,n3))
    deltas_gradient_2= np.zeros((m,n3))
    deltas_norm = np.zeros((m,n3))
    index = 0
    smooth_window = 2
    for i in range(n2):
        for j in range(i+1,n2):
            deltas[:,index] = price_means[:,i]-price_means[:,j]
            deltas_norm[:,index] = deltas[:,index]/price_means[:,j]
            deltas_gradient[1:,index] = np.divide(np.diff(deltas[:,index],axis=0),price_means[1:,j])*100
            
            temp = np.zeros((m))
            temp[1:] = np.nancumsum(np.diff(deltas[:,index],axis=0))
            p1 = temp[smooth_window:]
            p2 = temp[:-smooth_window]
            smoothed_gradient = np.zeros((m))
            smoothed_gradient[smooth_window:] = (p1-p2)/smooth_window
            
            
            deltas_gradient_2[1:,index] = np.divide(np.diff(smoothed_gradient,axis=0),price_means[1:,j])*100
            index += 1

    mean_price_indexed = price_means[-2,0].reshape((1,1))
    if np.isnan(last_mean_price):
        price_diff = np.array([0])
    else:
        price_diff = np.array((mean_price_indexed - last_mean_price)/mean_price_indexed).reshape((1,1))
    
    
    inputs = np.concatenate((price_diff.reshape((1,1)),norm_price[-2,:].reshape((1,len(windows))),
                             deltas_norm[-2,:].reshape((1,n3)),
                             price_gradient[-2,:].reshape((1,len(windows))),
                             deltas_gradient[-2,:].reshape((1,n3)),
                             deltas_gradient_2[-2,:].reshape((1,n3)),
                             price_gradient_2[-2,:].reshape((1,len(windows)))),axis=1)
    
    valid = np.all(np.isfinite(inputs),axis=1)
    
    inputs = inputs[valid,:]
    
    
    return inputs,mean_price_indexed

def format_training_data_V3(train_kline,margin):

    windows = [3,5,10,20,1]
    
    mean_time = np.mean(train_kline[:,[0,6]],axis=1,keepdims=True)
    
    m,n = train_kline.shape
    n2 = len(windows)
    # create uninterpolated data
    price_means = np.zeros((m,n2))
    price_means_ema = np.zeros((m,n2))
    volume_means = np.zeros((m,n2))
    mean_price = np.mean(train_kline[:,1:4],axis=1,keepdims=True);
    volumes = train_kline[:,5]
    
    dp_bought = train_kline[:,2]-train_kline[:,1] + train_kline[:,4]-train_kline[:,3]
    dp_sold = train_kline[:,1]-train_kline[:,3] + train_kline[:,2]-train_kline[:,4]
    buy_ratio = np.divide(dp_bought,dp_bought+dp_sold)
    sell_ratio = 1-buy_ratio
    buy_vol = np.multiply(buy_ratio,volumes)
    sell_vol = np.multiply(sell_ratio,volumes)
    
    dp_dv_buy = np.divide(dp_bought,buy_vol)
    dp_dv_sell = np.divide(dp_sold,sell_vol)
    dp_dv_buy[np.isnan(dp_dv_buy)] = 0
    dp_dv_sell[np.isnan(dp_dv_sell)] = 0
    buy_ratio[np.isnan(buy_ratio)] = 0.5
    buy_vol[np.isnan(buy_vol)] = 0
    sell_vol[np.isnan(sell_vol)] = 0
    
    dp_dv_buy_smoothed = np.zeros((m,n2))
    dp_dv_sell_smoothed = np.zeros((m,n2))
    buy_ratio_smoothed = np.zeros((m,n2))
    buy_vol_smoothed = np.zeros((m,n2))
    sell_vol_smoothed = np.zeros((m,n2))
    
    mean_price_cusum = np.cumsum(mean_price)
    volume_cusum = np.cumsum(volumes)
    dp_dv_buy_cusum = np.cumsum(dp_dv_buy)
    dp_dv_sell_cusum = np.cumsum(dp_dv_sell)
    buy_ratio_cusum  = np.cumsum(buy_ratio)
    buy_cusum = np.cumsum(buy_vol)
    sell_cusum = np.cumsum(sell_vol)
    for i in range(n2):
        
        
        p1 = mean_price_cusum[windows[i]:]
        p2 = mean_price_cusum[:-windows[i]]
        price_means[windows[i]:,i] = (p1-p2)/windows[i]
        price_means[:windows[i],i] = np.nan
        
        
        for j in range(m):
            if j < windows[i]-1:
                price_means_ema[j,i] = np.nan
            elif j == windows[i]-1:
                price_means_ema[j,i] = np.mean(mean_price[:j+1])
            else:
                price_means_ema[j,i] = (1-1/windows[i])*price_means_ema[j-1,i] + mean_price[j]*(1/windows[i])
        
        p1 = volume_cusum[windows[i]:]
        p2 = volume_cusum[:-windows[i]]
        volume_means[windows[i]:,i] = (p1-p2)/windows[i]
        volume_means[:windows[i],i] = np.nan
        
        p1 = buy_ratio_cusum[windows[i]:]
        p2 = buy_ratio_cusum[:-windows[i]]
        buy_ratio_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_ratio_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_buy_cusum[windows[i]:]
        p2 = dp_dv_buy_cusum[:-windows[i]]
        dp_dv_buy_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_buy_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_sell_cusum[windows[i]:]
        p2 = dp_dv_sell_cusum[:-windows[i]]
        dp_dv_sell_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_sell_smoothed[:windows[i],i] = np.nan
        
        p1 = buy_cusum[windows[i]:]
        p2 = buy_cusum[:-windows[i]]
        buy_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_vol_smoothed[:windows[i],i] = np.nan
        buy_vol_smoothed[:,i] = np.divide(buy_vol_smoothed[:,i],volume_means[:,i])
        
        p1 = sell_cusum[windows[i]:]
        p2 = sell_cusum[:-windows[i]]
        sell_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        sell_vol_smoothed[:windows[i],i] = np.nan
        sell_vol_smoothed[:,i] = np.divide(sell_vol_smoothed[:,i],volume_means[:,i])
    
    price_gradient = np.zeros((m,n2))
    price_gradient_smoothed = np.zeros((m,n2))
    price_gradient[1:,:] = np.divide(np.diff(price_means,axis=0),price_means[1:,:])*100
    norm_price = np.divide(price_means-mean_price,mean_price)*100
    norm_price_ema = np.divide(price_means_ema-mean_price,mean_price)*100

    price_gradient_2 = np.zeros((m,n2))
    smooth_window = 2
    for i in range(n2):
        gradient_cusum = np.zeros((m))
        gradient_cusum[1:] = np.nancumsum(np.diff(price_means[:,i],axis=0))
        p1 = gradient_cusum[smooth_window:]
        p2 = gradient_cusum[:-smooth_window]
        price_gradient_smoothed[smooth_window:,i] = (p1-p2)/smooth_window
        #price_gradient_2[:windows[i],i] = np.nan
        price_gradient_2[1:,i] = np.divide(np.diff(price_gradient_smoothed[:,i],axis=0),price_means[1:,i])*100
        
    price_gradient_smoothed[1:,:] = np.divide(price_gradient_smoothed[1:,:],price_means[1:,:])*100
    
    deltas_buy_ratio= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_gradient= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_gradient_smoothed = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_gradient_2= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_ratio_gradient= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_vol_gradient = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_vol_gradient_smoothed = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_norm = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    index = 0
    smooth_window = 2
    for i in range(n2):
        for j in range(i+1,n2):
            deltas[:,index] = price_means[:,i]-price_means[:,j]
            deltas_norm[:,index] = np.divide(deltas[:,index],price_means[:,j])
            deltas_gradient[1:,index] = np.divide(np.diff(deltas[:,index],axis=0),price_means[1:,j])*100
            
            temp = np.zeros((m))
            temp[1:] = np.nancumsum(np.diff(deltas[:,index],axis=0))
            p1 = temp[smooth_window:]
            p2 = temp[:-smooth_window]
            deltas_gradient_smoothed[smooth_window:,index] = (p1-p2)/smooth_window
            deltas_gradient_2[1:,index] = np.divide(np.diff(deltas_gradient_smoothed[:,index],axis=0),price_means[1:,j])*100
            
            deltas_buy_vol[:,index] = buy_vol_smoothed[:,i]-buy_vol_smoothed[:,j]
            deltas_buy_vol_gradient[1:,index] = np.diff(deltas_buy_vol[:,index],axis=0)
            
            gradient_cusum = np.zeros((m))
            gradient_cusum[1:] = np.nancumsum(np.diff(deltas_buy_vol_gradient[:,index],axis=0))
            p1 = gradient_cusum[smooth_window:]
            p2 = gradient_cusum[:-smooth_window]
            deltas_buy_vol_gradient_smoothed[smooth_window:,index] = (p1-p2)/smooth_window
            
            deltas_sell_vol[:,index] = sell_vol_smoothed[:,i]-sell_vol_smoothed[:,j]
            deltas_buy_ratio[:,index] = buy_ratio_smoothed[:,i]-buy_ratio_smoothed[:,j]
            deltas_buy_ratio_gradient[1:,index] = np.diff(deltas_buy_ratio[:,i])
            deltas_buy[:,index] = dp_dv_buy_smoothed[:,i]-dp_dv_buy_smoothed[:,j]
            deltas_sell[:,index] = dp_dv_sell_smoothed[:,i]-dp_dv_sell_smoothed[:,j]
            index += 1
    '''
    plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(mean_price,'.-',price_means,'-')
    ax2 = plt.subplot(312,sharex=ax1)
    plt.plot(price_gradient,'-')
    ax3= plt.subplot(313,sharex=ax1)
    plt.plot(price_gradient_2,'-')
    return p1
    '''
    
    amt = 1.0
    idx_cross = np.zeros((0,0))
    for i in range(1):
        # find the zero crosses
       # print(i)
        if idx_cross.shape[1]==0:
            idx_cross = np.array(np.where(np.logical_and(np.isfinite(deltas[:m-1,i]),np.diff(np.sign(deltas[:,i]))!=0)))+1
        else:
            idx_cross = np.unique(np.concatenate((idx_cross,np.array(np.where(np.logical_and(np.isfinite(deltas[:m-1,i]),np.diff(np.sign(deltas[:,i]))!=0)))),axis=1))
            idx_cross = idx_cross.reshape((1,idx_cross.shape[0]))
            
    
   
    time_ax = mean_time[idx_cross[0,:-1],:]
    mean_price_indexed = price_means[idx_cross[0,:],0].reshape((idx_cross.shape[1]),1)
    act_price_indexed = mean_price[idx_cross[0,:-1],:]
    price_diff = np.zeros((len(mean_price_indexed),1))
    price_diff[1:] = np.divide((mean_price_indexed[1:] - mean_price_indexed[:-1]),mean_price_indexed[1:])
    inputs = np.concatenate((price_diff[:-1],norm_price_ema[idx_cross[0,:-1],:],norm_price[idx_cross[0,:-1],:],deltas_norm[idx_cross[0,:-1],:],price_gradient_smoothed[idx_cross[0,:-1],:],deltas_gradient_smoothed[idx_cross[0,:-1],:],deltas_gradient_2[idx_cross[0,:-1],:],price_gradient_2[idx_cross[0,:-1],:],deltas_buy_vol_gradient_smoothed[idx_cross[0,:-1],:]),axis=1)
    outputs = np.zeros((idx_cross.shape[1]-1,3))
    for j in range(idx_cross.shape[1]-2):
        perc = (act_price_indexed[j+1] - act_price_indexed[j])/act_price_indexed[j] * 100
        if perc > margin:
            outputs[j,0] = 1.0
            amt *= 1+perc/100
        elif perc <= -margin:
            outputs[j,1] = 1.0
            #amt *= 1+perc/100
        else:
            outputs[j,2] = 1.0
        
    valid = np.logical_and(np.all(np.isfinite(inputs),axis=1),np.all(np.isfinite(outputs),axis=1))
    
    inputs = inputs[valid,:]
    outputs= outputs[valid,:]
    time_ax= time_ax[valid,:]
    act_price_indexed= act_price_indexed[valid,:]
    
    plt.figure()
    plt.plot(mean_time,mean_price,'.-',time_ax,act_price_indexed,'rx')
    plt.plot(mean_time,price_means,'-')
    
    return inputs,outputs,time_ax,act_price_indexed,mean_time,mean_price,amt
        
    
    


def format_training_data_V2(train_kline,dt,t_forecast):
    factor_high = 0.1
    factor_low = 0.90
    input_filter = 0.02
    windows = [15,30,60,240]
    
    time_ax = np.arange(np.mean(train_kline[0,[0,6]].reshape((1,2)),axis=1,keepdims=True),np.mean(train_kline[-1,[0,6]].reshape((1,2)),axis=1,keepdims=True),dt*1000)
    mean_time = np.mean(train_kline[:,[0,6]],axis=1,keepdims=True)
    m2 = len(time_ax)
    
    m,n = train_kline.shape
    n2 = len(windows)
    # create uninterpolated data
    price_means = np.zeros((m,n2))
    volume_means = np.zeros((m,n2))
    mean_price = np.mean(train_kline[:,1:4],axis=1,keepdims=True);
    volumes = train_kline[:,5]
    
    dp_bought = train_kline[:,2]-train_kline[:,1] + train_kline[:,4]-train_kline[:,3]
    dp_sold = train_kline[:,1]-train_kline[:,3] + train_kline[:,2]-train_kline[:,4]
    buy_ratio = np.divide(dp_bought,dp_bought+dp_sold)
    sell_ratio = 1-buy_ratio
    buy_vol = np.multiply(buy_ratio,volumes)
    sell_vol = np.multiply(sell_ratio,volumes)
    
    dp_dv_buy = np.divide(dp_bought,buy_vol)
    dp_dv_sell = np.divide(dp_sold,sell_vol)
    dp_dv_buy[np.isnan(dp_dv_buy)] = 0
    dp_dv_sell[np.isnan(dp_dv_sell)] = 0
    buy_ratio[np.isnan(buy_ratio)] = 0.5
    buy_vol[np.isnan(buy_vol)] = 0
    sell_vol[np.isnan(sell_vol)] = 0
    
    dp_dv_buy_smoothed = np.zeros((m,n2))
    dp_dv_sell_smoothed = np.zeros((m,n2))
    buy_ratio_smoothed = np.zeros((m,n2))
    buy_vol_smoothed = np.zeros((m,n2))
    sell_vol_smoothed = np.zeros((m,n2))
    
    mean_price_cusum = np.cumsum(mean_price)
    volume_cusum = np.cumsum(volumes)
    dp_dv_buy_cusum = np.cumsum(dp_dv_buy)
    dp_dv_sell_cusum = np.cumsum(dp_dv_sell)
    buy_ratio_cusum  = np.cumsum(buy_ratio)
    buy_cusum = np.cumsum(buy_vol)
    sell_cusum = np.cumsum(sell_vol)
    for i in range(n2):
        
        p1 = mean_price_cusum[windows[i]:]
        p2 = mean_price_cusum[:-windows[i]]
        price_means[windows[i]:,i] = (p1-p2)/windows[i]
        price_means[:windows[i],i] = np.nan
        
        p1 = volume_cusum[windows[i]:]
        p2 = volume_cusum[:-windows[i]]
        volume_means[windows[i]:,i] = (p1-p2)/windows[i]
        volume_means[:windows[i],i] = np.nan
        
        p1 = buy_ratio_cusum[windows[i]:]
        p2 = buy_ratio_cusum[:-windows[i]]
        buy_ratio_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_ratio_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_buy_cusum[windows[i]:]
        p2 = dp_dv_buy_cusum[:-windows[i]]
        dp_dv_buy_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_buy_smoothed[:windows[i],i] = np.nan
        
        p1 = dp_dv_sell_cusum[windows[i]:]
        p2 = dp_dv_sell_cusum[:-windows[i]]
        dp_dv_sell_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        dp_dv_sell_smoothed[:windows[i],i] = np.nan
        
        p1 = buy_cusum[windows[i]:]
        p2 = buy_cusum[:-windows[i]]
        buy_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        buy_vol_smoothed[:windows[i],i] = np.nan
        buy_vol_smoothed[:,i] = np.divide(buy_vol_smoothed[:,i],volume_means[:,i])
        
        p1 = sell_cusum[windows[i]:]
        p2 = sell_cusum[:-windows[i]]
        sell_vol_smoothed[windows[i]:,i] = (p1-p2)/windows[i]
        sell_vol_smoothed[:windows[i],i] = np.nan
        sell_vol_smoothed[:,i] = np.divide(sell_vol_smoothed[:,i],volume_means[:,i])
    
    
    price_gradient = np.divide(np.gradient(price_means,axis=0),price_means)
    price_gradient_2 = np.divide(np.gradient(np.gradient(price_means,axis=0,edge_order=2),axis=0,edge_order=2),price_means)
    price_norm_means = np.divide(price_means-mean_price,mean_price)
    
    
    deltas = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell = np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_ratio= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_buy_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    deltas_sell_vol= np.zeros((m,int(math.factorial(n2)/(math.factorial(2)*math.factorial(int(n2-2))))))
    index = 0
    for i in range(n2):
        for j in range(i+1,n2):
            deltas[:,index] = price_means[:,i]-price_means[:,j]
            deltas_buy[:,index] = dp_dv_buy_smoothed[:,i]-dp_dv_buy_smoothed[:,j]
            deltas_sell[:,index] = dp_dv_sell_smoothed[:,i]-dp_dv_sell_smoothed[:,j]
            deltas_buy_ratio[:,index] = buy_ratio_smoothed[:,i]-buy_ratio_smoothed[:,j]
            deltas_buy_vol[:,index] = buy_vol_smoothed[:,i]-buy_vol_smoothed[:,j]
            deltas_sell_vol[:,index] = sell_vol_smoothed[:,i]-sell_vol_smoothed[:,j]
            index += 1
    
    
    
    #deltas_buy_vol = np.divide(deltas_buy_vol,volume_means)
    #deltas_sell_vol = np.divide(deltas_sell_vol,volume_means)
    
    dp_buy_smoothed = np.multiply(np.gradient(buy_vol_smoothed,axis=0),dp_dv_buy_smoothed)
    
    
    '''
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(mean_price,'.-')
    plt.subplot(212,sharex=ax1)
    plt.plot(deltas_buy_vol,'.-',deltas_buy_ratio,'-')
    return
    '''
    # interpolation step
    price_delta_interp = np.zeros((m2,n2))
    buy_delta_interp = np.zeros((m2,n2))
    sell_delta_interp = np.zeros((m2,n2))
    ratio_delta_interp = np.zeros((m2,n2))
    dp_dv_buy_smoothed_interp = np.zeros((m2,n2))
    dp_dv_sell_smoothed_interp = np.zeros((m2,n2))
    price_gradient_interp = np.zeros((m2,n2))
    buy_vol_interp = np.zeros((m2,n2))
    sell_vol_interp = np.zeros((m2,n2))
    norm_price_interp = np.zeros((m2,n2))
    for i in range(n2):
        f = interp1d(mean_time.flatten(),deltas[:,i],axis=0)
        price_delta_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),deltas_buy[:,i],axis=0)
        buy_delta_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),deltas_sell[:,i],axis=0)
        sell_delta_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),deltas_buy_ratio[:,i],axis=0)
        ratio_delta_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),dp_dv_buy_smoothed[:,i],axis=0)
        dp_dv_buy_smoothed_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),dp_dv_sell_smoothed[:,i],axis=0)
        dp_dv_sell_smoothed_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),price_gradient[:,i],axis=0)
        price_gradient_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),deltas_buy_vol[:,i],axis=0)
        buy_vol_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),deltas_sell_vol[:,i],axis=0)
        sell_vol_interp[:,i] = f(time_ax)
        
        f = interp1d(mean_time.flatten(),price_norm_means[:,i],axis=0)
        norm_price_interp[:,i] = f(time_ax)
    
    
    f = interp1d(mean_time.flatten(),mean_price,axis=0)
    actual_price = f(time_ax)
    f = interp1d(mean_time.flatten(),volumes)
    actual_volume = f(time_ax)
    
    p_ax = np.array([-100,1],dtype='float')
    p_norm_ax = np.array([-1,1],dtype='float')
    p_len = 1 #len(p_ax)
    outputs_buy = np.zeros((m2,len(p_ax)));
    
    # set up inputs matrix
    max_price = np.mean(train_kline[:,[1,2,4]].reshape((m,3)),axis=1,keepdims=True)
    f = interp1d(mean_time.flatten(),max_price[:,0])
    max_price_interp = f(time_ax)
    inputs_buy = np.concatenate((norm_price_interp,price_gradient_interp,buy_vol_interp,sell_vol_interp,price_delta_interp,buy_delta_interp,sell_delta_interp,ratio_delta_interp,dp_dv_buy_smoothed_interp,dp_dv_sell_smoothed_interp),axis=1)
    # set up outputs matrix and feedback on inputs
    for i in range(m2):
        v = np.logical_and(train_kline[:,0]>=time_ax[i], train_kline[:,0]<=(time_ax[i]+t_forecast*1000*60))
        if not any(v):
            outputs_buy[i,:] = np.nan
            #if i > 0:
                #inputs_buy[i,:p_len] = inputs_buy[i-1,:p_len]
            
            continue
        norm_max_price = 100*np.divide(max_price[v,:]-actual_price[i],actual_price[i])
        idx = np.where(np.max(norm_max_price)>p_ax)
        outputs_buy[i,np.max(idx)] = 1.0
            
        #if i == 1:
            #inputs_buy[i,:p_len] = np.matmul(outputs_buy[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
        #elif i > 1:
            #if np.any(np.isnan(outputs_buy[i-1,:])):
                #inputs_buy[i,:p_len] = inputs_buy[i-1,:p_len]
            #else:
                #inputs_buy[i,:p_len] = (1-input_filter)*inputs_buy[i-1,:p_len] + input_filter*np.matmul(outputs_buy[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
                #inputs_buy[i,:p_len] = softmax(inputs_buy[i,:p_len].reshape((1,p_len)))
        
        
    p_ax = np.array([100,-1],dtype='float')
    p_norm_ax = np.array([1,-1],dtype='float')
    p_len = 1 #len(p_ax)
    outputs_sell = np.zeros((m2,len(p_ax)));
    
    # set up inputs matrix
    min_price = np.mean(train_kline[:,[1,3,4]].reshape((m,3)),axis=1,keepdims=True)
    f = interp1d(mean_time.flatten(),min_price[:,0])
    min_price_interp = f(time_ax)
    inputs_sell = np.concatenate((norm_price_interp,price_gradient_interp,buy_vol_interp,sell_vol_interp,price_delta_interp,buy_delta_interp,sell_delta_interp,ratio_delta_interp,dp_dv_buy_smoothed_interp,dp_dv_sell_smoothed_interp),axis=1)
    # set up outputs matrix and feedback on inputs
    for i in range(m2):
        v = np.logical_and(train_kline[:,0]>=time_ax[i], train_kline[:,0]<=(time_ax[i]+t_forecast*1000*60))
        if not any(v):
            outputs_sell[i,:] = np.nan
            #if i > 0:
                #inputs_sell[i,:p_len] = inputs_sell[i-1,:p_len]
            
            continue
        norm_max_price = 100*np.divide(min_price[v,:]-actual_price[i],actual_price[i])
        idx = np.where(np.min(norm_max_price)<p_ax)
        outputs_sell[i,np.max(idx)] = 1.0
            
        #if i == 1:
            #inputs_sell[i,:p_len] = np.matmul(outputs_sell[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
        #elif i > 1:
            #if np.any(np.isnan(outputs_sell[i-1,:])):
                #inputs_sell[i,:p_len] = inputs_sell[i-1,:p_len]
            #else:
                #inputs_sell[i,:p_len] = (1-input_filter)*inputs_sell[i-1,:p_len] + input_filter*np.matmul(outputs_sell[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
                #inputs_sell[i,:p_len] = softmax(inputs_sell[i,:p_len].reshape((1,p_len)))
    
    valid_buy = np.logical_and(np.all(np.isfinite(inputs_buy),axis=1),np.all(np.isfinite(outputs_buy),axis=1))
    #valid_buy[:max(scales)+1] = False
    valid_sell = np.logical_and(np.all(np.isfinite(inputs_sell),axis=1),np.all(np.isfinite(outputs_sell),axis=1))
    valid = np.logical_and(valid_buy,valid_sell)
    
    inputs_buy = inputs_buy[valid,:]
    #inputs_buy[:,0] = inputs_buy[:,0] + np.random.randn(len(inputs_buy))*0.6
    #inputs_buy[:,:p_len] = softmax(inputs_buy[:,:p_len].reshape((len(inputs_buy),p_len)))
    outputs_buy = outputs_buy[valid,:]
    inputs_sell = inputs_sell[valid,:]
    #inputs_sell[:,0] = inputs_sell[:,0] + np.random.randn(len(inputs_buy))*0.6
    #inputs_sell[:,:p_len] = softmax(inputs_sell[:,:p_len].reshape((len(inputs_sell),p_len)))
    outputs_sell = outputs_sell[valid,:]
    return inputs_buy,outputs_buy,inputs_sell,outputs_sell,actual_price[valid],max_price_interp[valid],min_price_interp[valid],time_ax[valid]



def format_training_data(train_kline,dt,t_forecast):
    factor_high = 0.1
    factor_low = 0.90
    input_filter = 0.02
    scales = np.array([3,12,48,96])
    windows = [5,20,100]
    price_factor = np.array([1-1/scales[0],1-1/scales[1],1-1/scales[2],1-1/scales[3]]).reshape(1,4)
    
    time_ax = np.arange(np.mean(train_kline[0,[0,6]].reshape((1,2)),axis=1,keepdims=True),np.mean(train_kline[-1,[0,6]].reshape((1,2)),axis=1,keepdims=True),dt*1000)
    mean_time = np.mean(train_kline[:,[0,6]],axis=1,keepdims=True)
    m2 = len(time_ax)
    
    m,n = train_kline.shape
    n2 = price_factor.shape[1]
    # create uninterpolated data
    price_means = np.zeros((m,n2))
    price_means_normal = np.zeros((m,n2))
    volume_means = np.zeros((m,n2))
    mean_price = np.mean(train_kline[:,1:4],axis=1,keepdims=True);
    price_range_high = np.zeros((m,1))
    price_range_low = np.zeros((m,1))
    volumes = train_kline[:,5]
    q_vol = np.cumsum(volumes,axis=0)
    price_delta = train_kline[:,4]-train_kline[:,1]
    price_sign = np.sign(price_delta)
    avg_strength = np.zeros((m,n2))
    buy_ratio_smoothed = np.zeros((m,n2))
    amt_bought_smoothed = np.zeros((m,n2))
    amt_sold_smoothed = np.zeros((m,n2))
    
    amt_bought = train_kline[:,2]-train_kline[:,1] + train_kline[:,4]-train_kline[:,3]
    amt_sold = train_kline[:,1]-train_kline[:,3] + train_kline[:,2]-train_kline[:,4]
    buy_ratio = np.divide(amt_bought,amt_bought+amt_sold)
    
    
    mean_price_cusum = np.cumsum(mean_price)
    n3 = len(windows)
    for i in range(n3):
        
        p1 = mean_price_cusum[windows[i]:]
        p2 = mean_price_cusum[:-windows[i]]
        
        price_means_normal[windows[i]:,i] = (p1-p2)/windows[i]
        price_means_normal[:windows[i],i] = np.nan

    
    deltas = np.zeros((m,int(math.factorial(n3)/(math.factorial(2)*math.factorial(int(n3-2))))))
    index = 0
    for i in range(n3):
        for j in range(i+1,n3):
            deltas[:,index] = price_means_normal[:,i]-price_means_normal[:,j]
            index += 1
    
    
    
    for i in range(m):
        curr_price = np.mean(train_kline[i,[1,4]].reshape((1,2)),axis=1,keepdims=True)
        if i == 0:
            price_means[i,:] = mean_price[i]
            volume_means[i,:] = train_kline[i,5]
            price_range_high[i,:] = train_kline[i,2]-curr_price
            price_range_low[i,:] = train_kline[i,3]-curr_price
            avg_strength[i,:] = 0
            buy_ratio_smoothed[i,:] = buy_ratio[i]
            amt_bought_smoothed[i,:] = amt_bought[i]
            amt_sold_smoothed[i,:] = amt_sold[i]
        else:
            price_means[i,:] = np.multiply(price_factor,price_means[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),mean_price[i])
            volume_means[i,:] = np.multiply(price_factor,volume_means[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),volumes[i])
            avg_strength[i,:] = np.multiply(price_factor,avg_strength[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),price_sign[i])
            
            amt_bought_smoothed[i,:] = np.multiply(price_factor,amt_bought_smoothed[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),amt_bought[i])
            amt_sold_smoothed[i,:] = np.multiply(price_factor,amt_sold_smoothed[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),amt_sold[i])
            
            if not np.isnan(buy_ratio[i]):
                buy_ratio_smoothed[i,:] = np.multiply(price_factor,buy_ratio_smoothed[i-1,:].reshape((1,n2))) + np.multiply((1-price_factor),buy_ratio[i])
            else:
                buy_ratio_smoothed[i,:] = buy_ratio_smoothed[i-1,:]
            
            
            if train_kline[i,2]-curr_price > price_range_high[i-1]:
                price_range_high[i] = np.multiply(factor_high,price_range_high[i-1]) + np.multiply((1-factor_high),train_kline[i,2]-curr_price)
            else:
                price_range_high[i] = np.multiply(factor_low,price_range_high[i-1]) + np.multiply((1-factor_low),train_kline[i,2]-curr_price)
            if train_kline[i,3]-curr_price < price_range_low[i-1]:
                price_range_low[i] = np.multiply(factor_high,price_range_low[i-1]) + np.multiply((1-factor_high),train_kline[i,3]-curr_price)
            else:
                price_range_low[i] = np.multiply(factor_low,price_range_low[i-1]) + np.multiply((1-factor_low),train_kline[i,3]-curr_price)
    
    
    
    # interpolation step
    
    f = interp1d(mean_time.flatten(),mean_price,axis=0)
    actual_price = f(time_ax)
    f = interp1d(mean_time.flatten(),volumes)
    actual_volume = f(time_ax)
    f = interp1d(mean_time.flatten(),price_range_high.flatten())
    price_range_high_interp = f(time_ax)
    f = interp1d(mean_time.flatten(),price_range_low.flatten())
    price_range_low_interp = f(time_ax)
    f = interp1d(mean_time.flatten(),q_vol.flatten())
    q_vol_interp = f(time_ax)
    
    prices_smoothed = np.zeros((m2,n2))
    volume_smoothed = np.zeros((m2,n2))
    strength_smoothed = np.zeros((m2,n2))
    prices_delta = np.zeros((m2,n2))
    volume_delta = np.zeros((m2,n2))
    strength_delta = np.zeros((m2,n2))
    buy_ratio_smoothed_interp = np.zeros((m2,n2))
    amt_bought_smoothed_interp = np.zeros((m2,n2))
    amt_sold_smoothed_interp = np.zeros((m2,n2))
    deltas_interp = np.zeros((m2,n3))
    for i in range(n3):
        f = interp1d(mean_time.flatten(),deltas[:,i])
        deltas_interp[:,i] = f(time_ax)
    for i in range(n2):
        f = interp1d(mean_time.flatten(),price_means[:,i])
        prices_smoothed[:,i] = f(time_ax)
        f = interp1d(mean_time.flatten(),volume_means[:,i])
        volume_smoothed[:,i] = f(time_ax)
        f = interp1d(mean_time.flatten(),avg_strength[:,i])
        strength_smoothed[:,i] = f(time_ax)
        f = interp1d(mean_time.flatten(),buy_ratio_smoothed[:,i])
        buy_ratio_smoothed_interp[:,i] = f(time_ax)
        f = interp1d(mean_time.flatten(),amt_bought_smoothed[:,i])
        amt_bought_smoothed_interp[:,i] = f(time_ax)
        f = interp1d(mean_time.flatten(),amt_sold_smoothed[:,i])
        amt_sold_smoothed_interp[:,i] = f(time_ax)
        
    
    prices_delta = np.divide(np.gradient(prices_smoothed,axis=0),prices_smoothed)
    volume_delta = np.divide(np.gradient(volume_smoothed,axis=0),volume_smoothed)
    strength_delta = np.gradient(strength_smoothed,axis=0)
    
    norm_price = np.divide(prices_smoothed-actual_price.reshape((m2,1)),actual_price.reshape((m2,1)))
    norm_volume = np.divide(volume_smoothed-actual_volume.reshape((m2,1)),actual_volume.reshape((m2,1)))
    
    #norm_price_high = np.divide(price_range_high_interp.reshape((m2,1)),actual_price.reshape((m2,1)))
    #norm_price_low = np.divide(price_range_low_interp.reshape((m2,1)),actual_price.reshape((m2,1)))
    
    #price_incr = np.divide(actual_price.reshape((m2,1))-prices_smoothed,prices_smoothed)
    
    vol_per_percent_incr = np.divide(amt_bought_smoothed_interp,np.multiply(np.multiply(volume_smoothed,scales),buy_ratio_smoothed_interp))
    
    vol_per_percent_decr = np.divide(amt_sold_smoothed_interp,np.multiply(np.multiply(volume_smoothed,scales),1-buy_ratio_smoothed_interp))
    
    vol_sens_ratio = np.divide(vol_per_percent_incr,vol_per_percent_decr)
    '''
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(time_ax/1000,actual_price,'.-')
    ax2 = plt.subplot(212,sharex=ax1)
    plt.plot(time_ax/1000,vol_sens_ratio[:,1:])
    '''
    p_ax = np.array([-100,-1,1,2,3,4,5],dtype='float')
    p_norm_ax = np.array([-2,-1,1,2,3,4,5],dtype='float')
    p_len = 1 #len(p_ax)
    outputs_buy = np.zeros((m2,len(p_ax)));
    
    # set up inputs matrix
    max_price = np.mean(train_kline[:,[1,2,4]].reshape((m,3)),axis=1,keepdims=True)
    f = interp1d(mean_time.flatten(),max_price[:,0])
    max_price_interp = f(time_ax)
    m,n= deltas.shape
    print('{} \t {}'.format(m,n))
    inputs_buy = np.concatenate((np.zeros((len(outputs_buy),1)),deltas_interp,norm_price,prices_delta[:,-1].reshape((m2,1)),strength_smoothed[:,-1].reshape((m2,1)),strength_delta[:,0].reshape((m2,1)),vol_per_percent_incr[:,-1].reshape((m2,1)),vol_per_percent_decr[:,-1].reshape((m2,1)),vol_sens_ratio[:,[0,3]].reshape((m2,2)),buy_ratio_smoothed_interp),axis=1)
    # set up outputs matrix and feedback on inputs
    for i in range(m2):
        v = np.logical_and(train_kline[:,0]>=time_ax[i], train_kline[:,0]<=(time_ax[i]+t_forecast*1000*60))
        if not any(v):
            outputs_buy[i,:] = np.nan
            if i > 0:
                inputs_buy[i,:p_len] = inputs_buy[i-1,:p_len]
            
            continue
        norm_max_price = 100*np.divide(max_price[v,:]-actual_price[i],actual_price[i])
        idx = np.where(np.max(norm_max_price)>p_ax)
        outputs_buy[i,np.max(idx)] = 1.0
            
        if i == 1:
            inputs_buy[i,:p_len] = np.matmul(outputs_buy[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
        elif i > 1:
            if np.any(np.isnan(outputs_buy[i-1,:])):
                inputs_buy[i,:p_len] = inputs_buy[i-1,:p_len]
            else:
                inputs_buy[i,:p_len] = (1-input_filter)*inputs_buy[i-1,:p_len] + input_filter*np.matmul(outputs_buy[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
                #inputs_buy[i,:p_len] = softmax(inputs_buy[i,:p_len].reshape((1,p_len)))
        
        
    p_ax = np.array([100,1,-1,-2,-3,-4,-5],dtype='float')
    p_norm_ax = np.array([2,1,-1,-2,-3,-4,-5],dtype='float')
    p_len = 1 #len(p_ax)
    outputs_sell = np.zeros((m2,len(p_ax)));
    
    # set up inputs matrix
    min_price = np.mean(train_kline[:,[1,3,4]].reshape((m,3)),axis=1,keepdims=True)
    f = interp1d(mean_time.flatten(),min_price[:,0])
    min_price_interp = f(time_ax)
    inputs_sell = np.concatenate((np.zeros((len(outputs_buy),1)),deltas_interp,norm_price,prices_delta[:,-1].reshape((m2,1)),strength_smoothed[:,-1].reshape((m2,1)),strength_delta[:,0].reshape((m2,1)),vol_per_percent_incr[:,-1].reshape((m2,1)),vol_per_percent_decr[:,-1].reshape((m2,1)),vol_sens_ratio[:,[0,3]].reshape((m2,2)),buy_ratio_smoothed_interp),axis=1)
    # set up outputs matrix and feedback on inputs
    for i in range(m2):
        v = np.logical_and(train_kline[:,0]>=time_ax[i], train_kline[:,0]<=(time_ax[i]+t_forecast*1000*60))
        if not any(v):
            outputs_sell[i,:] = np.nan
            if i > 0:
                inputs_sell[i,:p_len] = inputs_sell[i-1,:p_len]
            
            continue
        norm_max_price = 100*np.divide(min_price[v,:]-actual_price[i],actual_price[i])
        idx = np.where(np.min(norm_max_price)<p_ax)
        outputs_sell[i,np.max(idx)] = 1.0
            
        if i == 1:
            inputs_sell[i,:p_len] = np.matmul(outputs_sell[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
        elif i > 1:
            if np.any(np.isnan(outputs_sell[i-1,:])):
                inputs_sell[i,:p_len] = inputs_sell[i-1,:p_len]
            else:
                inputs_sell[i,:p_len] = (1-input_filter)*inputs_sell[i-1,:p_len] + input_filter*np.matmul(outputs_sell[i-1,:].reshape((1,len(p_norm_ax))),p_norm_ax)
                #inputs_sell[i,:p_len] = softmax(inputs_sell[i,:p_len].reshape((1,p_len)))
    
    valid_buy = np.logical_and(np.all(np.isfinite(inputs_buy),axis=1),np.all(np.isfinite(outputs_buy),axis=1))
    valid_buy[:max(scales)+1] = False
    valid_sell = np.logical_and(np.all(np.isfinite(inputs_sell),axis=1),np.all(np.isfinite(outputs_sell),axis=1))
    valid = np.logical_and(valid_buy,valid_sell)
    
    inputs_buy = inputs_buy[valid,:]
    inputs_buy = inputs_buy[:,1:]
    #inputs_buy[:,0] = inputs_buy[:,0] + np.random.randn(len(inputs_buy))*0.6
    #inputs_buy[:,:p_len] = softmax(inputs_buy[:,:p_len].reshape((len(inputs_buy),p_len)))
    outputs_buy = outputs_buy[valid,:]
    inputs_sell = inputs_sell[valid,:]
    inputs_sell = inputs_sell[:,1:]
    #inputs_sell[:,0] = inputs_sell[:,0] + np.random.randn(len(inputs_buy))*0.6
    #inputs_sell[:,:p_len] = softmax(inputs_sell[:,:p_len].reshape((len(inputs_sell),p_len)))
    outputs_sell = outputs_sell[valid,:]
    return inputs_buy,outputs_buy,inputs_sell,outputs_sell,actual_price[valid],max_price_interp[valid],min_price_interp[valid],time_ax[valid]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return e_x / np.sum(e_x,axis=1,keepdims=True)