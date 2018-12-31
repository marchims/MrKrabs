# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:52:01 2018

@author: marchims
"""

coin = coin_start
base = base_start
coin_best = coin_start
base_best = base_start
max_perc_trade = 0.02
min_coin_perc = 0.40
min_trade_amt = 0.002
model_out = softmax(eval_model(inputs,W,b))
m = inputs.shape[0]
model_index = np.zeros(m)
confidence =  np.zeros(m)
for  i in range(m):
    model_index[i], _ = max(enumerate(model_out[i]), key=operator.itemgetter(1))
    confidence[i] = 1#(np.max(model_out[i,:]))

plt.figure()
ax1 = plt.subplot(511)
plt.plot(mean_time,mean_price,'.-',time_ax,price_indexed,'rx')
plt.subplot(512,sharex=ax1)
plt.plot(time_ax,targets_index,'go',time_ax,model_index,'rx')
plt.subplot(513,sharex=ax1)
plt.subplot(514,sharex=ax1)
plt.subplot(515,sharex=ax1)
m = len(model_index)

for i in range(m):
    total = base + coin*price_indexed[i]
    total_best = base_best+coin_best*price_indexed[i]
    plt.subplot(513,sharex=ax1)
    plt.plot(time_ax[i],coin,'b.',time_ax[i],coin_best,'g.')
    plt.subplot(514,sharex=ax1)
    plt.plot(time_ax[i],base,'b.',time_ax[i],base_best,'g.')
    plt.subplot(515,sharex=ax1)
    plt.plot(time_ax[i],total/price_indexed[i],'b.',time_ax[i],total_best/price_indexed[i],'g.')
    #trade_amt = 0.1*base
    if model_index[i]==0 and confidence[i] > 0.75:
        # buy
        perc_trade = min(base/total,max_perc_trade)
        trade_amt =  1*perc_trade*total
        
        if base >= trade_amt and trade_amt > min_trade_amt:
            coin = trade_amt/price_indexed[i] + coin
            base = base - trade_amt
        '''
        if base*max_trade >= 0.002:
            coin = max_trade*base/mean_price[i] + coin
            base = base - base*max_trade
        '''
    elif model_index[i]==1 and confidence[i] > 0.75:
        perc_trade = min(max(0,coin*price_indexed[i]/total - min_coin_perc),max_perc_trade)
        trade_amt = 1*perc_trade*total
        if trade_amt/price_indexed[i] <= coin and trade_amt > min_trade_amt:
            base = trade_amt + base
            coin = coin - trade_amt/price_indexed[i]
        '''
        if max_trade*coin*mean_price[i] >= 0.002:
            base = max_trade*coin*mean_price[i] + base
            coin = coin - coin*max_trade
        '''
    
    #trade_amt = 0.1*base_best
    if outputs[i]==0:
        # buy
        perc_trade = min(base_best/total_best,max_perc_trade)
        trade_amt = 1*max_perc_trade*total_best
        if base_best >= trade_amt and trade_amt > min_trade_amt:
            coin_best = trade_amt/price_indexed[i] + coin_best
            base_best = base_best - trade_amt
        '''
        if base_best*max_trade >= 0.002:
            coin_best = max_trade*base_best/mean_price[i] + coin_best
            base_best = base_best - base_best*max_trade
        '''
    elif outputs[i]==1:
        perc_trade = min(max(0,coin_best*price_indexed[i]/total_best-min_coin_perc),max_perc_trade)
        trade_amt = 1*perc_trade*total_best
        if trade_amt/price_indexed[i] <= coin_best and trade_amt > min_trade_amt:
            base_best = trade_amt + base_best
            coin_best = coin_best - trade_amt/price_indexed[i]
        '''
        if max_trade*coin_best*mean_price[i] >= 0.002:
            base_best = max_trade*coin_best*mean_price[i] + base_best
            coin_best = coin_best - coin_best*max_trade
        '''
        