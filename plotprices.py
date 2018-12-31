# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 00:11:08 2018

@author: marchims
"""
import matplotlib.pyplot as plt

ax1 =plt.subplot(211)
plt.plot(prices[val_idx,0],prices[val_idx,1],'.')
ax2 = plt.subplot(212,sharex=ax1)
plt.plot(prices[val_idx,0],est_price,'g.',prices[val_idx,0],np.matmul(targets[val_idx],np.array([0.5,1.0,2.0,3.0,4.0,5.0,0.0])),'r.')