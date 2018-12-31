# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:05:40 2018

@author: marchims
"""


import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import operator


seed = 128
rng = np.random.RandomState(seed)

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

def forward_prop(x,W,b):
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

def scale_inputs(x):
    min_x = np.min(x,axis=0,keepdims=True)
    max_x = np.max(x,axis=0,keepdims=True)
    
    valid = max_x-min_x>0
    
    x[:,valid[0,:]] = np.divide(np.subtract(x[:,valid[0,:]],min_x[valid]),max_x[valid]-min_x[valid])
    return x
    
    
inputs = pd.read_excel('Inputs.xlsx',header=None).as_matrix()
targets = pd.read_excel('Outputs.xlsx',header=None).as_matrix()
prices = pd.read_excel('Price.xlsx',header=None).as_matrix()

input_scaled = scale_inputs(inputs)

# number of neurons in each layer
num_in = inputs.shape[1]
m = inputs.shape[0]
num_hidden = [10,10,10,10,10]
num_out = targets.shape[1]
epochs = 50000
batch_size = 128
learning_rate = 0.013
keep_prob = 1
val_ratio = 0.50
test_ratio = 0.01
val_counts = 0
val_count_max = 200

# define placeholders
x = tf.placeholder(tf.float32, [None, num_in])
y = tf.placeholder(tf.int64, [None])

W,b = initialize_weights(num_in,num_hidden,num_out)

Z,A = forward_prop(x,W,b)
output_layer = A['layer{}'.format(len(W)-1)]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

test_idx = int((1-test_ratio)*m)
idx = np.random.permutation(test_idx)
val_idx,train_idx = idx[:int(val_ratio*m)],idx[int(val_ratio*m):]

targets_index = np.zeros(m)
for  i in range(m):
    targets_index[i], _ = max(enumerate(targets[i]), key=operator.itemgetter(1))

init = tf.global_variables_initializer()

plt.figure(1)
plt.ion()
ax1 = plt.subplot(211)
plt.plot(prices[test_idx:,0],prices[test_idx:,1],'.')
ax2 = plt.subplot(212,sharex=ax1)

p_ax = [-1,0.5,1,2,3,4,5]

sess = tf.Session()

sess.run(init)

with sess.as_default():
    ### for each epoch, do:♦
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    min_val_cost = np.inf
    for epoch in range(epochs):
    
        _, avg_cost = sess.run([optimizer, cost], feed_dict = {x: input_scaled[train_idx], y: targets_index[train_idx]})
        
        # Validation
        val_cost = cost.eval({x: input_scaled[test_idx:], y: targets_index[test_idx:]})
        if val_cost < min_val_cost:
            min_val_cost = val_cost
            val_count = 0
        else:
            val_count += 1
            if val_count >= val_count_max:
                #print("Epoch:{} \t\t Cost:{:.9f} \t\t Val Cost:{:.9f}".format(epoch+1,avg_cost,min_val_cost))
                print("\nValidation minimized!")
                break
        
        
        if epoch%100==0:
            pred_temp = tf.equal(tf.argmax(output_layer, 1), y)
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            train_acc = accuracy.eval({x: input_scaled[train_idx], y: targets_index[train_idx]})
            val_acc = cost.eval({x: input_scaled[test_idx:], y: targets_index[test_idx:]})
            print("Epoch:{} \t\t Cost:{:.4f} \t\t Training Accuracy:{:.4f} \t\t Validation Accuracy:{:.4f}".format(epoch+1,avg_cost,train_acc,val_acc))
            
            predict_test = tf.nn.softmax(output_layer).eval({x: input_scaled[test_idx:], y: targets_index[test_idx:]})
            est_price = np.matmul(predict_test,np.array(p_ax))
            
            ax2.clear()
            plt.plot(prices[test_idx:,0],est_price,'g.',prices[test_idx:,0],np.matmul(targets[test_idx:],np.array(p_ax)),'r.')
            
            plt.pause(0.05)
            plt.draw()
            #print("Epoch:{} \t\t Cost:{:.9f} \t\t Val Cost:{:.9f} \t\t Count:{}".format(epoch+1,avg_cost,val_cost,val_count))
    
    print("\nTraining complete!")
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), y)
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    model_out = tf.nn.softmax(output_layer).eval({x: input_scaled, y: targets_index})
    train_acc = accuracy.eval({x: input_scaled[train_idx], y: targets_index[train_idx]})
    val_acc = accuracy.eval({x: input_scaled[val_idx], y: targets_index[val_idx]})
    test_acc = accuracy.eval({x: input_scaled[test_idx:], y: targets_index[test_idx:]})
    
    #layer0_Z = Z['layer0'].eval({x: input_scaled, y: targets_index})
    #layer1_Z = Z['layer1'].eval({x: input_scaled, y: targets_index})
    #layer0_A = A['layer0'].eval({x: input_scaled, y: targets_index})
    #layer1_A = A['layer1'].eval({x: input_scaled, y: targets_index})
    weights = W['layer0'].eval()
    bias = b['layer0'].eval()
    print("Training Accuracy:{} \nValidation Accuracy:{}".format(train_acc,test_acc))
    
    predict_train = tf.nn.softmax(output_layer).eval({x: input_scaled[train_idx], y: targets_index[train_idx]})
    predict_val = tf.nn.softmax(output_layer).eval({x: input_scaled[val_idx], y: targets_index[val_idx]})
    predict_test = tf.nn.softmax(output_layer).eval({x: input_scaled[test_idx:], y: targets_index[test_idx:]})
    

inputs_test = pd.read_excel('InputsTest.xlsx',header=None).as_matrix()
targets_test = pd.read_excel('OutputsTest.xlsx',header=None).as_matrix()
prices_test = pd.read_excel('PriceTest.xlsx',header=None).as_matrix()
 
input_scaled_test = scale_inputs(inputs_test)
   
with sess.as_default():
    
    input_row = input_scaled_test[0,:]
    input_row = input_row.reshape((1,len(input_row)))
    input_row[0,:7] = np.zeros((1,7))
    input_row[0,0] = 1
    L = len(targets_test)
    est_perc = np.zeros((L,1))
    p_arr = np.array(p_ax)
    index = 0
    plt.figure(1)
    plt.ion()
    ax1 = plt.subplot(211)
    plt.plot(prices_test[:,1],'.')
    ax2 = plt.subplot(212,sharex=ax1)
    plt.plot(np.matmul(targets_test,p_arr),'go')
    for i in range(L):
        model_out = tf.nn.softmax(output_layer).eval({x: input_row})
        est_perc[index] = np.matmul(model_out,p_arr)
        plt.plot(index,est_perc[index],'rx')
        index += 1
        input_row = input_scaled_test[i+1,:]
        input_row = input_row.reshape((1,len(input_row)))
        input_row[0,:model_out.shape[1]] = (input_row[0,:model_out.shape[1]].reshape((1,model_out.shape[1]))+model_out)/2
        
            
        plt.pause(0.05)
        plt.draw()
    
#pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
'''
input_row = input_scaled[test_idx,:]
input_row = input_row.reshape((1,len(input_row)))
input_row[0,:7] = np.zeros((1,7))
input_row[0,0] = 1
L = len(targets_index)-test_idx
est_perc = np.zeros((L,1))
p_arr = np.array(p_ax)
index = 0
plt.figure(2)
test_idx = test_idx + int(0.5*(len(targets_index)-test_idx))
plt.ion()
for i in range(test_idx,len(targets_index)-1):
    model_out = tf.nn.softmax(output_layer).eval({x: input_row})
    est_perc[index] = np.matmul(model_out,p_arr)
    plt.plot(index,est_perc[index],'rx',index,np.matmul(targets[i],p_arr),'go')
    index += 1
    input_row = input_scaled[i+1,:]
    input_row = input_row.reshape((1,len(input_row)))
    input_row[0,:model_out.shape[1]] = (input_row[0,:model_out.shape[1]].reshape((1,model_out.shape[1]))+model_out)/2
    
        
    plt.pause(0.05)
    plt.draw()
    
'''
'''
est_price = np.matmul(predict_val,np.array(p_ax))
plt.figure(1)
ax1 =plt.subplot(211)
plt.plot(prices[val_idx,0],prices[val_idx,1],'.')
ax2 = plt.subplot(212,sharex=ax1)
plt.plot(prices[val_idx,0],est_price,'g.',prices[val_idx,0],np.matmul(targets[val_idx],np.array(p_ax)),'r.')

est_price = np.matmul(predict_train,np.array(p_ax))
plt.figure(2)
ax1 =plt.subplot(211)
plt.plot(prices[train_idx,0],prices[train_idx,1],'.')
ax2 = plt.subplot(212,sharex=ax1)
plt.plot(prices[train_idx,0],est_price,'g.',prices[train_idx,0],np.matmul(targets[train_idx],np.array(p_ax)),'r.')

est_price = np.matmul(predict_test,np.array(p_ax))
plt.figure(3)
ax1 =plt.subplot(211)
plt.plot(prices[test_idx:,0],prices[test_idx:,1],'.')
ax2 = plt.subplot(212,sharex=ax1)
plt.plot(prices[test_idx:,0],est_price,'g.',prices[test_idx:,0],np.matmul(targets[test_idx:],np.array(p_ax)),'r.')
'''