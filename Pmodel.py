# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:40:53 2020

@author: Aiyun
"""
import numpy as np
from keras import backend as K
from keras.layers import MaxPooling1D,AveragePooling1D,Conv1D,MaxPool1D,add,Flatten,Dense,regularizers,Concatenate,Activation
import random
from keras.models import Sequential,load_model,Model

def model(inputs1,inputs2,inputs3):
    conv11 = Conv1D(filters=8,kernel_size=4,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv11=Activation('relu')(conv11)
    pool11=MaxPool1D(pool_size=2)(conv11)
    conv12 = Conv1D(filters=16,kernel_size=6,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(pool11)
    conv12=Activation('relu')(conv12)
    pool12=MaxPool1D(pool_size=2)(conv12)
    #F1=Flatten(pool12)
    
    conv21 = Conv1D(filters=8,kernel_size=6,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv21=Activation('relu')(conv21)
    pool21=MaxPool1D(pool_size=2)(conv21)
    conv22 = Conv1D(filters=16,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(pool21)
    conv22=Activation('relu')(conv22)
    pool22=MaxPool1D(pool_size=2)(conv22)
    #F2=Flatten(pool22)
    
    
    conv31 = Conv1D(filters=8,kernel_size=8,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(inputs1)
    conv31=Activation('relu')(conv31)
    pool31=MaxPool1D(pool_size=2)(conv31)
    conv32 = Conv1D(filters=16,kernel_size=10,kernel_initializer="he_uniform",strides=1,padding='same',
                   use_bias=False,kernel_regularizer=regularizers.l2(0.0001))(pool31)
    conv32=Activation('relu')(conv32)
    pool32=MaxPool1D(pool_size=2)(conv32)
    #F3=Flatten(pool32)
   
    P_C=Concatenate(axis=1)([pool12,pool22,pool32])

    F1=Flatten()(P_C)
    
    Dense1=Dense(256)(F1)
    
    Dense2=Dense(32)(Dense1)
    
    res = Dense(4, activation='softmax')(Dense2) 
    res = Model(inputs=[inputs1,inputs2,inputs3], outputs=[res], name="ResNet")
    return res
    
def cross_Kfolds(data1,data2,folds,jiange,start_index,end_index):
    #分割数据集;data1,data2分别为数据和标签
    df_Xtest=data1[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_Xtrain=data1[end_index*jiange:len(data1[:,0,0])*jiange]
    else:
        df_Xtrain00=data1[0*jiange:start_index*jiange]
        df_Xtrain01=data1[end_index*jiange:len(data1[:,0,0])*jiange]
        df_Xtrain=np.concatenate((df_Xtrain00,df_Xtrain01),axis=0)
    #分割标签
    df_ytest=data2[start_index*jiange:end_index*jiange] #数据刚好可以做k折交叉验证。
    if start_index==0:
        df_ytrain=data2[end_index*jiange:len(data2[:,0])*jiange]
    else:
        df_ytrain00=data2[0*jiange:start_index*jiange]
        df_ytrain01=data2[end_index*jiange:len(data2[:,0])*jiange]
        df_ytrain=np.concatenate((df_ytrain00,df_ytrain01),axis=0)
        
    return df_Xtrain,df_Xtest,df_ytrain,df_ytest  #每一折的训练和测试

#-----------------------------打乱数据集----------------------------------
def shuffle_set(data,label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data,Label
    