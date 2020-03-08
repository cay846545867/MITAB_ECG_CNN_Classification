# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:10:03 2020

@author: Aiyun
"""
import numpy as np
import random
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from keras.layers import Input
from keras  import optimizers
from keras.utils import np_utils
import Pmodel 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler  #选择保留最佳训练模型
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn import metrics#模型评估
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#----------------------导入数据------------
data=np.load('E:/..../Data.npy')
label=np.load('E:/.../Label.npy')

data = np.expand_dims(data, axis=2)
label=np_utils.to_categorical(label,4)  #-----------转化为one-hot标签
#------打乱数据------
Data,Label=Pmodel.shuffle_set(data,label)

#---------------------5折交叉验证训练和测试------
folds=5
jiange=int(Data.shape[0]/folds)
Con_Matr=[]  #存储每一折的混淆矩阵
F1=[]        #存储每一折的f1
Acc=[]        #存储每一折的acc
Loss=[]      #存储每一折的loss
for i in range(1,6):  
    X_train,X_test,y_train,y_test=Pmodel.cross_Kfolds(Data,Label,folds,jiange,i-1,i)
    inputs1=Input(shape=(251, 1 ))
    inputs2=Input(shape=(251, 1 ))
    inputs3=Input(shape=(251,1))
    model = Pmodel.model(inputs1,inputs2,inputs3)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy']
                  )
    
    filepath="E:/..../xxxx.hdf5"#保存模型的路径
    checkpoint = ModelCheckpoint(filepath, verbose=2,
                              monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')

    #lr_scheduler = LearningRateScheduler(lr_schedule)
    
    #lr_scheduler=0.001
    #callback_lists = [checkpoint, lr_scheduler]
    
    callback_lists = [checkpoint]
    
    
    history = model.fit([X_train,X_train,X_train],y_train,validation_data=([X_test,X_test,X_test],y_test),class_weight = 'auto',
                    callbacks=callback_lists,epochs=100,batch_size=64)
    
    inputs1=Input(shape=(251,1 ))
    inputs2=Input(shape=(251,1 ))
    inputs3=Input(shape=(251,1))
    model = Pmodel.model(inputs1,inputs2,inputs3)

    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy']
                  )
    print('\ntesting(i).....')

    #Evaluate the model with the metrics  we defined earlier
    loss,accuracy=model.evaluate([X_test,X_test,X_test],y_test)
    Acc.append(accuracy)
    Loss.append(loss)
    
    y_pred=model.predict([X_test,X_test,X_test])
    #f1_score和confusion_matrix不支持one_hot，只支持普通标签
    y_test=np.argmax(y_test,axis=1)
    y_pred=np.argmax(y_pred,axis=1)
    f1=metrics.f1_score(y_test, y_pred, average='macro')
    F1.append(f1)
    con_matr=confusion_matrix(y_test, y_pred)
    Con_Matr.append(con_matr)

#----------------------------------------总体评估----------------------------------------
print("%.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(Acc), np.std(Acc)))









