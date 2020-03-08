# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:42:38 2020

@author: Aiyun
"""
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt

#--------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt    


#-------------------------心拍截取-------------------
def heartbeat(file0):
    '''
    file0:下载的MITAB数据
    
    '''
    N_Seg=[]; SVEB_Seg=[];  VEB_Seg=[]; F_Seg=[] ; Q_Seg=[];
    #--------去掉指定的四个导联的头文件---------
    De_file=[panth[:-1]+'\\102.hea',panth[:-1]+'\\104.hea',panth[:-1]+'\\107.hea',panth[:-1]+'\\217.hea']
    file=list(set(file0).difference(set(De_file)))
    
    for f in range(len(file)) :
        annotation= wfdb.rdann(panth+file[f][-7:-4],'atr')
        record_name=annotation.record_name    #读取记录名称
        Record=wfdb.rdsamp(panth+record_name)[0][:,0] #一般只取一个导联
        record=WTfilt_1d(Record)         #小波去噪
        label=annotation.symbol  #心拍标签列表
        label_index=annotation.sample   #标签索引列表
        for j in range(len(label_index)):
            if label_index[j]>=144  and (label_index[j]+180)<=650000:
                if label[j]=='N' or label[j]=='.' or label[j]=='L' or label[j]=='R' or label[j]=='e' or label[j]=='j':
                    Seg=record[label_index[j]-144:label_index[j]+180]#R峰的前0.4s和后0.5s
                    segment=resample(Seg,251, axis=0)  #重采样到251
                    N_Seg.append(segment)
                    
                if label[j]=='A' or label[j]=='a' or label[j]=='J' or label[j]=='S':
                    
                    Seg=record[label_index[j]-144:label_index[j]+180]
                    segment=resample(Seg,251, axis=0) 
                    SVEB_Seg.append(segment)
                    
                if label[j]=='V' or label[j]=='E':
                   
                    Seg=record[label_index[j]-144:label_index[j]+180]
                    segment=resample(Seg,251, axis=0)  
                    VEB_Seg.append(segment)
                    
                if label[j]=='F':
                    
                    Seg=record[label_index[j]-144:label_index[j+1]+180]
                    segment=resample(Seg,251, axis=0)  
                    F_Seg.append(segment)
                if  label[j]=='/' or label[j]=='f' or label[j]=='Q':
                    
                    Seg=record[label_index[j]-144:label_index[j]+180]
                    segment=resample(Seg,251, axis=0)  
                    Q_Seg.append(segment)
                    
    N_segement=np.array(N_Seg)
    SVEB_segement=np.array(SVEB_Seg)
    VEB_segement=np.array(VEB_Seg)
    F_segement=np.array(F_Seg)
    Q_segement=np.array(Q_Seg)
    
    label_N=np.zeros(N_segement.shape[0])
    label_SVEB=np.ones(SVEB_segement.shape[0])
    label_VEB=np.ones(VEB_segement.shape[0])*2
    label_F=np.ones(F_segement.shape[0])*3
    label_Q=np.ones(Q_segement.shape[0])*4
                    
    Data=np.concatenate((N_segement,SVEB_segement,VEB_segement,F_segement),axis=0)
    Label=np.concatenate((label_N,label_SVEB,label_VEB,label_F,),axis=0)
    
    return  Data, Label

#-----------------------心拍截取和保存---------------------
#建议一次性截取和保存，不需要重复操作，下次训练和测试的时候，直接load
panth='E:/.../MIT_BIH/'
file = glob.glob(panth+'*.hea')
Data, Label=heartbeat(file)

Data=np.save('.../'+'Data',Data)
Label=np.save('..../'+'Label',Label)







