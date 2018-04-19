import numpy as np
import scipy
import pandas as pd

import time
#help("scipy")

import math
from copy import deepcopy
import sys
import os

#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation
from keras.callbacks import Callback
from keras.models import load_model

#from matplotlib import pyplot as plt
print('import end')


testDataPath=sys.argv[1]
modelPath='./_bestModel?dl=1'
outputpath=sys.argv[2]






#Read  path= path:'./train_X'   header=none,'infer'  
def readData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead)
    #print(data,'Raw data')
    #print(data.values,'Data values')
    print(type(data),'Type of data')
    return data


def seperateData(data):
    trainX=[]
    trainY=[]
    countId=0
    #dataValues=data.values
    timeStart=time.time()
    for i in range(data.shape[0]):
        onehot=np.zeros(7)
        #print(data[i][0])
        #onehot[data[i][0]]=1
        #print(onehot,'onehot')
        trainY.append(onehot)
        
        sep=[]
        sepMatrix=[]
        sep=np.array(data[i][1].split(' '))
        for j in range(48):
            #temp=sep[j*48:j*48+48]
            #print(temp[0],type(temp[0]),'before')
            temp=np.asarray(sep[j*48:j*48+48],dtype=np.float32)
            #print(type(temp[0]),'after')
            sepMatrix.append(temp)

        sepMatrix=np.asarray(sepMatrix)
#        print(sepMatrix.shape,'spm shape')
        trainX.append(sepMatrix)
            
        if i==0:
            print(type(np.array(data[i][1].split(' '))),'type Split')
    trainX=np.asarray(trainX)
    trainY=np.asarray(trainY)
    print(trainX.shape,'trainx shape ')
    print(trainY.shape,'trainy shape ')
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to sep Data")
    return trainX,trainY

testData=readData(testDataPath,'infer')
testData=testData.values
print(testData.shape,'TDS')
print(type(testData),'TDT')

model=load_model(modelPath)

testX,_=seperateData(testData)


testX=testX/255

testX=testX.reshape(testX.shape[0],48,48,1)
onehotAnswer=model.predict(testX)
print(type(onehotAnswer),'T OHans')
print(onehotAnswer.shape)
answer=[]
#temp=['id','label']
#answer.append(temp)
ans=onehotAnswer.argmax(axis=-1)
print(ans,'ans?')
print(type(ans),'T ans')
print(ans.shape,'ans shape')
for i in range(onehotAnswer.shape[0]):
    temp=[]
    temp.append(i)
    temp.append(ans[i])
    answer.append(temp)
"""
for i in range(onehotAnswer.shape[0]):
    label=-1
    for j in range(7):
        if onehotAnswer[i][j]>0.99:
            if label!=-1:
                print('Error? lable not -1')
            label=j
    temp=[]
    temp.append(i)
    temp.append(label)
    if label==-1:
        print('Error? lable is -1')
        print('The answer? is',onehotAnswer[i])
    answer.append(temp)
"""
print(answer[1],'ans0')
print(type(answer[1][0]),type(answer[1][1]))


print(answer)
col = ['id', 'label']
df_ans = pd.DataFrame(answer, index = None,columns=col)
print(df_ans)
df_ans.to_csv(outputpath, index=False,columns=None)
print('output success!, path is :',outputpath)