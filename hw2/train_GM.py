import numpy as np
import scipy
import pandas as pd

import time
#help("scipy")

import math
from copy import deepcopy
import sys
import os




def warmstart(path,numOfFeatures):
    theta=np.load(path)
    print(theta)
    print(type(theta))
    bias=theta[numOfFeatures]
    theta=np.delete(theta,numOfFeatures,0)
    return theta,bias

def sigmoid(x):
    #print(type(x))
    #print(x.shape)
    xarray=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i]<0:
            xarray[i]=1-1/(1+math.exp(x[i]))
        else:
            xarray[i]=1 / (1 + math.exp(-x[i]))

    return xarray

def loss_func(X,Y):
    loss=-1
    print("Loss function")
    
    return loss



def readData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead)
    #print(data)
    #print(type(data))
    return data
def NormMatrix():
    NM=[]
    NM.append(50)
    for i in range(9):
        NM.append(1)
    NM.append(1000000)
    for i in range(67):
        NM.append(1)
    NM.append(10000)
    NM.append(3000)
    NM.append(70)
    for i in range(42):
        NM.append(1)
    print(NM,'NM=?')
    NM=np.asarray(NM)
    print(NM.shape)
    return NM

def extendDataByFnlwgt(trainX,trainY):
    extendData=[]
    extendDataY=[]
    for i in range(len(trainX)):
        #print((int)(trainX[i][10]/1500),'fnlwgt/1500')
        for j in range((int)(trainX[i][10]/20000)):
            extendData.append(trainX[i])
            extendDataY.append(trainY[i])
    extendData=np.asarray(extendData)
    extendDataY=np.asarray(extendDataY)
    return extendData,extendDataY

def createFeatureMatrix(Xi):
    feature=[]
    for i in range(featureNum):
        tlist=np.zeros(len(Xi))
        tlist=np.asarray(tlist)
        feature.append(tlist)
    for i in range(Xi.shape[0]):
        if i%100000==0:
            print(i,'i in feature')
        for j in range(featureNum):
            feature[j][i]=Xi[i][j]
    feature=np.asarray(feature)
    
    #print(feature,'Feature')
    print(type(feature),'F T')
    print(feature.shape,'F S')
    print(feature[0][0],'F0')
    print(feature[0][54],'F54')
    return feature

def validation(testX,testY,theta,bias,start,end,mid):
    print('validation start')
    print(testX.shape,'T X S')
    print(theta,'theta')
    print(theta.shape,'theta S')
    
    y_=sigmoid(np.dot(testX,theta)+bias)
    print(y_,'y_')
    backupY=np.zeros(y_.shape[0])
    for i in range(y_.shape[0]):
        backupY[i]=y_[i]
    print(y_.shape,'y_ shape')
    print(testY.shape,'y shape')
    while start<end:
        #print(start,'start')
        for i in range(y_.shape[0]):
            y_[i]=backupY[i]
        for i in range(y_.shape[0]):
            if y_[i]>start:
                y_[i]=1
            else:
                y_[i]=0
        totalHit=0.0
        for i in range(y_.shape[0]):
            if y_[i]-testY[i]==0:
                totalHit=totalHit+1
        print(totalHit,'Total hit')
        print(totalHit/y_.shape[0],"Accuracy of theshold:",start)
        start=start+mid
def extendX2(trainX):
    X2=[]
    print("X2 and X3 X4!")
    for i in range(len(trainX)):
        tlist=np.zeros(24)
        tlist=np.asarray(tlist)
        X2.append(tlist)
    for i in range(len(trainX)):
        X2[i][0]=trainX[i][0]*trainX[i][0]
        X2[i][1]=trainX[i][78]*trainX[i][78]
        X2[i][2]=trainX[i][79]*trainX[i][79]
        X2[i][3]=trainX[i][80]*trainX[i][80]
        X2[i][4]=trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][5]=trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][6]=trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][7]=trainX[i][80]*trainX[i][80]*trainX[i][80]
        X2[i][8]=trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][9]=trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][10]=trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][11]=trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]
        X2[i][12]=trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][13]=trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][14]=trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][15]=trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]
        X2[i][16]=trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][17]=trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][18]=trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][19]=trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]
        
        X2[i][20]=trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][21]=trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][22]=trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][23]=trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]
        """
        X2[i][24]=trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]*trainX[i][0]
        X2[i][25]=trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]*trainX[i][78]
        X2[i][26]=trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]*trainX[i][79]
        X2[i][27]=trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]*trainX[i][80]
        """
    X2=np.asarray(X2)
    print(X2.shape,'X2S')
    print(trainX.shape,'tXS')
    merge=np.append(trainX,X2,axis=1)
    print(merge,'merge')
    return merge

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

def GD(trainX,trainY,iterNum,learningRate,isWarmStart,pathOfWarmStart,featureNum,isShuffle):
    #iterNum=100000
    #learningRate=0.001
#    dataValue=np.array(dataValue)

    NorM=NormMatrix()
    trainX=trainX.values
    trainY=trainY.values
    #trainX,trainY=extendDataByFnlwgt(trainX,trainY)
    if isShuffle==True:
        trainX,trainY=shuffle_in_unison_scary(trainX,trainY)

    trainX=np.multiply(trainX,1/NorM)
    trainX=extendX2(trainX)
    print(len(trainX))
    print(len(trainX[0]),'data0')
    print(trainX[0],'data0')
    print(trainX.shape,'txS')
    print(type(trainX[1]),' DV1',type(trainX[1][0]),' DV10')
    print(type(trainX[0]),' DV0',type(trainX[1][0]),' DV10')
    #normalize Matrix
    NormM=NormMatrix()
    #normalize end
    if isWarmStart==True:
        theta,bias=warmstart(pathOfWarmStart,featureNum)
    else:
        temp=0
        theta=np.zeros(featureNum)
        theta=np.asarray(theta)
        bias=0
   
    #theta2=np.zeros(featureNum)
    #theta2=np.asarray(theta2)
    
    


    tempTheta=np.zeros(featureNum)
    #tempTheta2=np.zeros(featureNum)
    
    tempErr=np.zeros(featureNum)
    #tempErr2=np.zeros(featureNum)
    tempBiasErr=0

    


    m=len(trainX)*21
    print(theta,'theta')
    print( tempTheta,'temptheta')
    print(len(trainX),'len data')
    #Start LR
    #test
#    print(trainX[0][0][0:9])
    #testEnd
    Xi=[]
    y=[]
  #  print(Xi,'Xi')
    Xi=trainX
    Xi=np.asarray(Xi)
    
    print(Xi[0],'Xi0')
    print(type(Xi),'TXi=?')
    print(Xi.shape)
    templr=(learningRate/(len(trainX)))
    y=trainY
    print(y[0],'Y0')
    print(y.shape)
    y=y.reshape(len(trainX),)
    print(y.shape,'Y SHAPE RESHAPE')
    

    #Gaussian  Initial
    u0=np.zeros(featureNum)
    u1=np.zeros(featureNum)
    countu0=0
    countu1=0
    for i in range(Xi.shape[0]):
        if y[i]==1:
            u1=u1+Xi[i]
            countu1+=1
        elif y[i]==0:
            u0=u0+Xi[i]
            countu0+=1
        else:
            print("error? y=",y[i])
    print(countu0,'CU0',countu1,"CU1")
    print(u0,'u0',u1,'u1')
    u0=u0/float(countu0)
    u1=u1/float(countu1)
    u0=u0.reshape(-1,1)
    u1=u1.reshape(-1,1)
    E0=np.matrix(np.zeros((featureNum,featureNum)))
    E1=np.matrix(np.zeros((featureNum,featureNum)))
    print(u0,'u0 AFTER',u1,'u1  AFTER')
    #Gaussian
    for i in range(Xi.shape[0]):
        Xg=np.matrix(Xi[i]).transpose()
        if y[i]==1:
            difu=Xg-u1
            E1=E1+difu*difu.transpose()
        else:
            difu=Xg-u0
            E0=E0+difu*difu.transpose()
    E=E1/countu1+E0/countu0
    print(E,'E')
    print(E.shape,'ES')
    print(np.linalg.det(E),'detE')
    Einverse=np.linalg.pinv(E)
    #u0=u0.reshape(u0.shape[0],1)
    #u1=u1.reshape(u1.shape[0],1)
    print(u1.shape,'u1s')
    print(u1.transpose().shape,'UTS')
    print(Einverse.shape,'EINV S')
    w=(u1-u0).transpose()*Einverse
    b=-0.5*u1.transpose()*Einverse*u1
    b=b+0.5*u0.transpose()*Einverse*u0
    b=b+np.log(countu1/countu0)
    print(type(w))
    w=np.array(w)
    w=w.reshape(147,)
    b=b.reshape(1,)
    print(b,'b')
    print(b.shape,'b S')
    print(type(bias))
    bias=b.item(0)
    print(type(bias),'BS T')
    print(w,'w')
    print(w.shape,'ws')
    print(bias,'bias')

    """
    RegularParam=1
    tempRegular=RegularParam*templr

    #feature
    feature=createFeatureMatrix(Xi)

    #Show the params
    print('Iteration times :',iterNum)
    print('Learning rate is:',learningRate)
    print('Warm Start is:',isWarmStart)
    print('Shape of input:',Xi.shape)
    print('Shape of feature matrix:',feature.shape)
    print('Regular Param and Pre-cal Param:',RegularParam,tempRegular)
    print('Is Shuffle:',isShuffle)

    #Main iter part
    for iter in range(iterNum):
        timeStart=time.time()
        if iter%5000==0:
            print(iter,'Num of iter now--------------------------------------------------')
        sumOfError=0
        y_=sigmoid(np.dot(Xi,theta)+bias)

        difY=y_-y
       # print(difY.shape,'dfy Shape')
        sumOfError=sumOfError+difY.sum()
        tempErr=tempErr+np.dot(feature,difY)
        tempBiasErr=tempBiasErr+difY.sum()
        timeMid=time.time()
        theta=theta-np.dot(templr,tempErr)-np.dot(theta,tempRegular)
        bias=bias-templr*tempBiasErr
        tempErr=np.zeros(featureNum)
        tempBiasErr=0
        if iter % 5000==0 or iter==iterNum-1:
  #      print(bias,'Bias')
            print(sumOfError,'SOE FINAL')       
        timeEnd=time.time()
   #     print(timeEnd-timeStart,'Time for 1 iter')
   """
    #validation
    validation(Xi,y,w,bias,0.1,0.9,0.05)
    #validation(Xi,y,theta,bias,0.1,0.9,0.05)
    #SAAVEMODEL
    theta=np.append(w,bias)
    print(theta)
    savepath='model_GM.npy'
    np.save(savepath,theta)
    
    print('Save Model Success, Model name=',savepath)



#LR(trainX,trainY,Epoch,LearningRate)
isWS=False
WSPath='./model_norm_Bias_3_NOfnlwgt_valid_X2_X3_X4.npy'
iterateNum=100000
LearningRate=0.1
featureNum=147
isShuffle=True

#tx_path='./train_X'
#ty_path='./train_Y'
tx_path=sys.argv[3]
ty_path=sys.argv[4]
GD(readData(tx_path,'infer'),readData(ty_path,None),iterateNum,LearningRate,isWS,WSPath,featureNum,isShuffle)
