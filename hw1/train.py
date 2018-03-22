import numpy as np
import scipy
import pandas as pd

import time
#help("scipy")


def loss_func(X,Y):
    loss=-1
    print("Loss function")
    loss=(X-Y)*(X-Y)

    return loss

def mergeByData(data):
    processData=[]
    tempY,tempM,tempD='2014','1','1'
    print(tempY,tempM,tempD)
    countId=0
    da=data.values
    dayDataSum=[]
    for i in range(data.shape[0]):
        #print(da[i,0])
        year,month,day=da[i,0].split('/')
#        print(year,month,day)
        if tempY==year:
            if tempM==month:
                if tempD==day:
#                    print(countId)
                    tempData=[]
                    for k in range(24):
                        if da[i,k+2] == 'NR':
                            da[i,k+2]=0
                        tempData.append(da[i,k+2])
#                    print(tempData)
                    tempData=np.array(tempData,dtype=float)
                    dayDataSum.append(tempData)
                else:
                    tempD=day
                    countId=countId+1
                    if len(dayDataSum)>1:
                        processData.append(dayDataSum)
 #                   print(len(dayDataSum),' LEN ')
                    dayDataSum=[]
                    tempData=[]
                    for k in range(24):
                        if da[i,k+2] == 'NR':
                            da[i,k+2]=0
                        tempData.append(da[i,k+2])
                        
                    tempData=np.array(tempData,dtype=float)
                    dayDataSum.append(tempData)
            else:
                tempM=month
                tempD=day
#                countId=countId+1
                countId=countId+1
                if len(dayDataSum)>1:
                    processData.append(dayDataSum)
#                   print(len(dayDataSum),' LEN ')
                dayDataSum=[]
                tempData=[]
                for k in range(24):
                    if da[i,k+2] == 'NR':
                        da[i,k+2]=0
                    tempData.append(da[i,k+2])    
                tempData=np.array(tempData,dtype=float)
                dayDataSum.append(tempData)
        else:
            tempY=year
#            countId=countId+1
    ArraydayDataSum=np.array(dayDataSum)
    processData.append(ArraydayDataSum)
 #   print(dayDataSum,'  Is here still any data?')
    print(countId)        
    print(tempY,tempM,tempD,'TEMP YMD')
    processData=np.array(processData)
    return processData

def NormMatrix():
    NM=[]
    ran=3
    for i in range(ran):
        NM.append(30)
    for i in range(ran):
        NM.append(2)
    for i in range(ran):
        NM.append(1)
    for i in range(ran):
        NM.append(0.5)
    for i in range(ran):
        NM.append(50)
    for i in range(ran):
        NM.append(50)
    for i in range(ran):
        NM.append(50)
    for i in range(ran):
        NM.append(55)
    for i in range(ran):
        NM.append(50)
    for i in range(ran):
        NM.append(50)
    for i in range(ran):
        NM.append(1)
    for i in range(ran):
        NM.append(80)
    for i in range(ran):
        NM.append(3)
    for i in range(ran):
        NM.append(3)
    for i in range(ran):
        NM.append(350)
    for i in range(ran):
        NM.append(10000000)
    for i in range(ran):
        NM.append(5)
    for i in range(ran):
        NM.append(3)
    print(NM,'NM=?')
    NM=np.asarray(NM)
    print(NM.shape)
    return NM
        


def LR(dataValue):
    iterNum=1000000
    learningRate=0.0001
    dataValue=np.array(dataValue)
    print(len(dataValue))
    print(len(dataValue[0]),'data0')
    print(type(dataValue[1]),' DV1',type(dataValue[1][0]),' DV10')
    #normalize Matrix
    NormM=NormMatrix()
    #normalize end

    temp=0
    theta=np.zeros(108)
    theta=np.asarray(theta)

    theta2=np.zeros(54)
    theta2=np.asarray(theta2)


    tempTheta=np.zeros(108)
    tempTheta2=np.zeros(54)
    
    bias=0
    tempErr=np.zeros(108)
    tempErr2=np.zeros(54)
    tempBiasErr=0
    m=len(dataValue)*21
    print(theta,'theta')
    print( tempTheta,'temptheta')
    print(len(dataValue),'len data')
    #Start LR
    #test
    print(dataValue[0][0][0:9])
    #testEnd
    Xi=[]
    y=[]
    for i in range(len((dataValue))):
       for j in range(21):
            tempXi=[]
            tempXi=[dataValue[i][q][j:j+3] for q in range(18)]
            tempXi=[item for items in tempXi for item in items]
            tempXi=np.asarray(tempXi)
            #Norm
            tempXi=np.multiply(tempXi,1/NormM)
            #Norm End
            tempXi2=np.multiply(tempXi,tempXi)
            tempXi=np.append(tempXi,tempXi2)
            tempy=dataValue[i][9][j+3]
            seli=(int)((i)/20)
            if(tempy>150):
                print(tempy,'TY',y[-1],'Y-1')
                tempy=y[-1]
                print(tempy,'TY')
            if(tempy<0):
                print(tempy,'<0')
                tempy=0
                print(tempy)
            if seli!=2 and seli!=3 and seli!=8 and seli!=11: 
            #    print(i,'i',(i)/20,'(i)/20',seli,'Select i')
                Xi.append(tempXi)
                y.append(tempy)
  #  print(Xi,'Xi')
    print(Xi[0],'Xi0')
    Xi=np.asarray(Xi)
    print(type(Xi),'TXi=?')
    print(Xi.shape)
    templr=(learningRate/(240*21))
    y=np.asarray(y)
    print(y,'Y')
    print(y.shape)
    feature=[]
    for i in range(108):
        tlist=[]
        tlist=np.asarray(tlist)
        feature.append(tlist)
    for i in range(3360):
        for j in range(108):
            feature[j]=np.append(feature[j],Xi[i][j])
    feature=np.asarray(feature)
    print(feature,'Feature')
    print(type(feature),'F T')
    print(feature.shape,'F S')
    print(feature[0],'F0')
    print(feature[54],'F54')
    for iter in range(iterNum):
        timeStart=time.time()
        if iter%10000==0:
            print(iter,'Num of iter now--------------------------------------------------')
        sumOfError=0
        y_=np.dot(Xi,theta)+bias
        difY=y_-y
        sumOfError=sumOfError+difY.sum()
        tempErr=tempErr+np.dot(feature,difY)
        tempBiasErr=tempBiasErr+difY.sum()
        timeMid=time.time()
        theta=theta-np.dot(templr,tempErr)
        bias=bias-templr*tempBiasErr
        tempErr=np.zeros(108)
        tempBiasErr=0
        if iter % 10000==0 or iter==iterNum-1:
  #      print(bias,'Bias')
            print(sumOfError,'SOE FINAL')       
        timeEnd=time.time()
   #     print(timeEnd-timeStart,'Time for 1 iter')
    #SAAVEMODEL
    theta=np.append(theta,bias)
    print(theta)
    np.save('model3HR_X2_norm_SelectMonth_Bias_2.npy',theta)
    print('Save Model Success, Model name=model3HR_X2_norm_SelectMonth_Bias')

input='./train.csv'
data = pd.read_csv(input,encoding = "ANSI",)

selectCol=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
data=data.iloc[:,selectCol]
data=pd.DataFrame(data=data.values)
print(data)
print(type(data))
processedData = mergeByData(data)

print(type(processedData))
print(data.shape)
LR(processedData)
