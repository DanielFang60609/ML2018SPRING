import numpy as np
import scipy
import pandas as pd
import math
import sys
import os

theta=np.load('model_GM.npy')
print(theta)
print(type(theta))
bias=theta[147]
theta=np.delete(theta,147,0)

#theta=theta.reshape(18,6)
print(theta)
print(theta.shape)
print(bias,'bias')


def sigmoid(x):
    #print(type(x))
    if x<0:
        #print(x,'x<0!')
        return 1-1/(1+math.exp(x))
    return 1 / (1 + math.exp(-x))


def NormMatrix():
    NM=[]
    NM.append(50)
    for i in range(9):
        NM.append(1)
    NM.append(10000000000000)
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

#input='test_X'
input=sys.argv[5]
data = pd.read_csv(input,header='infer')

#selectCol=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
#data=data.iloc[:,8:11]
#data=pd.DataFrame(data=data.values)
print(data)
print(type(data))
print(data.shape)
"""
for i in range(4680):
    for k in range(3):
        if data.values[i][k] == "NR":
            data.values[i][k]=0
        data.values[i][k]=float(data.values[i][k])
"""
#print(data)
ans=[]
NormM=NormMatrix()
for i in range(len(data)):
    #計算第I筆資料的值
    Xi=data.values[i]
    X2=np.zeros(24)
    #print(Xi,'Xi',i,' I')
    #NormM=NormM.reshape(18,3)
    #print(Xi.shape,'XIS',NormM.shape,'NMS')
    Xi=np.multiply(Xi,1/NormM)
    X2[0]=Xi[0]*Xi[0]
    X2[1]=Xi[78]*Xi[78]
    X2[2]=Xi[79]*Xi[79]
    X2[3]=Xi[80]*Xi[80]
    X2[4]=Xi[0]*Xi[0]*Xi[0]
    X2[5]=Xi[78]*Xi[78]*Xi[78]
    X2[6]=Xi[79]*Xi[79]*Xi[79]
    X2[7]=Xi[80]*Xi[80]*Xi[80]
    X2[8]=Xi[0]*Xi[0]*Xi[0]*Xi[0]
    X2[9]=Xi[78]*Xi[78]*Xi[78]*Xi[78]
    X2[10]=Xi[79]*Xi[79]*Xi[79]*Xi[79]
    X2[11]=Xi[80]*Xi[80]*Xi[80]*Xi[80]
    X2[12]=Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]
    X2[13]=Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]
    X2[14]=Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]
    X2[15]=Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]
    X2[16]=Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]
    X2[17]=Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]
    X2[18]=Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]
    X2[19]=Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]
    
    X2[20]=Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]
    X2[21]=Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]
    X2[22]=Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]
    X2[23]=Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]
    """
    X2[24]=Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]*Xi[0]
    X2[25]=Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]*Xi[78]
    X2[26]=Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]*Xi[79]
    X2[27]=Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]*Xi[80]
    """
    Xi=np.append(Xi,X2)
    #print(Xi.shape)
    """
    Xi2=np.multiply(Xi,Xi)
    Xi=np.append(Xi,Xi2)
    """
    #print(Xi,'Xi1+2',i,' I')
    #pm25=np.dot(Xi,theta_t)
    income=np.multiply(Xi,theta).sum()+bias
#    print(i,'i')
#    print(pm25,'pm25')
    #print(i)
    income=sigmoid(income)
    if income>0.5:
        income=1
    else:
        income=0
    ans.append(income)
print(ans)
col = ['id', 'label']
#path="./ans_GM_Threshold50.csv"
path=sys.argv[6]
ans_sheet = []
for i in range(len(data)):
    #if ans[i] < 0:
    #    ans[i] = 0
    ans_sheet.append((i+1, ans[i]))
df_ans = pd.DataFrame(ans_sheet, index = None, columns = col)
df_ans.to_csv(path, index=False)
print(">>> ans SAVE SUCCEED path=",path)