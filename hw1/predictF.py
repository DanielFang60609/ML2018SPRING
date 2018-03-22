import numpy as np
import scipy
import pandas as pd
import sys
import os

theta=np.load('model3HR_X2_norm_SelectMonth_Bias.npy')
print(theta)
print(type(theta))
bias=theta[108]
theta=np.delete(theta,108,0)

#theta=theta.reshape(18,6)
print(theta)
print(theta.shape)
print(bias,'bias')

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

#input='./test.csv'
input=sys.argv[1]
data = pd.read_csv(input,encoding = "ANSI",header=None)

#selectCol=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
data=data.iloc[:,8:11]
data=pd.DataFrame(data=data.values)
print(data)
print(type(data))
print(data.shape)
for i in range(4680):
    for k in range(3):
        if data.values[i][k] == "NR":
            data.values[i][k]=0
        data.values[i][k]=float(data.values[i][k])
print(data)
ans=[]
NormM=NormMatrix()
for i in range(260):
    #計算第I筆資料的值
    Xi=data.values[18*i:18*i+18,:]
    #print(Xi,'Xi',i,' I')
    NormM=NormM.reshape(18,3)
    #print(Xi.shape,'XIS',NormM.shape,'NMS')
    Xi=np.multiply(Xi,1/NormM)
    Xi2=np.multiply(Xi,Xi)
    Xi=np.append(Xi,Xi2)
    #print(Xi,'Xi1+2',i,' I')
    #pm25=np.dot(Xi,theta_t)
    pm25=np.multiply(Xi,theta).sum()+bias
#    print(i,'i')
#    print(pm25,'pm25')
    ans.append(pm25)
print(ans)
col = ['id', 'value']
#path="./ans1D3HR_X2_norm_SelectMonth_Bias_ISTHISCORRECT.csv"
path=sys.argv[2]
ans_sheet = []
for i in range(260):
    if ans[i] < 0:
        ans[i] = 0
    ans_sheet.append(('id_' + str(i), ans[i]))
df_ans = pd.DataFrame(ans_sheet, index = None, columns = col)
df_ans.to_csv(path, index=False)
print(">>> ans SAVE SUCCEED path=",path)