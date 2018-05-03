import numpy as np
import scipy
import pandas as pd
import math
import sys
import os

theta=np.load('model_norm_Bias_3_NOfnlwgt_valid_X2_X3_X4_X5.npy')
print(theta)
print(type(theta))
bias=theta[139]
theta=np.delete(theta,139,0)

#theta=theta.reshape(18,6)
print(theta)
print(theta.shape)
print(bias,'bias')




theta2=np.load('model_norm_Bias_3_NOfnlwgt_valid_X6_Nreg.npy')
bias2=theta2[143]
theta2=np.delete(theta2,143,0)

theta3=np.load('model_norm_Bias_3_NOfnlwgt_valid_X7_reg.npy')
bias3=theta3[147]
theta3=np.delete(theta3,147,0)






def sigmoid(x):
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

input='test_X'
#input=sys.argv[5]
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
    
    
    #Xi1
    Xi=data.values[i]
    X2=np.zeros(16)
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

    Xi=np.append(Xi,X2)

    #Xi2
    Xi2=data.values[i]
    X2=np.zeros(20)
    Xi2=np.multiply(Xi2,1/NormM)
    X2[0]=Xi2[0]*Xi2[0]
    X2[1]=Xi2[78]*Xi2[78]
    X2[2]=Xi2[79]*Xi2[79]
    X2[3]=Xi2[80]*Xi2[80]
    X2[4]=Xi2[0]*Xi2[0]*Xi2[0]
    X2[5]=Xi2[78]*Xi2[78]*Xi2[78]
    X2[6]=Xi2[79]*Xi2[79]*Xi2[79]
    X2[7]=Xi2[80]*Xi2[80]*Xi2[80]
    X2[8]=Xi2[0]*Xi2[0]*Xi2[0]*Xi2[0]
    X2[9]=Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]
    X2[10]=Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]
    X2[11]=Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]
    X2[12]=Xi2[0]*Xi2[20]*Xi2[0]*Xi2[0]*Xi2[0]
    X2[13]=Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]
    X2[14]=Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]
    X2[15]=Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]
    X2[16]=Xi2[0]*Xi2[20]*Xi2[0]*Xi2[0]*Xi2[0]*Xi2[0]
    X2[17]=Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]*Xi2[78]
    X2[18]=Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]*Xi2[79]
    X2[19]=Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]*Xi2[80]

    Xi2=np.append(Xi2,X2)
    #Xi3

    Xi3=data.values[i]
    X2=np.zeros(24)
    Xi3=np.multiply(Xi3,1/NormM)
    X2[0]=Xi3[0]*Xi3[0]
    X2[1]=Xi3[78]*Xi3[78]
    X2[2]=Xi3[79]*Xi3[79]
    X2[3]=Xi3[80]*Xi3[80]
    X2[4]=Xi3[0]*Xi3[0]*Xi3[0]
    X2[5]=Xi3[78]*Xi3[78]*Xi3[78]
    X2[6]=Xi3[79]*Xi3[79]*Xi3[79]
    X2[7]=Xi3[80]*Xi3[80]*Xi3[80]
    X2[8]=Xi3[0]*Xi3[0]*Xi3[0]*Xi3[0]
    X2[9]=Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]
    X2[10]=Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]
    X2[11]=Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]
    X2[12]=Xi3[0]*Xi3[20]*Xi3[0]*Xi3[0]*Xi3[0]
    X2[13]=Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]
    X2[14]=Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]
    X2[15]=Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]
    X2[16]=Xi3[0]*Xi3[20]*Xi3[0]*Xi3[0]*Xi3[0]*Xi3[0]
    X2[17]=Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]
    X2[18]=Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]
    X2[19]=Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]
    X2[20]=Xi3[0]*Xi3[20]*Xi3[0]*Xi3[0]*Xi3[0]*Xi3[0]*Xi3[0]
    X2[21]=Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]*Xi3[78]
    X2[22]=Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]*Xi3[79]
    X2[23]=Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]*Xi3[80]

    Xi3=np.append(Xi3,X2)



    income=np.multiply(Xi,theta).sum()+bias

    income2=np.multiply(Xi2,theta2).sum()+bias2
    income3=np.multiply(Xi3,theta3).sum()+bias3
    
#    print(i,'i')
#    print(pm25,'pm25')
    income=sigmoid(income)
    income2=sigmoid(income2)
    income3=sigmoid(income3)
    """
    if income>0.5:
        income=1
    else:
        income=0
    if income2>0.5:
        income2=1
    else:
        income2=0
    if income3>0.5:
        income3=1
    else:
        income3=0
    """
    print(income+income2+income3,'income!',i)
    if (income+income2+income3)>1.5:
        finalincome=1
    else:
        finalincome=0
    ans.append(finalincome)
print(ans)
col = ['id', 'label']
path="./ans_ensemble.csv"
#path=sys.argv[6]
ans_sheet = []
for i in range(len(data)):
    #if ans[i] < 0:
    #    ans[i] = 0
    ans_sheet.append((i+1, ans[i]))
df_ans = pd.DataFrame(ans_sheet, index = None, columns = col)
df_ans.to_csv(path, index=False)
print(">>> ans SAVE SUCCEED path=",path)