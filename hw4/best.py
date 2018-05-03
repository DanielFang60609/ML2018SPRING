import numpy as np
import scipy
#import pandas as pd

import time
import sklearn

import sys
from sklearn import cluster,datasets

from sklearn.decomposition import PCA

print('import end')

def readData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead)
    #print(data,'Raw data')
    #print(data.values,'Data values')
    print(type(data),'Type of data')
    return data

def readNpyData(filepath):
    data=np.load(filepath)
    return data



def processPCA(data):
    #PCA 
    print('PCA start')
    PCAStart=time.time()
    pca=PCA(n_components=350,whiten=True)    
    data=pca.fit_transform(data)
    print(data.shape,'data SHape after PCA')
    PCAEnd=time.time()
    
    print('PCA END spent',PCAEnd-PCAStart,'time')    


    return data

def normalizeData(data):
    timeStart=time.time()
    alreadyNorm=False
    if alreadyNorm==True:
        data_afterNorm=np.load('./Norm2.npy')
    else:
        data_afterNorm=np.zeros((data.shape[0],784))

        _means=np.zeros(784)
        _variance=np.zeros(784)
        for i in range(784):
            _means[i]=np.mean(data[:,i])
            _variance[i]=np.var(data[:,i])
            if _means[i]==0:
                print(i,'=0 means')
            if _variance[i]==0:
                print(i,'=0, var')
                _variance=1
        print(_means,_variance,'Mean Vari')
        for i in range(data.shape[0]):
            for j in range(784):
                if _variance[j]!=0:
                    data_afterNorm[i][j]=(data[i][j]-_means[j])/_variance[j]
                elif _variance[j]==0:
                    data_afterNorm[i][j]=(data[i][j]-_means[j])
            if i%1000==0:
                print(i,'---------Norm')
        np.save('./Norm2.npy',data_afterNorm)
                

    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to Nprmalized Data")
    return data_afterNorm

def main(npyDataPath,testDataPath):
    timeStart=time.time()
    data=readNpyData(npyDataPath)

    print(data)
    print(data.shape)
    print(type(data))
    #data=normalizeData(data)
    data=np.array(data,dtype='float32')
    #change to 28*28
    """
    newData=[]
    for i in range(data.shape[0]):
        temp2Darray=[]
        for j in range(28):
            temp1D=[]
            for k in range(28):
               temp1D.append(data[i][j*28+k])
            temp1D=np.asarray(temp1D)
            temp2Darray.append(temp1D)
        temp2Darray=np.asarray(temp2Darray)
        newData.append(temp2Darray)
    data=np.asarray(newData)
    print(data.shape,'data shape')
    #print(type(data))
    """
    #cov
    """
    covMatrix=np.cov(data.T)
    print(covMatrix,'COVMATRIX')
    print(covMatrix.shape,'CM shape')
    df=pd.DataFrame(covMatrix)
    df.to_csv('COV M.csv', index=False,columns=None)
    for i in range(784):
        for j in range(784):
            if covMatrix[i][j]<0.000001 and covMatrix[i][j]>-0.000001:
                #print(covMatrix[i][j])
                covMatrix[i][j]=0.0001
            if covMatrix[i][j]=='NaN':
                print(covMatrix[i][j])
    w,v=np.linalg.eig(covMatrix)
    print(w.shape,v.shape,'WV SHAPE')
    """
    #PCA 
    data=processPCA(data)

    #k means

    kmeans_fit = cluster.KMeans(n_clusters = 2).fit(data)
    clusterlabel=kmeans_fit.labels_
    print(clusterlabel,'label')
    print(type(kmeans_fit))
    pred=kmeans_fit.predict(data)
    print(pred,'pred')
    print(type(pred),'pred type')
    print(pred.shape,'pred shape')
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to cluster Data")

    testData=readData(testDataPath,'infer')
    testData=testData.values
    print(type(testData))
    print(testData.shape,'test Data shape')
    ans=[]
    for i in range(testData.shape[0]):
        temp=[]
        temp.append(i)
        #test1=data[testData[i][1]].reshape(1,-1)
        #test2=data[testData[i][2]].reshape(1,-1)
        
        if i%2000==0:
            print(i,'-----i')
            print(pred[testData[i][1]],pred[testData[i][2]],'predictresult of t1 t2')
        if pred[testData[i][1]]==pred[testData[i][2]]:
            temp.append(1)
            temp=np.asarray(temp)
            ans.append(temp)
        else:
            temp.append(0)
            temp=np.asarray(temp)
            ans.append(temp)
    ans=np.asarray(ans)
    print(ans,'ans')
    print(type(ans))
    outputpath=sys.argv[3]
    col = ['ID', 'Ans']
    df_ans = pd.DataFrame(ans, index = None,columns=col)
    print(df_ans)
    df_ans.to_csv(outputpath, index=False,columns=None)
    print('output success!, path is :',outputpath)
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time for main")



testDataPath=sys.argv[2]
npyDataPath=sys.argv[1]
main(npyDataPath,testDataPath)