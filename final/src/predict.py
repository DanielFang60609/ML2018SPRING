import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import pandas as pd

import time
import math
from copy import deepcopy
import sys
import os
#ADD
import pickle
from matplotlib import pyplot as plt
#from random_eraser import get_random_eraser


#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model



print('import end')


def readTrainData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead)
    return data

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

#uni length hop
def unify_hop_length(data,uni_length,hop_distance=1):
    newData=[]
    newLabel=[]
    countID=0
    print('unify start')
    for i in range(len(data)):
        #print(data[i].shape,i,'data i shape')
        if data[i].shape[1]<uni_length:
            L = abs(data[i].shape[1] - uni_length)
            unified  = np.pad(data[i], ((0, 0), (0, L)), 'constant')
            #print(unified.shape,'uni shape')
            newData.append(unified)
            newLabel.append(countID)
        elif data[i].shape[1] == uni_length:
            unified  = data[i]
            #print(unified.shape,'uni shape')
            newData.append(unified)
            newLabel.append(countID)
        elif data[i].shape[1] > uni_length:
            #L = float(abs(data[i].shape[1] - uni_length))
            L=float(data[i].shape[1])
            L=int(L/hop_distance)+1
            #print(L,uni_length,hop_distance,'L/ uni/distance')
            for j in range(0,L):
                if data[i].shape[1]>uni_length*(j+1):
                    unified = data[i][:,0+j*uni_length :uni_length*(j+1)]
                    #print(unified.shape,'uni shape')
                    newData.append(unified)
                    newLabel.append(countID)
                else:
                    #print('else')
                    #print(data[i].shape[1]-uni_length,data[i].shape[1],'len')
                    unified = data[i][:,data[i].shape[1]-uni_length :data[i].shape[1]]
                    #print(unified.shape,'uni shape')
                    newData.append(unified)
                    newLabel.append(countID)
        #unified = data[:, :dest_length]
        else:
            print('error in unify hop length')
        countID+=1
    #print(newData,'newData')
    newData=np.array(newData)
    newLabel=np.array(newLabel)
    print(newData.shape,'newData in hop shape')
    print(newLabel.shape,'newLabel in hop shape')
    return newData,newLabel


#daisukelab
def tf_wave_to_melspectrogram(wave, sr):
    spectrogram = lb.feature.melspectrogram(wave, sr=sr, n_mels=40, hop_length=160, n_fft=400, fmin=20, fmax=4000)
    spectrogram = lb.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram
def wave_padding(data, sr, minimum_sec):
    min_data_size = int(np.ceil(sr * minimum_sec))
    if len(data) < min_data_size:
        L = abs(len(data) - min_data_size)
        start = L // 2
        data  = np.pad(data, (start, L-start), 'constant')
    return data           
def unify_2d_length(data, dest_length):
    d_len = data.shape[1]
    if d_len < dest_length:
        L = abs(d_len - dest_length)
        unified  = np.pad(data, ((0, 0), (0, L)), 'symmetric')
    elif dest_length < d_len:
        unified = data[:, :dest_length]
    else:
        unified = data
    return unified
def get_2d_mode_length(mels_set):
    print(len(mels_set))
    #[print(x.shape) for x in mels_set]
    _length = stats.mode([x.shape[1] for x in mels_set])[0][0] # mode value
    return _length
from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))






def main(savepath):

    timeStart=time.time()

    col_name=np.load('colName.npy')
    print(col_name,'col name')
    #fileName=os.listdir("./audio_test/audio_test")
    samplerate=44100
    hop=160
    _fmin=20
    _fmax=16000
    _nfft=400
    bands=64
    targetMel='./sr'+str(samplerate)+'hops'+str(hop)+'bands'+str(bands)+'fft_fmin_fmax'+str(_nfft)+'_'+str(_fmin)+'_'+str(_fmax)+'HoptestData.npy'
    targetID='./sr'+str(samplerate)+'hops'+str(hop)+'bands'+str(bands)+'fft_fmin_fmax'+str(_nfft)+'_'+str(_fmin)+'_'+str(_fmax)+'HoptestID.npy'
    loadNp=True
    """
    if os.path.exists(targetMel) and os.path.exists(targetID) and loadNp==True:
        #features=np.load(targetMel)
        dataID=np.load(targetID)
        print('load success')
    else:
        #load audio
        timeMid=time.time()

        log_specgrams=[]
        
        print(len(fileName) ,'f num? ')
        count=0
        for f in fileName:
            fn="./audio_test/audio_test/"+f
            sound_clip,sr=lb.load(fn,sr=samplerate)
            count+=1
            if count % 10 ==0:
                print(count,'--------------------')


            sound_clip=wave_padding(sound_clip,sr,1.)

            spectrogram = lb.feature.melspectrogram(sound_clip,sr=sr,n_mels=bands,hop_length=hop,n_fft=_nfft,fmin=_fmin,fmax=_fmax)
            spectrogram = lb.power_to_db(spectrogram)
            spectrogram = spectrogram.astype(np.float32)

            log_specgrams.append(spectrogram)
           
            if count<1:
                print(fn,count,'Fname count')
           
        timeEnd=time.time()
        print("Using ",timeEnd-timeMid,"time for Load")
	"""        
	"""
        with open('TestPre', 'wb') as handle:
                pickle.dump(log_specgrams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
	"""
        temp=len(log_specgrams)
        #uni_length = get_2d_mode_length(log_specgrams)
        uni_length=512

        #data=np.array([unify_2d_length(x, uni_length) for x in log_specgrams])
        data,dataID=unify_hop_length(log_specgrams,uni_length,uni_length)
        print(data.shape,' log shape')
        print(dataID.shape,'ID shape')
        features=data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
        print(features.shape,' features shape')
        #print(features,'features')

        np.save(targetMel,features)
        np.save(targetID,dataID)
    """
   

    
    dataID=np.load('./dataID.npy')

    #print(features.shape,' features shape')
    #model ensemble
    onehotans=np.zeros((33586,41))
    countModel=0
    for m in range(17):
        """
        model=load_model(m)

        #Image generator
        
        print('im gen')
        
        with open('./log/Test_data_gen_11', 'rb') as handle:
            test_datagen=pickle.load(handle)
        
        test_gen=test_datagen.flow(
            features,
            batch_size=64,
            shuffle=False
        )
        print('image gen end')
        
        print(m,'m')

        model.summary()

        step=int(features.shape[0]/64)+1
        print(step,'Step')
        tempAns=model.predict_generator(test_gen,verbose=1,steps=step)
        """
        #np.save('./predict/'+str(countModel),tempAns)
        tempAns=np.load('./'+str(m)+'.npy')
        print(m,'--------------------')
        
        onehotans+=tempAns
        """
        for i in range(41):
            print(tempAns[0][i])
        """    


    for i in range(41):
        print(onehotans[0][i])

    print(onehotans,'one hot ans')
    print(onehotans.shape,'one hot shape')
    
    #ans=np.argsort(onehotans)
    ans=onehotans
    print(ans,'ans')
    sumArg=np.zeros((9400,41))
    answer=[]
    for i in range(dataID.shape[0]):
        sumArg[dataID[i]]+=ans[i]
    ans=np.argsort(sumArg)
    print(sumArg[0][ans[0][40]],'sumarg 1')
    print(sumArg[0][ans[0][39]],'sumarg 2')
    a=sumArg[0][ans[0][40]]
    b=sumArg[0][ans[0][39]]
    print((a>b).all)
    for i in range(ans.shape[0]):
        a=sumArg[i][ans[i][40]]
        b=sumArg[i][ans[i][39]]
        #c=np.array(15.0)
        #print( (a > 2*b).all,'a > 2*b).all')
        #print((a>c).all)
        if  a > 4*b:
            temp=[]
            temp.append(fileName[i])
            #temp.append(col_name[ans[i][40]]+' '+col_name[ans[i][39]]+' '+col_name[ans[i][38]])
            temp.append(col_name[ans[i][40]])
            
            answer.append(temp)
        #else:
        #    print(a,b,i,'a b')
    answer=np.array(answer)
    
    print(answer.shape,'answer shape')      
    print(answer)
    
    col = ['fname', 'label']
    df_ans = pd.DataFrame(answer, index = None,columns=col)
    print(df_ans)
    df_ans.to_csv(savepath, index=False,columns=None)
    print('output success!, path is :',savepath)
    
        




savepath='./test_self_11_44100_mel_hop_33model_ensemble.csv'

#files=os.listdir("./modelEnsemble")
#print(files)
print(type(files))
print(len(files))
modelpath=[]
"""
for i in files:
    fname="./modelEnsemble/"+i
    print(fname)
    modelpath.append(fname)
"""
#print(trainCsv,'t csv')
#print(type(trainCsv),'type t csv')

main(savepath)
