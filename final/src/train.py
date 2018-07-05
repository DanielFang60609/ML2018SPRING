import librosa as lb
import librosa.display
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
from random_eraser import get_random_eraser


#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation,Reshape,Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



print('import end')




def readTrainData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead)
    return data


def kai_CNN_Model(inputShape,output_dim):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2, 2), padding='same',input_shape=(inputShape)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(256, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
def buildKerasCNNModel(inputShape,output_dim):
    model=Sequential()
    act = PReLU(alpha_initializer='zeros', weights=None)
    dp1=0.25
    dp2=0.35
    # act = LeakyReLU(alpha=0.3)
    # model.add(act)
    #block1
    model.add(Conv2D(64,(2,2),
        border_mode='same',
        kernel_initializer='glorot_normal',
        input_shape=inputShape
    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', weights=None))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))

   
    model.add(Conv2D(128,(2,2),
        border_mode='same',
        kernel_initializer='glorot_normal'
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))

    model.add(Flatten())
    
    model.add(Dense(256))   
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', weights=None))
    model.add(Dropout(dp2))

    model.add(Dense(256))   
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(Dropout(dp2))
    # optimizer=keras.optimizers.Adam(lr=0.0001),
    model.add(Dense(output_dim,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', 
                optimizer = keras.optimizers.Adam(lr=0.01), 
                metrics = ['accuracy'])

    return model


def buildKerasCNNModel_Alexnet(inputShape,output_dim):
    model = Sequential()
 
    model.add(Conv2D(48, 11,  input_shape=inputShape, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
    return model

def CRNNTestModel(inputShape,output_dim):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2, 2), padding='same',
                     input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./10))
    model.add(MaxPooling2D(pool_size=(3, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./10))
    model.add(MaxPooling2D(pool_size=(3, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./10))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./10))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    """
    model.add(Conv2D(512, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.5))
    """
    #model.add(Reshape((1, -1)))
    model.add(Reshape((32, -1)))
    #model.add(Permute((2, 1)))


    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(LSTM(output_dim,return_sequences=False,activation='softmax'))
    """
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    """
    #model.add(Dense(output_dim, activation='softmax'))



    #optimizer=keras.optimizers.Adam(lr=0.0001),
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    return model


def buildKerasCRNNModel_Alexnet_LSTM(inputShape,output_dim):
    model = Sequential()
 
    model.add(Conv2D(48, 5,  input_shape=inputShape, kernel_initializer='glorot_normal', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(1,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, kernel_initializer='glorot_normal', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(1,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(1,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    #model.add(Flatten())
    """
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    """
    model.add(Reshape((128, -1)))
    model.add(Permute((2, 1)))


    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256,return_sequences=False)))    
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(output_dim, activation='softmax'))




    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    return model
    
def onehotLabel(label):
    n_unique_labels=41
    print(label.shape[0])
    onehot=np.zeros((label.shape[0],n_unique_labels))
    print(np.arange(label.shape[0]))
    onehot[np.arange(label.shape[0]),label]=1
    

    return onehot

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)
#uni length hop
def unify_hop_length(data,label,uni_length,hop_distance=1):
    newData=[]
    newLabel=[]
    print('unify start')
    for i in range(len(data)):
        #print(data[i].shape,i,'data i shape')
        if data[i].shape[1]<uni_length:
            L = abs(data[i].shape[1] - uni_length)
            unified  = np.pad(data[i], ((0, 0), (0, L)), 'wrap')
            #print(unified.shape,'uni shape')
            newData.append(unified)
            newLabel.append(label[i])
        elif data[i].shape[1] == uni_length:
            unified  = data[i]
            #print(unified.shape,'uni shape')
            newData.append(unified)
            newLabel.append(label[i])
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
                    newLabel.append(label[i])
                else:
                    #print('else')
                    #print(data[i].shape[1]-uni_length,data[i].shape[1],'len')
                    unified = data[i][:,data[i].shape[1]-uni_length :data[i].shape[1]]
                    #print(unified.shape,'uni shape')
                    newData.append(unified)
                    newLabel.append(label[i])
        #unified = data[:, :dest_length]
        else:
            print('error in unify hop length')
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
        data  = np.pad(data, (start, L-start), 'wrap')
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
    #print(x.shape) for x in mels_set
    lenArr=[x.shape[1] for x in mels_set]
    lenArr=np.array(lenArr)
    lenArr=np.sort(lenArr)
    print(lenArr)
    print(np.average(lenArr),'AVG of Len')
    print(np.min(lenArr),'min')
    print(np.max(lenArr),'np.max')
    for i in range(10):
        print(np.percentile(lenArr,i*10))
    _length = stats.mode([x.shape[1] for x in mels_set])[0][0] # mode value
    np.save('./lenOfTheMel44100',lenArr)
    return _length
from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))

def main(trainCsv,savepath):

    timeStart=time.time()
    print(trainCsv.shape,'tdl shape')
    print(trainCsv[0],'tdl 0')

    label=trainCsv[:,1]
    fileName=trainCsv[:,0]

    print(label.shape,'label shape')
    print(fileName.shape,'filename shape')

    #onehot label

    df=pd.DataFrame(label)
    #print(pd.get_dummies(label))
    label=pd.get_dummies(label)

    print(label.columns.values,'Label index')

    col_name=label.columns.values
    print(type(col_name))
    np.save('colName.npy',col_name)
    print(label.columns.values.shape,'S')

    label=label.values
    #label=to_categorical(label)

    print(label.shape,'label shape')
    print(label[0:20])

    if not os.path.exists('./log'):
        os.makedirs('./log')

    #param
    frames=41
    window_size=512*(frames-1)
    num_channels=2

    samplerate=44100
    hop=160
    _fmin=20
    _fmax=16000
    _nfft=400
    bands=64
    print('param')
    print(window_size,'window size')
    print(samplerate,'sr')
    print(hop,'hop')
    print(_fmin,'fmin')
    print(_fmax,'fmax')
    print(_nfft,'nfft')
    print(bands,'bands')

    #160/20/16000/400/64 seems better?
    targetMel='./sr'+str(samplerate)+'hops'+str(hop)+'bands'+str(bands)+'fft_fmin_fmax'+str(_nfft)+'_'+str(_fmin)+'_'+str(_fmax)+'mel512repeatData.npy'
    targetLabel='./sr'+str(samplerate)+'hops'+str(hop)+'bands'+str(bands)+'fft_fmin_fmax'+str(_nfft)+'_'+str(_fmin)+'_'+str(_fmax)+'label512repeatData.npy'
    
    loadNp=True
    if os.path.exists(targetMel) and os.path.exists(targetLabel) and loadNp==True:
        print('Find Exist data...  Start loading...')
        timeMid=time.time()
        features=np.load(targetMel)
        newLabel=np.load(targetLabel)
        print('load success')
        timeEnd=time.time()
        print("Using ",timeEnd-timeMid,"time for Load")
    else:
        #load audio
        timeMid=time.time()

        newLabel=[]
        log_specgrams=[]

        print(int(fileName.shape[0]) ,'Times? ')
        #for i in range(30):
        for i in range(fileName.shape[0]):
            #print(lb.audioread(fileName[i]))
            fn='./audio_train/'+fileName[i]
            #print(fn,'fn')
            sound_clip,sr=lb.load(fn,sr=samplerate)
            if i % 100 ==0:
                print(i,'--------------------')


            sound_clip=wave_padding(sound_clip,sr,1.)

            spectrogram = lb.feature.melspectrogram(sound_clip,sr=sr,n_mels=bands,hop_length=hop,n_fft=_nfft,fmin=_fmin,fmax=_fmax)
            spectrogram = lb.power_to_db(spectrogram)
            spectrogram = spectrogram.astype(np.float32)

            """
            print(spectrogram.shape,'Spec shape')
            print(type(spectrogram),'type Spec')

            print(data,'data')
            print(data.shape,'data shape')
            print(type(data),'type data')
            """
            log_specgrams.append(spectrogram)
            newLabel.append(label[i])
           

            if i<1:
                print(fileName[i],i,'Fname i')
                print(label[i],'label')
        
        timeEnd=time.time()
        print("Using ",timeEnd-timeMid,"time for Load")
        
        temp=len(log_specgrams)
        #log_specgrams=np.array(log_specgrams)
        uni_length = get_2d_mode_length(log_specgrams)
        #return [np.array([util.unify_2d_length(x, uni_length) for x in Xs]) for Xs in Xss]
        print(uni_length,'uni length')
        uni_length=512
        print(uni_length,'new unilength')
        data,newLabel=unify_hop_length(log_specgrams,newLabel,uni_length,uni_length)
        

        #data=np.array([unify_2d_length(x, uni_length) for x in log_specgrams])

        #data=unify_2d_length(spectrogram,256)
        print(data.shape,' log shape')
        """
        log_specgrams=log_specgrams.reshape(temp,bands,frames,1)
        features=np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
        for i in range(len(features)):
            features[i, :, :, 1] = lb.feature.delta(features[i, :, :, 0])
        """
        newLabel=np.array(newLabel,dtype=np.int)
        features=data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
        print(newLabel.shape,'nl shape')
        print(features.shape,' features shape')
        #print(features,'features')

        np.save(targetMel,features)
        np.save(targetLabel,newLabel)

   
    distri=np.zeros(41)
    for i in range(newLabel.shape[0]):
        for j in range(newLabel.shape[1]):
            if newLabel[i][j]==1:
                distri[j]=distri[j]+1


     #Image generator
    
    print('Draw')
    lb.display.specshow(features[0,:,:,0],y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('hops'+str(hop)+'bands'+str(bands)+'fft/fmin/fmax'+str(_nfft)+'/'+str(_fmin)+'/'+str(_fmax))
    #plt.show()
    plt.savefig('drawFirst.png')
    

    
    print("Test Split")
    from sklearn.model_selection import train_test_split
    features, val_x, newLabel, val_label = train_test_split(features, newLabel, test_size=0.15)
    

    print('save the split data for further training...')
    np.save('./log/features',features)
    np.save('./log/val_x',val_x)
    np.save('./log/newLabel',newLabel)
    np.save('./log/val_label',val_label)
    print('im gen')
    timeMid=time.time()
    train_data_gen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=0,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=np.min(features), v_h=np.max(features)) # RANDOM ERASER
        )
    train_data_gen.fit(features)
    val_data_gen = ImageDataGenerator()
    train_gen = train_data_gen.flow(
        features, 
        newLabel,
        batch_size = 64
    )

    """
    with open('Train_data_gen_6', 'wb') as handle:
            pickle.dump(train_data_gen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """        
    test_datagen = ImageDataGenerator(
            featurewise_center=train_data_gen.featurewise_center,
            featurewise_std_normalization=train_data_gen.featurewise_std_normalization
        )
    test_datagen.mean, test_datagen.std = train_data_gen.mean, train_data_gen.std
    val_gen=test_datagen.flow(
        val_x,
        val_label,
        batch_size=32
    )

    with open('./log/Test_data_gen_11', 'wb') as handle:
            pickle.dump(test_datagen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    train_gen = train_data_gen.flow(
        features, 
        newLabel,
        batch_size = 32
    )
    """
    timeEnd=time.time()
    print('image gen end')
    print("Using ",timeEnd-timeMid,"time for Image Gen")
    


    print(distri,'distribute')
    print(features.shape,' features shape')
    #model
    inputShape=(features.shape[1],features.shape[2],1)
    
    #model=buildKerasCRNNModel_Alexnet_LSTM(inputShape,41)
    #model=CRNNTestModel(inputShape,41)
    
    model=kai_CNN_Model(inputShape,41)
    model.summary()

    BATCHSIZE=64
    EPOCHS=500
    logPath='./log'
    CSVlogName=logPath+'/log.csv'
    checkPointPath=logPath+'/model.{epoch:04d}-{val_acc:.4f}.h5'
    callback=[
                TensorBoard(log_dir=logPath),
                CSVLogger(CSVlogName,append=True),
                ModelCheckpoint(checkPointPath, period=10),
                ModelCheckpoint(logPath+'/modelBest.h5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=False)
                
            ]
    #samples_per_epoch = trainX.shape[0]/2,    
    """
    
    """
    plt.close()
    history = model.fit_generator(
            train_gen,            
            epochs = EPOCHS,
            steps_per_epoch=169,
            #shuffle=True,
            #validation_split = 0.2,
            validation_data=val_gen,
            validation_steps=64,
            callbacks=callback
        )
    
    #history = model.fit(features,newLabel,batch_size=BATCHSIZE,epochs=EPOCHS,validation_split=0.2,callbacks=callback)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    

    model.save(savepath)
    print('model save success , path is :',savepath)

        




trainCsvPath=sys.argv[1]
savepath='./model_11_500'
trainCsv=readTrainData(trainCsvPath,'infer')
trainCsv=trainCsv.values
#print(trainCsv,'t csv')
#print(type(trainCsv),'type t csv')

main(trainCsv,savepath)