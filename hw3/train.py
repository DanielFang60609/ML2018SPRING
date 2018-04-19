import numpy as np
import scipy
import pandas as pd

import time
#help("scipy")

import math
from copy import deepcopy
import sys
import os

#from matplotlib import pyplot as plt

#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

#picture process
from PIL import Image  
from PIL import ImageFilter 

print('import end')


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
        onehot[data[i][0]]=1
        trainY.append(onehot)
        
        sep=[]
        sepMatrix=[]
        sep=np.array(data[i][1].split(' '))
        sepMatrix2=[]
        
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

    sY=pd.DataFrame(trainY, index = None)
    #sY.to_csv('whatsY', index=False,columns=None)
    print(trainX.shape,'trainx shape ')
    print(trainY.shape,'trainy shape ')
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to sep Data")
    return trainX,trainY

"""
timeStart=time.time()
timeEnd=time.time()
print("Using ",timeEnd-timeStart,"time to ")

"""
def shuffle_in_unison_scary(a, b):
    print('Shuffle!')
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b



def CNNusingKeras(trainX,trainY,modelPath,savepath,EPOCHS,warmstart):
    print('start CNN using Keras')
    timeStart=time.time()
    #plt.imshow(trainX[0],cmap='gray')
    #plt.show()
    #plt.imshow(trainX[1],cmap='gray')
    #plt.show()
    #trainX,trainY=shuffle_in_unison_scary(trainX,trainY)
    trainX=trainX.reshape(trainX.shape[0],48,48,1)
    print(trainX[0],'Before')
    trainX=trainX/255
    print(trainX[0],'After')

    from sklearn.model_selection import train_test_split
    trainX, x_test, trainY, y_test = train_test_split(trainX, trainY, test_size=0.1)
    print("Test Split")
    #trainY=trainY.reshape(trainY.shape[0],1)
    
    train_data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )
    
    train_data_gen.fit(trainX)
    val_data_gen = ImageDataGenerator()
    train_gen = train_data_gen.flow(
        trainX, 
        trainY,
        batch_size = 128
    )
    PATIENCE=50
    callback = [
            TensorBoard(),
            CSVLogger('log.csv', append=True),
            ModelCheckpoint('./temp', period=10),
            ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PATIENCE/4), verbose=1),
            EarlyStopping(patience = PATIENCE)
            ]

    train_gen = train_data_gen.flow(
        trainX, 
        trainY,
        batch_size = 32
    )

    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to Set train gen")
    print('train_gen End')


    #WARMSTART
    
    if warmstart==True:
        from keras.models import load_model
        #modelPath='./testModel_AfterWarmStart_More_add100_200'
        model2=load_model(modelPath)

        #model2.add(Dense(64))   
        #model2.add(BatchNormalization())
        #model2.add(Activation('relu'))
        #model2.add(Dropout(0.25))
        print(model2.summary(),'WS summary?')



    model=Sequential()
    act = PReLU(alpha_initializer='zeros', weights=None)
    dp1=0.25
    dp2=0.35
    # act = LeakyReLU(alpha=0.3)
    # model.add(act)
    #block1
    model.add(Conv2D(64,(5,5),
        border_mode='same',
        kernel_initializer='glorot_normal',
        input_shape=(48,48,1)
    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', weights=None))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))

   
    model.add(Conv2D(128,(3,3),
        border_mode='same',
        kernel_initializer='glorot_normal'
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))

    model.add(Conv2D(512,(3,3),
        border_mode='same',
        kernel_initializer='glorot_normal'
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))

    model.add(Conv2D(512,(3,3),
        border_mode='same',
        kernel_initializer='glorot_normal'
    ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dp1))



    model.add(Flatten())
    
    model.add(Dense(512))   
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', weights=None))
    model.add(Dropout(dp2))

    model.add(Dense(512))   
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(Dropout(dp2))

    model.add(Dense(7,activation='softplus'))
    model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    if warmstart==False:
        print(model.summary())

    """
    model.fit_generator(
        train_gen,
        samples_per_epoch = trainX.shape[0],    
        epochs = 15
    )
    """
    print(trainX.shape,' TX shape',trainY.shape,'TY shape')

    print('Warm Start:',warmstart)
    print('Epochs:',EPOCHS)
    print('Model save as:',savepath)
    

    if warmstart==True:
        print('Read model:',modelPath)
        model2.fit_generator(
            train_gen,
            #samples_per_epoch = trainX.shape[0]/2,    
            epochs = EPOCHS,
            shuffle=True,
            validation_data = (x_test, y_test)
        )
    else:
        model.fit_generator(
            train_gen,
            samples_per_epoch = trainX.shape[0]/2,    
            epochs = EPOCHS,
            shuffle=True,
            validation_data = (x_test, y_test)
        )

    #model.fit(x=trainX,y=trainY,validation_data=(x_test,y_test),epochs=150,batch_size=256,shuffle=True)
    #savepath='./testModel_3_filter'
    if warmstart==True:
        model2.save(savepath)
    else:
        model.save(savepath)
    print('Save model as',savepath)
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time to CNN using Keras")

def main():
    #Read  path= path:'./train_X'   header=none,'infer'  
    trainDataPath=sys.argv[1]
    trainData=readData(trainDataPath,'infer')
    trainData=trainData.values
    print(trainData.shape,'TDS')
    print(type(trainData),'TDT')

    trainX,trainY=seperateData(trainData)
    savepath='./testModel_7_Prelu_New2_500'
    modelPath='./testModel_5_Prelu_NewShuffle_350'
    EPOCHS=700
    warmstart=False
    CNNusingKeras(trainX,trainY,modelPath,savepath,EPOCHS,warmstart)


main()