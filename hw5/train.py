import numpy as np
import pandas as pd
import time
import gensim as gs
from gensim.models import Word2Vec as w2v
import sys

#from sklearn.model_selection import train_test_split
#ADD
import pickle
#from matplotlib import pyplot as plt


#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
#from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation
from keras.layers import Dense, Flatten,Dropout, BatchNormalization,Activation
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



print(gs.__version__,'gs version')
print(keras.__version__,'keras version')



print('import end')

def readData(filepath,isHead):
    data = pd.read_csv(filepath,sep=" ",header=isHead,delimiter='\t')
    #print(data,'Raw data')
    #print(data.values,'Data values')
    print(type(data),'Type of data')
    return data

def sepData(data):
    print('sep data start')
    label=[]
    newData=[]
    print(data.shape[0],'data Shape')
    for i in range(data.shape[0]):
        #print(data[i].shape)
        tempSep=data[i][0].split(' ')
        #print(tempSep)
        #print(type(tempSep),'type')
        label.append(tempSep[0])
        del tempSep[0:2]
        tempSep=" ".join(str(x) for x in tempSep)
        newData.append(tempSep)
        """
        if i < 10:
            print(tempSep,'tempSep')
            print(type(tempSep),'ts type')
        """
    label=np.array(label)
    newData=np.array(newData)
    print(newData[0],' Newdata 0 ',label[0],' Label')
    print(label.shape)
    print(newData.shape)
    
    print('sep data end')
    return newData,label

def replaceStr(data):
    timeStart=time.time()
    print('Start to replace')
    newData=[]
    for i in range(data.shape[0]):
        tempStr=data[i]

        for q in range(7):
            #replace symbol 
            tempStr=tempStr.replace('..','.').replace('!!','!').replace('??','?')
            #replace repeat
            tempStr=tempStr.replace('aa','a').replace('bb','b').replace('cc','c').replace('dd','d').replace('ee','e')
            tempStr=tempStr.replace('ff','f').replace('gg','g').replace('hh','h').replace('ii','i').replace('jj','j')
            tempStr=tempStr.replace('kk','k').replace('ll','l').replace('mm','m').replace('nn','n').replace('oo','o')
            tempStr=tempStr.replace('pp','p').replace('qq','q').replace('rr','r').replace('ss','s').replace('tt','t')
            tempStr=tempStr.replace('uu','u').replace('vv','v').replace('ww','w').replace('xx','x').replace('yy','y')
            tempStr=tempStr.replace('zz','z')

            #replace common short
            tempStr=tempStr.replace('can \' t',' can not').replace('won \' t',' will not').replace('c \' mon','come on')
            tempStr=tempStr.replace('n \' t',' not').replace('\' m',' am').replace('\' d',' had').replace('\' ve',' have')
            tempStr=tempStr.replace('\' s',' is').replace('\' ll',' will').replace('\' re',' are')

            #replace uncommon short
            tempStr=tempStr.replace(' b4 ',' before ').replace(' u ',' you ').replace(' r ',' are ').replace(' w ',' with ')
            tempStr=tempStr.replace(' 2b ',' to be ').replace(' k ',' ok ').replace(' n ',' and ').replace('ive','i have')
            
        newData.append(tempStr)
    newData=np.array(newData)
    print(newData.shape,'newdata shape')
    print('Replace end')
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time for Replace")
    return newData


def buildKerasLSTMModel(num_word, embedding_matrix, emb_dim = 100):
    model=Sequential()
    model.add(Embedding(num_word, emb_dim, weights = embedding_matrix, trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128,return_sequences=False)))    
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['acc'])

    return model


def main(trainDataLabel,Label,savepath,EPOCHS):
    timeStart=time.time()
    print(trainDataLabel.shape,'tdl shape')
    print(trainDataLabel[0],'tdl 0')
    print(len(trainDataLabel[0]),'tdl 0 length')
    print(type(trainDataLabel),'tdl type')

    #replace---------------
    trainDataLabel=replaceStr(trainDataLabel)

    print(trainDataLabel[0:10],'tdl 0~10')
    testReplace=['rrrrrrrrrr today u r sooooooooooooooo cute !!!!!!!!!!!','plzzzzzz 2b here k ?','ive so maaaaaannnnyyyy cake !!!!!!!']
    testReplace=np.array(testReplace)
    print(testReplace,'test Replace')
    testReplace=replaceStr(testReplace)
    print(testReplace,'test Replace  after ')

    #replace end---------------

    #stem---------------
    stemStart=time.time()
    
    stemmer = gs.parsing.porter.PorterStemmer()
    s = "been there ,,, still there .,... but he can ' t complain when he runs out of clean clothes i taught mine how to do his at 10"
    #print("after stemming: ",stemmer.stem_sentence(s))
    #print("a stem type :",type(stemmer.stem_sentence(s)))
    #trainDataLabelAfterStem=stemmer.stem_sentence(trainDataLabel)
    trainDataLabelAfterStem=[]

    for i in range(trainDataLabel.shape[0]):
        trainDataLabelAfterStem.append(stemmer.stem_sentence(trainDataLabel[i]))
    trainDataLabelAfterStem=np.array(trainDataLabelAfterStem)
    print(trainDataLabelAfterStem.shape,'TDLAS SHAPE')
    print(trainDataLabelAfterStem[0],'TDAS 0')
    stemEnd=time.time()
    print("Using ",stemEnd-stemStart,"time for stem")

    #stem end---------------

    #w2v---------------

    trainDataLabelList = [l.split(" ") for l in trainDataLabelAfterStem]
    w2v_model=w2v(trainDataLabelList,size=400,window=5,min_count=5,workers=5)
    print(w2v_model.most_similar("cat"))
    emb_size = len(w2v_model["dog"])
    print("embedding size", emb_size)
    print("gensim model vocab size:", len(w2v_model.wv.vocab))


    #w2v end---------------

    tokenizer = Tokenizer(num_words=None,filters="\n\t")
    tokenizer.fit_on_texts(trainDataLabelAfterStem)
    sequences = tokenizer.texts_to_sequences(trainDataLabelAfterStem)
    word_index=tokenizer.word_index
    with open('tokenizer_10_AnotherReplace__400_semi.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(type(sequences))
    print(len(sequences))
    print('Found %s unique tokens.' % len(word_index))
    embedding_matrix = np.zeros((len(word_index), emb_size))

    oov_count=0
    for word, i in word_index.items():
        try:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector
        except:
            oov_count +=1
            #print(word)
    print("oov count: ", oov_count)
    max_length = np.max([len(i) for i in sequences])
    print("max length:", max_length)

    embedding_layer = Embedding(len(word_index),output_dim= emb_size,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    print("embedd matrix shape: ",embedding_matrix.shape)


    model=buildKerasLSTMModel(num_word=len(word_index),embedding_matrix = [embedding_matrix],emb_dim=400)

    train_X = pad_sequences(sequences, maxlen=max_length)
    train_Y = to_categorical(np.asarray(Label))

    trainX=train_X
    trainY=train_Y
    #trainX, x_val, trainY, y_val = train_test_split(train_X, train_Y, test_size=0.2)
    #print("Test Split")

    _batchsize=64
    logPath='./log'
    CSVlogName=logPath+'/log.csv'
    checkPointPath=logPath+'/model.{epoch:04d}-{val_acc:.4f}.h5'
    callback=[
                TensorBoard(log_dir=logPath),
                CSVLogger(CSVlogName,append=True),
                ModelCheckpoint(checkPointPath)

            ]


    #show the param
    print(savepath,'model save path')
    print(EPOCHS,'epochs')
    print(_batchsize,'batch size')
    print(trainX.shape,'trainX shape')
    print(trainY.shape,'trainY shape')
    #print(x_val.shape,'x val shape')
    #print(y_val.shape,'y val shape')
    
    #show the param end
    



    history = model.fit(trainX,trainY,epochs=EPOCHS,batch_size=_batchsize,validation_split=0.2,callbacks=callback)
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    """

    model.save(savepath)
    print('model save success , path is :',savepath)

    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time for main")


EPOCHS=20
savepath='./testModel_10_Anotherreplace_mod'
trainingLabelDataPath=sys.argv[1]
trainingNoLabelDataPath=sys.argv[2]
trainDataLabel=readData(trainingLabelDataPath,None)
trainDataLabel=trainDataLabel.values
print(trainDataLabel)
print(trainDataLabel.shape,'shape')
print(trainDataLabel[0])
trainDataLabel,Label=sepData(trainDataLabel)

print(trainDataLabel)
print(trainDataLabel.shape,'shape')
print(trainDataLabel[0])
print(Label,'label')
print(Label.shape,' label shape')


#trainDataNoLabel=readData(trainingNoLabelDataPath,None)
#trainDataNoLabel=trainDataNoLabel.values
#trainDataNoLabel,Label2=sepData(trainDataNoLabel)

print(trainDataNoLabel)
print(trainDataNoLabel.shape,'shape tdnl')
print(trainDataNoLabel[0])
print(Label2.shape,' Label2 shape')


#trainDataLabel=np.append(trainDataLabel,trainDataNoLabel,axis=0)
#Label=np.append(Label,Label2,axis=0)

print(trainDataLabel)
print(trainDataLabel.shape,'shape')
print(trainDataLabel[0])
print(Label,'label')
print(Label.shape,' label shape')



main(trainDataLabel,Label,savepath,EPOCHS)
