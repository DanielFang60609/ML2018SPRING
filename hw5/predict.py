import numpy as np
import pandas as pd
import time
import gensim as gs
from gensim.models import Word2Vec as w2v
import pickle
import sys

#from sklearn.model_selection import train_test_split

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
from keras.models import load_model



print(gs.__version__,'gs version')
print(keras.__version__,'keras version')



print('import end')

def readData(filepath,isHead):
    data = pd.read_csv(filepath,header=isHead,delimiter='\t')
    #data = pd.read_csv(filepath,sep=" ",header=isHead)
    #print(data,'Raw data')
    #print(data.values,'Data values')
    print(type(data),'Type of data')
    return data

def sepTestData(data):
    print('sep data start')
    #label=[]
    newData=[]
    print(data.shape[0],'data Shape')
    for i in range(data.shape[0]):
        #print(data[i].shape)
        tempSep=data[i][0].split(',')
        #print(tempSep)
        #print(type(tempSep),'type')
        #label.append(tempSep[0])
        del tempSep[0]
        tempSep=",".join(str(x) for x in tempSep)
        newData.append(tempSep)
        
        if i < 10:
            print(tempSep,'tempSep')
            print(type(tempSep),'ts type')
        
    #label=np.array(label)
    newData=np.array(newData)
    print(newData[0],' Newdata 0 ')
    #print(label.shape)
    print(newData.shape)
    print('sep data end')
    return newData

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


def main(testData,outputPath,modelPath):
    timeStart=time.time()
    model=load_model(modelPath)

    model.summary()
    

    testData=sepTestData(testData)

    testData=replaceStr(testData)
    #stem---------------
    stemStart=time.time()
    
    stemmer = gs.parsing.porter.PorterStemmer()
    s = "been there ,,, still there .,... but he can ' t complain when he runs out of clean clothes i taught mine how to do his at 10"
    #print("after stemming: ",stemmer.stem_sentence(s))
    #print("a stem type :",type(stemmer.stem_sentence(s)))
    #trainDataLabelAfterStem=stemmer.stem_sentence(trainDataLabel)
    #trainDataLabelAfterStem=[]
    testDataAfterStem=[]

    for i in range(testData.shape[0]):
        testDataAfterStem.append(stemmer.stem_sentence(testData[i]))
    testDataAfterStem=np.array(testDataAfterStem)
    print(testDataAfterStem.shape,'TDAS SHAPE')
    print(testDataAfterStem[0],'TDAS 0')
    stemEnd=time.time()
    print("Using ",stemEnd-stemStart,"time for stem")

    #stem end---------------
    #w2v---------------

    testDataList = [l.split(" ") for l in testDataAfterStem]
    w2v_model=w2v(testDataList,size=400,window=5,min_count=5,workers=5)
    print(w2v_model.most_similar("cat"))
    emb_size = len(w2v_model["dog"])
    print("embedding size", emb_size)
    print("gensim model vocab size:", len(w2v_model.wv.vocab))


    #w2v end---------------
    with open('tokenizer_10_AnotherReplace__400_semi.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #tokenizer = Tokenizer(num_words=None,filters="\n\t")
    #tokenizer.fit_on_texts(testDataAfterStem)
    sequences = tokenizer.texts_to_sequences(testDataAfterStem)
    word_index=tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    embedding_matrix = np.zeros((len(word_index), emb_size))

    max_length = np.max([len(i) for i in sequences])
    print("max length:", max_length)

    testX = pad_sequences(sequences, maxlen=max_length)



    print('predict!')
    Answer=model.predict(testX,batch_size=128,verbose=1)

    print(Answer,'Answer')
    print(type(Answer),'Answer type')

    
    ans=Answer.argmax(axis=-1)

    #print(ans,' ans (argmax)')
    Answer=[]
    for i in range(ans.shape[0]):
        temp=[]
        temp.append(i)
        temp.append(ans[i])
        Answer.append(temp)
    #print(Answer,'i label')
    col = ['id', 'label']
    df_ans = pd.DataFrame(Answer, index = None,columns=col)
    print(df_ans)
    df_ans.to_csv(outputPath, index=False,columns=None)
    print('output success!, path is :',outputPath)
        
    timeEnd=time.time()
    print("Using ",timeEnd-timeStart,"time for main")



modelPath='./bestModel.h5'
outputPath=sys.argv[2]
#trainingLabelDataPath='./training_label.txt'
#trainingNoLabelDataPath='./training_nolabel.txt'
testDataPath=sys.argv[1]
testData=readData(testDataPath,'infer')
testData=testData.values
print(testData)
print(testData.shape,'shape')
print(testData[0])

#trainDataLabel=readData(trainingLabelDataPath,None)
#trainDataLabel=trainDataLabel.values
#print(trainDataLabel)
#print(trainDataLabel.shape,'shape')
#print(trainDataLabel[0])
#trainDataLabel,Label=sepData(trainDataLabel)

main(testData,outputPath,modelPath)
