
#import Keras
import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
#from keras.layers import Dense, Flatten,Convolution1D, Conv2D, MaxPooling2D, Dropout, BatchNormalization,Activation
from keras.layers import Dense, Flatten,Dropout, BatchNormalization,Activation,Input, Lambda, Concatenate
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dot
from keras.initializers import Zeros
from keras.models import Model, load_model
from keras.utils import get_custom_objects
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras import optimizers



from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import time

import pickle
import sys

print('import end')


modelpath='./model.h5'
savepath=sys.argv[2]
def rmse(y, y_pred):
    return keras.backend.sqrt(keras.backend.mean((y_pred-y) ** 2))


# load model 
model=load_model(modelpath,custom_objects={'rmse':rmse})

test_data = pd.read_csv(sys.argv[1])

with open('user2id', 'rb') as handle:
    user2id=pickle.load(handle)
with open('movie2id', 'rb') as handle:
    movie2id=pickle.load(handle)


test_user = np.array([user2id[i] for i in test_data["UserID"]])
test_movie = np.array([movie2id[i] for i in test_data["MovieID"]])

print(test_user,'test U')
print(test_movie,'test M')


pred_rating = model.predict([test_user,test_movie],verbose=1)


std=1.116897661146206
mean=3.5817120860388076

#pred_rating=pred_rating*std+mean
print(pred_rating)
#pred_rating=np.around(pred_rating)
#print(pred_rating)

answer=[]
for i in range(pred_rating.shape[0]):
    temp=[]
    temp.append(int(i+1))
    #print(type(temp[0]))

    temp.append(np.clip(pred_rating[i][0],1,5))
    #temp.append(pred_rating[i])
    answer.append(temp)
#answer=np.array(answer,dtype=np.int)
#answer=np.array(answer,dtype=)
#print(answer.shape,'answer shape')      
#print(answer)
col = ['TestDataID', 'Rating']
df_ans = pd.DataFrame(answer, index = None,columns=col)
print(df_ans)
df_ans.to_csv(savepath, index=None,columns=None)
print('output success!, path is :',savepath)