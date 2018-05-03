import numpy as np
import skimage
from skimage import io
import sys
import os

#import matplotlib.pyplot as plt

print('import end')
inputpath=sys.argv[1]
testimage=[]


#read file
filepath=os.listdir(inputpath)
for f in filepath:
     testimage.append(skimage.io.imread(os.path.join(inputpath,f),None))
"""
for i in range(415):
    path=inputpath+str(i)+'.jpg'
    testimage.append(skimage.io.imread(path))
"""
    #plt.imshow(testimage[i])
    #plt.show()

#mean
testimage=np.array(testimage)
testimage=testimage.astype(int)

print(testimage[:].shape,'test?')
print(type(testimage[1][0]))
meanImage=np.zeros((600,600,3))

meanImage=np.mean(testimage[:],axis=0)
meanImage=meanImage.astype(np.uint8)

data=testimage
data=data-meanImage

data=data.reshape(testimage.shape[0],-1)
print(data.shape,'data shape 2')

#cov eig

U,S,V=np.linalg.svd(data.T,full_matrices=False)
#S=np.load('S.npy')
#U=np.load('U.npy')
#V=np.load('V.npy')
#sortEval=argsort(-eval)
#np.save('./U.npy',U)
#np.save('./S.npy',S)
#np.save('./V.npy',V)
print(U.shape,'U shape ')
print(S.shape,'S shape ')
print(V.shape,'V shape ')

#print(S,'S')

#print(S[0]/np.sum(S),'S0 %')
#print(S[1]/np.sum(S),'S1 %')
#print(S[2]/np.sum(S),'S2 %')
#print(S[3]/np.sum(S),'S3 %')


#print(sorted(S),'sort S')
#print(U[:,:4])
#print(S[0],' S0')
#print(U.T[0].astype(np.uint8).reshape(600,600,3))
#print((U.T[0].reshape(600,600,3)*S[0]).astype(np.uint8)+meanImage ,' X x0')
#print(pd.DataFrame((U.T[0].reshape(600,600,3)*S[0]+meanImage).astype(np.uint8)))
"""
temp=(U.T[0].reshape(600,600,3)*S[0])
temp-=np.min(temp)
temp/=np.max(temp)
temp=(temp*255).astype(np.uint8)
plt.imshow(temp)
plt.show()
temp=(U.T[1].reshape(600,600,3)*S[0])
temp-=np.min(temp)
temp/=np.max(temp)
temp=(temp*255).astype(np.uint8)
plt.imshow(temp)
plt.show()
temp=(U.T[2].reshape(600,600,3)*S[0])
temp-=np.min(temp)
temp/=np.max(temp)
temp=(temp*255).astype(np.uint8)
plt.imshow(temp)
plt.show()
temp=(U.T[3].reshape(600,600,3)*S[0])
temp-=np.min(temp)
temp/=np.max(temp)
temp=(temp*255).astype(np.uint8)
plt.imshow(temp)
plt.show()
"""
print("RECO!")
meanImage=meanImage.flatten()
#Reconstruct 4 picture 0 25 75 100

inputNum=int(sys.argv[2])
RecoImg=testimage[inputNum].flatten()-meanImage

w=np.dot(RecoImg,U[:,:4])
RecoImg=np.dot(w,U[:,:4].T)+meanImage
RecoImg=RecoImg-np.min(RecoImg)
RecoImg=RecoImg/np.max(RecoImg)
RecoImg=(RecoImg*255).astype(np.uint8)

savePath='./reconstruction.png'
skimage.io.imsave(savePath,RecoImg.reshape(600,600,3))
"""
plt.imshow(RecoImg.reshape(600,600,3))
plt.show()
"""

