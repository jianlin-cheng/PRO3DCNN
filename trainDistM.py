import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from numpy import array
import random
import math
import os

def getDatas(indices):
    x=[]
    y=[]
    for i in indices:
        xadd,yadd = getData(i)
        x+=xadd
        y+=yadd
    return array(x),array(y)

def getData(index):
    global loadPath
    x = []
    y = []
    batchNum = math.floor(index/1000)
    proteinNum = index-batchNum*1000

    cwd = loadPath+str(batchNum)+'/'+str(proteinNum)
    n = len(os.listdir(cwd))
    for i in range(n):
        data = unpickle(cwd+'/'+str(i))
        xt = data[0]
        if type(xt) is list:
            xt= data[0][0]

        x.append(xt)
        y.append(data[1])
    return x,y


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

##########################################
######DATA PREP#######
##########################################
windowSize=100
num_classes=1232+1

##########################################
###### MODEL 1 #######
###########################################
#Define the model
input_shape=(windowSize,windowSize,1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


##########################################

###### Parameters #######
dataNumber = 18913
epoch = 20
batchNum = 20
global loadPath
loadPath = 'croppedMats/'
###### Parameters #######

ordering = np.arange(dataNumber)
np.random.shuffle(ordering)
trainN = int(dataNumber*.7)
valN = math.floor((dataNumber-trainN)/2)
testN = math.ceil((dataNumber-trainN)/2)

trainI = ordering[0:trainN]
ntrainBatch = math.ceil(trainN/batchNum)
valI = ordering[trainN:trainN+valN]
nvalBatch = math.ceil(valN/batchNum)
testI = ordering[trainN+valN:]
ntestBatch = math.ceil(testN/batchNum)

for e in range(epoch):
    model.save('distMCNN.h5')
    print('epoch: '+str(e)+' ----------------------------')

    # Training
    np.random.shuffle(trainI)
    trainBatches = np.array_split(trainI,ntrainBatch)
    sumAcc = 0
    sumLoss = 0
    for j,indices in enumerate(trainBatches):
        x,y = getDatas(indices)
        x = x.reshape(len(x),windowSize,windowSize,1)
        y = keras.utils.to_categorical(y, num_classes)
        results = model.train_on_batch(x,y)
        print('                                                                                        ',end="\r")
        print(str(results[1])+'        '+str(j)+'/'+str(len(trainBatches)), end="\r")
        sumAcc += results[1]
        sumLoss += results[0]
    print("\nAcc: "+str(round(sumAcc/ntrainBatch,5)) + " Loss: " + str(round(sumLoss/ntrainBatch,5)))

    #Validation
    sumAcc = 0
    sumLoss = 0
    np.random.shuffle(valI)
    valBatches = np.array_split(valI,nvalBatch)
    for j,indices in enumerate(valBatches):
        x,y = getDatas(indices)
        x = x.reshape(len(x),windowSize,windowSize,1)
        y = keras.utils.to_categorical(y, num_classes)
        results = model.test_on_batch(x,y)
        print('                                                                                        ',end="\r")
        print(str(results[1])+'        '+str(j)+'/'+str(len(valBatches)), end="\r")
        sumAcc += results[1]
        sumLoss += results[0]
    print("\nValAcc: "+str(round(sumAcc/nvalBatch,5)) + " Loss: " + str(round(sumLoss/nvalBatch,5)))

#Testing
sumAcc = 0
sumLoss = 0
np.random.shuffle(testI)
testBatches = np.array_split(testI,nvalBatch)
for j,indices in enumerate(testBatches):
    x,y = getDatas(indices)
    x = x.reshape(len(x),windowSize,windowSize,1)
    y = keras.utils.to_categorical(y, num_classes)
    results = model.test_on_batch(x,y)
    print('                                                                                        ',end="\r")
    print(str(results[1])+'        '+str(j)+'/'+str(len(testBatches)), end="\r")
    sumAcc += results[1]
    sumLoss += results[0]
print("\nTestAcc: "+str(round(sumAcc/nvalBatch,5)) + " Loss: " + str(round(sumLoss/nvalBatch,5)))


