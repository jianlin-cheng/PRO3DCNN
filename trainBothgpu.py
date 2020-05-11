import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import Input
import numpy as np
from numpy import array
import random
import math
import os
import pickle

def getDistMDatas(indices):
    x=[]
    y=[]
    for i in indices:
        xadd,yadd = getDistMData(i)
        x+=xadd
        y+=yadd
    return array(x),array(y)

def getDistMData(index):
    global loadDistMPath
    x = []
    y = []
    batchNum = math.floor(index/1000)
    proteinNum = index-batchNum*1000

    cwd = loadDistMPath+str(batchNum)+'/'+str(proteinNum)
    n = len(os.listdir(cwd))
    for i in range(n):
        data = unpickle(cwd+'/'+str(i))
        xt = data[0]
        if type(xt) is list:
            xt= data[0][0]

        x.append(xt)
        y.append(data[1])
    return x,y

def getHomDatas(indices):
    x=[]
    y=[]
    for i in indices:
        data = getHomData(i)
        x.append(data[b'x'])
    return array(x)

def getHomData(index):
    global loadHomPath

    batchNum = math.floor(index/1000)
    fileNum = index-batchNum*1000
    data = unpickle(loadHomPath+str(batchNum)+'/'+str(fileNum))
    return data #(id,x,y)

def getDatas(indices):
    x1 = []
    x2 = []
    y = []
    for i in indices:
        x1add, yadd = getDistMData(i)
        x1 += x1add
        y += yadd
        data = getHomData(i)
        x2add = data[b'x']
        x2add = x2add*len(x1add)
        x2 += x2add
    return array(x1),array(x2),array(y)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pick(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f, protocol = 2)
    f.close()

##########################################
######DATA PREP#######
##########################################
windowSize=100
num_classes=1232+1

##########################################
###### MODEL 1 #######
###########################################
#Define the model
# input_shape=(windowSize,windowSize,1)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.50))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01),
#               metrics=['accuracy'])
##########################################
# Multi input model
input_shape=(windowSize,windowSize,1)
distMinput = Input(shape = input_shape, name = 'distM')
barcodeinput = Input(shape = input_shape, name = 'barcodes')
x = keras.layers.concatenate([distMinput, barcodeinput])
x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.50)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs = [distMinput, barcodeinput], outputs = output)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


###### Parameters #######
dataNumber = 231822
epoch = 20
batchNum = 50
global loadDistMPath, loadHomPath
loadDistMPath = 'croppedMats207/'
loadHomPath = 'barcodeImgs207/'
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
    model.save('hybridCNN207.h5')
    print('epoch: '+str(e)+' ----------------------------')

    # Training
    np.random.shuffle(trainI)
    trainBatches = np.array_split(trainI,ntrainBatch)
    sumAcc = 0
    sumLoss = 0
    for j,indices in enumerate(trainBatches):
        # x1,y = getDistMDatas(indices)
        # x2 = getHomDatas(indices)
        x1, x2, y = getDatas(indices)
        x1 = x1.reshape(len(x1),windowSize,windowSize,1)
        x2 = x2.reshape(len(x2),windowSize,windowSize,1)
        y = keras.utils.to_categorical(y, num_classes)
        results = model.train_on_batch([x1, x2],y)
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
        # x1,y = getDistMDatas(indices)
        # x2 = getHomDatas(indices)
        x1, x2, y = getDatas(indices)
        x1 = x1.reshape(len(x1),windowSize,windowSize,1)
        x2 = x2.reshape(len(x2),windowSize,windowSize,1)
        y = keras.utils.to_categorical(y, num_classes)
        results = model.test_on_batch([x1, x2],y)
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
    # x1,y = getDistMDatas(indices)
    # x2 = getHomDatas(indices)
    x1, x2, y = getDatas(indices)
    x1 = x1.reshape(len(x1),windowSize,windowSize,1)
    x2 = x2.reshape(len(x2),windowSize,windowSize,1)
    y = keras.utils.to_categorical(y, num_classes)
    results = model.test_on_batch([x1, x2],y)
    print('                                                                                        ',end="\r")
    print(str(results[1])+'        '+str(j)+'/'+str(len(testBatches)), end="\r")
    sumAcc += results[1]
    sumLoss += results[0]
print("\nTestAcc: "+str(round(sumAcc/nvalBatch,5)) + " Loss: " + str(round(sumLoss/nvalBatch,5)))
