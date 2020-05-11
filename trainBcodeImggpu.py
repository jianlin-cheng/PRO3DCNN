# Hello
# # If using tensorflow, set image dimensions order
# from keras import backend as K
# import keras
# if K.backend()=='tensorflow':
#     K.set_image_dim_ordering("th")
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from numpy import array
import random
import math
import pickle


def getDatas(indices):
    x=[]
    y=[]
    for i in indices:
        data = getData(i)
        x.append(data[b'x'])
        y.append(data[b'y'])
    return array(x),array(y)

def getData(index):
    global loadPath

    batchNum = math.floor(index/1000)
    fileNum = index-batchNum*1000
    data = unpickle(loadPath+str(batchNum)+'/'+str(fileNum))
    return data #(id,x,y)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pick(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f, protocol = 2)
    f.close()



windowSize=100
num_classes=1231+1+1
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
# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])




##########################################
###### Parameters #######
# windowSize=100
# dataNumber = 231822
# epoch = 20
# batchNum = 50
# global loadPath
# loadPath = 'barcodeImgs207/'

windowSize=100
dataNumber = 18913
epoch = 20
batchNum = 50
global loadPath
loadPath = 'barcodeImgs/'
###### Parameters #######

nsplit = math.ceil(dataNumber/batchNum)
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


ordering = np.arange(dataNumber)

for e in range(epoch):
    #model.save('homCNN207.h5')
    model.save('homCNN_h.h5')
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

##########################################
###### Validation #######
##########################################

# plot model history
#plot_model_history(model_info)
#start = time.time()
#end = time.time()
#print ("Model took seconds to train" + str(end - start))
# compute test accuracy
#print ("Accuracy on test data is:" + str(accuracy(test_features, test_labels, model)))
