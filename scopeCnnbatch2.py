# Hello
# # If using tensorflow, set image dimensions order
# from keras import backend as K
# import keras
# if K.backend()=='tensorflow':
#     K.set_image_dim_ordering("th")
#

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from numpy import array
    

def reshapeImage(imageArray):
    r = imageArray[0:1024].reshape((1,32,32))
    g = imageArray[1024:2048].reshape((1,32,32))
    b = imageArray[2048:3073].reshape((1,32,32))
    data = np.concatenate([r,g,b])
    return data
def savepickle(data,filename):
    import pickle
    f = open(filename,"wb")
    pickle.dump(data,f,protocol=2)
    f.close()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def SQCrops(array,windowSize):
    batches = []
    testSample = []
    dim = array.shape[0]
    m = 0
    n = 0
    shift = 50
    while m+windowSize <= dim:
        while n+windowSize <= dim:
            batches.append(array[m:m+windowSize,n:n+windowSize])
            n+=1*shift
        m+=1*shift
    return batches
    
def convertLabelsToInts(array):
    list=[]
    for label in array:
        list.append(int(label[1:]))
    return list

def generateTrainingBatchData():
    #provide this beforehand
    
    num_classes = 600
    windowSize=100
    nInBatch = 3000
    ##################SCope Data######################
    upToBatchNum = 24
    dataIndex = 0

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    countNum = 0
    for i in range(upToBatchNum+1):
        data = unpickle("./intBatch"+str(i))
        x = data[b'x']
        y = data[b'y']
        for j,matrix in enumerate(x):
            
            if matrix.shape[0]>=windowSize:
                croppedData = SQCrops(matrix,windowSize)
                label = y[j]
                #separate 10% for training
                if(countNum%10==0):
                    x_test+=croppedData
                    y_test+=[label]*len(croppedData)
                else:
                    x_train+=croppedData
                    y_train+=[label]*len(croppedData)
            else:
                a=windowSize
                #use buffer matrix
            countNum+=1
            if countNum == nInBatch:

                countNum=0

    num_classes=y[len(y)-1]+1
    dataNumber = len(x_train)
    testNumber = len(x_test)

    x_train = array(x_train)
    y_train = array(y_train)
    x_train = x_train.reshape(dataNumber,windowSize,windowSize,1)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_test = array(x_test)
    y_test = array(y_test)
    x_test = x_test.reshape(testNumber,windowSize,windowSize,1)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    #x_train = array(x_train)
    #y_train = array(y_train)
    print(x_train.shape)
    pass



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
###### MODEL 2 #######
##########################################
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(num_classes+1, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])





##########################################
###### Training Standard #######
##########################################
#model.fit(x_train, y_train, epochs=5)
# model.fit_generator(generate_arrays_from_file('/my_file.txt'),steps_per_epoch=10000, epochs=10,verbose=1)
# model_info = model.fit(x_train, y_train,batch_size=128, nb_epoch=40,validation_data = (x_test, y_test),verbose=1)



##########################################
###### Training Batch #######
##########################################
#
# nb_of_epochs = 10
# nb_of_batches =
# batches_generator = get_cropped_batches_generator(INPUTDATA, BATCHSIZE=16)
# losses = list()
# for epoch_nb in range(nb_of_epochs):
#     epoch_losses = list()
#     for batch_nb in range(nb_of_batches):
#         # cropped_x has a different shape for different batches (in general)
#         cropped_x, cropped_y = next(batches_generator)
#         current_loss = model.train_on_batch(cropped_x, cropped_y)
#         epoch_losses.append(current_loss)
#     losses.append(epoch_losses.sum() / (1.0 * len(epoch_losses))
# final_loss = losses.sum() / (1.0 * len(losses))
#
#





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
