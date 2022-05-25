from pandas.core.tools.datetimes import to_datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Bidirectional

import numpy as np
import pandas as pd
from sklearn import preprocessing as prc

#Data preprocessing - Formats the data file to a usable format for the model
def prepareData(trainFile, step):
    trainData = pd.read_csv(trainFile).drop(['Type', 'Time'], axis=1)
    #trainData['Time'] = pd.to_numeric(pd.to_datetime(trainData['Time']))
    #trainNormal = prc.normalize(trainData)
    #trainData = pd.DataFrame(trainNormal)
    #print(trainData.head())
    #print(trainData.describe())
    trainData = trainData.values

    feature = []
    label = []
    for i in range(len(trainData)):
        end = i + step
        if end > len(trainData) - 1:
            break
        seq_x, seq_y = trainData[i:end, ], trainData[end, 1:]
        feature.append(seq_x)
        label.append(seq_y)

    feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.float32)
    print("Shape of feature: {}".format(feature.shape))
    return feature, label

#Model creation - Assembles and compiles the whole network model to be used
def createModel(shape):
    #CNN LSTM model creation
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=2, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(30, activation='relu', return_sequences=True)))
    model.add(Dense(5))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    print("Model created!")
    return model

#Training - Fits the model into the data and begins the training
def startTrain(model, train_feature, train_label, validation_feature, validation_label):
    dataInfo = model.fit(train_feature, train_label, epochs=30, batch_size=16, #batch size multiple of 2^x, early stopping
                         validation_data=(validation_feature, validation_label))
    metric = open(metricFile, 'a')
    for i in range(len(metricInfo.history['loss'])):
        metric.write('{}, {}, {}, {}, {}\n'.format(i + 1, dataInfo.history['loss'][i],
                                                   dataInfo.history['val_loss'][i],
                                                   dataInfo.history['root_mean_squared_error'][i],
                                                   dataInfo.history['val_root_mean_squared_error'][i]))
    metric.close()
    return model

#Function that contains all the process to train the model
def trainModel():
    dataDir = ".//"
    trainFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    testFile = ["KITCHEN_test.csv", "LIVINGROOM_test.csv", 
                 "CENTRALIZED_test.csv"]

    outputFile = ["KITCHEN_result", "LIVINGROOM_result", "CENTRALIZED_result"]

    nSteps = [5, 7, 9]
    for i in range(len(trainFile)):
        for step in nSteps:
            train_feature, train_label = prepareData(dataDir + trainFile[i], step)
            test_feature, test_label = prepareData(dataDir + testFile[i], step)
            model = createModel(train_feature.shape)
            model = startTrain(model, train_feature, train_label, test_feature, test_label)

if __name__ == "__main__":
    trainModel()