from types import DynamicClassAttribute
from pandas.io import parsers
import tensorflow as tf
import numpy as np
import pandas as pd

#Data preprocessing - Formats the data file to a usable format for the model
def prepareData(trainFile, step):
    trainData = pd.read_csv(trainFile, parse_dates=['Time']).drop(['Type'], axis=1)
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

    feature, label = np.array(feature), np.array(label)
    print("Shape of feature: {}".format(feature.shape))
    return feature, label

#Model creation - Assembles and compiles the whole network model to be used
def createModel(shape):
    #CNN LSTM model creation
    model = tf.keras.Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam)
    print("Model created!")
    return model

#Training - Fits the model into the data and begins the training
def startTrain(model, train_feature, train_label, validation_feature, validation_label):
    dataInfo = model.fit(train_feature, train_label, epochs=30, batch_size=10,
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
            trainX, trainY = prepareData(dataDir + trainFile[i], step)
            #print(trainX.shape)
            #model = createModel(trainX.shape)

           #print(trainX)
 
    #createModel()
    #startTrain()

if __name__ == "__main__":
    trainModel()
