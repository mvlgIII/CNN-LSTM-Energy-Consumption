from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.tools.datetimes import to_datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from sklearn import preprocessing as prc
from sklearn.preprocessing import MinMaxScaler

#Data preprocessing - Formats the data file to a usable format for the model
def prepareData(trainFile, step):
    trainData = pd.read_csv(trainFile).drop(['Type', 'Time'], axis=1)
    #trainData['Time'] = trainData['Time'].apply(lambda x: datetime.strptime(trainData['Time'][x]).timestamp(), axis=1)
    #trainData['Time'] = pd.to_datetime(trainData['Time'])
    #for i in range(len(trainData)):
    #    trainData['Time'][i] = trainData['Time'][i].timestamp()
    #    print(trainData['Time'][i])
    #trainNormal = prc.normalize(trainData)
    #trainData = pd.DataFrame(trainNormal)
    #print(trainData.head())
    #print(trainData.describe())
    scaler = MinMaxScaler()
    trainData = scaler.fit_transform(trainData)
    trainData = pd.DataFrame(trainData)
    print(trainData.head())
    trainData = trainData.values

    feature = []
    label = []
    for i in range(len(trainData)):
        end = i + step
        if end > len(trainData) - 1:
            break
        seq_x, seq_y = trainData[i:end, ], trainData[end, 0] * trainData[end, 1]
        feature.append(seq_x)
        label.append(seq_y)
        
    feature, label = np.array(feature).astype(np.float64), np.array(label).astype(np.float64)
    #print("Shape of feature: {}".format(feature.shape))
    return feature, label

#Model creation - Assembles and compiles the whole network model to be used
def createTeacherModel(shape):
    #CNN LSTM model creation
    tf.random.set_seed(1)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='sigmoid', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(300, activation='sigmoid', return_sequences=True)))
    model.add(Dense(1))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adadelta', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    #print("Model created!")
    return model

def createStudentModel(shape):
    tf.random.set_seed(5)

    model = Sequential()
    model.add(Conv1D(filters=24, kernel_size=2, activation='relu', input_shape=(shape[1], shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True)))
    model.add(Dense(1))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss=tf.keras.losses.MeanSquaredError,
                  optimizer='adamax',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model

#Training - Fits the model into the data and begins the training while recording metric values
def startTrain(model, train_feature, train_label, validation_feature, validation_label, metricFile):
    dataInfo = model.fit(train_feature, train_label, epochs=100, batch_size=128, #batch size multiple of 2^x, early stopping00
                         validation_data=(validation_feature, validation_label),
                         callbacks=[EarlyStopping(monitor='loss', patience=3)])
    #metric = model.evaluate(validation_feature, validation_label)  
    #print(metric)
    metric = open(metricFile, 'a')
    for i in range(len(dataInfo.history['loss'])):
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

    outputFile = ["KITCHEN_result", "LIVINGROOM_result",
                 "CENTRALIZED_result"]
    
    model = 0

    nSteps = [9]
    #Centralized learning approach:
    for step in nSteps:
        print("Preparing data {}_{}".format(trainFile[2], step))
        train_feature, train_label = prepareData(dataDir + trainFile[2], step)
        test_feature, test_label = prepareData(dataDir + testFile[2], step)
        #print("Test data: {}".format(test_feature))
        print("Feature: {} Label: {}".format(train_feature.shape, test_label.shape))
        model = createTeacherModel(train_feature.shape)
        metricFile = dataDir + outputFile[2] + "_CNN-LSTM_" + str(step) + "_steps.csv"
        print("Training " + metricFile)
        fle = open(metricFile, 'w')
        fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
        fle.close()
        model = startTrain(model, train_feature, train_label, test_feature, test_label, metricFile)
        #model.save(dataDir + outputFile[2] + '_CNN-LSTM_' + str(step) + "_steps_model")
        predictions = model.predict(train_feature)
        predictions = np.array(predictions).astype(np.float64)
        print("Shape of predictions: {}".format(predictions.shape))
        #studentModel = createStudentModel(train_feature.shape)

    #student_train_feature
if __name__ == "__main__":
    trainModel()