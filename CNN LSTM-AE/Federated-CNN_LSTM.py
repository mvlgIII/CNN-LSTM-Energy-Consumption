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
from CNN_LSTM_AE import prepareData, createTeacherModel, createStudentModel, startTrain

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
    all_features = []
    all_labels = []
    all_test_features = []
    all_test_labels = []

    nSteps = [9]
    for i in range(len(trainFile)):
        for step in nSteps:
            train_feature, train_label = prepareData(dataDir + trainFile[i], step)
            test_feature, test_label = prepareData(dataDir + testFile[i], step)
            all_features.append(train_feature)
            all_labels.append(train_label)
            all_test_features.append(test_feature)
            all_test_labels.append(test_label)
            metricFile = dataDir + outputFile[i] + "_CNN-LSTM_" + str(step) + "_steps.csv"
            print("Training " + metricFile)
            fle = open(metricFile, 'w')
            fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
            fle.close()
    
    model = createTeacherModel(all_feature[0].shape)
    globalModel = []
    for i in range(len(trainFile) - 1):
        globalModel.append(model)
    for epoch in range(100):
        subModels = []

    
    #Centralized learning approach:
    for step in nSteps:
        print("Preparing data {}_{}".format(trainFile[2], step))
        
        print("Test data: {}".format(test_feature))
        print("Feature: {} Label: {}".format(train_feature.shape, test_label.shape))
        model = createTeacherModel(train_feature.shape)
        metricFile = dataDir + outputFile[2] + "_CNN-LSTM_" + str(step) + "_steps.csv"
        print("Training " + metricFile)
        fle = open(metricFile, 'w')
        fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
        fle.close()
        model = startTrain(model, train_feature, train_label, test_feature, test_label, metricFile)
        #model.save(dataDir + outputFile[2] + '_CNN-LSTM_' + str(step) + "_steps_model")
        predictions = model.predict(test_feature, verbose=False)
        print("Shape of predictions: {}".format(predictions.shape))
        studentModel = createStudentModel(train_feature.shape)
        studentMetrics = studentModel.fit(test_feature, predictions, epochs=5, batch_size=64)
        print(studentMetrics)

    #student_train_feature
if __name__ == "__main__":
    trainModel()
