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
from Centralized_CNN_LSTM import prepareData, createTeacherModel, createStudentModel

def startTrain(model, train_feature, train_label, validation_feature, validation_label, metricFile):
    dataInfo = model.fit(train_feature, train_label, epochs=1, batch_size=128, #batch size multiple of 2^x, early stopping00
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
    all_features = []
    all_labels = []
    all_test_features = []
    all_test_labels = []

    nSteps = [9]
    for i in range(len(trainFile) - 1):
        for step in nSteps:
            print("Preparing {} with {} steps".format(trainFile[i], step))
            train_feature, train_label = prepareData(dataDir + trainFile[i], step)
            test_feature, test_label = prepareData(dataDir + testFile[i], step)
            all_features.append(train_feature)
            all_labels.append(train_label)
            all_test_features.append(test_feature)
            all_test_labels.append(test_label)
            metricFile = dataDir + "Federated_" + outputFile[i] + "_CNN-LSTM_" + str(step) + "_steps.csv"
            #print("Training " + metricFile)
            fle = open(metricFile, 'w')
            fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
            fle.close()
    
    model = createTeacherModel(all_features[0].shape)
    globalModel = []
    for i in range(len(trainFile) - 1):
        globalModel.append(model)
    for epoch in range(100):
        subModels = []
        for i in range(len(trainFile) - 1):
            metricFile = dataDir + "Federated_" + outputFile[i] + "_CNN-LSTM_" + str(step) + "_steps.csv"
            print("Training " + metricFile)
            devModel = startTrain(globalModel[i],
                                  all_features[i],
                                  all_labels[i],
                                  all_test_features[i],
                                  all_test_labels[i],
                                  metricFile)
            subModels.append(devModel)

        sumWeights = []
        for x in range(len(subModels)):
            weights = subModels[x].get_weights()
            if x == 0:
                for numWeights in range(len(weights)):
                    weightVal = weights[numWeights] / len(subModels)
                    sumWeights.append(weightVal)
            else:
                for numWeights in range(len(weights)):
                    weightVal = sumWeights[numWeights] + (weights[numWeights] / len(subModels))
                    sumWeights[numWeights] = weightVal

        model.set_weights(sumWeights)
        globalModel.clear()
        for i in range(len(trainFile) - 1):
            globalModel.append(model)
        print("Finished round {}".format(epoch+1))
            
    print("Federated learning section finished.")
    
    #Insert student learning here
    
if __name__ == "__main__":
    trainModel()
