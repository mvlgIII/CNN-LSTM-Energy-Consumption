from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.tools.datetimes import to_datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import datetime, copy
from datetime import datetime
from sklearn import preprocessing as prc
from sklearn.preprocessing import MinMaxScaler
from Centralized_CNN_LSTM import prepareData, createStudentModel, rmse

def startTrain(model, train_feature, train_label, validation_feature, validation_label, metricFile, epoch):
    dataInfo = model.fit(train_feature, train_label, epochs=1, batch_size=32, #batch size multiple of 2^x, early stopping00
                         validation_data=(validation_feature, validation_label),
                         callbacks=[EarlyStopping(monitor='loss', patience=3)])
    #metric = model.evaluate(validation_feature, validation_label)  
    #print(metric)
    metric = open(metricFile, 'a')
    for i in range(len(dataInfo.history['loss'])):
        metric.write('{}, {}, {}, {}, {}\n'.format(epoch + 1, dataInfo.history['loss'][i],
                                                   dataInfo.history['val_loss'][i],
                                                   dataInfo.history['root_mean_squared_error'][i],
                                                   dataInfo.history['val_root_mean_squared_error'][i]))
    metric.close()
    return model

#Function that contains all the process to train the model
def trainModel():
    tf.random.set_seed(1)
    dataDir = ".//"
    trainFile = ["KITCHEN_data_record.csv", "LIVINGROOM_data_record.csv", 
                 "CENTRALIZED_data_record.csv"]

    testFile = ["KITCHEN_test.csv", "LIVINGROOM_test.csv", 
                 "CENTRALIZED_test.csv"]

    #augmentFile = ["data_recordNEW-KITCHEN.csv", "data_recordNEW-LIVINGROOM.csv",
    #               "data_recordNEW-CENTRALIZED.csv"]

    outputFile = ["KITCHEN_result", "LIVINGROOM_result",
                 "CENTRALIZED_result"]
    
    teacherModel = keras.models.load_model('CENTRALIZED_result_CNN-LSTM_7_steps_model')
    all_features = []
    all_labels = []
    all_test_features = []
    all_test_labels = []
    all_predictions = []

    allKitchenRmse = []
    allLivingRmse = []

    nSteps = [7]
    for i in range(len(trainFile) - 1):
        for step in nSteps:
            print("Preparing Federated {} with {} steps".format(trainFile[i], step))
            train_feature, train_label = prepareData(dataDir + trainFile[i], step)
            test_feature, test_label = prepareData(dataDir + testFile[i], step)
            predictions = teacherModel.predict(train_feature, verbose=False)
            all_features.append(train_feature)
            all_labels.append(train_label)
            all_test_features.append(test_feature)
            all_test_labels.append(test_label)
            all_predictions.append(predictions)
            metricFile = dataDir + "Federated_" + outputFile[i] + "_CNN-LSTM_" + str(step) + "_steps.csv"
            #print("Training " + metricFile)
            fle = open(metricFile, 'w')
            fle.write('epoch, trainloss, validationloss, trainRMSE, validationRMSE\n')
            fle.close()
    
    studentModelTemplate = createStudentModel(all_features[0].shape)
    globalStudentModel = []
    for i in range(len(trainFile) - 1):
        #clone student model here
        studentModel = keras.models.clone_model(studentModelTemplate)
        studentModel.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer='adadelta',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
        studentModel.set_weights(studentModelTemplate.get_weights())
        globalStudentModel.append(studentModel)
    for epoch in range(200):
        subModels = []
        total = 0
        for i in range(len(trainFile) - 1):
            metricFile = dataDir + "Federated_" + outputFile[i] + "_CNN-LSTM_" + str(step) + "_steps.csv"
            print("Training " + metricFile)
            devModel = startTrain(globalStudentModel[i],
                                  all_features[i],
                                  all_predictions[i],
                                  all_test_features[i],
                                  all_test_labels[i],
                                  metricFile,
                                  epoch)
            total += len(all_features[i])
            subModels.append(devModel)

        sumWeights = []
        for x in range(len(subModels)):
            weights = subModels[x].get_weights()
            if x == 0:
                for numWeights in range(len(weights)):
                    weightVal = weights[numWeights] / len(subModels)
                    # weightVal = (len(all_features[x])/total) * weights[numWeights]
                    sumWeights.append(weightVal)
            else:
                for numWeights in range(len(weights)):
                    weightVal = sumWeights[numWeights] + (weights[numWeights] / len(subModels))
                    # weightVal = sumWeights[numWeights] + (len(all_features[x])/total) * weights[numWeights]
                    sumWeights[numWeights] = weightVal

        # Aggregated weights result          
        studentModelTemplate.set_weights(sumWeights)
        for i in range(len(subModels)):
            subModels.save(dataDir + "_Federated_" + outputFile[i] + '_CNN-LSTM_' + str(step) + "_steps_{}_epoch_model".format(epoch+1))

        # Validation in kitchen and living using the studentModelTemplate 
        kitchenPrediction = studentModel.predict(all_test_features[0], verbose=False)
        livingPrediction = studentModel.predict(all_test_features[1], verbose=False)
        kitchenRmse = rmse(kitchenPrediction, all_test_labels[0])
        livingRmse = rmse(livingPrediction, all_test_labels[1])
        allKitchenRmse.append(kitchenRmse)
        allLivingRmse.append(livingRmse)

        globalStudentModel.clear()
        for i in range(len(trainFile) - 1):
            studentModel = keras.models.clone_model(studentModelTemplate)
            studentModel.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer='adadelta',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
            studentModel.set_weights(studentModelTemplate.get_weights())
            globalStudentModel.append(studentModel)
        print("Finished round {}".format(epoch+1))

    resultFile = pd.read_csv(metricFile)
    resultFile['kitchenRMSE'] = allKitchenRmse
    resultFile['livingRMSE'] = allLivingRmse
    resultFile.to_csv(metricFile, mode='w', header=True, index=False)

    idx = range(len(allKitchenRmse))
    plt.figure()
    plt.plot(idx, allKitchenRmse)
    plt.plot(idx, allLivingRmse)
    plt.legend(['Kitchen', 'Living Room'])
    plt.title('Federated Learning Scheme')
    plt.show()
        
    print("Federated learning section finished.")
    #model.evaluate()
    
if __name__ == "__main__":
    trainModel()